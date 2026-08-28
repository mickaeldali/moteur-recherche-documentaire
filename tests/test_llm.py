import json

import embeddings
import llm


def _faux_resultat(nom, texte_chunk, score, id_doc=1):
    """
    Construit un tuple avec la même forme que ceux retournés par
    database.rechercher_par_sens() :
    (id, nom, type_fichier, categorie, date_ajout, chemin, commentaire, texte_chunk, score)
    """
    return (id_doc, nom, "pdf", "Contrats", "2026-08-28", "documents/x.pdf", None, texte_chunk, score)


# ---------------------------------------------------------------------------
# filtrer_chunks_pertinents : fonction pure, aucun appel API.
# ---------------------------------------------------------------------------

def test_filtrer_garde_les_scores_superieurs_au_seuil():
    resultats = [_faux_resultat("Bail.pdf", "texte", 0.5)]
    assert llm.filtrer_chunks_pertinents(resultats) == resultats


def test_filtrer_exclut_les_scores_inferieurs_au_seuil():
    resultats = [_faux_resultat("Bail.pdf", "texte", 0.05)]
    assert llm.filtrer_chunks_pertinents(resultats) == []


def test_filtrer_garde_exactement_au_seuil():
    resultats = [_faux_resultat("Bail.pdf", "texte", llm.SEUIL_SIMILARITE_MINIMAL)]
    assert llm.filtrer_chunks_pertinents(resultats) == resultats


def test_filtrer_melange_scores_hauts_et_bas():
    pertinent = _faux_resultat("Bail.pdf", "clause pertinente", 0.4)
    non_pertinent = _faux_resultat("Statuts.docx", "hors sujet", 0.02)
    assert llm.filtrer_chunks_pertinents([pertinent, non_pertinent]) == [pertinent]


# ---------------------------------------------------------------------------
# construire_messages : fonction pure, aucun appel API.
# ---------------------------------------------------------------------------

def test_construire_messages_contient_la_question_et_le_contexte():
    chunks = [_faux_resultat("Bail.pdf", "Clause de résiliation du bail.", 0.6)]

    messages = llm.construire_messages("Que dit la clause de résiliation ?", chunks)

    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"
    assert "Que dit la clause de résiliation ?" in messages[1]["content"]
    assert "Clause de résiliation du bail." in messages[1]["content"]
    assert "Bail.pdf" in messages[1]["content"]


def test_construire_messages_numerote_plusieurs_passages():
    chunks = [
        _faux_resultat("Bail.pdf", "premier passage", 0.6),
        _faux_resultat("Contrat.pdf", "deuxieme passage", 0.5),
    ]

    messages = llm.construire_messages("question", chunks)

    assert "Passage 1" in messages[1]["content"]
    assert "Passage 2" in messages[1]["content"]


# ---------------------------------------------------------------------------
# generer_reponse : mocke l'appel à l'API OpenAI, aucun réseau nécessaire.
# ---------------------------------------------------------------------------

def test_generer_reponse_sans_cle_api_leve_une_erreur(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    chunks = [_faux_resultat("Bail.pdf", "texte", 0.6)]

    try:
        llm.generer_reponse("question", chunks)
        assert False, "CleApiManquante aurait dû être levée"
    except embeddings.CleApiManquante:
        pass


def test_generer_reponse_appelle_l_api_et_parse_le_json(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "cle-de-test")

    contenu_json = json.dumps({
        "reponse_document": "Le bail peut être résilié avec un préavis de trois mois.",
        "complement_connaissance_generale": None,
    })

    class FauxMessage:
        def __init__(self, content):
            self.content = content

    class FauxChoix:
        def __init__(self, content):
            self.message = FauxMessage(content)

    class FausseReponse:
        def __init__(self, content):
            self.choices = [FauxChoix(content)]

    appels = {}

    class FauxCompletions:
        def create(self, model, messages, response_format):
            appels["model"] = model
            appels["response_format"] = response_format
            appels["messages"] = messages
            return FausseReponse(contenu_json)

    class FauxChat:
        def __init__(self):
            self.completions = FauxCompletions()

    class FauxClient:
        def __init__(self, api_key):
            self.chat = FauxChat()

    monkeypatch.setattr(embeddings, "OpenAI", FauxClient)

    chunks = [_faux_resultat("Bail.pdf", "Clause de résiliation.", 0.6)]
    resultat = llm.generer_reponse("Comment résilier le bail ?", chunks)

    assert appels["model"] == llm.MODELE_CHAT
    assert appels["response_format"]["json_schema"] == llm.SCHEMA_REPONSE
    assert resultat["reponse_document"] == "Le bail peut être résilié avec un préavis de trois mois."
    assert resultat["complement_connaissance_generale"] is None


def test_generer_reponse_normalise_la_chaine_null_en_vrai_none(monkeypatch):
    # Certains modèles renvoient parfois la CHAÎNE "null" au lieu du JSON
    # null attendu par le schéma : ce cas a été observé en conditions
    # réelles et doit être normalisé, sinon l'app afficherait le mot "null"
    # à l'utilisateur comme s'il s'agissait d'un vrai complément.
    monkeypatch.setenv("OPENAI_API_KEY", "cle-de-test")

    contenu_json = json.dumps({
        "reponse_document": "Réponse basée sur le document.",
        "complement_connaissance_generale": "null",
    })

    class FauxMessage:
        def __init__(self, content):
            self.content = content

    class FauxChoix:
        def __init__(self, content):
            self.message = FauxMessage(content)

    class FausseReponse:
        def __init__(self, content):
            self.choices = [FauxChoix(content)]

    class FauxCompletions:
        def create(self, model, messages, response_format):
            return FausseReponse(contenu_json)

    class FauxChat:
        def __init__(self):
            self.completions = FauxCompletions()

    class FauxClient:
        def __init__(self, api_key):
            self.chat = FauxChat()

    monkeypatch.setattr(embeddings, "OpenAI", FauxClient)

    chunks = [_faux_resultat("Bail.pdf", "texte", 0.6)]
    resultat = llm.generer_reponse("question", chunks)

    assert resultat["complement_connaissance_generale"] is None
