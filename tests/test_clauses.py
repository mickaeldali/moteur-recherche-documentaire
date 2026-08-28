import json

import clauses
import embeddings
import llm


def _faux_resultat(nom, texte_chunk, score, id_doc=1):
    """Même forme que les tuples retournés par database.rechercher_par_sens()."""
    return (id_doc, nom, "pdf", "Contrats", "2026-08-28", "documents/x.pdf", None, texte_chunk, score)


def _mocker_client_openai(monkeypatch, contenu_dict, appels=None):
    """
    Simule embeddings.OpenAI(...).chat.completions.create(...) pour que
    clauses.analyser_clause() reçoive contenu_dict (encodé en JSON) sans
    aucun appel réseau réel. `appels`, si fourni, reçoit les arguments
    passés à create() pour vérification par le test appelant.
    """
    monkeypatch.setenv("OPENAI_API_KEY", "cle-de-test")
    contenu_json = json.dumps(contenu_dict)

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
            if appels is not None:
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


# ---------------------------------------------------------------------------
# construire_messages_clause : fonction pure, aucun appel API.
# ---------------------------------------------------------------------------

def test_construire_messages_contient_le_type_de_clause_et_le_contexte():
    chunks = [_faux_resultat("Contrat.pdf", "Le salarie ne peut pas travailler pour un concurrent.", 0.5)]

    messages = clauses.construire_messages_clause("non-concurrence", chunks)

    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"
    assert "non-concurrence" in messages[1]["content"]
    assert "Le salarie ne peut pas travailler pour un concurrent." in messages[1]["content"]


def test_construire_messages_numerote_plusieurs_extraits():
    chunks = [
        _faux_resultat("Contrat.pdf", "premier extrait", 0.5),
        _faux_resultat("Contrat.pdf", "deuxieme extrait", 0.4),
    ]

    messages = clauses.construire_messages_clause("confidentialité", chunks)

    assert "Extrait 1" in messages[1]["content"]
    assert "Extrait 2" in messages[1]["content"]


# ---------------------------------------------------------------------------
# analyser_clause : mocke l'appel à l'API OpenAI, aucun réseau nécessaire.
# ---------------------------------------------------------------------------

def test_analyser_clause_sans_cle_api_leve_une_erreur(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    chunks = [_faux_resultat("Contrat.pdf", "texte", 0.5)]

    try:
        clauses.analyser_clause("non-concurrence", chunks)
        assert False, "CleApiManquante aurait dû être levée"
    except embeddings.CleApiManquante:
        pass


def test_analyser_clause_appelle_l_api_et_parse_le_json(monkeypatch):
    appels = {}
    _mocker_client_openai(monkeypatch, {
        "clause_correspond": True,
        "analyse": "La clause interdit toute activité concurrente pendant 12 mois, sans contrepartie financière.",
        "niveau_risque": "élevé",
    }, appels=appels)

    chunks = [_faux_resultat("Contrat.pdf", "Le salarie ne peut pas travailler pour un concurrent.", 0.5)]
    resultat = clauses.analyser_clause("non-concurrence", chunks)

    assert appels["response_format"]["json_schema"] == clauses.SCHEMA_ANALYSE_CLAUSE
    assert appels["model"] == llm.MODELE_CHAT
    assert resultat["clause_correspond"] is True
    assert resultat["niveau_risque"] == "élevé"
    assert "activité concurrente" in resultat["analyse"]


def test_analyser_clause_signale_une_fausse_detection(monkeypatch):
    # Cas découvert en conditions réelles : la détection préliminaire (par
    # similarité de sens) peut retrouver un candidat qui ne correspond PAS
    # vraiment au type de clause demandé (deux clauses de même domaine
    # juridique). analyser_clause() doit pouvoir renvoyer clause_correspond
    # à False dans ce cas, pour que app.py traite la clause comme absente
    # plutôt que d'enregistrer une analyse hors sujet.
    _mocker_client_openai(monkeypatch, {
        "clause_correspond": False,
        "analyse": "Les extraits fournis portent sur la non-concurrence, pas sur la propriété intellectuelle.",
        "niveau_risque": "faible",
    })

    chunks = [_faux_resultat("Contrat.pdf", "Le salarie ne peut pas travailler pour un concurrent.", 0.4)]
    resultat = clauses.analyser_clause("propriété intellectuelle", chunks)

    assert resultat["clause_correspond"] is False
