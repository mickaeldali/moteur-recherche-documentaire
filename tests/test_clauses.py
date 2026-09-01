import json

import clauses
import database
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


def test_construire_messages_inclut_verification_et_red_flags_si_fournis():
    chunks = [_faux_resultat("Contrat.pdf", "texte", 0.5)]

    messages = clauses.construire_messages_clause(
        "résiliation", chunks,
        ce_qu_il_faut_verifier="préavis, motifs de résiliation",
        red_flags="résiliation unilatérale sans préavis",
    )

    assert "préavis, motifs de résiliation" in messages[1]["content"]
    assert "résiliation unilatérale sans préavis" in messages[1]["content"]


# ---------------------------------------------------------------------------
# detecter_et_analyser : orchestration des deux étapes, ne lève jamais
# d'exception. database.rechercher_par_sens est mockée directement (pas de
# vraie base SQLite nécessaire pour ces tests).
# ---------------------------------------------------------------------------

def test_detecter_et_analyser_detection_indisponible(monkeypatch):
    def fausse_recherche(*args, **kwargs):
        raise embeddings.CleApiManquante("pas de clé")
    monkeypatch.setattr(database, "rechercher_par_sens", fausse_recherche)

    resultat = clauses.detecter_et_analyser(1, "non-concurrence")

    assert resultat["statut"] == "detection_indisponible"
    assert "erreur" in resultat


def test_detecter_et_analyser_absente_aucun_candidat(monkeypatch):
    monkeypatch.setattr(database, "rechercher_par_sens", lambda *a, **k: [])

    resultat = clauses.detecter_et_analyser(1, "confidentialité")

    assert resultat["statut"] == "absente"
    assert resultat["clause_presente"] is False
    assert resultat["texte_extrait"] is None


def test_detecter_et_analyser_absente_rejetee_par_verification(monkeypatch):
    chunks = [_faux_resultat("Contrat.pdf", "texte hors sujet", 0.4)]
    monkeypatch.setattr(database, "rechercher_par_sens", lambda *a, **k: chunks)
    _mocker_client_openai(monkeypatch, {
        "clause_correspond": False,
        "analyse": "Ne correspond pas.",
        "niveau_risque": "faible",
    })

    resultat = clauses.detecter_et_analyser(1, "propriété intellectuelle")

    assert resultat["statut"] == "absente"
    assert resultat["clause_presente"] is False


def test_detecter_et_analyser_presente(monkeypatch):
    chunks = [_faux_resultat("Contrat.pdf", "Duree de douze mois de non-concurrence.", 0.5)]
    monkeypatch.setattr(database, "rechercher_par_sens", lambda *a, **k: chunks)
    _mocker_client_openai(monkeypatch, {
        "clause_correspond": True,
        "analyse": "Clause large, risque important.",
        "niveau_risque": "élevé",
    })

    resultat = clauses.detecter_et_analyser(1, "non-concurrence")

    assert resultat["statut"] == "presente"
    assert resultat["clause_presente"] is True
    assert resultat["texte_extrait"] == "Duree de douze mois de non-concurrence."
    assert resultat["analyse"] == "Clause large, risque important."
    assert resultat["niveau_risque"] == "élevé"


def test_detecter_et_analyser_detectee_non_analysee(monkeypatch):
    chunks = [_faux_resultat("Contrat.pdf", "texte candidat", 0.5)]
    monkeypatch.setattr(database, "rechercher_par_sens", lambda *a, **k: chunks)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    resultat = clauses.detecter_et_analyser(1, "non-concurrence")

    assert resultat["statut"] == "detectee_non_analysee"
    assert resultat["clause_presente"] is True
    assert resultat["texte_extrait"] == "texte candidat"
    assert resultat["analyse"] is None
    assert resultat["niveau_risque"] is None
    assert "erreur" in resultat


def test_detecter_et_analyser_transmet_verification_et_red_flags(monkeypatch):
    chunks = [_faux_resultat("Contrat.pdf", "texte", 0.5)]
    monkeypatch.setattr(database, "rechercher_par_sens", lambda *a, **k: chunks)
    appels = {}
    _mocker_client_openai(monkeypatch, {
        "clause_correspond": True,
        "analyse": "ok",
        "niveau_risque": "faible",
    }, appels=appels)

    clauses.detecter_et_analyser(
        1, "résiliation",
        ce_qu_il_faut_verifier="préavis", red_flags="résiliation abusive",
    )

    contenu_envoye = appels["messages"][1]["content"]
    assert "préavis" in contenu_envoye
    assert "résiliation abusive" in contenu_envoye
