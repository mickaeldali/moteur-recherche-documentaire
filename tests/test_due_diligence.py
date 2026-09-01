import json

import clauses
import database
import due_diligence
import embeddings


# ---------------------------------------------------------------------------
# La liste des 24 catégories : un test structurel, pas de logique.
# ---------------------------------------------------------------------------

def test_categories_due_diligence_bien_formees():
    assert len(due_diligence.CATEGORIES_DUE_DILIGENCE) == 24

    noms = [categorie["nom"] for categorie in due_diligence.CATEGORIES_DUE_DILIGENCE]
    assert len(noms) == len(set(noms))  # pas de doublon

    for categorie in due_diligence.CATEGORIES_DUE_DILIGENCE:
        assert set(categorie.keys()) == {"nom", "verification", "red_flags"}
        assert categorie["nom"].strip()
        assert categorie["verification"].strip()
        assert categorie["red_flags"].strip()


def test_noms_categories_correspond_a_categories_due_diligence():
    assert due_diligence.NOMS_CATEGORIES == [c["nom"] for c in due_diligence.CATEGORIES_DUE_DILIGENCE]


# ---------------------------------------------------------------------------
# analyser_case : wrapper de clauses.detecter_et_analyser, avec un filet de
# sécurité supplémentaire contre toute exception imprévue.
# ---------------------------------------------------------------------------

def test_analyser_case_transmet_les_champs_de_la_categorie(monkeypatch):
    appels = {}

    def faux_detecter_et_analyser(doc_id, type_clause, ce_qu_il_faut_verifier=None, red_flags=None):
        appels["doc_id"] = doc_id
        appels["type_clause"] = type_clause
        appels["ce_qu_il_faut_verifier"] = ce_qu_il_faut_verifier
        appels["red_flags"] = red_flags
        return {"statut": "absente", "doc_id": doc_id, "type_clause": type_clause,
                "clause_presente": False, "texte_extrait": None, "analyse": None, "niveau_risque": None}

    monkeypatch.setattr(clauses, "detecter_et_analyser", faux_detecter_et_analyser)

    categorie = due_diligence.CATEGORIES_DUE_DILIGENCE[0]
    due_diligence.analyser_case(42, categorie)

    assert appels["doc_id"] == 42
    assert appels["type_clause"] == categorie["nom"]
    assert appels["ce_qu_il_faut_verifier"] == categorie["verification"]
    assert appels["red_flags"] == categorie["red_flags"]


def test_analyser_case_ne_leve_jamais_meme_sur_bug_imprevu(monkeypatch):
    def detecter_qui_plante(*args, **kwargs):
        raise RuntimeError("bug totalement imprévu")
    monkeypatch.setattr(clauses, "detecter_et_analyser", detecter_qui_plante)

    resultat = due_diligence.analyser_case(1, due_diligence.CATEGORIES_DUE_DILIGENCE[0])

    assert resultat["statut"] == "detection_indisponible"
    assert "erreur" in resultat


# ---------------------------------------------------------------------------
# lancer_analyse_data_room : ThreadPoolExecutor sur une petite liste de
# documents/catégories factices. clauses.detecter_et_analyser est mockée
# (pas d'appel réseau réel) ; database.lire_documents/enregistrer_analyse_clause
# passent par une vraie base de test temporaire.
# ---------------------------------------------------------------------------

def test_lancer_analyse_data_room_traite_toutes_les_combinaisons(tmp_path, monkeypatch):
    base_test = tmp_path / "test_documents.db"
    monkeypatch.setattr(database, "DB_PATH", base_test)
    database.initialiser_base()

    id1 = database.ajouter_document("A.pdf", "pdf", "Contrats", "documents/A.pdf")
    id2 = database.ajouter_document("B.pdf", "pdf", "Contrats", "documents/B.pdf")

    # Catégories factices, réduites à 2 pour un test rapide.
    monkeypatch.setattr(due_diligence, "CATEGORIES_DUE_DILIGENCE", [
        {"nom": "Résiliation", "verification": "v1", "red_flags": "r1"},
        {"nom": "Confidentialité", "verification": "v2", "red_flags": "r2"},
    ])

    def faux_detecter_et_analyser(doc_id, type_clause, ce_qu_il_faut_verifier=None, red_flags=None):
        return {
            "statut": "presente", "doc_id": doc_id, "type_clause": type_clause,
            "clause_presente": True, "texte_extrait": "extrait", "analyse": "analyse", "niveau_risque": "faible",
        }
    monkeypatch.setattr(clauses, "detecter_et_analyser", faux_detecter_et_analyser)

    progressions = []
    resultats = due_diligence.lancer_analyse_data_room(
        max_workers=2, on_resultat=lambda fait, total: progressions.append((fait, total)),
    )

    assert len(resultats) == 4  # 2 documents x 2 catégories
    assert progressions[-1] == (4, 4)

    analyses = database.lire_analyses_clauses()
    assert len(analyses) == 4


def test_lancer_analyse_data_room_resilience_du_lot(tmp_path, monkeypatch):
    # Une des tâches échoue (détection indisponible) : les autres doivent
    # quand même s'exécuter et s'enregistrer normalement.
    base_test = tmp_path / "test_documents.db"
    monkeypatch.setattr(database, "DB_PATH", base_test)
    database.initialiser_base()

    id1 = database.ajouter_document("A.pdf", "pdf", "Contrats", "documents/A.pdf")

    monkeypatch.setattr(due_diligence, "CATEGORIES_DUE_DILIGENCE", [
        {"nom": "Résiliation", "verification": "v1", "red_flags": "r1"},
        {"nom": "Confidentialité", "verification": "v2", "red_flags": "r2"},
        {"nom": "Audit", "verification": "v3", "red_flags": "r3"},
    ])

    def faux_detecter_et_analyser(doc_id, type_clause, ce_qu_il_faut_verifier=None, red_flags=None):
        if type_clause == "Confidentialité":
            raise RuntimeError("panne réseau simulée")
        return {
            "statut": "absente", "doc_id": doc_id, "type_clause": type_clause,
            "clause_presente": False, "texte_extrait": None, "analyse": None, "niveau_risque": None,
        }
    monkeypatch.setattr(clauses, "detecter_et_analyser", faux_detecter_et_analyser)

    resultats = due_diligence.lancer_analyse_data_room(max_workers=2)

    assert len(resultats) == 3  # les 3 tâches ont bien un résultat
    statuts = [r["statut"] for r in resultats]
    assert statuts.count("detection_indisponible") == 1
    assert statuts.count("absente") == 2

    # Les 2 tâches réussies sont bien enregistrées ; celle en échec ne l'est pas.
    analyses = database.lire_analyses_clauses()
    assert len(analyses) == 2


# ---------------------------------------------------------------------------
# construire_matrice : fonction pure, aucun accès base/réseau.
# ---------------------------------------------------------------------------

def _fausse_ligne_analyse(nom, type_clause, presente, niveau_risque, date_analyse, id_analyse=1):
    return (id_analyse, nom, type_clause, int(presente), "extrait" if presente else None,
            "analyse" if niveau_risque else None, niveau_risque, date_analyse)


def test_construire_matrice_cellules_selon_le_statut():
    analyses = [
        _fausse_ligne_analyse("A.pdf", "Résiliation", True, "élevé", "2026-09-01 10:00:00"),
        _fausse_ligne_analyse("A.pdf", "Confidentialité", False, None, "2026-09-01 10:00:00"),
        _fausse_ligne_analyse("A.pdf", "Audit", True, None, "2026-09-01 10:00:00"),
        # "Compliance" jamais analysée pour A.pdf -> "—"
    ]

    matrice = due_diligence.construire_matrice(
        analyses, noms_categories=["Résiliation", "Confidentialité", "Audit", "Compliance"],
    )

    assert matrice.loc["A.pdf", "Résiliation"] == "Élevé"
    assert matrice.loc["A.pdf", "Confidentialité"] == "Absente"
    assert matrice.loc["A.pdf", "Audit"] == "Détectée (non analysée)"
    assert matrice.loc["A.pdf", "Compliance"] == "—"


def test_construire_matrice_garde_seulement_la_plus_recente(monkeypatch):
    # Deux analyses pour le même document + même catégorie (relance du
    # scan) : seule la plus récente (déjà en tête, la liste est triée par
    # database.lire_analyses_clauses) doit compter.
    analyses = [
        _fausse_ligne_analyse("A.pdf", "Résiliation", True, "faible", "2026-09-01 11:00:00", id_analyse=2),
        _fausse_ligne_analyse("A.pdf", "Résiliation", True, "élevé", "2026-09-01 10:00:00", id_analyse=1),
    ]

    matrice = due_diligence.construire_matrice(analyses, noms_categories=["Résiliation"])

    assert matrice.loc["A.pdf", "Résiliation"] == "Faible"


# ---------------------------------------------------------------------------
# agreger_resultats_data_room / generer_synthese
# ---------------------------------------------------------------------------

def test_agreger_resultats_data_room_utilise_compter_analyses_par_risque(tmp_path, monkeypatch):
    base_test = tmp_path / "test_documents.db"
    monkeypatch.setattr(database, "DB_PATH", base_test)
    database.initialiser_base()

    id1 = database.ajouter_document("A.pdf", "pdf", "Contrats", "documents/A.pdf")
    database.enregistrer_analyse_clause(id1, "Résiliation", True, "extrait", "analyse", "élevé")
    database.enregistrer_analyse_clause(id1, "recherche libre hors 24 categories", True, "x", "y", "moyen")

    texte = due_diligence.agreger_resultats_data_room()

    assert "Résiliation" in texte
    assert "élevé" in texte
    assert "recherche libre hors 24 categories" not in texte


def test_construire_messages_synthese_contient_le_comptage():
    messages = due_diligence.construire_messages_synthese("Résiliation : 3 document(s) -- élevé")

    assert messages[0]["role"] == "system"
    assert "Résiliation : 3 document(s) -- élevé" in messages[1]["content"]


def test_generer_synthese_sans_cle_api_leve_une_erreur(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    try:
        due_diligence.generer_synthese("comptage factice")
        assert False, "CleApiManquante aurait dû être levée"
    except embeddings.CleApiManquante:
        pass


def test_generer_synthese_appelle_l_api_sans_schema_json(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "cle-de-test")

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
        def create(self, **kwargs):
            appels.update(kwargs)
            return FausseReponse("Synthèse rédigée du data room.")

    class FauxChat:
        def __init__(self):
            self.completions = FauxCompletions()

    class FauxClient:
        def __init__(self, api_key):
            self.chat = FauxChat()

    monkeypatch.setattr(embeddings, "OpenAI", FauxClient)

    resultat = due_diligence.generer_synthese("Résiliation : 3 document(s) -- élevé")

    assert resultat == "Synthèse rédigée du data room."
    assert "response_format" not in appels  # pas de schema JSON impose ici
