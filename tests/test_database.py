import database
import embeddings


def test_ajouter_et_lire_document(tmp_path, monkeypatch):
    base_test = tmp_path / "test_documents.db"
    monkeypatch.setattr(database, "DB_PATH", base_test)

    database.initialiser_base()
    id_document = database.ajouter_document(
        nom="Contrat_test.pdf",
        type_fichier="pdf",
        categorie="Contrats",
        chemin="documents/Contrat_test.pdf",
        commentaire="Document de test",
    )

    documents = database.lire_documents()

    assert len(documents) == 1
    assert documents[0][1] == "Contrat_test.pdf"
    assert documents[0][3] == "Contrats"
    assert id_document == documents[0][0]


def test_commentaire_optionnel(tmp_path, monkeypatch):
    base_test = tmp_path / "test_documents.db"
    monkeypatch.setattr(database, "DB_PATH", base_test)

    database.initialiser_base()
    database.ajouter_document("Statuts.docx", "docx", "Corporate", "documents/Statuts.docx")

    documents = database.lire_documents()
    assert documents[0][6] is None


def test_modifier_document(tmp_path, monkeypatch):
    base_test = tmp_path / "test_documents.db"
    monkeypatch.setattr(database, "DB_PATH", base_test)

    database.initialiser_base()
    database.ajouter_document("Doc.txt", "txt", "Autre", "documents/Doc.txt")
    id_doc = database.lire_documents()[0][0]

    database.modifier_document(id_doc, "Fiscal", "Commentaire modifié")
    documents = database.lire_documents()

    assert documents[0][3] == "Fiscal"
    assert documents[0][6] == "Commentaire modifié"


def test_supprimer_document(tmp_path, monkeypatch):
    base_test = tmp_path / "test_documents.db"
    monkeypatch.setattr(database, "DB_PATH", base_test)

    database.initialiser_base()
    id_document = database.ajouter_document("Doc.txt", "txt", "Autre", "documents/Doc.txt")
    database.indexer_document(id_document, "Doc.txt", "un peu de texte")

    database.supprimer_document(id_document)
    documents = database.lire_documents()

    assert len(documents) == 0
    # l'index doit aussi être nettoyé : une recherche ne doit plus rien trouver
    assert database.rechercher("texte") == []


def test_indexer_et_rechercher(tmp_path, monkeypatch):
    base_test = tmp_path / "test_documents.db"
    monkeypatch.setattr(database, "DB_PATH", base_test)

    database.initialiser_base()
    id_document = database.ajouter_document("Bail.pdf", "pdf", "Immobilier", "documents/Bail.pdf")
    database.indexer_document(
        id_document, "Bail.pdf", "Le présent bail commercial est conclu entre les parties."
    )

    resultats = database.rechercher("bail")

    assert len(resultats) == 1
    assert resultats[0][0] == id_document
    assert resultats[0][7] is not None  # un extrait est retourné


def test_rechercher_par_prefixe(tmp_path, monkeypatch):
    base_test = tmp_path / "test_documents.db"
    monkeypatch.setattr(database, "DB_PATH", base_test)

    database.initialiser_base()
    id_document = database.ajouter_document("Contrat.pdf", "pdf", "Contrats", "documents/Contrat.pdf")
    database.indexer_document(id_document, "Contrat.pdf", "Clause de non-concurrence contractuelle")

    # "concurr" doit retrouver "concurrence" grâce à la recherche par préfixe
    resultats = database.rechercher("concurr")

    assert len(resultats) == 1


def test_rechercher_filtre_par_categorie(tmp_path, monkeypatch):
    base_test = tmp_path / "test_documents.db"
    monkeypatch.setattr(database, "DB_PATH", base_test)

    database.initialiser_base()
    id1 = database.ajouter_document("Bail.pdf", "pdf", "Immobilier", "documents/Bail.pdf")
    database.indexer_document(id1, "Bail.pdf", "clause de résiliation du bail")
    id2 = database.ajouter_document("Contrat.pdf", "pdf", "Contrats", "documents/Contrat.pdf")
    database.indexer_document(id2, "Contrat.pdf", "clause de résiliation du contrat")

    resultats = database.rechercher("résiliation", categorie="Immobilier")

    assert len(resultats) == 1
    assert resultats[0][0] == id1


def test_rechercher_sans_texte_retourne_tout(tmp_path, monkeypatch):
    base_test = tmp_path / "test_documents.db"
    monkeypatch.setattr(database, "DB_PATH", base_test)

    database.initialiser_base()
    database.ajouter_document("Doc.txt", "txt", "Autre", "documents/Doc.txt")

    resultats = database.rechercher("")
    assert len(resultats) == 1
    assert resultats[0][7] is None  # pas d'extrait sans recherche


# ---------------------------------------------------------------------------
# Recherche par sens (embeddings). L'appel à l'API OpenAI est mocké via
# embeddings.calculer_embedding : ces tests ne font aucun appel réseau réel.
# ---------------------------------------------------------------------------

def test_indexer_embeddings_stocke_un_vecteur_par_chunk(tmp_path, monkeypatch):
    base_test = tmp_path / "test_documents.db"
    monkeypatch.setattr(database, "DB_PATH", base_test)
    monkeypatch.setattr(embeddings, "calculer_embedding", lambda texte: [1.0, 0.0])

    database.initialiser_base()
    id_document = database.ajouter_document("Bail.pdf", "pdf", "Immobilier", "documents/Bail.pdf")

    nb_chunks = database.indexer_embeddings(
        id_document, "Le bail commercial est conclu entre les parties."
    )

    assert nb_chunks == 1


def test_indexer_embeddings_texte_vide_ne_fait_rien(tmp_path, monkeypatch):
    base_test = tmp_path / "test_documents.db"
    monkeypatch.setattr(database, "DB_PATH", base_test)
    monkeypatch.setattr(embeddings, "calculer_embedding", lambda texte: [1.0, 0.0])

    database.initialiser_base()
    id_document = database.ajouter_document("Doc.txt", "txt", "Autre", "documents/Doc.txt")

    nb_chunks = database.indexer_embeddings(id_document, "")

    assert nb_chunks == 0


def test_rechercher_par_sens_retourne_le_plus_proche(tmp_path, monkeypatch):
    base_test = tmp_path / "test_documents.db"
    monkeypatch.setattr(database, "DB_PATH", base_test)

    database.initialiser_base()
    id_bail = database.ajouter_document("Bail.pdf", "pdf", "Immobilier", "documents/Bail.pdf")
    id_societe = database.ajouter_document("Statuts.docx", "docx", "Corporate", "documents/Statuts.docx")

    # Vecteurs orthogonaux pour bien distinguer les deux documents entre eux.
    monkeypatch.setattr(embeddings, "calculer_embedding", lambda texte: [1.0, 0.0])
    database.indexer_embeddings(id_bail, "Clause de résiliation du bail commercial.")

    monkeypatch.setattr(embeddings, "calculer_embedding", lambda texte: [0.0, 1.0])
    database.indexer_embeddings(id_societe, "Répartition du capital social entre associés.")

    # La requête est volontairement proche du vecteur du document "bail".
    monkeypatch.setattr(embeddings, "calculer_embedding", lambda texte: [0.9, 0.1])
    resultats = database.rechercher_par_sens("résiliation de bail")

    assert len(resultats) == 2
    assert resultats[0][0] == id_bail  # le plus proche en sens arrive en premier
    assert resultats[0][-1] > resultats[1][-1]  # score (dernière colonne) décroissant


def test_rechercher_par_sens_filtre_par_categorie(tmp_path, monkeypatch):
    base_test = tmp_path / "test_documents.db"
    monkeypatch.setattr(database, "DB_PATH", base_test)
    monkeypatch.setattr(embeddings, "calculer_embedding", lambda texte: [1.0, 0.0])

    database.initialiser_base()
    id_bail = database.ajouter_document("Bail.pdf", "pdf", "Immobilier", "documents/Bail.pdf")
    id_societe = database.ajouter_document("Statuts.docx", "docx", "Corporate", "documents/Statuts.docx")
    database.indexer_embeddings(id_bail, "Clause de résiliation du bail commercial.")
    database.indexer_embeddings(id_societe, "Répartition du capital social entre associés.")

    resultats = database.rechercher_par_sens("résiliation", categorie="Immobilier")

    assert len(resultats) == 1
    assert resultats[0][0] == id_bail


def test_rechercher_par_sens_texte_vide_retourne_rien(tmp_path, monkeypatch):
    base_test = tmp_path / "test_documents.db"
    monkeypatch.setattr(database, "DB_PATH", base_test)

    database.initialiser_base()
    # Aucun mock de calculer_embedding : une requête vide ne doit déclencher
    # aucun appel API, la fonction doit s'arrêter avant.
    assert database.rechercher_par_sens("") == []
    assert database.rechercher_par_sens("   ") == []


def test_rechercher_par_sens_filtre_par_doc_id(tmp_path, monkeypatch):
    # Utilisé par le projet 5 : détecter une clause dans UN document précis,
    # pas dans tout le corpus.
    base_test = tmp_path / "test_documents.db"
    monkeypatch.setattr(database, "DB_PATH", base_test)
    monkeypatch.setattr(embeddings, "calculer_embedding", lambda texte: [1.0, 0.0])

    database.initialiser_base()
    id_bail = database.ajouter_document("Bail.pdf", "pdf", "Immobilier", "documents/Bail.pdf")
    id_contrat = database.ajouter_document("Contrat.pdf", "pdf", "Contrats", "documents/Contrat.pdf")
    database.indexer_embeddings(id_bail, "Clause de resiliation du bail.")
    database.indexer_embeddings(id_contrat, "Clause de resiliation du contrat.")

    resultats = database.rechercher_par_sens("resiliation", doc_id=id_contrat)

    assert len(resultats) == 1
    assert resultats[0][0] == id_contrat


# ---------------------------------------------------------------------------
# Historique des analyses de clauses (projet 5). Aucun appel réseau.
# ---------------------------------------------------------------------------

def test_enregistrer_et_lire_analyse_clause_presente(tmp_path, monkeypatch):
    base_test = tmp_path / "test_documents.db"
    monkeypatch.setattr(database, "DB_PATH", base_test)

    database.initialiser_base()
    id_doc = database.ajouter_document("Contrat.pdf", "pdf", "Contrats", "documents/Contrat.pdf")

    database.enregistrer_analyse_clause(
        id_doc, "non-concurrence", clause_presente=True,
        texte_extrait="Le salarié ne peut pas travailler pour un concurrent.",
        analyse="Clause large, sans contrepartie financière.",
        niveau_risque="élevé",
    )

    analyses = database.lire_analyses_clauses(id_doc)

    assert len(analyses) == 1
    assert analyses[0][1] == "Contrat.pdf"
    assert analyses[0][2] == "non-concurrence"
    assert analyses[0][3] == 1  # clause_presente stockée en INTEGER
    assert analyses[0][6] == "élevé"


def test_enregistrer_analyse_clause_absente_sans_analyse(tmp_path, monkeypatch):
    base_test = tmp_path / "test_documents.db"
    monkeypatch.setattr(database, "DB_PATH", base_test)

    database.initialiser_base()
    id_doc = database.ajouter_document("Contrat.pdf", "pdf", "Contrats", "documents/Contrat.pdf")

    database.enregistrer_analyse_clause(
        id_doc, "confidentialité", clause_presente=False, texte_extrait=None,
    )

    analyses = database.lire_analyses_clauses(id_doc)

    assert len(analyses) == 1
    assert analyses[0][3] == 0
    assert analyses[0][4] is None  # texte_extrait
    assert analyses[0][5] is None  # analyse
    assert analyses[0][6] is None  # niveau_risque


def test_lire_analyses_clauses_sans_doc_id_retourne_tout(tmp_path, monkeypatch):
    base_test = tmp_path / "test_documents.db"
    monkeypatch.setattr(database, "DB_PATH", base_test)

    database.initialiser_base()
    id_bail = database.ajouter_document("Bail.pdf", "pdf", "Immobilier", "documents/Bail.pdf")
    id_contrat = database.ajouter_document("Contrat.pdf", "pdf", "Contrats", "documents/Contrat.pdf")
    database.enregistrer_analyse_clause(id_bail, "résiliation", True, "extrait bail")
    database.enregistrer_analyse_clause(id_contrat, "confidentialité", True, "extrait contrat")

    assert len(database.lire_analyses_clauses()) == 2
    assert len(database.lire_analyses_clauses(id_bail)) == 1


def test_supprimer_document_nettoie_aussi_les_analyses_de_clauses(tmp_path, monkeypatch):
    base_test = tmp_path / "test_documents.db"
    monkeypatch.setattr(database, "DB_PATH", base_test)

    database.initialiser_base()
    id_doc = database.ajouter_document("Contrat.pdf", "pdf", "Contrats", "documents/Contrat.pdf")
    database.enregistrer_analyse_clause(id_doc, "non-concurrence", True, "extrait")

    database.supprimer_document(id_doc)

    assert database.lire_analyses_clauses() == []
