import database


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
