import recherche


def test_construire_requete_simple():
    assert recherche.construire_requete_fts("bail") == '"bail"*'


def test_construire_requete_plusieurs_mots():
    resultat = recherche.construire_requete_fts("clause résiliation")
    assert resultat == '"clause"* "résiliation"*'


def test_construire_requete_vide():
    assert recherche.construire_requete_fts("") is None
    assert recherche.construire_requete_fts("   ") is None


def test_construire_requete_echappe_les_guillemets():
    resultat = recherche.construire_requete_fts('bail "commercial"')
    assert resultat == '"bail"* """commercial"""*'
