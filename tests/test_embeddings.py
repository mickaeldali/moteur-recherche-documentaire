import numpy as np

import embeddings


# ---------------------------------------------------------------------------
# decouper_en_chunks : ne fait aucun appel API, testable directement.
# ---------------------------------------------------------------------------

def test_decouper_texte_vide():
    assert embeddings.decouper_en_chunks("") == []
    assert embeddings.decouper_en_chunks("   ") == []


def test_decouper_texte_court_ne_fait_qu_un_chunk():
    texte = "Le présent contrat est conclu entre les parties."
    assert embeddings.decouper_en_chunks(texte, taille=700) == [texte]


def test_decouper_texte_long_produit_plusieurs_chunks():
    # Un texte fait de nombreuses phrases courtes, largement plus long que
    # la taille cible : doit être découpé en plusieurs morceaux.
    phrase = "Ceci est une phrase de test pour la découpe en chunks. "
    texte = phrase * 40  # largement plus long que 700 caractères

    chunks = embeddings.decouper_en_chunks(texte, taille=200, chevauchement=20)

    assert len(chunks) > 1
    # Aucun chunk ne doit dépasser largement la taille demandée.
    assert all(len(chunk) <= 250 for chunk in chunks)
    # Rien n'est perdu : chaque chunk est un texte non vide.
    assert all(chunk.strip() for chunk in chunks)


def test_decouper_ne_boucle_pas_indefiniment_sans_espaces():
    # Texte long sans aucun espace ni ponctuation : cas limite où la
    # fonction doit quand même terminer (le test échouerait par timeout
    # sinon) et produire plusieurs chunks non vides.
    texte = "a" * 3000
    chunks = embeddings.decouper_en_chunks(texte, taille=200, chevauchement=20)
    assert len(chunks) > 1
    assert all(chunk.strip() for chunk in chunks)


# ---------------------------------------------------------------------------
# similarite_cosinus : calcul numpy pur, aucun appel API.
# ---------------------------------------------------------------------------

def test_similarite_cosinus_vecteurs_identiques():
    v = [1.0, 2.0, 3.0]
    assert embeddings.similarite_cosinus(v, v) == 1.0


def test_similarite_cosinus_vecteurs_opposes():
    v1 = [1.0, 0.0]
    v2 = [-1.0, 0.0]
    assert embeddings.similarite_cosinus(v1, v2) == -1.0


def test_similarite_cosinus_vecteurs_orthogonaux():
    v1 = [1.0, 0.0]
    v2 = [0.0, 1.0]
    assert embeddings.similarite_cosinus(v1, v2) == 0.0


def test_similarite_cosinus_vecteur_nul():
    # Un vecteur nul n'a pas de direction : par convention, similarité 0
    # plutôt qu'une division par zéro.
    v1 = [0.0, 0.0]
    v2 = [1.0, 1.0]
    assert embeddings.similarite_cosinus(v1, v2) == 0.0


def test_similarite_cosinus_coherente_avec_numpy():
    rng = np.random.default_rng(42)
    v1 = rng.random(10)
    v2 = rng.random(10)

    resultat = embeddings.similarite_cosinus(v1, v2)
    attendu = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

    assert resultat == attendu


# ---------------------------------------------------------------------------
# calculer_embedding : mocke l'appel à l'API OpenAI, aucun réseau nécessaire.
# ---------------------------------------------------------------------------

def test_calculer_embedding_sans_cle_api_leve_une_erreur(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    try:
        embeddings.calculer_embedding("texte quelconque")
        assert False, "CleApiManquante aurait dû être levée"
    except embeddings.CleApiManquante:
        pass


def test_calculer_embedding_appelle_l_api_openai(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "cle-de-test")

    class FauxeDonnee:
        def __init__(self, embedding):
            self.embedding = embedding

    class FausseReponse:
        def __init__(self, embedding):
            self.data = [FauxeDonnee(embedding)]

    class FauxEmbeddings:
        def create(self, model, input):
            assert model == embeddings.MODELE_EMBEDDING
            assert input == "clause de non-concurrence"
            return FausseReponse([0.1, 0.2, 0.3])

    class FauxClient:
        def __init__(self, api_key):
            self.embeddings = FauxEmbeddings()

    monkeypatch.setattr(embeddings, "OpenAI", FauxClient)

    resultat = embeddings.calculer_embedding("clause de non-concurrence")

    assert resultat == [0.1, 0.2, 0.3]
