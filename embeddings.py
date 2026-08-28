"""
embeddings.py

Recherche par sens : transforme un texte en vecteur numérique (embedding)
via l'API OpenAI, pour pouvoir comparer des textes par proximité de SENS
plutôt que par mots en commun (ce que fait déjà FTS5 dans recherche.py).

Un embedding est une liste de nombres (1536 pour le modèle utilisé ici) qui
représente le sens d'un texte : deux textes de sens proche ont des vecteurs
"proches" au sens mathématique, mesuré ici par la similarité cosinus. Ce
fichier isole tout ce qui touche à l'API OpenAI et au calcul numérique, pour
rester testable sans dépendre d'un vrai appel réseau payant (voir
tests/test_embeddings.py, qui mocke calculer_embedding).
"""

import os

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

MODELE_EMBEDDING = "text-embedding-3-small"

# Taille cible d'un chunk, en caractères, et chevauchement entre deux chunks
# consécutifs pour ne pas perdre une clause coupée en deux par le découpage.
TAILLE_CHUNK = 700
CHEVAUCHEMENT = 100

# Charge le fichier .env (à la racine du projet) dans les variables
# d'environnement, s'il existe. N'écrase jamais une variable déjà définie
# par le système.
load_dotenv()


class CleApiManquante(Exception):
    """Levée quand OPENAI_API_KEY n'est pas définie (ni dans .env, ni dans l'environnement)."""


def client_openai():
    """
    Construit un client OpenAI, ou lève CleApiManquante si aucune clé n'est
    configurée. Publique (pas de préfixe _) car réutilisée par llm.py, qui a
    besoin du même client pour appeler le modèle de chat : pas de raison de
    dupliquer la logique de lecture de la clé API à deux endroits.
    """
    cle = os.getenv("OPENAI_API_KEY")
    if not cle:
        raise CleApiManquante(
            "Aucune clé API OpenAI trouvée. Ajoute une ligne "
            "OPENAI_API_KEY=... dans le fichier .env à la racine du projet."
        )
    return OpenAI(api_key=cle)


def decouper_en_chunks(texte, taille=TAILLE_CHUNK, chevauchement=CHEVAUCHEMENT):
    """
    Découpe un texte en chunks (morceaux) d'environ `taille` caractères.

    Pourquoi découper : un embedding résume tout le texte qu'on lui donne en
    un seul vecteur. Sur un document entier de plusieurs pages, ce résumé
    serait trop dilué pour retrouver un passage précis (une clause au milieu
    d'un contrat, par exemple). En découpant en chunks plus petits, chaque
    vecteur représente un passage plus ciblé, et la recherche peut pointer
    directement dessus.

    Le découpage essaie de couper à une fin de phrase (. ! ?), sinon à un
    saut de ligne, sinon à un espace, pour ne jamais trancher en plein
    milieu d'un mot ou d'une phrase. `chevauchement` fait recommencer le
    chunk suivant un peu avant la fin du précédent, pour ne pas perdre une
    clause à cheval sur la coupure.

    Retourne une liste de chaînes de texte (liste vide si `texte` est vide).
    """
    texte = texte.strip()
    if not texte:
        return []

    if len(texte) <= taille:
        return [texte]

    chunks = []
    debut = 0
    while debut < len(texte):
        fin = debut + taille
        if fin >= len(texte):
            chunks.append(texte[debut:].strip())
            break

        fenetre = texte[debut:fin]
        coupure = max(fenetre.rfind(". "), fenetre.rfind("! "), fenetre.rfind("? "))
        if coupure == -1:
            coupure = fenetre.rfind("\n")
        if coupure == -1:
            coupure = fenetre.rfind(" ")
        if coupure == -1:
            # Aucun espace ni ponctuation dans la fenêtre : on tranche brut,
            # plutôt que de boucler indéfiniment.
            coupure = len(fenetre) - 1

        fin_reelle = debut + coupure + 1
        morceau = texte[debut:fin_reelle].strip()
        if morceau:
            chunks.append(morceau)

        nouveau_debut = fin_reelle - chevauchement
        # Le chevauchement ne doit jamais faire reculer `debut` : sinon la
        # boucle ne progresse plus et tourne indéfiniment.
        debut = nouveau_debut if nouveau_debut > debut else fin_reelle

    return chunks


def calculer_embedding(texte):
    """
    Appelle l'API OpenAI pour transformer un texte en vecteur numérique
    (liste de float).

    Peut lever CleApiManquante (pas de clé configurée) ou une exception
    venant de la librairie openai (pas de réseau, quota dépassé...). Ces
    deux cas sont volontairement laissés remonter : c'est à l'appelant
    (database.py, puis app.py) de décider comment réagir à un échec.
    """
    reponse = client_openai().embeddings.create(model=MODELE_EMBEDDING, input=texte)
    return reponse.data[0].embedding


def similarite_cosinus(vecteur1, vecteur2):
    """
    Mesure à quel point deux vecteurs "pointent dans la même direction",
    entre -1 (sens opposés) et 1 (identiques). Deux textes de sens proche
    ont des embeddings dont la similarité cosinus est proche de 1.

    Formule : produit scalaire des deux vecteurs, divisé par le produit de
    leurs normes (longueurs). numpy est utilisé pour la clarté et la
    rapidité du calcul, plutôt qu'une boucle Python manuelle.
    """
    v1 = np.array(vecteur1, dtype=float)
    v2 = np.array(vecteur2, dtype=float)

    norme1 = np.linalg.norm(v1)
    norme2 = np.linalg.norm(v2)
    if norme1 == 0 or norme2 == 0:
        return 0.0

    return float(np.dot(v1, v2) / (norme1 * norme2))
