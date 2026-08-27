"""
recherche.py

Construction de la requête de recherche plein texte à partir du texte tapé
par l'utilisateur.

Séparé de database.py pour deux raisons pédagogiques :
- pouvoir tester la construction de la requête sans toucher à SQLite ;
- isoler la seule partie du projet qui connaît la syntaxe particulière de
  FTS5 (Full-Text Search 5), le moteur de recherche plein texte intégré à
  SQLite. FTS5 construit un index inversé (mot -> documents qui le
  contiennent), ce qui permet de retrouver un mot dans de gros volumes de
  texte bien plus vite qu'en relisant chaque document avec un simple `in`.
  C'est une recherche "classique" : aucun modèle de langage, aucun vecteur,
  uniquement des mots comparés à des mots.
"""


def construire_requete_fts(texte_utilisateur):
    """
    Transforme le texte tapé par l'utilisateur en requête FTS5.

    Chaque mot est mis entre guillemets (pour être traité comme du texte
    littéral, pas comme un opérateur FTS5 tel que AND/OR/NOT) puis suivi
    de '*' pour activer la recherche par préfixe : taper "contr" retrouve
    "contrat", "contrats", "contractuel", etc. Les mots sont implicitement
    combinés par un ET logique, qui est le comportement par défaut de FTS5
    quand plusieurs termes sont séparés par un espace.

    Retourne None si le texte est vide (rien à rechercher).
    """
    mots = texte_utilisateur.strip().split()
    if not mots:
        return None

    mots_echappes = []
    for mot in mots:
        # Un guillemet double à l'intérieur d'un mot doit être doublé pour
        # être traité comme du texte littéral et non comme la fin de la
        # chaîne (règle d'échappement standard de FTS5).
        mot_echappe = mot.replace('"', '""')
        mots_echappes.append(f'"{mot_echappe}"*')

    return " ".join(mots_echappes)
