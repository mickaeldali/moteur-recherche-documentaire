"""
export_excel.py

Petits utilitaires pour préparer des données avant un export Excel.

Problème résolu : Excel (via openpyxl) refuse dans une cellule certains
caractères de contrôle invisibles (codes ASCII 0 à 8, 11, 12 et 14 à 31).
Or le texte d'une analyse générée par le LLM, ou un extrait lu dans un PDF,
peut en contenir (ex. "\\x0b", une tabulation verticale). Sans nettoyage,
l'export plante avec `IllegalCharacterError`.

On nettoie UNIQUEMENT au moment de l'export : les données stockées en base
ne sont jamais modifiées.
"""

from openpyxl.cell.cell import ILLEGAL_CHARACTERS_RE


def nettoyer_texte_excel(valeur):
    """Supprime les caractères interdits par Excel d'une valeur.

    - Si `valeur` est du texte : retourne le texte sans les caractères
      illégaux. Les retours à la ligne, tabulations et accents sont
      conservés (ils ne font pas partie des caractères interdits).
    - Sinon (None, nombre...) : retourne la valeur inchangée.
    """
    if isinstance(valeur, str):
        return ILLEGAL_CHARACTERS_RE.sub("", valeur)
    return valeur


def nettoyer_dataframe_pour_excel(df):
    """Retourne une COPIE du DataFrame dont toutes les cellules texte sont
    nettoyées. Le DataFrame d'origine n'est pas modifié."""
    return df.map(nettoyer_texte_excel)
