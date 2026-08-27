"""
fichiers.py

Gestion physique des fichiers importés :
- sauvegarde d'un fichier dans le dossier documents/
- gestion des collisions de noms (deux fichiers importés avec le même nom)
- suppression d'un fichier physique

Identique au projet 2 : ce projet est indépendant et reprend telle quelle
la brique de stockage physique, déjà écrite et testée.
"""

from pathlib import Path

DOSSIER_DOCUMENTS = Path("documents")


def sauvegarder_fichier(nom_fichier, contenu_binaire):
    """
    Sauvegarde un fichier importé dans le dossier documents/.

    nom_fichier : str, ex. "Contrat.pdf" (nom original du fichier importé)
    contenu_binaire : bytes, le contenu du fichier

    Si un fichier du même nom existe déjà, un suffixe numérique est ajouté
    (Contrat.pdf -> Contrat_1.pdf -> Contrat_2.pdf ...) pour ne jamais écraser
    un document existant.

    Retourne le chemin (str) où le fichier a été sauvegardé.
    """
    DOSSIER_DOCUMENTS.mkdir(exist_ok=True)

    chemin = DOSSIER_DOCUMENTS / nom_fichier
    nom_sans_extension = chemin.stem
    extension = chemin.suffix
    compteur = 1

    while chemin.exists():
        nouveau_nom = f"{nom_sans_extension}_{compteur}{extension}"
        chemin = DOSSIER_DOCUMENTS / nouveau_nom
        compteur += 1

    with open(chemin, "wb") as f:
        f.write(contenu_binaire)

    return str(chemin)


def supprimer_fichier(chemin):
    """
    Supprime un fichier physique s'il existe.
    Ne provoque pas d'erreur si le fichier a déjà été supprimé manuellement.
    """
    chemin_obj = Path(chemin)
    if chemin_obj.exists():
        chemin_obj.unlink()
