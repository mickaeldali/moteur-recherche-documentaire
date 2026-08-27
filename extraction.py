"""
extraction.py

Extraction du texte des documents selon leur format (PDF, DOCX, TXT).

Identique au projet 2. C'est ce texte extrait qui sera indexé pour la
recherche plein texte (voir database.py / recherche.py) : sans cette brique,
il n'y aurait rien à indexer d'autre que les métadonnées.

Deux erreurs possibles, distinguées volontairement pour donner un message
clair à l'utilisateur :
- FormatNonSupporte : le type de fichier n'est pas géré par l'extraction.
- ErreurExtraction : le format est géré, mais l'extraction a échoué
  (fichier corrompu, illisible, introuvable...).
"""

from pathlib import Path
from pypdf import PdfReader
from docx import Document

FORMATS_GERES = ("pdf", "docx", "txt")


class FormatNonSupporte(Exception):
    pass


class ErreurExtraction(Exception):
    pass


def extraire_texte(chemin, type_fichier):
    """
    Extrait le texte d'un document.

    chemin : str, chemin du fichier sur le disque
    type_fichier : str, "pdf", "docx" ou "txt"

    Retourne une chaîne de caractères (peut être vide si le document ne
    contient pas de texte détectable, par ex. un PDF scanné en image).
    Lève FormatNonSupporte ou ErreurExtraction en cas de problème.
    """
    type_fichier = type_fichier.lower()

    if type_fichier not in FORMATS_GERES:
        raise FormatNonSupporte(f"Le format '{type_fichier}' n'est pas pris en charge pour l'extraction.")

    try:
        if type_fichier == "pdf":
            return _extraire_texte_pdf(chemin)
        elif type_fichier == "docx":
            return _extraire_texte_docx(chemin)
        else:
            return _extraire_texte_txt(chemin)
    except FileNotFoundError:
        raise ErreurExtraction("Le fichier est introuvable sur le disque.")
    except Exception as erreur:
        raise ErreurExtraction(f"Impossible d'extraire le texte de ce fichier : {erreur}")


def _extraire_texte_pdf(chemin):
    lecteur = PdfReader(chemin)
    morceaux = [page.extract_text() or "" for page in lecteur.pages]
    return "\n".join(morceaux)


def _extraire_texte_docx(chemin):
    document = Document(chemin)
    paragraphes = [p.text for p in document.paragraphs]
    return "\n".join(paragraphes)


def _extraire_texte_txt(chemin):
    return Path(chemin).read_text(encoding="utf-8", errors="ignore")
