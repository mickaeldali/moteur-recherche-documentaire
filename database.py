"""
database.py

Communication avec la base SQLite : métadonnées des documents (table
`documents`, identique au projet 2) et index de recherche plein texte
(table virtuelle FTS5 `documents_fts`, la nouveauté du projet 3).

Le fichier réel (PDF/DOCX/TXT) n'est jamais stocké ici : seule son adresse
(le champ `chemin`) est mémorisée. Le fichier physique est géré par
fichiers.py, le texte qu'il contient par extraction.py. `documents_fts`
mémorise ce texte extrait pour pouvoir le rechercher rapidement : c'est un
index séparé de la table `documents`, relié à elle par `doc_id`.
"""

import sqlite3
from pathlib import Path

import recherche

DB_PATH = Path("data") / "documents.db"


def initialiser_base():
    """Crée le dossier data/, la table documents et l'index FTS5 s'ils n'existent pas encore."""
    DB_PATH.parent.mkdir(exist_ok=True)
    connexion = sqlite3.connect(DB_PATH)
    curseur = connexion.cursor()

    curseur.execute("""
        CREATE TABLE IF NOT EXISTS documents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            nom TEXT NOT NULL,
            type_fichier TEXT,
            categorie TEXT,
            date_ajout TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            chemin TEXT NOT NULL,
            commentaire TEXT
        )
    """)

    # Table virtuelle FTS5 : chaque ligne indexe le contenu d'un document.
    # doc_id est UNINDEXED (stocké mais jamais recherché) : il sert juste de
    # référence vers la ligne correspondante dans `documents`.
    curseur.execute("""
        CREATE VIRTUAL TABLE IF NOT EXISTS documents_fts USING fts5(
            doc_id UNINDEXED,
            nom,
            contenu
        )
    """)

    connexion.commit()
    connexion.close()


def ajouter_document(nom, type_fichier, categorie, chemin, commentaire=None):
    """
    Insère les métadonnées d'un nouveau document.

    Retourne l'id généré par SQLite, nécessaire pour indexer ensuite son
    contenu avec indexer_document().
    """
    connexion = sqlite3.connect(DB_PATH)
    curseur = connexion.cursor()
    curseur.execute("""
        INSERT INTO documents (nom, type_fichier, categorie, chemin, commentaire)
        VALUES (?, ?, ?, ?, ?)
    """, (nom, type_fichier, categorie, chemin, commentaire))
    connexion.commit()
    id_document = curseur.lastrowid
    connexion.close()
    return id_document


def indexer_document(id_document, nom, texte):
    """
    Ajoute (ou remplace, si déjà présent) le contenu texte d'un document
    dans l'index de recherche.

    texte peut être une chaîne vide si l'extraction a échoué ou n'a rien
    trouvé (ex. PDF scanné en image) : le document reste dans la base,
    mais ne sera simplement retrouvé par aucune recherche de contenu.
    """
    connexion = sqlite3.connect(DB_PATH)
    curseur = connexion.cursor()
    curseur.execute("DELETE FROM documents_fts WHERE doc_id = ?", (id_document,))
    curseur.execute("""
        INSERT INTO documents_fts (doc_id, nom, contenu)
        VALUES (?, ?, ?)
    """, (id_document, nom, texte))
    connexion.commit()
    connexion.close()


def lire_documents():
    """Retourne tous les documents, du plus récent au plus ancien, sous forme de liste de tuples."""
    connexion = sqlite3.connect(DB_PATH)
    curseur = connexion.cursor()
    curseur.execute("SELECT * FROM documents ORDER BY date_ajout DESC")
    documents = curseur.fetchall()
    connexion.close()
    return documents


def modifier_document(id_document, categorie, commentaire):
    """Met à jour la catégorie et le commentaire d'un document (le contenu indexé ne change pas)."""
    connexion = sqlite3.connect(DB_PATH)
    curseur = connexion.cursor()
    curseur.execute("""
        UPDATE documents
        SET categorie = ?, commentaire = ?
        WHERE id = ?
    """, (categorie, commentaire, id_document))
    connexion.commit()
    connexion.close()


def supprimer_document(id_document):
    """Supprime le document : sa ligne de métadonnées et sa ligne d'index de recherche."""
    connexion = sqlite3.connect(DB_PATH)
    curseur = connexion.cursor()
    curseur.execute("DELETE FROM documents WHERE id = ?", (id_document,))
    curseur.execute("DELETE FROM documents_fts WHERE doc_id = ?", (id_document,))
    connexion.commit()
    connexion.close()


def rechercher(texte_recherche, categorie=None):
    """
    Recherche dans le contenu texte des documents (et pas seulement leurs
    métadonnées, contrairement au projet 2).

    texte_recherche : mots-clés tapés par l'utilisateur.
    categorie : si fourni (et différent de "Toutes"), restreint la
        recherche à cette catégorie.

    Retourne une liste de tuples :
    (id, nom, type_fichier, categorie, date_ajout, chemin, commentaire, extrait)

    - Si texte_recherche contient des mots, les résultats sont triés par
      pertinence (BM25, le score de classement standard utilisé par les
      moteurs de recherche plein texte) et `extrait` contient un passage du
      document autour des mots trouvés, avec les mots encadrés par **.
    - Si texte_recherche est vide, retourne tous les documents (du plus
      récent au plus ancien), sans notion de pertinence : une recherche
      vide n'exclut rien. `extrait` vaut alors None.
    """
    requete_fts = recherche.construire_requete_fts(texte_recherche)

    connexion = sqlite3.connect(DB_PATH)
    curseur = connexion.cursor()

    if requete_fts is None:
        sql = """
            SELECT id, nom, type_fichier, categorie, date_ajout, chemin, commentaire, NULL
            FROM documents
        """
        parametres = []
        if categorie and categorie != "Toutes":
            sql += " WHERE categorie = ?"
            parametres.append(categorie)
        sql += " ORDER BY date_ajout DESC"
    else:
        sql = """
            SELECT d.id, d.nom, d.type_fichier, d.categorie, d.date_ajout, d.chemin, d.commentaire,
                   snippet(documents_fts, 2, '**', '**', '…', 12) AS extrait
            FROM documents_fts
            JOIN documents d ON d.id = documents_fts.doc_id
            WHERE documents_fts MATCH ?
        """
        parametres = [requete_fts]
        if categorie and categorie != "Toutes":
            sql += " AND d.categorie = ?"
            parametres.append(categorie)
        sql += " ORDER BY bm25(documents_fts)"

    curseur.execute(sql, parametres)
    resultats = curseur.fetchall()
    connexion.close()
    return resultats
