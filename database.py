"""
database.py

Communication avec la base SQLite : métadonnées des documents (table
`documents`, identique au projet 2), index de recherche plein texte (table
virtuelle FTS5 `documents_fts`, projet 3), index de recherche par sens
(table `documents_embeddings`, étape embeddings) et historique des analyses
de clauses (table `documents_clauses`, projet 5).

Le fichier réel (PDF/DOCX/TXT) n'est jamais stocké ici : seule son adresse
(le champ `chemin`) est mémorisée. Le fichier physique est géré par
fichiers.py, le texte qu'il contient par extraction.py.

- `documents_fts` mémorise le texte extrait pour la recherche par MOTS
  (FTS5, index inversé mot -> documents).
- `documents_embeddings` mémorise, pour chaque morceau (chunk) de texte, son
  vecteur numérique (embedding), pour la recherche par SENS : on y compare
  le vecteur de la requête à celui de chaque chunk avec une similarité
  cosinus (voir embeddings.py).
- `documents_clauses` mémorise, pour chaque analyse de clause demandée par
  l'utilisateur (un type de clause + un document), le résultat obtenu :
  présence, extrait, analyse rédigée et niveau de risque (voir clauses.py).

Les trois index sont reliés à `documents` par `doc_id`.
"""

import json
import sqlite3
from pathlib import Path

import embeddings
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

    # Table (classique, pas virtuelle) qui stocke un vecteur par chunk de
    # texte. `vecteur` est une liste de nombres sérialisée en JSON : SQLite
    # n'a pas de type "tableau de nombres" natif, JSON est le moyen le plus
    # simple de le stocker dans une colonne TEXT.
    curseur.execute("""
        CREATE TABLE IF NOT EXISTS documents_embeddings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            doc_id INTEGER NOT NULL,
            chunk_id INTEGER NOT NULL,
            texte_chunk TEXT NOT NULL,
            vecteur TEXT NOT NULL
        )
    """)

    # Historique des analyses de clauses (projet 5). clause_presente est
    # stockée en INTEGER (0 ou 1) : SQLite n'a pas de type booléen natif.
    # analyse et niveau_risque peuvent rester NULL : soit la clause est
    # absente (rien à analyser), soit l'appel au LLM a échoué après une
    # détection réussie (voir clauses.py) -- dans les deux cas, la ligne
    # est quand même enregistrée plutôt que perdue.
    curseur.execute("""
        CREATE TABLE IF NOT EXISTS documents_clauses (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            doc_id INTEGER NOT NULL,
            type_clause TEXT NOT NULL,
            clause_presente INTEGER NOT NULL,
            texte_extrait TEXT,
            analyse TEXT,
            niveau_risque TEXT,
            date_analyse TIMESTAMP DEFAULT CURRENT_TIMESTAMP
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


def indexer_embeddings(id_document, texte):
    """
    Découpe `texte` en chunks et calcule un embedding pour chacun via
    l'API OpenAI (embeddings.py), pour alimenter la recherche par sens.

    Remplace les chunks déjà indexés pour ce document, s'il y en avait.
    Ne fait rien (retourne 0) si le texte est vide : rien à indexer.

    Ne rattrape volontairement aucune erreur : si l'API OpenAI échoue (pas
    de clé, pas de réseau, quota dépassé...), l'exception remonte jusqu'à
    l'appelant (app.py), qui décide comment réagir sans bloquer l'import du
    document. C'est la même logique de résilience que pour l'extraction de
    texte au projet 2/3.

    Retourne le nombre de chunks indexés.
    """
    chunks = embeddings.decouper_en_chunks(texte)
    if not chunks:
        return 0

    connexion = sqlite3.connect(DB_PATH)
    curseur = connexion.cursor()
    curseur.execute("DELETE FROM documents_embeddings WHERE doc_id = ?", (id_document,))

    for chunk_id, chunk in enumerate(chunks):
        vecteur = embeddings.calculer_embedding(chunk)
        curseur.execute("""
            INSERT INTO documents_embeddings (doc_id, chunk_id, texte_chunk, vecteur)
            VALUES (?, ?, ?, ?)
        """, (id_document, chunk_id, chunk, json.dumps(vecteur)))

    connexion.commit()
    connexion.close()
    return len(chunks)


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
    """Supprime le document : sa ligne de métadonnées et ses lignes dans les trois index/historiques."""
    connexion = sqlite3.connect(DB_PATH)
    curseur = connexion.cursor()
    curseur.execute("DELETE FROM documents WHERE id = ?", (id_document,))
    curseur.execute("DELETE FROM documents_fts WHERE doc_id = ?", (id_document,))
    curseur.execute("DELETE FROM documents_embeddings WHERE doc_id = ?", (id_document,))
    curseur.execute("DELETE FROM documents_clauses WHERE doc_id = ?", (id_document,))
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


def rechercher_par_sens(texte_requete, categorie=None, doc_id=None, top_n=5):
    """
    Recherche par SENS : retrouve les chunks dont le sens se rapproche le
    plus de `texte_requete`, même s'ils ne partagent aucun mot avec elle
    (contrairement à rechercher(), qui compare des mots).

    Principe : calcule l'embedding de la requête, puis le compare (par
    similarité cosinus) à l'embedding de chaque chunk déjà indexé. Pas
    d'index vectoriel spécialisé (type FAISS) : à l'échelle d'un outil
    personnel, comparer chaque chunk un par un en Python est largement
    assez rapide, et bien plus simple à comprendre.

    texte_requete : la question/le passage tapé par l'utilisateur.
    categorie : si fourni (et différent de "Toutes"), restreint la
        recherche aux documents de cette catégorie.
    doc_id : si fourni, restreint la recherche aux chunks de CE document
        (utilisé par le projet 5 : détecter une clause dans un document
        précis, pas dans tout le corpus).
    top_n : nombre maximum de chunks retournés (les plus proches en sens).

    Retourne une liste de tuples, du plus proche au moins proche :
    (id, nom, type_fichier, categorie, date_ajout, chemin, commentaire,
     texte_chunk, score)
    où score est la similarité cosinus (entre -1 et 1, proche de 1 = sens
    très proche).

    Peut lever embeddings.CleApiManquante ou une exception de la librairie
    openai si l'appel API pour la requête échoue : à l'appelant (app.py) de
    décider comment l'afficher.
    """
    if not texte_requete.strip():
        return []

    vecteur_requete = embeddings.calculer_embedding(texte_requete)

    connexion = sqlite3.connect(DB_PATH)
    curseur = connexion.cursor()

    sql = """
        SELECT d.id, d.nom, d.type_fichier, d.categorie, d.date_ajout, d.chemin, d.commentaire,
               e.texte_chunk, e.vecteur
        FROM documents_embeddings e
        JOIN documents d ON d.id = e.doc_id
    """
    conditions = []
    parametres = []
    if categorie and categorie != "Toutes":
        conditions.append("d.categorie = ?")
        parametres.append(categorie)
    if doc_id is not None:
        conditions.append("e.doc_id = ?")
        parametres.append(doc_id)
    if conditions:
        sql += " WHERE " + " AND ".join(conditions)

    curseur.execute(sql, parametres)
    lignes = curseur.fetchall()
    connexion.close()

    resultats_scores = []
    for ligne in lignes:
        *infos_document, texte_chunk, vecteur_json = ligne
        vecteur_chunk = json.loads(vecteur_json)
        score = embeddings.similarite_cosinus(vecteur_requete, vecteur_chunk)
        resultats_scores.append((*infos_document, texte_chunk, score))

    resultats_scores.sort(key=lambda resultat: resultat[-1], reverse=True)
    return resultats_scores[:top_n]


def enregistrer_analyse_clause(
    doc_id, type_clause, clause_presente, texte_extrait, analyse=None, niveau_risque=None,
):
    """
    Enregistre le résultat d'une analyse de clause (projet 5) : présente ou
    non, avec (si présente) son texte exact, et (si l'appel au LLM a
    réussi) l'analyse rédigée et le niveau de risque.

    analyse et niveau_risque restent None si la clause est absente, ou si
    la détection a réussi mais que l'analyse par le LLM a échoué (clé
    absente, quota, réseau...) : la ligne est quand même enregistrée plutôt
    que perdue, avec de quoi repérer une clause "détectée mais pas encore
    analysée" dans l'export.
    """
    connexion = sqlite3.connect(DB_PATH)
    curseur = connexion.cursor()
    curseur.execute("""
        INSERT INTO documents_clauses
            (doc_id, type_clause, clause_presente, texte_extrait, analyse, niveau_risque)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (doc_id, type_clause, int(clause_presente), texte_extrait, analyse, niveau_risque))
    connexion.commit()
    connexion.close()


def lire_analyses_clauses(doc_id=None):
    """
    Retourne l'historique des analyses de clauses, du plus récent au plus
    ancien, jointe à `documents` pour le nom du fichier.

    doc_id : si fourni, restreint l'historique à ce document ; sinon,
    retourne l'historique complet (utilisé pour l'export Excel global).

    Retourne une liste de tuples :
    (id, nom, type_clause, clause_presente, texte_extrait, analyse,
     niveau_risque, date_analyse)
    """
    connexion = sqlite3.connect(DB_PATH)
    curseur = connexion.cursor()

    sql = """
        SELECT c.id, d.nom, c.type_clause, c.clause_presente, c.texte_extrait,
               c.analyse, c.niveau_risque, c.date_analyse
        FROM documents_clauses c
        JOIN documents d ON d.id = c.doc_id
    """
    parametres = []
    if doc_id is not None:
        sql += " WHERE c.doc_id = ?"
        parametres.append(doc_id)
    sql += " ORDER BY c.date_analyse DESC"

    curseur.execute(sql, parametres)
    resultats = curseur.fetchall()
    connexion.close()
    return resultats
