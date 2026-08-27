# Moteur de recherche documentaire

Projet n°3 du parcours LegalTech — application locale qui permet de rechercher dans le **contenu texte** des documents juridiques importés, et non plus seulement dans leurs métadonnées comme au projet 2. Toujours aucune IA, aucun embedding, aucune recherche vectorielle : la recherche s'appuie sur FTS5, le moteur de recherche plein texte classique intégré à SQLite.

## Fonctionnalités

- Import de documents PDF, DOCX et TXT (repris du projet 2 : `fichiers.py`, `extraction.py`)
- Indexation automatique du texte extrait à chaque import, dans une table virtuelle FTS5
- Recherche plein texte dans le contenu des documents, avec :
  - recherche par préfixe (« résili » retrouve « résiliation », « résiliable »...)
  - tri par pertinence (score BM25)
  - extrait du passage trouvé, avec les mots correspondants surlignés
  - filtre par catégorie juridique, combinable avec la recherche
- Une recherche vide affiche tous les documents (comme au projet 2)
- Aperçu du texte extrait, modification (catégorie/commentaire), suppression (base + fichier + index)
- Import résilient : si l'extraction échoue (format non supporté, PDF scanné en image...), le document est quand même importé, avec un message clair indiquant qu'il ne sera pas trouvable par une recherche de contenu

## Structure du projet

```text
moteur_recherche_documentaire/
├── app.py            # interface Streamlit
├── database.py       # SQLite : table documents + index plein texte FTS5
├── recherche.py       # construction de la requête FTS5 à partir du texte tapé
├── fichiers.py         # sauvegarde/suppression physique des fichiers (repris du projet 2)
├── extraction.py       # extraction du texte PDF/DOCX/TXT (repris du projet 2)
├── documents/           # fichiers importés (non versionné)
├── data/
│   └── documents.db     # base SQLite (non versionnée)
├── tests/
│   ├── test_database.py
│   ├── test_recherche.py
│   ├── test_fichiers.py
│   └── test_extraction.py
├── requirements.txt
└── readme.md
```

## Installation

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Lancer l'application

```powershell
streamlit run app.py
```

## Lancer les tests

```powershell
python -m pytest -v
```

## Note d'architecture : pourquoi FTS5

Une recherche « classique » naïve relirait le texte de chaque document à chaque recherche (comme `if mot in texte`, utilisé pour les métadonnées au projet 2) : ça devient lent dès que le nombre ou la taille des documents grandit. FTS5 (Full-Text Search 5) construit à la place un **index inversé** : pour chaque mot, il mémorise à l'avance la liste des documents qui le contiennent. Une recherche consiste alors à consulter cet index plutôt qu'à relire tous les documents.

Deux tables SQLite coexistent :
- `documents` : les métadonnées (identique au projet 2).
- `documents_fts` : une table **virtuelle** FTS5 qui indexe le nom et le contenu texte de chaque document, reliée à `documents` par un identifiant (`doc_id`).

`database.rechercher()` interroge `documents_fts` avec l'opérateur `MATCH`, trie les résultats par pertinence avec `bm25()`, et récupère un extrait surligné avec `snippet()`. La construction de la requête (échappement, recherche par préfixe) est isolée dans `recherche.py` pour rester testable indépendamment de SQLite.

C'est cette étape — la recherche plein texte classique — qui prépare le terrain pour le projet suivant : la recherche **vectorielle** (embeddings), qui ne cherche plus des mots mais du sens.
