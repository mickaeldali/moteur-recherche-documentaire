# Moteur de recherche documentaire

Projet n°3 du parcours LegalTech — application locale qui permet de rechercher dans le **contenu texte** des documents juridiques importés, et non plus seulement dans leurs métadonnées comme au projet 2. Deux moteurs de recherche coexistent : FTS5 (recherche par mots-clés, plein texte classique) et une recherche par **sens**, ajoutée dans une étape ultérieure grâce aux embeddings OpenAI. Toujours aucun LLM (pas de génération de texte, pas de résumé, pas de raisonnement) : les embeddings servent uniquement à comparer des textes par proximité de sens.

## Fonctionnalités

- Import de documents PDF, DOCX et TXT (repris du projet 2 : `fichiers.py`, `extraction.py`)
- Indexation automatique du texte extrait à chaque import, à la fois dans l'index FTS5 (mots-clés) et dans l'index d'embeddings (sens)
- **Recherche par mots-clés** (FTS5) :
  - recherche par préfixe (« résili » retrouve « résiliation », « résiliable »...)
  - tri par pertinence (score BM25)
  - extrait du passage trouvé, avec les mots correspondants surlignés
- **Recherche par sens** (embeddings OpenAI, modèle `text-embedding-3-small`) :
  - retrouve les passages dont le sens se rapproche de la requête, même sans mot commun
  - découpe chaque document en chunks (~700 caractères, avec léger chevauchement) pour cibler un passage précis plutôt que tout le document
  - compare la requête à chaque chunk par similarité cosinus (calcul numpy, sans index vectoriel spécialisé — largement suffisant à l'échelle d'un usage personnel)
  - affiche le document, le passage correspondant et un score de proximité
- Filtre par catégorie juridique, combinable avec les deux modes de recherche
- Une recherche vide (mode mots-clés) affiche tous les documents (comme au projet 2)
- Aperçu du texte extrait, modification (catégorie/commentaire), suppression (base + fichier + les deux index)
- Import résilient à deux niveaux :
  - si l'extraction échoue (format non supporté, PDF scanné en image...), le document est quand même importé, avec un message clair indiquant qu'il ne sera pas trouvable par une recherche de contenu
  - si l'indexation par sens échoue (pas de clé API OpenAI, pas de réseau, quota dépassé...), le document reste importé et trouvable par mots-clés, avec un message clair indiquant qu'il ne sera pas trouvable par une recherche par sens

## Structure du projet

```text
moteur_recherche_documentaire/
├── app.py            # interface Streamlit
├── database.py       # SQLite : table documents + index FTS5 + index embeddings
├── recherche.py       # construction de la requête FTS5 à partir du texte tapé
├── embeddings.py       # découpage en chunks, appel à l'API OpenAI, similarité cosinus
├── fichiers.py         # sauvegarde/suppression physique des fichiers (repris du projet 2)
├── extraction.py       # extraction du texte PDF/DOCX/TXT (repris du projet 2)
├── documents/           # fichiers importés (non versionné)
├── data/
│   └── documents.db     # base SQLite (non versionnée)
├── .env                 # OPENAI_API_KEY (non versionné, à créer soi-même)
├── tests/
│   ├── test_database.py
│   ├── test_recherche.py
│   ├── test_embeddings.py
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

Pour activer la recherche par sens, créer un fichier `.env` à la racine du projet avec :

```
OPENAI_API_KEY=sk-...
```

Sans cette clé, l'application fonctionne quand même : la recherche par mots-clés reste pleinement disponible, seule la recherche par sens affiche un message d'indisponibilité.

## Lancer l'application

```powershell
streamlit run app.py
```

## Lancer les tests

```powershell
python -m pytest -v
```

Les tests ne font aucun appel réseau réel à l'API OpenAI : `calculer_embedding` est mocké dans `test_embeddings.py` et `test_database.py`.

## Note d'architecture : pourquoi FTS5

Une recherche « classique » naïve relirait le texte de chaque document à chaque recherche (comme `if mot in texte`, utilisé pour les métadonnées au projet 2) : ça devient lent dès que le nombre ou la taille des documents grandit. FTS5 (Full-Text Search 5) construit à la place un **index inversé** : pour chaque mot, il mémorise à l'avance la liste des documents qui le contiennent. Une recherche consiste alors à consulter cet index plutôt qu'à relire tous les documents.

Deux tables SQLite coexistent :
- `documents` : les métadonnées (identique au projet 2).
- `documents_fts` : une table **virtuelle** FTS5 qui indexe le nom et le contenu texte de chaque document, reliée à `documents` par un identifiant (`doc_id`).

`database.rechercher()` interroge `documents_fts` avec l'opérateur `MATCH`, trie les résultats par pertinence avec `bm25()`, et récupère un extrait surligné avec `snippet()`. La construction de la requête (échappement, recherche par préfixe) est isolée dans `recherche.py` pour rester testable indépendamment de SQLite.

## Note d'architecture : la recherche par sens (embeddings)

FTS5 compare des **mots** : chercher « résiliation » ne retrouve pas un passage qui dit « mettre fin au contrat » s'il n'utilise pas ce mot précis. La recherche par sens répond à cette limite en comparant des **vecteurs numériques** (embeddings) plutôt que des mots.

Principe :
1. À l'import, le texte extrait est découpé en chunks (`embeddings.decouper_en_chunks`), des morceaux d'environ 700 caractères coupés aux limites de phrases, avec un léger chevauchement pour ne pas perdre une clause à cheval sur une coupure. On découpe car un embedding résume tout le texte qu'on lui donne en un seul vecteur : sur un document entier, ce résumé serait trop dilué pour retrouver un passage précis.
2. Chaque chunk est envoyé à l'API OpenAI (`embeddings.calculer_embedding`, modèle `text-embedding-3-small`), qui retourne un vecteur de 1536 nombres représentant son sens. Ce vecteur est stocké en JSON dans la table `documents_embeddings` (`doc_id`, `chunk_id`, `texte_chunk`, `vecteur`).
3. Au moment de la recherche, la requête tapée par l'utilisateur est elle aussi transformée en vecteur, puis comparée à chaque chunk stocké avec une **similarité cosinus** (`embeddings.similarite_cosinus`) : plus deux vecteurs « pointent dans la même direction », plus les textes qu'ils représentent ont un sens proche.
4. Les chunks sont triés par score décroissant, et les meilleurs sont retournés (`database.rechercher_par_sens`).

Pas d'index vectoriel spécialisé (type FAISS ou Chroma) : à l'échelle d'un outil personnel (quelques dizaines/centaines de documents), comparer la requête à chaque chunk un par un en Python est largement assez rapide, et bien plus simple à comprendre qu'un index dédié. Un tel index deviendrait pertinent à beaucoup plus grande échelle.

La clé API OpenAI est chargée depuis un fichier `.env` (jamais commité sur Git — un fichier `.env` contient un secret qui donnerait accès au compte OpenAI de l'utilisateur, et donc à sa facturation, à quiconque le lirait, même sur un dépôt privé) via `python-dotenv`. Si la clé est absente, ou si l'appel API échoue pour toute autre raison, l'import du document et la recherche par mots-clés continuent de fonctionner normalement : seule la recherche par sens est indisponible, avec un message clair.

C'est cette étape qui prépare le terrain pour le projet suivant : la première intégration d'un **LLM** (modèle de langage), cette fois pour générer du texte et raisonner sur le contenu — ce que les embeddings, volontairement, ne font pas.
