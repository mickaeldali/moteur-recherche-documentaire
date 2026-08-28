# Moteur de recherche documentaire

Projet n°3 du parcours LegalTech — application locale qui permet d'interroger le **contenu texte** des documents juridiques importés, et non plus seulement leurs métadonnées comme au projet 2. Trois façons de chercher coexistent : FTS5 (mots-clés, plein texte classique), une recherche par **sens** (embeddings OpenAI), et depuis le projet 4, la possibilité de **poser une question** à laquelle un LLM rédige une vraie réponse à partir des passages retrouvés (RAG -- retrieval-augmented generation), avec citation systématique des extraits bruts pour rester vérifiable.

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
- **Poser une question** (RAG, modèle de chat `gpt-4o-mini`) :
  - retrouve les 3 passages les plus pertinents par sens (score cosinus ≥ 0,15), puis les envoie à un LLM qui rédige une réponse
  - format de sortie JSON strict imposé au modèle (schéma avec deux champs distincts) : une réponse ancrée dans le document, et un éventuel complément de connaissance générale du LLM, uniquement si la question déborde du document
  - le texte cité en source n'est **jamais** demandé au LLM : c'est le `texte_chunk` déjà stocké en base à l'indexation, affiché séparément dans une section « Sources », pour rester vérifiable même si le LLM paraphrase
  - affichage visuellement distinct entre la réponse ancrée (« trouvée dans le document ») et le complément de connaissance générale (« à vérifier »)
- Filtre par catégorie juridique, combinable avec les trois modes de recherche
- Une recherche vide (mode mots-clés) affiche tous les documents (comme au projet 2)
- Aperçu du texte extrait, modification (catégorie/commentaire), suppression (base + fichier + les deux index)
- Import résilient à deux niveaux :
  - si l'extraction échoue (format non supporté, PDF scanné en image...), le document est quand même importé, avec un message clair indiquant qu'il ne sera pas trouvable par une recherche de contenu
  - si l'indexation par sens échoue (pas de clé API OpenAI, pas de réseau, quota dépassé...), le document reste importé et trouvable par mots-clés, avec un message clair indiquant qu'il ne sera pas trouvable par une recherche par sens
- Résilience du mode « Poser une question » : si la génération de la réponse échoue (clé absente, quota, réseau), les passages bruts retrouvés restent affichés, seule la synthèse rédigée est indisponible, avec un message clair

## Structure du projet

```text
moteur_recherche_documentaire/
├── app.py            # interface Streamlit
├── database.py       # SQLite : table documents + index FTS5 + index embeddings
├── recherche.py       # construction de la requête FTS5 à partir du texte tapé
├── embeddings.py       # découpage en chunks, appel à l'API OpenAI (embeddings), similarité cosinus
├── llm.py               # RAG : prompt, appel au modèle de chat, schéma JSON imposé (projet 4)
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
│   ├── test_llm.py
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

Les tests ne font aucun appel réseau réel à l'API OpenAI : `calculer_embedding` est mocké dans `test_embeddings.py` et `test_database.py`, `generer_reponse` (chat) est mocké dans `test_llm.py`.

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

## Note d'architecture : le RAG (projet 4, `llm.py`)

La recherche par sens retrouve des passages ; elle ne rédige rien. Le RAG (Retrieval-Augmented Generation) ajoute une étape : une fois les passages retrouvés, on les donne à un LLM comme contexte et on lui demande de rédiger une vraie réponse -- sans jamais lui laisser deviner le contenu des documents à partir de sa mémoire générale, puisqu'il n'a accès qu'à ce qu'on lui envoie explicitement dans le prompt.

Principe (`llm.generer_reponse`) :
1. `database.rechercher_par_sens()` retrouve jusqu'à `llm.NB_CHUNKS_CONTEXTE` (3) chunks, puis `llm.filtrer_chunks_pertinents()` élimine ceux en dessous de `llm.SEUIL_SIMILARITE_MINIMAL` (0,15) : mieux vaut répondre "pas d'information pertinente" que fournir un contexte hors-sujet au modèle.
2. `llm.construire_messages()` assemble un message système (les règles ci-dessous) et un message utilisateur (la question + les extraits numérotés, chacun rattaché au nom de son document).
3. L'appel au modèle de chat (`gpt-4o-mini`) impose un **schéma JSON strict** (`llm.SCHEMA_REPONSE`, via le paramètre `response_format` de l'API) : le modèle ne peut pas répondre en texte libre. Deux champs, obligatoires :
   - `reponse_document` : la réponse, rédigée uniquement à partir du contexte fourni ;
   - `complement_connaissance_generale` : `null`, sauf si la question déborde volontairement du document (ex. « quel est le régime juridique général d'une clause de non-concurrence » en plus de « que dit cette clause dans ce contrat »).

   Un format libre ne suffirait pas : le code ne peut pas distinguer de façon fiable, dans un texte en prose, ce qui vient du document de ce qui vient de la culture générale du modèle. Le schéma JSON force cette séparation. **Constaté en conditions réelles pendant le développement** : même avec un schéma strict, le modèle renvoie parfois la chaîne de caractères `"null"` au lieu du JSON `null` attendu quand il n'a rien à ajouter -- `generer_reponse()` normalise ce cas explicitement, pour ne jamais afficher le mot "null" à l'utilisateur comme s'il s'agissait d'un vrai complément.
4. **La citation de la source n'est jamais demandée au LLM** (risque de légère reformulation). L'extrait affiché dans la section « Sources » de `app.py` est le `texte_chunk` déjà stocké tel quel dans `documents_embeddings` à l'indexation -- la même donnée que pour la recherche par sens. Principe retenu pour ce projet : tout ce que le code sait déjà de façon fiable ne doit pas être redemandé au LLM.
5. Résilience : `generer_reponse()` ne rattrape aucune erreur (comme `calculer_embedding`) -- `embeddings.CleApiManquante` ou une exception `openai` (réseau, quota...) remonte jusqu'à `app.py`, qui affiche un message clair à la place de la synthèse, sans empêcher l'affichage des passages bruts déjà retrouvés.

C'est cette étape qui ouvre la voie au projet 5 : l'extraction structurée de clauses et la qualification de risques, en s'appuyant sur le même principe (schéma JSON imposé, citation vérifiable) plutôt que sur du texte libre.
