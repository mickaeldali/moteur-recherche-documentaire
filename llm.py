"""
llm.py

Projet 4 : première IA qui RÉDIGE une réponse (RAG -- Retrieval-Augmented
Generation), là où embeddings.py + database.rechercher_par_sens() se
contentaient de RETROUVER des passages.

Principe du RAG : le LLM ne connaît PAS les documents importés. On lui
donne, à chaque question, les quelques passages les plus pertinents
(retrouvés par recherche par sens) directement dans le prompt, et on lui
demande de répondre à partir de ce contexte -- rien d'autre. C'est ce qui
rend la réponse vérifiable : contrairement à ChatGPT en général, on sait
exactement quel texte le modèle avait sous les yeux.

Comme embeddings.py, ce fichier isole tout ce qui touche à l'API OpenAI
(ici le modèle de CHAT, pas les embeddings) pour rester testable sans appel
réseau réel (voir tests/test_llm.py, qui mocke generer_reponse).
"""

import json

import embeddings

MODELE_CHAT = "gpt-4o-mini"

# Nombre de chunks envoyés en contexte au LLM (top-k) : ni un seul (une
# clause peut être coupée entre deux chunks malgré le chevauchement), ni le
# document entier (coût, limite de taille de contexte, dilution du signal --
# le même arbitrage que pour la taille des chunks dans embeddings.py).
NB_CHUNKS_CONTEXTE = 3

# Score de similarité cosinus en dessous duquel un chunk est jugé trop
# éloigné de la question pour être envoyé au LLM comme contexte fiable.
SEUIL_SIMILARITE_MINIMAL = 0.15

# Schéma JSON strict imposé à la réponse de l'API : le modèle ne peut pas
# répondre en texte libre, uniquement selon cette structure. C'est ce qui
# permet au code de séparer, de façon fiable, ce qui vient du document (le
# contexte fourni) de ce qui viendrait de la culture générale du modèle --
# une distinction impossible à garantir en analysant du texte libre après coup.
SCHEMA_REPONSE = {
    "name": "reponse_rag",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "reponse_document": {
                "type": "string",
                "description": (
                    "Réponse rédigée UNIQUEMENT à partir des extraits de "
                    "contexte fournis. Si le contexte ne permet pas de "
                    "répondre à la question, l'indiquer explicitement ici "
                    "plutôt que d'inventer une réponse."
                ),
            },
            "complement_connaissance_generale": {
                "type": ["string", "null"],
                "description": (
                    "Complément de connaissance générale du LLM, en dehors "
                    "du contexte fourni, UNIQUEMENT si la question déborde "
                    "du document (ex. le régime juridique général d'un "
                    "type de clause, au-delà de ce que dit ce contrat "
                    "précis). Mettre null si non applicable."
                ),
            },
        },
        "required": ["reponse_document", "complement_connaissance_generale"],
        "additionalProperties": False,
    },
}

INSTRUCTIONS_SYSTEME = (
    "Tu es un assistant juridique qui répond à des questions à partir "
    "d'extraits de documents fournis en contexte. Règles strictes :\n"
    "1. Le champ reponse_document doit se baser UNIQUEMENT sur le contexte "
    "fourni ci-dessous. Si le contexte ne permet pas de répondre, dis-le "
    "clairement dans ce champ au lieu d'inventer une réponse.\n"
    "2. Si la question comporte une partie qui déborde du contexte fourni "
    "(une notion juridique générale non présente dans les extraits, par "
    "exemple le régime juridique d'un type de clause au-delà de ce que dit "
    "ce document précis), réponds à cette partie dans le champ "
    "complement_connaissance_generale -- ne l'ignore pas et ne l'intègre "
    "pas non plus dans reponse_document.\n"
    "3. N'utilise le champ complement_connaissance_generale QUE dans ce "
    "cas précis. Sinon, mets la valeur JSON null (jamais la chaîne de "
    "caractères \"null\", jamais une chaîne vide) : le champ doit rester "
    "vide, pas contenir le mot null écrit en toutes lettres.\n"
    "4. Ne mélange jamais les deux champs : le contenu du contexte reste "
    "dans reponse_document, tes connaissances générales restent dans "
    "complement_connaissance_generale."
)


def filtrer_chunks_pertinents(resultats_recherche):
    """
    Garde, parmi les résultats de database.rechercher_par_sens(), ceux dont
    le score de similarité (dernier élément de chaque tuple) dépasse
    SEUIL_SIMILARITE_MINIMAL.

    Sert à éviter d'envoyer au LLM des passages trop éloignés du sujet
    quand aucun document ne traite vraiment de la question posée (ex. tous
    les scores sont proches de 0) : mieux vaut répondre "pas d'information
    pertinente" que de fournir un contexte hors-sujet au modèle.
    """
    return [resultat for resultat in resultats_recherche if resultat[-1] >= SEUIL_SIMILARITE_MINIMAL]


def construire_messages(question, chunks_pertinents):
    """
    Assemble les messages envoyés à l'API : les instructions système, puis
    la question accompagnée du contexte (les chunks pertinents, numérotés
    et rattachés au nom de leur document).

    Fonction pure (aucun appel réseau) : testable directement.
    """
    passages = []
    for i, resultat in enumerate(chunks_pertinents):
        nom_document = resultat[1]
        texte_chunk = resultat[-2]
        passages.append(f"Passage {i + 1} (document « {nom_document} ») :\n{texte_chunk}")
    contexte = "\n\n".join(passages)

    message_utilisateur = f"Question : {question}\n\nContexte (extraits de documents) :\n{contexte}"

    return [
        {"role": "system", "content": INSTRUCTIONS_SYSTEME},
        {"role": "user", "content": message_utilisateur},
    ]


def generer_reponse(question, chunks_pertinents):
    """
    Envoie la question et son contexte (chunks_pertinents) au modèle de
    chat, avec un format de sortie JSON imposé (SCHEMA_REPONSE).

    Retourne un dict {"reponse_document": str, "complement_connaissance_generale": str | None}.

    Ne rattrape volontairement aucune erreur (comme embeddings.calculer_embedding) :
    CleApiManquante ou une exception de la librairie openai (réseau, quota...)
    remonte jusqu'à l'appelant (app.py), qui décide comment l'afficher sans
    empêcher l'affichage des chunks bruts déjà trouvés par la recherche.
    """
    messages = construire_messages(question, chunks_pertinents)

    reponse = embeddings.client_openai().chat.completions.create(
        model=MODELE_CHAT,
        messages=messages,
        response_format={"type": "json_schema", "json_schema": SCHEMA_REPONSE},
    )

    contenu = json.loads(reponse.choices[0].message.content)

    # Le schéma JSON strict garantit le bon TYPE (string ou null), mais pas
    # que le modèle choisisse la bonne VALEUR : il arrive qu'il renvoie la
    # chaîne de caractères "null" (ou une chaîne vide) au lieu du JSON null
    # attendu quand il n'a rien à ajouter. On normalise ce cas ici plutôt
    # que de faire confiance au modèle, pour ne jamais afficher ce mot à
    # l'utilisateur comme s'il s'agissait d'un vrai complément.
    complement = contenu.get("complement_connaissance_generale")
    if isinstance(complement, str) and complement.strip().lower() in ("", "null"):
        contenu["complement_connaissance_generale"] = None

    return contenu
