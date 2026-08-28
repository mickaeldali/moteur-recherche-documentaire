"""
clauses.py

Projet 5 : mini assistant d'analyse contractuelle. Étant donné un TYPE de
clause tapé librement par l'utilisateur (ex. "non-concurrence",
"confidentialité") et un document précis, on détecte si cette clause existe
dans le document, puis un LLM l'analyse et lui attribue un niveau de risque.

Ce fichier ne fait QUE la partie "analyse par le LLM" (comme llm.py fait la
partie "génération de réponse" du RAG). La détection PRÉLIMINAIRE (retrouver
des candidats, décider s'ils sont au-dessus du seuil de pertinence)
réutilise ce qui existe déjà : database.rechercher_par_sens() (avec le
paramètre doc_id, pour rester sur UN document) et llm.filtrer_chunks_pertinents().

Détection en DEUX étapes, pas une seule -- point important découvert en
testant en conditions réelles pendant le développement : la seule
similarité cosinus ne suffit pas à décider fiablement si une clause est
présente. Sur un contrat de test, demander "clause de brevet et de droits
d'auteur" (absente du document) obtenait un score plus élevé (0,39) que la
clause réellement présente "non-concurrence" ailleurs (0,34) : deux clauses
de même domaine juridique peuvent avoir un score proche même quand l'une
des deux n'existe pas dans le document. Une similarité au-dessus du seuil
n'est donc qu'un CANDIDAT, jamais une confirmation :
1. database.rechercher_par_sens() + llm.filtrer_chunks_pertinents() retrouvent des
   candidats (orientés "ne rien rater" -- rappel élevé) ;
2. analyser_clause() demande au LLM de vérifier lui-même, en lisant le
   contexte, si ces candidats correspondent VRAIMENT au type de clause
   demandé (champ clause_correspond du schéma, orienté précision) avant de
   produire une analyse. Si ce n'est pas le cas, l'appelant (app.py) doit
   traiter la clause comme absente, même si l'étape 1 avait trouvé un
   candidat au-dessus du seuil.
"""

import json

import embeddings
import llm

# Les 3 seules valeurs autorisées pour le niveau de risque : une échelle
# fixe, contrairement à `analyse` qui reste du texte libre. C'est ce qui
# rend ce champ exploitable (filtrable, triable) dans l'export Excel.
NIVEAUX_RISQUE = ["faible", "moyen", "élevé"]

SCHEMA_ANALYSE_CLAUSE = {
    "name": "analyse_clause",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "clause_correspond": {
                "type": "boolean",
                "description": (
                    "true si les extraits fournis correspondent VRAIMENT au "
                    "type de clause demandé. false si, après lecture, ils "
                    "traitent d'un autre sujet -- la détection automatique "
                    "en amont se base sur une similarité de sens qui peut "
                    "se tromper entre deux clauses de même domaine "
                    "juridique (ex. confondre une clause de non-concurrence "
                    "avec une clause de propriété intellectuelle)."
                ),
            },
            "analyse": {
                "type": "string",
                "description": (
                    "Si clause_correspond est true : analyse en texte "
                    "libre de la clause, adaptée à son type précis "
                    "(délais, périmètre, obligations, préavis, pénalités, "
                    "etc. selon ce qui est pertinent pour ce type de "
                    "clause), basée UNIQUEMENT sur les extraits fournis. "
                    "Si clause_correspond est false : une phrase expliquant "
                    "brièvement pourquoi les extraits ne correspondent pas "
                    "au type de clause demandé."
                ),
            },
            "niveau_risque": {
                "type": "string",
                "enum": NIVEAUX_RISQUE,
                "description": (
                    "Si clause_correspond est true : niveau de risque "
                    "juridique de la clause pour la partie qui la subit "
                    "(pas pour celle qui l'impose). Si clause_correspond "
                    "est false : mettre 'faible' (aucune clause de ce type "
                    "n'a été trouvée, donc aucun risque associé)."
                ),
            },
        },
        "required": ["clause_correspond", "analyse", "niveau_risque"],
        "additionalProperties": False,
    },
}

INSTRUCTIONS_SYSTEME_CLAUSE = (
    "Tu es un assistant juridique qui analyse une clause précise d'un "
    "contrat, à partir d'extraits de ce contrat fournis en contexte. Ces "
    "extraits ont été présélectionnés par une recherche de similarité de "
    "sens, qui peut se tromper entre deux clauses de même domaine "
    "juridique. Règles strictes :\n"
    "1. Vérifie D'ABORD si les extraits correspondent VRAIMENT au type de "
    "clause demandé (champ clause_correspond). Ne suppose jamais que "
    "c'est le cas seulement parce qu'ils t'ont été fournis pour cette "
    "clause.\n"
    "2. Si clause_correspond est true : base ton analyse UNIQUEMENT sur le "
    "contexte fourni, jamais sur une clause type générique que tu "
    "connaîtrais par ailleurs. Adapte les critères de ton analyse au type "
    "de clause demandé : les points pertinents pour une clause de "
    "non-concurrence (durée, périmètre géographique, contrepartie "
    "financière...) ne sont pas ceux d'une clause de confidentialité ou "
    "de résiliation, par exemple.\n"
    "3. Le niveau de risque s'apprécie du point de vue de la partie qui "
    "subit la clause (celle à qui elle est opposée), pas de celle qui "
    "l'impose."
)


def construire_messages_clause(type_clause, chunks_pertinents):
    """
    Assemble les messages envoyés à l'API : instructions système, puis le
    type de clause demandé accompagné du contexte (les chunks pertinents
    retrouvés dans le document, numérotés).

    Fonction pure (aucun appel réseau) : testable directement.
    """
    passages = []
    for i, resultat in enumerate(chunks_pertinents):
        texte_chunk = resultat[-2]
        passages.append(f"Extrait {i + 1} :\n{texte_chunk}")
    contexte = "\n\n".join(passages)

    message_utilisateur = (
        f"Type de clause à analyser : {type_clause}\n\n"
        f"Extraits du contrat (contexte) :\n{contexte}"
    )

    return [
        {"role": "system", "content": INSTRUCTIONS_SYSTEME_CLAUSE},
        {"role": "user", "content": message_utilisateur},
    ]


def analyser_clause(type_clause, chunks_pertinents):
    """
    Envoie le type de clause et son contexte (chunks_pertinents, déjà
    filtrés au-dessus du seuil de pertinence) au modèle de chat, avec un
    format de sortie JSON imposé (SCHEMA_ANALYSE_CLAUSE).

    Retourne un dict {"analyse": str, "niveau_risque": "faible"|"moyen"|"élevé"}.

    Ne rattrape volontairement aucune erreur (même principe que
    llm.generer_reponse) : CleApiManquante ou une exception de la
    librairie openai remonte jusqu'à l'appelant (app.py), qui décide
    comment réagir sans perdre la détection déjà obtenue.
    """
    messages = construire_messages_clause(type_clause, chunks_pertinents)

    reponse = embeddings.client_openai().chat.completions.create(
        model=llm.MODELE_CHAT,
        messages=messages,
        response_format={"type": "json_schema", "json_schema": SCHEMA_ANALYSE_CLAUSE},
    )

    return json.loads(reponse.choices[0].message.content)
