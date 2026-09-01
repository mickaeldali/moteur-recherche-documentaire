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
   produire une analyse. Si ce n'est pas le cas, la clause est traitée
   comme absente, même si l'étape 1 avait trouvé un candidat au-dessus du
   seuil.

detecter_et_analyser() orchestre ces deux étapes pour UN document et UN
type de clause, sans jamais lever d'exception (contrairement à
analyser_clause() elle-même) : c'est la fonction partagée entre le bouton
"Analyser une clause" de app.py (projet 5, un document à la fois, type de
clause tapé librement) et due_diligence.py (projet final, exécutée en
parallèle sur tous les documents × les 24 catégories fixes, où une
exception non rattrapée dans une tâche ferait perdre le résultat des
autres).
"""

import json

import database
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


def construire_messages_clause(type_clause, chunks_pertinents, ce_qu_il_faut_verifier=None, red_flags=None):
    """
    Assemble les messages envoyés à l'API : instructions système, puis le
    type de clause demandé accompagné du contexte (les chunks pertinents
    retrouvés dans le document, numérotés).

    ce_qu_il_faut_verifier et red_flags sont optionnels : quand fournis
    (par due_diligence.py, pour une des 24 catégories fixes), ils enrichissent
    le message pour guider une analyse plus précise et cohérente d'une
    catégorie à l'autre, plutôt qu'un simple nom de clause tapé librement
    (usage projet 5, ces deux paramètres restent à None).

    Fonction pure (aucun appel réseau) : testable directement.
    """
    passages = []
    for i, resultat in enumerate(chunks_pertinents):
        texte_chunk = resultat[-2]
        passages.append(f"Extrait {i + 1} :\n{texte_chunk}")
    contexte = "\n\n".join(passages)

    message_utilisateur = f"Type de clause à analyser : {type_clause}\n"
    if ce_qu_il_faut_verifier:
        message_utilisateur += f"\nCe qu'il faut vérifier : {ce_qu_il_faut_verifier}\n"
    if red_flags:
        message_utilisateur += f"\nRed flags typiques : {red_flags}\n"
    message_utilisateur += f"\nExtraits du contrat (contexte) :\n{contexte}"

    return [
        {"role": "system", "content": INSTRUCTIONS_SYSTEME_CLAUSE},
        {"role": "user", "content": message_utilisateur},
    ]


def analyser_clause(type_clause, chunks_pertinents, ce_qu_il_faut_verifier=None, red_flags=None):
    """
    Envoie le type de clause et son contexte (chunks_pertinents, déjà
    filtrés au-dessus du seuil de pertinence) au modèle de chat, avec un
    format de sortie JSON imposé (SCHEMA_ANALYSE_CLAUSE).

    ce_qu_il_faut_verifier/red_flags : voir construire_messages_clause().

    Retourne un dict {"clause_correspond": bool, "analyse": str, "niveau_risque": "faible"|"moyen"|"élevé"}.

    Ne rattrape volontairement aucune erreur (même principe que
    llm.generer_reponse) : CleApiManquante ou une exception de la
    librairie openai remonte jusqu'à l'appelant, qui décide comment réagir
    sans perdre la détection déjà obtenue.
    """
    messages = construire_messages_clause(type_clause, chunks_pertinents, ce_qu_il_faut_verifier, red_flags)

    reponse = embeddings.client_openai().chat.completions.create(
        model=llm.MODELE_CHAT,
        messages=messages,
        response_format={"type": "json_schema", "json_schema": SCHEMA_ANALYSE_CLAUSE},
    )

    return json.loads(reponse.choices[0].message.content)


def detecter_et_analyser(doc_id, type_clause, ce_qu_il_faut_verifier=None, red_flags=None):
    """
    Orchestre les deux étapes de détection (voir docstring du fichier) pour
    UN document et UN type de clause, sans jamais lever d'exception : les
    échecs sont capturés et reflétés dans la valeur de retour, jamais
    propagés. C'est ce qui permet de réutiliser cette fonction aussi bien
    depuis un bouton de app.py (projet 5, un appel à la fois) que depuis un
    thread de due_diligence.py (potentiellement des dizaines d'appels en
    parallèle, où une exception non rattrapée ferait perdre le résultat
    des autres tâches du lot).

    Retourne toujours un dict avec une clé "statut" :
    - "detection_indisponible" : l'étape 1 (recherche des candidats) a
      échoué (clé API absente, réseau, quota) -- rien à enregistrer, le
      dict contient aussi "erreur" (message explicatif).
    - "absente" : aucun candidat trouvé, ou candidat(s) trouvé(s) mais le
      LLM a confirmé (clause_correspond=false) qu'ils ne correspondent
      pas réellement au type de clause demandé.
    - "presente" : clause confirmée et analysée.
    - "detectee_non_analysee" : un candidat a été trouvé mais l'étape 2
      (vérification + analyse par le LLM) a échoué -- le dict contient
      aussi "erreur", "analyse"/"niveau_risque" restant None (résilience :
      la détection n'est pas perdue).

    Dans tous les cas sauf "detection_indisponible", le dict contient aussi
    doc_id, type_clause, clause_presente, texte_extrait, analyse,
    niveau_risque : prêt à être passé à database.enregistrer_analyse_clause().
    """
    try:
        resultats_bruts = database.rechercher_par_sens(type_clause, doc_id=doc_id, top_n=llm.NB_CHUNKS_CONTEXTE)
        chunks_candidats = llm.filtrer_chunks_pertinents(resultats_bruts)
    except Exception as erreur:
        return {"statut": "detection_indisponible", "erreur": str(erreur)}

    resultat = {
        "doc_id": doc_id, "type_clause": type_clause,
        "clause_presente": False, "texte_extrait": None, "analyse": None, "niveau_risque": None,
    }

    if not chunks_candidats:
        resultat["statut"] = "absente"
        return resultat

    try:
        resultat_llm = analyser_clause(type_clause, chunks_candidats, ce_qu_il_faut_verifier, red_flags)
    except Exception as erreur:
        resultat["clause_presente"] = True
        resultat["texte_extrait"] = "\n\n".join(r[-2] for r in chunks_candidats)
        resultat["statut"] = "detectee_non_analysee"
        resultat["erreur"] = str(erreur)
        return resultat

    if not resultat_llm["clause_correspond"]:
        resultat["statut"] = "absente"
        return resultat

    resultat["clause_presente"] = True
    resultat["texte_extrait"] = "\n\n".join(r[-2] for r in chunks_candidats)
    resultat["analyse"] = resultat_llm["analyse"]
    resultat["niveau_risque"] = resultat_llm["niveau_risque"]
    resultat["statut"] = "presente"
    return resultat
