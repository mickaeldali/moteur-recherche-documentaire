"""
due_diligence.py

Projet final : passer d'une analyse de clause document par document (projet
5) à une analyse de DATA ROOM ENTIER -- une matrice de risques croisée
(documents en lignes, catégories juridiques en colonnes) sur une liste fixe
de 24 catégories standards de due diligence contractuelle, plus une synthèse
textuelle globale des points d'attention.

Ce fichier isole ce qui est propre à cette étape : la liste des 24
catégories, le lancement en parallèle de la détection+analyse (déjà en
place dans clauses.py) sur tous les documents x toutes les catégories, la
construction de la matrice pour l'affichage, et la synthèse rédigée par le
LLM à partir d'une agrégation SQL -- jamais des analyses brutes, voir
database.compter_analyses_par_risque().

Parallélisme (concurrent.futures.ThreadPoolExecutor) : chaque case de la
matrice (un document x une catégorie) passe par clauses.detecter_et_analyser(),
qui fait un ou deux appels à l'API OpenAI. ces appels sont I/O-bound :
l'essentiel du temps est passé à ATTENDRE une réponse réseau, pas à
calculer localement. Plusieurs peuvent donc être lancés en même temps sans
ralentir le CPU de la machine -- contrairement à un traitement CPU-bound
(calcul intensif), où le parallélisme n'aiderait pas au-delà du nombre de
coeurs disponibles, et où un GPU pourrait faire une différence (pas ici :
le calcul se fait côté serveurs OpenAI, pas localement). MAX_WORKERS limite
quand même le nombre d'appels simultanés, pour respecter les limites de
débit ("rate limits") de l'API OpenAI : au-delà, l'API renvoie des erreurs
HTTP 429.

Écriture en base : les threads ne font QUE des appels API (et des lectures
SQLite, sans risque en lecture concurrente). L'écriture (enregistrer_analyse_clause)
se fait uniquement dans le thread principal, au fur et à mesure que les
résultats arrivent (as_completed) -- ça évite complètement toute question
d'écriture SQLite concurrente entre threads.
"""

from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

import clauses
import database
import embeddings
import llm

MAX_WORKERS = 8

# Liste fixe de 24 catégories juridiques standards de due diligence
# contractuelle. Chacune a trois éléments : un nom (utilisé comme
# type_clause dans documents_clauses, et comme colonne de la matrice), ce
# qu'il faut vérifier, et des red flags typiques -- ces deux derniers sont
# envoyés au LLM (clauses.py) pour guider une analyse plus précise et plus
# cohérente d'une catégorie à l'autre qu'un simple nom de clause tapé
# librement (usage du projet 5).
CATEGORIES_DUE_DILIGENCE = [
    {
        "nom": "Parties / périmètre",
        "verification": "Identifier précisément les parties au contrat (raison sociale, forme juridique, adresse), leur qualité, et le périmètre exact couvert par l'accord (filiales incluses ou non, entités affiliées).",
        "red_flags": "Partie mal identifiée ou non signataire réelle du groupe cible ; périmètre flou ne précisant pas si les filiales sont couvertes ; asymétrie entre les parties nommées dans le préambule et celles qui signent.",
    },
    {
        "nom": "Objet / scope",
        "verification": "Vérifier que l'objet du contrat est défini de façon précise et correspond à l'activité réellement exercée ; identifier les prestations/produits couverts et ceux explicitement exclus.",
        "red_flags": "Objet rédigé de façon trop vague ou trop large par rapport à l'activité réelle ; incohérence entre l'objet et les annexes techniques ; silence sur le périmètre géographique.",
    },
    {
        "nom": "Durée",
        "verification": "Relever la date de prise d'effet, la durée initiale, les modalités de reconduction (tacite ou expresse) et le préavis de non-reconduction.",
        "red_flags": "Reconduction tacite avec préavis de non-renouvellement très court ou difficile à respecter ; durée anormalement longue sans possibilité de sortie anticipée ; absence de date de fin claire.",
    },
    {
        "nom": "Résiliation",
        "verification": "Identifier les cas de résiliation (pour faute, pour convenance, de plein droit), le préavis applicable à chacun, et les formalités à respecter (mise en demeure, lettre recommandée).",
        "red_flags": "Résiliation unilatérale possible par la contrepartie sans préavis ni motif ; absence de droit de résiliation pour l'acquéreur en cas de changement de circonstances ; conditions de résiliation asymétriques entre les parties.",
    },
    {
        "nom": "Change of control",
        "verification": "Vérifier l'existence d'une clause de changement de contrôle, ses conditions de déclenchement (seuil de détention, changement d'actionnaire majoritaire) et ses conséquences (résiliation, agrément préalable, information).",
        "red_flags": "Clause de changement de contrôle permettant à la contrepartie de résilier immédiatement l'accord dès l'annonce de l'opération -- risque direct pour la transaction en cours ; absence totale de clause alors que le contrat est stratégique.",
    },
    {
        "nom": "Cession",
        "verification": "Vérifier si le contrat est librement cessible, cessible sous condition d'accord préalable, ou incessible ; distinguer cession du contrat et cession de créance.",
        "red_flags": "Interdiction totale de cession sans possibilité de dérogation ; accord préalable de la contrepartie requis sans encadrement de délai de réponse (risque de blocage).",
    },
    {
        "nom": "Prix",
        "verification": "Relever le mécanisme de prix (fixe, indexé, variable), les modalités de révision, les conditions de paiement (délais, pénalités de retard) et la devise.",
        "red_flags": "Clause d'indexation déséquilibrée ou peu claire ; absence de plafond sur les révisions de prix ; pénalités de retard de paiement asymétriques.",
    },
    {
        "nom": "Engagements de volume / exclusivité",
        "verification": "Identifier les engagements minimaux d'achat/vente, les éventuelles clauses d'exclusivité (territoriale, produit) et leurs sanctions en cas de non-respect.",
        "red_flags": "Engagement de volume minimal disproportionné par rapport à l'activité réelle ; exclusivité de longue durée sans contrepartie équilibrée ; pénalités automatiques élevées en cas de non-atteinte des volumes.",
    },
    {
        "nom": "Responsabilité",
        "verification": "Vérifier l'existence et le niveau des plafonds de responsabilité, les exclusions (dommages indirects, perte de profit) et leur réciprocité entre les parties.",
        "red_flags": "Absence de plafond de responsabilité pour l'une des parties ; plafond asymétrique nettement défavorable ; exclusions de responsabilité rédigées de façon trop large en faveur d'une seule partie.",
    },
    {
        "nom": "Indemnisation",
        "verification": "Relever les clauses d'indemnisation (garantie de passif, indemnisation croisée), leurs plafonds, franchises, et durée de validité des demandes.",
        "red_flags": "Absence de plafond ou de durée limitée à l'obligation d'indemnisation ; franchise trop basse ou absente rendant la clause difficilement gérable ; procédure de réclamation mal définie.",
    },
    {
        "nom": "Garanties",
        "verification": "Identifier les garanties données (conformité, éviction, garanties spécifiques métier) et leur étendue, ainsi que les recours en cas de manquement.",
        "red_flags": "Garanties limitées ou expressément exclues sur des points sensibles pour l'activité ; durée de garantie anormalement courte ; absence de garantie de conformité aux normes applicables.",
    },
    {
        "nom": "Propriété intellectuelle",
        "verification": "Vérifier la titularité des droits de PI (créés avant/pendant le contrat), les licences concédées (portée, exclusivité, durée) et le sort des droits en fin de contrat.",
        "red_flags": "Cession ou licence de PI stratégique à des conditions déséquilibrées ; absence de clause sur la titularité des développements réalisés dans le cadre du contrat ; licence non transférable en cas de changement de contrôle.",
    },
    {
        "nom": "Confidentialité",
        "verification": "Vérifier le périmètre des informations couvertes, la durée de l'obligation (y compris après la fin du contrat) et les exceptions prévues (obligations légales, informations déjà publiques).",
        "red_flags": "Durée de confidentialité perpétuelle ou au contraire trop courte pour protéger des informations sensibles ; définition des informations confidentielles trop vague ; absence de sanction en cas de manquement.",
    },
    {
        "nom": "Données personnelles / cybersécurité",
        "verification": "Vérifier la conformité au RGPD (rôles responsable de traitement/sous-traitant, base légale, durée de conservation), et les engagements de sécurité (mesures techniques, notification d'incident).",
        "red_flags": "Absence de clause RGPD alors que des données personnelles sont traitées ; transferts de données hors UE sans garanties appropriées ; absence d'obligation de notification en cas de violation de données.",
    },
    {
        "nom": "Sous-traitance",
        "verification": "Identifier si la sous-traitance est autorisée, sous quelles conditions (accord préalable, liste de sous-traitants agréés) et le régime de responsabilité du sous-traitant.",
        "red_flags": "Sous-traitance libre sans contrôle ni information du cocontractant ; absence de responsabilité solidaire ou de contrôle sur les sous-traitants successifs (chaîne de sous-traitance non maîtrisée).",
    },
    {
        "nom": "SLA",
        "verification": "Relever les niveaux de service engagés (disponibilité, délais d'intervention, temps de résolution), les modalités de mesure et les pénalités associées.",
        "red_flags": "SLA sans mécanisme de mesure objectif ni pénalité réelle en cas de non-respect ; niveaux de service très en-deçà des standards du secteur ; absence de clause de service credit ou de plan de remédiation.",
    },
    {
        "nom": "Audit",
        "verification": "Vérifier l'existence d'un droit d'audit (financier, technique, sécurité), ses modalités (fréquence, préavis, périmètre) et qui en supporte le coût.",
        "red_flags": "Absence de tout droit d'audit sur un contrat sensible (sous-traitance de données, prestataire critique) ; droit d'audit soumis à un accord préalable de la contrepartie qui peut le refuser.",
    },
    {
        "nom": "Non-concurrence",
        "verification": "Relever la durée, le périmètre géographique et matériel de l'engagement de non-concurrence, ainsi que l'existence d'une contrepartie financière.",
        "red_flags": "Absence de contrepartie financière pour une clause contraignante ; durée ou périmètre disproportionnés par rapport à l'objet du contrat ; clause potentiellement inopposable pour non-conformité aux exigences légales applicables.",
    },
    {
        "nom": "Compliance",
        "verification": "Vérifier les engagements de conformité (anti-corruption, sanctions internationales, droit de la concurrence) et les mécanismes de contrôle associés (déclarations, audits, formation).",
        "red_flags": "Absence totale de clause de conformité anti-corruption sur un contrat international ; absence de clause de sanctions économiques ; silence sur les conséquences d'un manquement à la conformité.",
    },
    {
        "nom": "Assurance",
        "verification": "Identifier les obligations d'assurance (responsabilité civile professionnelle, dommages), les montants de garantie exigés et l'obligation de justificatif.",
        "red_flags": "Absence d'obligation d'assurance alors que l'activité présente un risque significatif ; montants de garantie insuffisants au regard de l'exposition réelle ; absence de justificatif périodique exigé.",
    },
    {
        "nom": "Force majeure",
        "verification": "Vérifier la définition de la force majeure retenue (large/restrictive), les événements couverts et les conséquences (suspension, résiliation) et leurs délais.",
        "red_flags": "Définition de la force majeure extrêmement large permettant à une partie de s'exonérer facilement de ses obligations ; absence de clause de force majeure ; absence de délai avant résiliation en cas de force majeure prolongée.",
    },
    {
        "nom": "Droit applicable",
        "verification": "Relever le droit applicable au contrat et la juridiction ou le mode de règlement des litiges compétent (tribunal, arbitrage, médiation préalable).",
        "red_flags": "Droit applicable étranger inhabituel ou défavorable sans justification claire ; juridiction éloignée ou coûteuse à saisir ; absence de clause de règlement des litiges.",
    },
    {
        "nom": "Modification du contrat",
        "verification": "Vérifier les modalités de modification du contrat (avenant écrit signé des deux parties, procédure spécifique) et l'existence d'un droit de modification unilatérale.",
        "red_flags": "Clause permettant à une seule partie de modifier unilatéralement les conditions (prix, périmètre) sans accord de l'autre ; absence d'exigence de formalisme écrit pour les modifications.",
    },
    {
        "nom": "Survie des obligations",
        "verification": "Identifier les clauses désignées comme survivant à la fin du contrat (confidentialité, non-concurrence, propriété intellectuelle, garanties) et leur durée de survie.",
        "red_flags": "Absence de clause de survie pour des obligations essentielles après la fin du contrat (confidentialité notamment) ; durée de survie non précisée, source d'insécurité juridique.",
    },
]

NOMS_CATEGORIES = [categorie["nom"] for categorie in CATEGORIES_DUE_DILIGENCE]

INSTRUCTIONS_SYSTEME_SYNTHESE = (
    "Tu es un assistant juridique qui rédige une synthèse globale des "
    "points d'attention d'un data room, à partir d'un comptage déjà "
    "calculé par catégorie et niveau de risque (pas des documents "
    "eux-mêmes, tu n'y as pas accès). Mets en avant les catégories avec le "
    "plus de documents à risque élevé, et celles jamais couvertes ou pas "
    "encore analysées. Reste factuel : base-toi uniquement sur les "
    "chiffres fournis, n'invente aucun détail sur le contenu réel des "
    "documents."
)


def analyser_case(doc_id, categorie):
    """
    Une case de la matrice : un document x une catégorie. Appelle
    clauses.detecter_et_analyser(), qui ne lève déjà normalement aucune
    exception (les échecs API connus sont capturés et reflétés dans le
    statut retourné). Le `try/except` ici est un filet de sécurité
    supplémentaire, pour un bug totalement imprévu : cette fonction est
    exécutée dans un thread du ThreadPoolExecutor de
    lancer_analyse_data_room(), et une exception qui s'en échapperait
    ferait perdre le résultat de TOUTES les autres tâches du lot (pas
    seulement celle-ci) en interrompant la boucle `as_completed` du thread
    principal -- inacceptable pour un lot de plusieurs dizaines de cases.
    """
    try:
        return clauses.detecter_et_analyser(
            doc_id, categorie["nom"], categorie["verification"], categorie["red_flags"],
        )
    except Exception as erreur:
        return {"statut": "detection_indisponible", "erreur": str(erreur)}


def lancer_analyse_data_room(max_workers=MAX_WORKERS, on_resultat=None):
    """
    Lance analyser_case() pour chaque combinaison (document, catégorie) en
    parallèle, avec au plus `max_workers` appels simultanés.

    on_resultat, si fourni, est appelé après chaque résultat obtenu avec
    (nombre_traite, nombre_total) -- permet à app.py d'afficher une barre
    de progression pendant le traitement, qui peut prendre plusieurs
    minutes sur un data room de plusieurs documents.

    Les écritures en base (database.enregistrer_analyse_clause) se font
    uniquement ici, dans le thread principal, au fur et à mesure que les
    résultats arrivent (as_completed) -- jamais dans les threads eux-mêmes.

    Retourne la liste de tous les résultats (dicts retournés par
    clauses.detecter_et_analyser), dans l'ordre où ils ont été obtenus (pas
    forcément l'ordre des tâches soumises, puisqu'elles s'exécutent en
    parallèle).
    """
    documents = database.lire_documents()
    taches = [(doc[0], categorie) for doc in documents for categorie in CATEGORIES_DUE_DILIGENCE]

    resultats = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(analyser_case, doc_id, categorie) for doc_id, categorie in taches]
        for future in as_completed(futures):
            resultat = future.result()

            if resultat["statut"] != "detection_indisponible":
                database.enregistrer_analyse_clause(
                    resultat["doc_id"], resultat["type_clause"], resultat["clause_presente"],
                    resultat["texte_extrait"], resultat["analyse"], resultat["niveau_risque"],
                )

            resultats.append(resultat)
            if on_resultat:
                on_resultat(len(resultats), len(taches))

    return resultats


def construire_matrice(analyses, noms_categories=NOMS_CATEGORIES):
    """
    Construit la matrice croisée documents x catégories à partir de
    database.lire_analyses_clauses() (tuples déjà triés du plus récent au
    plus ancien -- voir son ORDER BY). Ne garde que la ligne la plus
    récente par (document, type_clause), pour la même raison que
    database.compter_analyses_par_risque() : une relance du scan ne doit
    pas laisser d'anciennes valeurs traîner dans la matrice affichée.

    Fonction pure (aucun appel réseau, aucun accès direct à la base) :
    testable avec une liste de tuples construite à la main.

    Retourne un pandas.DataFrame (index = nom du document, colonnes =
    noms_categories), chaque cellule valant "Élevé"/"Moyen"/"Faible"
    (analysée), "Absente", "Détectée (non analysée)" (résilience, l'étape
    2 a échoué après une détection réussie) ou "—" (jamais vérifiée pour
    ce document).
    """
    plus_recentes = {}
    for ligne in analyses:
        _, nom, type_clause, presente, texte_extrait, analyse_texte, niveau_risque, date_analyse = ligne
        cle = (nom, type_clause)
        # `analyses` est déjà trié du plus récent au plus ancien : la
        # première occurrence rencontrée pour une clé donnée est la bonne,
        # les suivantes (plus anciennes) sont ignorées.
        if cle not in plus_recentes:
            plus_recentes[cle] = (presente, niveau_risque)

    documents = sorted({nom for (nom, _type_clause) in plus_recentes})

    lignes = []
    for nom_doc in documents:
        ligne = {}
        for type_clause in noms_categories:
            valeurs = plus_recentes.get((nom_doc, type_clause))
            if valeurs is None:
                ligne[type_clause] = "—"
            else:
                presente, niveau_risque = valeurs
                if not presente:
                    ligne[type_clause] = "Absente"
                elif niveau_risque:
                    ligne[type_clause] = niveau_risque.capitalize()
                else:
                    ligne[type_clause] = "Détectée (non analysée)"
        lignes.append(ligne)

    return pd.DataFrame(lignes, index=documents, columns=noms_categories)


def agreger_resultats_data_room():
    """
    Agrège par SQL (database.compter_analyses_par_risque -- un vrai GROUP
    BY, pas un comptage fait par le LLM) les résultats les plus récents
    pour les 24 catégories fixes, et renvoie un texte compact prêt à être
    envoyé au LLM pour rédiger une synthèse -- jamais les analyses brutes.
    """
    comptages = database.compter_analyses_par_risque(types_clause=NOMS_CATEGORIES)

    lignes = []
    for type_clause, niveau_risque, nombre in comptages:
        etat = niveau_risque if niveau_risque else "absente ou non analysée"
        lignes.append(f"{type_clause} : {nombre} document(s) -- {etat}")
    return "\n".join(lignes)


def construire_messages_synthese(comptages_texte):
    """Fonction pure (aucun appel réseau) : testable directement."""
    return [
        {"role": "system", "content": INSTRUCTIONS_SYSTEME_SYNTHESE},
        {"role": "user", "content": f"Comptage des analyses du data room :\n{comptages_texte}"},
    ]


def generer_synthese(comptages_texte):
    """
    Envoie le comptage chiffré (pas les analyses brutes) à un appel LLM
    séparé, qui rédige un texte de synthèse global des points d'attention.

    Contrairement aux autres appels LLM du projet, PAS de schéma JSON
    strict ici : la sortie attendue est un seul texte narratif, il n'y a
    aucun champ à distinguer les uns des autres (pas de séparation
    contexte/connaissance générale comme au RAG, pas de vérification comme
    aux clauses) -- imposer une structure n'apporterait rien ici.

    Ne rattrape aucune erreur (même principe que les autres appels LLM du
    projet) : à l'appelant (app.py) de décider comment réagir.
    """
    messages = construire_messages_synthese(comptages_texte)
    reponse = embeddings.client_openai().chat.completions.create(
        model=llm.MODELE_CHAT,
        messages=messages,
    )
    return reponse.choices[0].message.content
