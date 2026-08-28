"""
app.py

Interface Streamlit du moteur de recherche documentaire (projet 3 + étapes
recherche par sens et RAG / projet 4).

Reprend l'import de documents du projet 2 (fichiers.py, extraction.py,
database.py) et propose trois façons d'interroger le CONTENU TEXTUEL des
documents (pas seulement leurs métadonnées comme au projet 2) :
- par MOTS-CLÉS, grâce à l'index plein texte FTS5 (projet 3) ;
- par SENS, grâce aux embeddings OpenAI (étape embeddings) : retrouve un
  passage même s'il ne partage aucun mot avec la requête, tant que le sens
  est proche ;
- en POSANT UNE QUESTION (projet 4, RAG) : retrouve les passages pertinents
  par sens, puis les envoie à un LLM qui rédige une vraie réponse -- tout en
  affichant séparément les extraits bruts, pour rester vérifiable.
"""

from pathlib import Path

import streamlit as st

import database
import embeddings
import extraction
import fichiers
import llm

CATEGORIES = [
    "Corporate", "Contrats", "Contentieux", "Fiscal", "Social",
    "Immobilier", "Financement", "Propriété intellectuelle",
    "Réglementaire", "Autre",
]

FORMATS_ACCEPTES = ["pdf", "docx", "txt"]

database.initialiser_base()

st.title("Moteur de recherche documentaire")

# ----------------------------------------------------------------------
# Import d'un document
# ----------------------------------------------------------------------
st.header("Importer un document")

fichier_uploade = st.file_uploader(
    "Choisir un fichier (PDF, DOCX ou TXT)",
    type=FORMATS_ACCEPTES,
)

categorie_import = st.selectbox("Catégorie juridique", CATEGORIES)
commentaire_import = st.text_input("Commentaire (optionnel)")

if st.button("Importer le document"):
    if fichier_uploade is None:
        st.warning("Choisis d'abord un fichier.")
    else:
        contenu = fichier_uploade.getvalue()
        if len(contenu) == 0:
            st.error("Ce fichier est vide, il n'a pas été importé.")
        else:
            type_fichier = Path(fichier_uploade.name).suffix.lower().lstrip(".")
            chemin = fichiers.sauvegarder_fichier(fichier_uploade.name, contenu)
            commentaire_final = commentaire_import if commentaire_import else None

            id_document = database.ajouter_document(
                nom=fichier_uploade.name,
                type_fichier=type_fichier,
                categorie=categorie_import,
                chemin=chemin,
                commentaire=commentaire_final,
            )

            # L'extraction alimente l'index de recherche par mots-clés (FTS5).
            # Une erreur ici n'empêche pas l'import : le document existe et
            # reste consultable, il ne sera simplement pas trouvable par une
            # recherche de contenu.
            try:
                texte = extraction.extraire_texte(chemin, type_fichier)
                database.indexer_document(id_document, fichier_uploade.name, texte)
                if texte.strip() == "":
                    st.warning(
                        f"Document '{fichier_uploade.name}' importé, mais aucun texte n'a pu "
                        "en être extrait (PDF scanné en image ?) : il ne sera pas trouvable "
                        "par une recherche de contenu."
                    )
                else:
                    st.success(f"Document '{fichier_uploade.name}' importé et indexé avec succès.")

                    # Indexation pour la recherche par SENS (appel à l'API
                    # OpenAI). Séparée de l'indexation FTS5 ci-dessus : si
                    # elle échoue (pas de clé API, pas de réseau, quota
                    # dépassé...), le document reste importé et trouvable
                    # par mots-clés, il ne sera simplement pas trouvable par
                    # une recherche par sens.
                    try:
                        nb_chunks = database.indexer_embeddings(id_document, texte)
                        st.info(f"{nb_chunks} passage(s) indexé(s) pour la recherche par sens.")
                    except embeddings.CleApiManquante:
                        st.info(
                            "Recherche par sens non disponible pour ce document (aucune clé "
                            "API OpenAI configurée dans .env) : il reste trouvable par mots-clés."
                        )
                    except Exception as erreur:
                        st.info(
                            f"Recherche par sens non disponible pour ce document ({erreur}) : "
                            "il reste trouvable par mots-clés."
                        )
            except (extraction.FormatNonSupporte, extraction.ErreurExtraction) as erreur:
                database.indexer_document(id_document, fichier_uploade.name, "")
                st.warning(f"Document importé, mais non indexé : {erreur}")

            st.rerun()

st.divider()

# ----------------------------------------------------------------------
# Recherche dans le contenu des documents, trois modes possibles. Les
# trois interrogent le même contenu texte, mais pas de la même façon :
# mots-clés compare des mots, par sens compare des vecteurs numériques
# représentant le sens, et poser une question ajoute une étape de
# génération : le LLM rédige une réponse à partir des passages retrouvés
# par sens (RAG -- retrieval-augmented generation, projet 4).
# ----------------------------------------------------------------------
st.header("Rechercher dans les documents")

mode_recherche = st.radio(
    "Mode de recherche",
    ["Mots-clés (FTS5)", "Par sens (embeddings)", "Poser une question (IA)"],
    horizontal=True,
    help=(
        "Mots-clés : retrouve les mots tapés tels quels (ou leur préfixe). "
        "Par sens : retrouve les passages dont le SENS se rapproche de la "
        "requête, même avec des mots différents. Poser une question : en "
        "plus de retrouver les passages, un LLM rédige une réponse à "
        "partir d'eux. Les deux derniers modes nécessitent une clé API "
        "OpenAI configurée dans .env."
    ),
)
recherche_texte = st.text_input(
    "Rechercher dans le contenu, ou poser une question",
    placeholder="ex. clause de non-concurrence, ou : quel est le préavis de résiliation du bail ?",
)
categorie_filtre = st.selectbox("Filtrer par catégorie", ["Toutes"] + CATEGORIES)

if mode_recherche == "Mots-clés (FTS5)":
    resultats = database.rechercher(recherche_texte, categorie_filtre)

    st.write(f"{len(resultats)} document(s) trouvé(s)")

    # ------------------------------------------------------------------
    # Affichage des résultats, aperçu, modification, suppression
    # ------------------------------------------------------------------
    for resultat in resultats:
        id_doc, nom, type_fichier, categorie, date_ajout, chemin, commentaire, extrait = resultat

        with st.expander(f"{nom} — {categorie} — {date_ajout}"):
            st.write(f"Type : {type_fichier}")
            st.write(f"Chemin : {chemin}")

            if extrait:
                st.markdown(f"Extrait correspondant : {extrait}")

            if st.button("Voir un aperçu du texte", key=f"apercu_{id_doc}"):
                try:
                    texte = extraction.extraire_texte(chemin, type_fichier)
                    if texte.strip() == "":
                        st.warning("Aucun texte n'a pu être extrait de ce document.")
                    else:
                        st.text_area("Aperçu", texte[:1000], height=200, key=f"texte_{id_doc}")
                except extraction.FormatNonSupporte:
                    st.info("L'aperçu n'est pas disponible pour ce format.")
                except extraction.ErreurExtraction as erreur:
                    st.error(str(erreur))

            st.subheader("Modifier")
            index_categorie = CATEGORIES.index(categorie) if categorie in CATEGORIES else 0
            nouvelle_categorie = st.selectbox(
                "Catégorie", CATEGORIES, index=index_categorie, key=f"cat_{id_doc}",
            )
            nouveau_commentaire = st.text_input(
                "Commentaire", value=commentaire or "", key=f"com_{id_doc}",
            )
            if st.button("Enregistrer les modifications", key=f"save_{id_doc}"):
                database.modifier_document(id_doc, nouvelle_categorie, nouveau_commentaire or None)
                st.success("Document mis à jour.")
                st.rerun()

            st.subheader("Supprimer")
            confirmer = st.checkbox("Confirmer la suppression", key=f"confirm_{id_doc}")
            if st.button("Supprimer ce document", key=f"delete_{id_doc}"):
                if confirmer:
                    database.supprimer_document(id_doc)
                    fichiers.supprimer_fichier(chemin)
                    st.success("Document supprimé.")
                    st.rerun()
                else:
                    st.warning("Coche la case de confirmation avant de supprimer.")

elif mode_recherche == "Par sens (embeddings)":
    # Un appel API est nécessaire pour calculer l'embedding de la requête
    # elle-même, donc les mêmes erreurs qu'à l'import peuvent survenir (clé
    # absente, réseau, quota). On les affiche clairement au lieu de faire
    # planter la page.
    if not recherche_texte.strip():
        resultats_sens = []
    else:
        try:
            resultats_sens = database.rechercher_par_sens(recherche_texte, categorie_filtre)
        except embeddings.CleApiManquante as erreur:
            st.error(str(erreur))
            resultats_sens = []
        except Exception as erreur:
            st.error(f"Recherche par sens indisponible : {erreur}")
            resultats_sens = []

    st.write(f"{len(resultats_sens)} passage(s) trouvé(s)")

    for resultat in resultats_sens:
        id_doc, nom, type_fichier, categorie, date_ajout, chemin, commentaire, texte_chunk, score = resultat

        with st.expander(f"{nom} — {categorie} — proximité de sens {score:.2f}"):
            st.write(f"Type : {type_fichier}")
            st.write(f"Chemin : {chemin}")
            st.markdown(f"Passage correspondant : *{texte_chunk}*")

else:
    # Mode "Poser une question" (RAG, projet 4) : on retrouve d'abord les
    # passages pertinents par sens (comme le mode précédent), puis on les
    # envoie à un LLM qui rédige une réponse à partir d'eux -- jamais à
    # partir de sa propre mémoire des documents, puisqu'il n'en a aucune.
    if not recherche_texte.strip():
        chunks_pertinents = []
    else:
        try:
            resultats_bruts = database.rechercher_par_sens(
                recherche_texte, categorie_filtre, top_n=llm.NB_CHUNKS_CONTEXTE,
            )
            chunks_pertinents = llm.filtrer_chunks_pertinents(resultats_bruts)
        except embeddings.CleApiManquante as erreur:
            st.error(str(erreur))
            chunks_pertinents = []
        except Exception as erreur:
            st.error(f"Recherche indisponible : {erreur}")
            chunks_pertinents = []

    if recherche_texte.strip() and not chunks_pertinents:
        st.info("Aucun passage suffisamment pertinent n'a été trouvé pour répondre à cette question.")

    if chunks_pertinents:
        # La génération de la réponse est un second appel API, séparé de la
        # recherche des passages ci-dessus : s'il échoue (clé absente,
        # quota, réseau...), les passages bruts restent affichés plus bas,
        # seule la synthèse rédigée est indisponible.
        try:
            reponse = llm.generer_reponse(recherche_texte, chunks_pertinents)

            st.success(f"Réponse trouvée dans le document : {reponse['reponse_document']}")

            if reponse["complement_connaissance_generale"]:
                st.warning(
                    "Connaissance générale du LLM (hors document, à vérifier) : "
                    f"{reponse['complement_connaissance_generale']}"
                )
        except embeddings.CleApiManquante as erreur:
            st.error(str(erreur))
        except Exception as erreur:
            st.error(f"Impossible de générer une réponse rédigée : {erreur}")

        # Sources : les extraits bruts tels que stockés en base, jamais
        # reformulés par le LLM -- c'est sur eux que la réponse peut être
        # vérifiée.
        st.subheader("Sources")
        for resultat in chunks_pertinents:
            id_doc, nom, type_fichier, categorie, date_ajout, chemin, commentaire, texte_chunk, score = resultat
            with st.expander(f"{nom} — {categorie} — proximité de sens {score:.2f}"):
                st.write(f"Type : {type_fichier}")
                st.write(f"Chemin : {chemin}")
                st.markdown(f"Extrait original : *{texte_chunk}*")
