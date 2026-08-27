"""
app.py

Interface Streamlit du moteur de recherche documentaire (projet 3).

Reprend l'import de documents du projet 2 (fichiers.py, extraction.py,
database.py) et ajoute la nouveauté du projet 3 : la recherche dans le
CONTENU TEXTUEL des documents -- pas seulement leurs métadonnées comme au
projet 2 -- grâce à l'index plein texte (FTS5) construit à chaque import.
"""

from pathlib import Path

import streamlit as st

import database
import extraction
import fichiers

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

            # L'extraction alimente l'index de recherche. Une erreur ici
            # n'empêche pas l'import : le document existe et reste
            # consultable, il ne sera simplement pas trouvable par une
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
            except (extraction.FormatNonSupporte, extraction.ErreurExtraction) as erreur:
                database.indexer_document(id_document, fichier_uploade.name, "")
                st.warning(f"Document importé, mais non indexé : {erreur}")

            st.rerun()

st.divider()

# ----------------------------------------------------------------------
# Recherche dans le contenu des documents
# ----------------------------------------------------------------------
st.header("Rechercher dans les documents")

recherche_texte = st.text_input(
    "Rechercher dans le contenu",
    placeholder="ex. clause de non-concurrence",
)
categorie_filtre = st.selectbox("Filtrer par catégorie", ["Toutes"] + CATEGORIES)

resultats = database.rechercher(recherche_texte, categorie_filtre)

st.write(f"{len(resultats)} document(s) trouvé(s)")

# ----------------------------------------------------------------------
# Affichage des résultats, aperçu, modification, suppression
# ----------------------------------------------------------------------
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
