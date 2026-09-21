import io

import pandas as pd
import pytest
from openpyxl.utils.exceptions import IllegalCharacterError

import export_excel


def test_nettoyer_texte_supprime_les_caracteres_de_controle():
    # \x0b (tabulation verticale) et \x00, \x1f sont interdits par Excel
    assert export_excel.nettoyer_texte_excel("est \x0b faible") == "est  faible"
    assert export_excel.nettoyer_texte_excel("a\x00b\x1fc") == "abc"


def test_nettoyer_texte_conserve_accents_sauts_de_ligne_et_tabulations():
    texte = "Clause de non-concurrence\navec préavis de 3 mois\tà compter de la rupture, é è à ç œ €"
    assert export_excel.nettoyer_texte_excel(texte) == texte


def test_nettoyer_texte_laisse_les_valeurs_non_texte_inchangees():
    assert export_excel.nettoyer_texte_excel(None) is None
    assert export_excel.nettoyer_texte_excel(42) == 42


def test_nettoyer_dataframe_nettoie_les_cellules_sans_modifier_l_original():
    df = pd.DataFrame(
        {"Analyse": ["risque \x0b faible\nà surveiller", None], "Nombre": [1, 2]}
    )

    df_propre = export_excel.nettoyer_dataframe_pour_excel(df)

    assert df_propre.loc[0, "Analyse"] == "risque  faible\nà surveiller"
    assert pd.isna(df_propre.loc[1, "Analyse"])
    assert list(df_propre["Nombre"]) == [1, 2]
    # Le DataFrame d'origine (donc les données de départ) n'est pas modifié
    assert df.loc[0, "Analyse"] == "risque \x0b faible\nà surveiller"


def test_export_excel_plante_sans_nettoyage_et_fonctionne_avec():
    df = pd.DataFrame({"Analyse": ["le risque est \x0b faible à moyen"]})

    # Sans nettoyage : c'est le bug d'origine
    with pytest.raises(IllegalCharacterError):
        with pd.ExcelWriter(io.BytesIO(), engine="openpyxl") as writer:
            df.to_excel(writer, index=False)

    # Avec nettoyage : l'export réussit
    df_propre = export_excel.nettoyer_dataframe_pour_excel(df)
    with pd.ExcelWriter(io.BytesIO(), engine="openpyxl") as writer:
        df_propre.to_excel(writer, index=False)
