import pytest

import extraction


def test_extraire_texte_txt(tmp_path):
    fichier = tmp_path / "test.txt"
    fichier.write_text("Contenu juridique de test", encoding="utf-8")

    texte = extraction.extraire_texte(str(fichier), "txt")

    assert texte == "Contenu juridique de test"


def test_format_non_supporte(tmp_path):
    fichier = tmp_path / "test.xyz"
    fichier.write_text("peu importe")

    with pytest.raises(extraction.FormatNonSupporte):
        extraction.extraire_texte(str(fichier), "xyz")


def test_fichier_introuvable():
    with pytest.raises(extraction.ErreurExtraction):
        extraction.extraire_texte("chemin/inexistant.txt", "txt")
