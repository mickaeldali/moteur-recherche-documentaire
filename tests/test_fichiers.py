from pathlib import Path

import fichiers


def test_sauvegarder_fichier(tmp_path, monkeypatch):
    monkeypatch.setattr(fichiers, "DOSSIER_DOCUMENTS", tmp_path / "documents")

    chemin = fichiers.sauvegarder_fichier("test.txt", b"contenu de test")

    assert Path(chemin).exists()
    assert Path(chemin).read_bytes() == b"contenu de test"


def test_collision_de_nom(tmp_path, monkeypatch):
    monkeypatch.setattr(fichiers, "DOSSIER_DOCUMENTS", tmp_path / "documents")

    chemin1 = fichiers.sauvegarder_fichier("test.txt", b"premier")
    chemin2 = fichiers.sauvegarder_fichier("test.txt", b"second")

    assert chemin1 != chemin2
    assert Path(chemin1).read_bytes() == b"premier"
    assert Path(chemin2).read_bytes() == b"second"


def test_supprimer_fichier(tmp_path, monkeypatch):
    monkeypatch.setattr(fichiers, "DOSSIER_DOCUMENTS", tmp_path / "documents")

    chemin = fichiers.sauvegarder_fichier("test.txt", b"contenu")
    fichiers.supprimer_fichier(chemin)

    assert not Path(chemin).exists()
