import chemins


def test_chemin_existant_est_conserve(tmp_path):
    img = tmp_path / "a.png"
    img.write_bytes(b"x")
    assert chemins.resoudre_image(str(img)) == img


def test_chemin_d_une_autre_machine_est_resolu_par_nom(tmp_path, monkeypatch):
    monkeypatch.setattr(chemins, "IMAGES_DIR", tmp_path)
    (tmp_path / "fig1.png").write_bytes(b"x")
    resolu = chemins.resoudre_image("/home/autre/projet/resultats_pdf/images/fig1.png")
    assert resolu == tmp_path / "fig1.png" and resolu.is_file()


def test_pas_de_remontee_de_repertoire(tmp_path, monkeypatch):
    monkeypatch.setattr(chemins, "IMAGES_DIR", tmp_path)
    assert chemins.resoudre_image("../../etc/passwd").parent == tmp_path