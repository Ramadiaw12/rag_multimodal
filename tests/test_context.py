#   FIchier pour tester les fonctions de context.py
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

import context as c


def src(id_, type_="texte", page=1, sim=0.8, contenu="un texte"):
    return {"id": id_, "type": type_, "pdf": "p.pdf", "page": page, "contenu": contenu, "similarite": sim}


def texte_long(n):
    return " ".join(f"mot{i}" for i in range(n))

# --- conversation utils
def conv(n, reponse=lambda i: f"r{i}"):
    msgs = []
    for i in range(n):
        msgs += [HumanMessage(f"q{i}"), AIMessage(reponse(i))]
    return msgs


# ---- retrieval filtering
def test_filtrage_ecarte_les_sources_trop_eloignees_du_meilleur():
    sources = [src("a", sim=0.80), src("b", page=2, contenu="autre", sim=0.60)]
    assert [s["id"] for s in c.filtrer_sources(sources)] == ["a"]


def test_filtrage_compare_texte_et_image_separement():
    sources = [src("t", sim=0.80), src("i", "image", sim=0.30)]
    assert {s["id"] for s in c.filtrer_sources(sources)} == {"t", "i"}


def test_filtrage_dedoublonne():
    sources = [src("a", contenu="même texte", sim=0.80), src("b", contenu="même texte", sim=0.79)]
    assert len(c.filtrer_sources(sources)) == 1


def test_filtrage_seuil_absolu(monkeypatch):
    monkeypatch.setattr(c, "SIM_MIN_TEXTE", 0.9)
    assert c.filtrer_sources([src("a", sim=0.8)]) == []


# ---- compression
TEXTE = ("The model uses residual connections. Multi-head attention allows the model to attend to "
         "different subspaces. Dropout is applied to the output. We trained on WMT 2014 data. ") * 4


def test_compression_respecte_la_taille():
    assert len(c.compresser_passage(TEXTE, "multi-head attention", 200)) <= 200


def test_compression_garde_la_phrase_pertinente():
    assert "Multi-head attention" in c.compresser_passage(TEXTE, "multi-head attention", 200)


def test_texte_court_inchange():
    assert c.compresser_passage("Court.", "x", 450) == "Court."


# ---- fenêtre de conversation
def test_fenetre_garde_les_derniers_echanges():
    msgs = conv(6) + [HumanMessage("courante")]
    assert [m.content for m in c.fenetre(msgs, tours=2)] == ["q4", "r4", "q5", "r5", "courante"]


def test_fenetre_sans_reponse_orpheline_en_tete():
    out = c.fenetre([AIMessage("orphelin"), HumanMessage("q")], tours=2)
    assert [m.content for m in out] == ["q"]


# ---- budget de tokens
def etat_avec_outils():
    return conv(4) + [
        HumanMessage("courante"),
        AIMessage("", tool_calls=[{"name": "rechercher_sources", "args": {"question": "x"}, "id": "1"}]),
        ToolMessage(texte_long(400), tool_call_id="1"),
        AIMessage("", tool_calls=[{"name": "rechercher_sources", "args": {"question": "y"}, "id": "2"}]),
        ToolMessage(texte_long(400), tool_call_id="2"),
    ]


def test_budget_ne_modifie_pas_l_etat():
    msgs = etat_avec_outils()
    avant = [m.content for m in msgs]
    c.ajuster_au_budget(msgs, "système", 0, budget=1400)
    assert [m.content for m in msgs] == avant


def test_budget_tronque_les_anciens_resultats_et_garde_le_dernier():
    envoyes, stats = c.ajuster_au_budget(etat_avec_outils(), "système", 0, budget=1400)
    outils = [m for m in envoyes if isinstance(m, ToolMessage)]
    assert len(outils[0].content) < len(texte_long(400))
    assert outils[-1].content == texte_long(400)
    assert stats["outils_tronques"] >= 1 and stats["apres"] <= stats["budget"]


def test_budget_oublie_les_vieux_echanges_si_necessaire():
    msgs = conv(4, reponse=lambda i: texte_long(200)) + [HumanMessage("courante")]
    envoyes, stats = c.ajuster_au_budget(msgs, "système", 0, budget=600)
    assert stats["echanges_perdus"] >= 1 and stats["apres"] <= 600
    assert envoyes[-1].content == "courante"


def test_comptage_de_tokens_positif_et_croissant():
    assert 0 < c.count_tokens("bonjour") < c.count_tokens(texte_long(50))