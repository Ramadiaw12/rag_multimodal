import asyncio, html, tempfile
from pathlib import Path
import streamlit as st

st.set_page_config(page_title="RAG multimodal", page_icon="🔦", layout="wide")
st.markdown("""
<style>
.passage{border-left:5px solid #ffd23f;background:#fff;padding:.7rem 1rem;margin-bottom:.7rem;border-radius:2px;color:#0f1b2d}
.passage .src{font-size:.82rem;color:#3a4a63;margin-bottom:.3rem}
.badge{background:#0f1b2d;color:#ffd23f;font-weight:800;font-size:.75rem;padding:.05rem .45rem;border-radius:2px;margin-right:.5rem}
[data-testid="stImage"] img{border:6px solid #0f1b2d;border-radius:2px}
</style>""", unsafe_allow_html=True)

MODES = {"RAG direct": "direct", "Agent avec outil local": "local", "Agent via MCP": "mcp"}


@st.cache_resource
def get_agent(mode: str):
    from agent import build_agent, local_tools, mcp_tools
    return build_agent(local_tools() if mode == "local" else asyncio.run(mcp_tools()))


def render_sources(sources, show_ref=True):
    if not sources:
        return
    textes = [s for s in sources if s["type"] == "texte"]
    images = [s for s in sources if s["type"] == "image"]
    tab_t, tab_i = st.tabs([f"Extraits ({len(textes)})", f"Images ({len(images)})"])
    with tab_t:
        for s in textes:
            badge = f'<span class="badge">{s["reference"]}</span>' if show_ref else ""
            st.markdown(f'<div class="passage"><div class="src">{badge}{html.escape(str(s["pdf"]))}, '
                        f'p.{s["page"]} ({s["similarite"]:.0%})</div>'
                        f'{html.escape((s["contenu"] or "")[:600])}</div>', unsafe_allow_html=True)
    with tab_i:
        cols = st.columns(min(len(images), 3) or 1)
        for i, s in enumerate(images):
            with cols[i % 3]:
                ref = f'{s["reference"]} : ' if show_ref else ""
                try:
                    st.image(s["source"], caption=f'{ref}{s["pdf"]}, p.{s["page"]} ({s["similarite"]:.0%})')
                except Exception:
                    st.caption(f'Image introuvable : {s["source"]}')


with st.sidebar:
    mode = MODES[st.radio("Mode", list(MODES))]
    k = st.slider("Nombre de sources (k)", 1, 15, 6)
    type_res = {"Texte et images": None, "Texte seul": "texte", "Images seules": "image"}[
        st.selectbox("Type de résultats", ["Texte et images", "Texte seul", "Images seules"])]
    max_images = st.slider("Images envoyées au modèle", 0, 5, 3)
    upload = st.file_uploader("Image comme requête (mode direct)", type=["png", "jpg", "jpeg"])
    if st.button("Nouvelle conversation"):
        st.session_state.msgs = []
        st.rerun()

st.title("RAG multimodal")
st.session_state.setdefault("msgs", [])

for m in st.session_state.msgs:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        for s in m.get("steps", []):
            st.caption(f"{s['tool']} {s['args']} → {s.get('summary', 'ok')}")
        for w in m.get("warnings", []):
            st.warning(w)
        render_sources(m.get("sources", []), show_ref=m.get("mode") == "direct")

if prompt := st.chat_input("Votre question"):
    history = [{"role": m["role"], "content": m["content"]} for m in st.session_state.msgs[-6:]]
    st.session_state.msgs.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        try:
            steps, warnings = [], []
            if mode == "direct":
                from rag_core import repondre
                image_requete = None
                if upload:  # adaptez si rechercher(image=...) attend un autre format qu'un chemin
                    image_requete = str(Path(tempfile.gettempdir()) / upload.name)
                    Path(image_requete).write_bytes(upload.getvalue())
                with st.spinner("Recherche et génération…"):
                    res = repondre(prompt, k, type_res, image_requete, max_images)
                answer, sources, warnings = res["reponse"], res["sources"], res["avertissements"]
                st.markdown(answer)
            else:
                from agent import run_agent
                with st.status("L'agent travaille…", expanded=True) as s:
                    def on_step(kind, d):
                        s.write(f"Appel de **{d['tool']}** `{d['args']}`" if kind == "call"
                                else f"Résultat de {d['tool']} : {d['summary']}")
                    answer, steps, sources = asyncio.run(run_agent(get_agent(mode), prompt, history, on_step))
                    s.update(label=f"Terminé en {len(steps)} appel(s)", state="complete", expanded=False)
                st.markdown(answer)
            for w in warnings:
                st.warning(w)
            render_sources(sources, show_ref=mode == "direct")
            st.session_state.msgs.append({"role": "assistant", "content": answer, "sources": sources,
                                          "steps": steps, "warnings": warnings, "mode": mode})
        except Exception as exc:
            st.error(f"Échec : {exc}")