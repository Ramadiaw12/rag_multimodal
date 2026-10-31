import asyncio, html
import streamlit as st

st.set_page_config(page_title="OmniRAG", page_icon="🔦", layout="wide")
st.markdown("""
<style>
.passage{border-left:5px solid #ffd23f;background:#fff;padding:.7rem 1rem;margin-bottom:.7rem;border-radius:2px}
.passage .src{font-size:.82rem;color:#3a4a63;margin-bottom:.3rem}
.badge{background:#0f1b2d;color:#ffd23f;font-weight:800;font-size:.75rem;padding:.05rem .45rem;border-radius:2px;margin-right:.5rem}
[data-testid="stImage"] img{border:6px solid #0f1b2d;border-radius:2px}
</style>""", unsafe_allow_html=True)

MODES = {"RAG direct": "direct", "Agent avec outil local": "local", "Agent via MCP": "mcp"}


@st.cache_resource
def get_agent(mode: str):
    from agent import build_agent, local_tools, mcp_tools
    return build_agent(local_tools() if mode == "local" else asyncio.run(mcp_tools()))


def render_evidence(evidence):
    texts = [e for e in evidence if e["kind"] == "text"]
    imgs = [e for e in evidence if e["kind"] == "image"]
    if not evidence:
        return
    tab_t, tab_i = st.tabs([f"Passages ({len(texts)})", f"Images ({len(imgs)})"])
    with tab_t:
        for i, e in enumerate(texts, 1):
            st.markdown(f'<div class="passage"><div class="src"><span class="badge">T{i}</span>'
                        f'{html.escape(e["source"])}, p.{e["page"]} ({e["score"]:.0%})</div>'
                        f'{html.escape(e["text"][:600])}</div>', unsafe_allow_html=True)
    with tab_i:
        cols = st.columns(min(len(imgs), 3) or 1)
        for i, e in enumerate(imgs, 1):
            with cols[(i - 1) % 3]:
                st.image(e["image_path"], caption=f'I{i} : {e["source"]}, p.{e["page"]} ({e["score"]:.0%})')


with st.sidebar:
    mode = MODES[st.radio("Mode", list(MODES))]
    k_t = st.slider("Passages de texte", 1, 10, 4)
    k_i = st.slider("Images", 0, 6, 3)
    if st.button("Nouvelle conversation"):
        st.session_state.msgs = []
        st.rerun()

st.title("OmniRAG")
st.session_state.setdefault("msgs", [])

for m in st.session_state.msgs:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        for s in m.get("steps", []):
            st.caption(f"{s['tool']} {s['args']} → {s.get('summary', 'ok')}")
        render_evidence(m.get("evidence", []))

if prompt := st.chat_input("Votre question"):
    history = [{"role": m["role"], "content": m["content"]} for m in st.session_state.msgs[-6:]]
    st.session_state.msgs.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        try:
            if mode == "direct":
                import rag_core
                with st.status("Recherche multimodale…") as s:
                    evidence = rag_core.retrieve(prompt, k_t, k_i)
                    s.update(label=f"{len(evidence)} preuves trouvées", state="complete")
                answer = st.write_stream(rag_core.stream_answer(prompt, evidence))
                steps = []
            else:
                from agent import run_agent
                with st.status("L'agent travaille…", expanded=True) as s:
                    def on_step(kind, d):
                        s.write(f"Appel de **{d['tool']}** `{d['args']}`" if kind == "call"
                                else f"Résultat de {d['tool']} : {d['summary']}")
                    answer, steps, evidence = asyncio.run(run_agent(get_agent(mode), prompt, history, on_step))
                    s.update(label=f"Terminé en {len(steps)} appel(s)", state="complete", expanded=False)
                st.markdown(answer)
            render_evidence(evidence)
            st.session_state.msgs.append({"role": "assistant", "content": answer,
                                          "evidence": evidence, "steps": steps})
        except Exception as exc:
            st.error(f"Échec : {exc}")