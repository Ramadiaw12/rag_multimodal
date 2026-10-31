import asyncio, html, tempfile
from pathlib import Path
import streamlit as st

# ═══════════════════════════════════════════════════════════════════════
# CONFIGURATION GLOBALE
# ═══════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="RAG multimodal",
    page_icon="🔦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ═══════════════════════════════════════════════════════════════════════
# DESIGN SYSTEM — palette + animations
# ═══════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
:root{
  --bg:#0a0f1c;
  --bg-soft:#0f1b2d;
  --card:#121f33;
  --gold:#ffd23f;
  --gold-soft:#ffe98a;
  --txt:#e8eefc;
  --txt-dim:#8ea0bf;
  --border:rgba(255,210,63,.18);
  --radius:14px;
}

/* ─── fond global ─── */
.stApp{
  background:
    radial-gradient(1200px 600px at 10% -10%, rgba(255,210,63,.10), transparent 60%),
    radial-gradient(1000px 500px at 100% 0%, rgba(80,120,255,.10), transparent 60%),
    var(--bg);
  color: var(--txt);
}
[data-testid="stHeader"]{background:transparent}

/* ─── typographie ─── */
h1,h2,h3,h4{color:var(--txt)!important;letter-spacing:-.02em}
.stMarkdown p, .stMarkdown li{color:var(--txt);line-height:1.55}

/* ─── animations d'entrée ─── */
@keyframes fadeUp{from{opacity:0;transform:translateY(10px)}to{opacity:1;transform:translateY(0)}}
@keyframes glow{0%,100%{box-shadow:0 0 0 0 rgba(255,210,63,.0)}50%{box-shadow:0 0 0 6px rgba(255,210,63,.08)}}
@keyframes slideIn{from{opacity:0;transform:translateX(-8px)}to{opacity:1;transform:translateX(0)}}
@keyframes shimmer{0%{background-position:-200% 0}100%{background-position:200% 0}}

[data-testid="stChatMessage"]{animation:fadeUp .35s ease both}
.passage{animation:slideIn .3s ease both}

/* ─── cartes de passage ─── */
.passage{
  position:relative;
  border-left:4px solid var(--gold);
  background:linear-gradient(180deg, var(--card), var(--bg-soft));
  padding:1rem 1.2rem;
  margin-bottom:.85rem;
  border-radius:var(--radius);
  color:var(--txt);
  box-shadow:0 4px 20px rgba(0,0,0,.25);
  transition:transform .2s ease, box-shadow .2s ease, border-color .2s ease;
}
.passage:hover{
  transform:translateY(-2px);
  box-shadow:0 8px 30px rgba(255,210,63,.12);
  border-left-color:var(--gold-soft);
}
.passage .src{
  font-size:.82rem;
  color:var(--txt-dim);
  margin-bottom:.45rem;
  display:flex;align-items:center;gap:.4rem;flex-wrap:wrap;
}

/* ─── badge de référence ─── */
.badge{
  background:linear-gradient(135deg,var(--gold),var(--gold-soft));
  color:#0a0f1c;
  font-weight:800;
  font-size:.72rem;
  padding:.15rem .55rem;
  border-radius:999px;
  letter-spacing:.02em;
  box-shadow:0 2px 8px rgba(255,210,63,.35);
}

/* ─── images cadrées ─── */
[data-testid="stImage"] img{
  border:3px solid var(--bg-soft);
  border-radius:var(--radius);
  transition:transform .25s ease, border-color .25s ease, box-shadow .25s ease;
}
[data-testid="stImage"]:hover img{
  transform:scale(1.015);
  border-color:var(--gold);
  box-shadow:0 10px 40px rgba(255,210,63,.15);
}

/* ─── sidebar ─── */
[data-testid="stSidebar"]{
  background:linear-gradient(180deg, var(--bg-soft), var(--bg));
  border-right:1px solid var(--border);
}
[data-testid="stSidebar"] .stRadio label,
[data-testid="stSidebar"] .stSelectbox label{color:var(--txt-dim);font-size:.85rem}

/* ─── barre de saisie ─── */
[data-testid="stChatInput"]{
  background:var(--card);
  border-top:1px solid var(--border);
  border-radius:var(--radius) var(--radius) 0 0;
  padding:.4rem;
}

/* ─── titre animé (shimmer) ─── */
.hero-title{
  font-size:2.6rem;font-weight:900;letter-spacing:-.03em;margin-bottom:.2rem;
  background:linear-gradient(90deg,var(--txt) 20%, var(--gold) 50%, var(--txt) 80%);
  background-size:200% auto;
  -webkit-background-clip:text;background-clip:text;-webkit-text-fill-color:transparent;
  animation:shimmer 5s linear infinite;
}
.hero-sub{color:var(--txt-dim);font-size:.95rem;margin-bottom:1.4rem}

/* ─── pills / chips de statut ─── */
.chip{
  display:inline-flex;align-items:center;gap:.35rem;
  background:rgba(255,210,63,.08);
  border:1px solid var(--border);
  color:var(--gold-soft);
  font-size:.78rem;padding:.25rem .65rem;border-radius:999px;
  margin-right:.35rem;margin-bottom:.4rem;
  animation:glow 3s ease-in-out infinite;
}

/* ─── warning stylé ─── */
[data-testid="stAlert"]{border-radius:var(--radius)!important}

/* ─── tabs ─── */
[data-testid="stTabs"] [data-baseweb="tab-list"]{gap:.4rem;border-bottom:1px solid var(--border)}
[data-testid="stTabs"] [data-baseweb="tab"]{
  border-radius:999px;padding:.35rem .9rem;color:var(--txt-dim);transition:all .2s;
}
[data-testid="stTabs"] [aria-selected="true"]{
  background:rgba(255,210,63,.12);color:var(--gold)!important;
}
</style>
""", unsafe_allow_html=True)

MODES = {"RAG direct": "direct", "Agent avec outil local": "local", "Agent via MCP": "mcp"}


# ═══════════════════════════════════════════════════════════════════════
# LOGIQUE (inchangée)
# ═══════════════════════════════════════════════════════════════════════
@st.cache_resource
def get_agent(mode: str):
    from agent import build_agent, local_tools, mcp_tools
    return build_agent(local_tools() if mode == "local" else asyncio.run(mcp_tools()))


def render_sources(sources, show_ref=True):
    if not sources:
        return
    textes = [s for s in sources if s["type"] == "texte"]
    images = [s for s in sources if s["type"] == "image"]
    tab_t, tab_i = st.tabs([f"📄 Extraits ({len(textes)})", f"🖼️ Images ({len(images)})"])
    with tab_t:
        for s in textes:
            badge = f'<span class="badge">{s["reference"]}</span>' if show_ref else ""
            st.markdown(
                f'<div class="passage">'
                f'<div class="src">{badge}'
                f'<span>{html.escape(str(s["pdf"]))} · p.{s["page"]} '
                f'<span style="color:var(--gold-soft)">({s["similarite"]:.0%} pertinence)</span></span>'
                f'</div>{html.escape((s["contenu"] or "")[:600])}</div>',
                unsafe_allow_html=True,
            )
    with tab_i:
        cols = st.columns(min(len(images), 3) or 1)
        for i, s in enumerate(images):
            with cols[i % 3]:
                ref = f'{s["reference"]} : ' if show_ref else ""
                try:
                    st.image(s["source"], caption=f'{ref}{s["pdf"]}, p.{s["page"]} ({s["similarite"]:.0%})')
                except Exception:
                    st.caption(f'Image introuvable : {s["source"]}')


# ═══════════════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown('<div style="font-size:1.5rem;font-weight:900;letter-spacing:-.03em;">'
                '🔦 <span style="color:var(--gold)">RAG</span> Multimodal</div>',
                unsafe_allow_html=True)
    st.markdown('<div style="color:var(--txt-dim);font-size:.8rem;margin-bottom:1rem;">'
                'Recherche augmentée · Texte &amp; Images</div>', unsafe_allow_html=True)

    st.markdown("#### ⚙️ Configuration")
    mode = MODES[st.radio("Mode", list(MODES), label_visibility="collapsed")]
    st.markdown(
        f'<span class="chip">{"🟢" if mode=="direct" else "🟡"} {mode}</span>',
        unsafe_allow_html=True,
    )

    k = st.slider("Nombre de sources (k)", 1, 15, 6)
    type_res = {"Texte et images": None, "Texte seul": "texte", "Images seules": "image"}[
        st.selectbox("Type de résultats", ["Texte et images", "Texte seul", "Images seules"])]
    max_images = st.slider("Images envoyées au modèle", 0, 5, 3)

    st.markdown("---")
    upload = st.file_uploader("🖼️ Image comme requête (mode direct)", type=["png", "jpg", "jpeg"])

    st.markdown("---")
    if st.button("🗑️ Nouvelle conversation", use_container_width=True):
        st.session_state.msgs = []
        st.rerun()


# 
# HEADER
# 
st.markdown('<div class="hero-title">RAG multimodal</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="hero-sub">Posez une question — le système retrouve les passages et les images '
    'pertinents dans vos PDF puis génère une réponse sourcée.</div>',
    unsafe_allow_html=True,
)

st.session_state.setdefault("msgs", [])

# 
# HISTORIQUE
# 
for m in st.session_state.msgs:
    with st.chat_message(m["role"], avatar="🧑" if m["role"] == "user" else "🤖"):
        st.markdown(m["content"])
        for s in m.get("steps", []):
            st.caption(f"🛠️ `{s['tool']}` {s['args']} → {s.get('summary', 'ok')}")
        for w in m.get("warnings", []):
            st.warning(w)
        render_sources(m.get("sources", []), show_ref=m.get("mode") == "direct")

# ═══════════════════════════════════════════════════════════════════════
# ENTRÉE UTILISATEUR
# ═══════════════════════════════════════════════════════════════════════
if prompt := st.chat_input("Votre question…"):
    history = [{"role": m["role"], "content": m["content"]} for m in st.session_state.msgs[-6:]]
    st.session_state.msgs.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="🧑"):
        st.markdown(prompt)
    with st.chat_message("assistant", avatar="🤖"):
        try:
            steps, warnings = [], []
            if mode == "direct":
                from rag_core import repondre
                image_requete = None
                if upload:
                    image_requete = str(Path(tempfile.gettempdir()) / upload.name)
                    Path(image_requete).write_bytes(upload.getvalue())
                with st.spinner("🔍 Recherche et génération…"):
                    res = repondre(prompt, k, type_res, image_requete, max_images)
                answer, sources, warnings = res["reponse"], res["sources"], res["avertissements"]
                st.markdown(answer)
            else:
                from agent import run_agent
                with st.status("🤖 L'agent travaille…", expanded=True) as s:
                    def on_step(kind, d):
                        s.write(f"🛠️ Appel de **{d['tool']}** `{d['args']}`" if kind == "call"
                                else f"✅ Résultat de {d['tool']} : {d['summary']}")
                    answer, steps, sources = asyncio.run(run_agent(get_agent(mode), prompt, history, on_step))
                    s.update(label=f"✅ Terminé en {len(steps)} appel(s)", state="complete", expanded=False)
                st.markdown(answer)
            for w in warnings:
                st.warning(w)
            render_sources(sources, show_ref=mode == "direct")
            st.session_state.msgs.append({"role": "assistant", "content": answer, "sources": sources,
                                          "steps": steps, "warnings": warnings, "mode": mode})
        except Exception as exc:
            st.error(f"❌ Échec : {exc}")