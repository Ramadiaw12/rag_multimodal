import json, os
from langchain.agents import create_agent
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import tool
from langchain_groq import ChatGroq

MCP_URL = os.getenv("MCP_URL", "http://127.0.0.1:8000/mcp")

SYSTEM_PROMPT = """Tu es un analyste documentaire. Outils : rechercher_sources, repondre_question.
1. Question complexe ou comparative : décompose-la en sous-questions, un appel à repondre_question par sous-question.
2. Pour vérifier ou recouper un fait : utilise rechercher_sources.
3. Ne réponds jamais de mémoire sur le contenu des documents. Si rien n'est trouvé, dis-le.
4. Réponse finale en français, synthétique. Les références [S1], [S2] changent à chaque appel d'outil :
   ne les reprends pas, cite les sources sous la forme (nom du pdf, p.N)."""


def build_llm():
    # GROQ_API_KEY est lue dans l'environnement. Modèle avec tool calling fiable.
    return ChatGroq(model=os.getenv("AGENT_MODEL", "llama-3.3-70b-versatile"), temperature=0)


def local_tools():
    """Mode 1 : le RAG est un tool LangChain dans le même processus."""
    import rag_core

    @tool
    def rechercher_sources(question: str, k: int = 6, type_resultat: str = "") -> str:
        """Recherche les sources (texte ET images) SANS rédiger de réponse.
        type_resultat : "texte", "image", ou chaîne vide pour les deux."""
        return rag_core.outil_rechercher(question, k, type_resultat)

    @tool
    def repondre_question(question: str, k: int = 6, type_resultat: str = "") -> str:
        """Répond à UNE question précise à partir des documents (texte, graphiques, schémas)."""
        return rag_core.outil_repondre(question, k, type_resultat)

    return [rechercher_sources, repondre_question]


async def mcp_tools():
    """Mode 2 : l'agent est séparé du RAG, il ne connaît que l'URL MCP (aucun import de rag_core)."""
    from langchain_mcp_adapters.client import MultiServerMCPClient
    client = MultiServerMCPClient({"rag": {"url": MCP_URL, "transport": "streamable_http"}})
    return await client.get_tools()

# build_agent, _text et run_agent : inchangés


def build_agent(tools):
    return create_agent(build_llm(), tools, system_prompt=SYSTEM_PROMPT)


def _text(content) -> str:
    if isinstance(content, str):
        return content
    return "".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in content)


async def run_agent(agent, question: str, history=(), on_step=None):
    """Retourne (réponse, étapes, sources)."""
    messages = [*history, {"role": "user", "content": question}]
    steps, sources, final = {}, {}, ""
    async for update in agent.astream({"messages": messages}, stream_mode="updates"):
        for node_out in update.values():
            for m in (node_out or {}).get("messages", []):
                if isinstance(m, AIMessage):
                    for c in m.tool_calls:
                        steps[c["id"]] = {"tool": c["name"], "args": c["args"]}
                        on_step and on_step("call", steps[c["id"]])
                    if not m.tool_calls:
                        final = _text(m.content)
                elif isinstance(m, ToolMessage):
                    try:
                        payload = json.loads(_text(m.content))
                    except ValueError:
                        payload = {}
                    found = payload.get("sources", []) if isinstance(payload, dict) else []
                    for s in found:
                        sources[s["id"]] = s
                    step = steps.get(m.tool_call_id, {"tool": m.name, "args": {}})
                    step["summary"] = f"{len(found)} source(s)"
                    on_step and on_step("result", dict(step))
    return final, list(steps.values()), list(sources.values())