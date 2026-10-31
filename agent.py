# agent LangChain (outils locaux ou MCP)

import json
from langchain.agents import create_agent
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import tool
from langchain_ollama import ChatOllama

MCP_URL = "http://127.0.0.1:8000/mcp"

SYSTEM_PROMPT = """Tu es un analyste documentaire.
1. Question complexe ou comparative : décompose-la en sous-questions, un appel à ask_documents par sous-question.
2. Pour vérifier un fait : utilise search_knowledge_base.
3. Ne réponds jamais de mémoire sur le contenu des documents. Si rien n'est trouvé, dis-le.
4. Réponse finale synthétique, avec les sources sous la forme (fichier, p.N)."""


def local_tools():
    """Mode 1 : le RAG est un tool LangChain dans le même processus."""
    import rag_core

    @tool
    def search_knowledge_base(query: str, k_text: int = 4, k_images: int = 3) -> str:
        """Recherche dans les documents (texte ET images) SANS rédiger de réponse."""
        return json.dumps({"evidence": rag_core.retrieve(query, k_text, k_images)}, ensure_ascii=False)

    @tool
    def ask_documents(question: str) -> str:
        """Répond à UNE question précise à partir des documents (texte, graphiques, schémas)."""
        return json.dumps(rag_core.ask(question), ensure_ascii=False)

    return [search_knowledge_base, ask_documents]


async def mcp_tools():
    """Mode 2 : l'agent est séparé du RAG et ne connaît que l'URL MCP (aucun import de rag_core)."""
    from langchain_mcp_adapters.client import MultiServerMCPClient
    client = MultiServerMCPClient({"rag": {"url": MCP_URL, "transport": "streamable_http"}})
    return await client.get_tools()


def build_agent(tools, model: str = "qwen3:8b"):
    llm = ChatOllama(model=model, base_url="http://localhost:11434", temperature=0)
    return create_agent(llm, tools, system_prompt=SYSTEM_PROMPT)


def _text(content) -> str:
    if isinstance(content, str):
        return content
    return "".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in content)


async def run_agent(agent, question: str, history=(), on_step=None):
    """Retourne (réponse, étapes, preuves)."""
    messages = [*history, {"role": "user", "content": question}]
    steps, evidence, final = {}, {}, ""
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
                    found = payload.get("evidence", []) if isinstance(payload, dict) else []
                    for e in found:
                        evidence[(e["kind"], e.get("image_path") or e.get("text", "")[:80])] = e
                    step = steps.get(m.tool_call_id, {"tool": m.name, "args": {}})
                    step["summary"] = f"{len(found)} preuve(s)"
                    on_step and on_step("result", dict(step))
    return final, list(steps.values()), list(evidence.values())