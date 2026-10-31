# Le RAG exposé via MCP
import json
from mcp.server.fastmcp import FastMCP
import rag_core

mcp = FastMCP("OmniRAG", host="127.0.0.1", port=8000)


@mcp.tool()
def search_knowledge_base(query: str, k_text: int = 4, k_images: int = 3) -> str:
    """Recherche dans les documents (texte ET images) SANS rédiger de réponse.
    Utile pour explorer, vérifier un fait ou comparer des passages."""
    return json.dumps({"evidence": rag_core.retrieve(query, k_text, k_images)}, ensure_ascii=False)


@mcp.tool()
def ask_documents(question: str) -> str:
    """Répond à UNE question précise à partir des documents (texte, graphiques, schémas).
    Retourne une réponse sourcée avec les preuves utilisées."""
    return json.dumps(rag_core.ask(question), ensure_ascii=False)


if __name__ == "__main__":
    mcp.run(transport="streamable-http")  # endpoint : http://127.0.0.1:8000/mcp