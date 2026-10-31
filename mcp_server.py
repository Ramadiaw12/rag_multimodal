from mcp.server.fastmcp import FastMCP
import rag_core

mcp = FastMCP("RAG multimodal", host="127.0.0.1", port=8000)


@mcp.tool()
def rechercher_sources(question: str, k: int = 6, type_resultat: str = "") -> str:
    """Recherche les sources les plus pertinentes (extraits de texte ET images) SANS rédiger de réponse.
    Utile pour explorer, vérifier un fait ou comparer des passages.
    type_resultat : "texte", "image", ou chaîne vide pour les deux."""
    return rag_core.outil_rechercher(question, k, type_resultat)


@mcp.tool()
def repondre_question(question: str, k: int = 6, type_resultat: str = "") -> str:
    """Répond à UNE question précise à partir des documents (texte, graphiques, schémas).
    Retourne la réponse sourcée, les sources utilisées et d'éventuels avertissements.
    type_resultat : "texte", "image", ou chaîne vide pour les deux."""
    return rag_core.outil_repondre(question, k, type_resultat)


if __name__ == "__main__":
    mcp.run(transport="streamable-http")  # http://127.0.0.1:8000/mcp