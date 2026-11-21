# Fichier mcp_server.py : serveur MCP pour exposer les outils RAG multimodal via HTTP
import os

from mcp.server.fastmcp import FastMCP

import rag_core

mcp = FastMCP("RAG multimodal", host=os.getenv("MCP_HOST", "127.0.0.1"),
              port=int(os.getenv("MCP_PORT", "8000")))



# Tools exposés par le serveur MCP : ces fonctions sont accessibles via l'API MCP, pour l'agent distant ou d'autres clients MCP. Elles appellent les fonctions de rag_core pour effectuer la recherche et la génération de réponses.
@mcp.tool()
def rechercher_sources(question: str, k: int = 6, type_resultat: str = "") -> str:
    """Interroge la base vectorielle et renvoie les passages de texte et les images les plus proches
    (id, pdf, page, contenu, similarité), SANS rédiger de réponse. La base est indexée en ANGLAIS :
    formule la requête en anglais. type_resultat : "texte", "image", ou chaîne vide pour les deux."""
    return rag_core.outil_rechercher(question, k, type_resultat)

# Tool pour analyser une image trouvée par rechercher_sources, afin de fournir une observation textuelle sur le schéma, graphique ou tableau. L'ID de l'image est fourni par la source renvoyée par rechercher_sources.
@mcp.tool()
def analyser_image(id_image: str, question: str) -> str:
    """Lit une image trouvée par rechercher_sources (schéma, graphique, tableau) et renvoie une
    observation en texte. id_image = champ 'id' d'une source de type image."""
    return rag_core.outil_analyser_image(id_image, question)


@mcp.tool()
def repondre_question(question: str, k: int = 6, type_resultat: str = "") -> str:
    """Pipeline RAG complet : recherche puis réponse rédigée par un modèle. Pour d'autres clients MCP."""
    return rag_core.outil_repondre(question, k, type_resultat)


if __name__ == "__main__":
    mcp.run(transport="streamable-http")  # http://127.0.0.1:8000/mcp