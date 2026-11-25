# Ce fichier contient le serveur MCP qui expose les outils de l'agent à tout client MCP (agent, Claude Desktop, IDE…).
"""Serveur MCP : expose les outils de outils.py à tout client MCP (agent, Claude Desktop, IDE…).
Lancer : uv run python mcp_server.py   ->   http://127.0.0.1:8000/mcp"""
import os

from mcp.server.fastmcp import FastMCP

import outils

# Exposion des outils de l'agent à tout client MCP pour qu'il puisse les invoquer par HTTP. Le serveur MCP ne connaît pas l'agent, il ne fait que relayer les appels
mcp = FastMCP("RAG multimodal", host=os.getenv("MCP_HOST", "127.0.0.1"),
              port=int(os.getenv("MCP_PORT", "8000")))

# On expose les fonctions de outils.py telles quelles, avec leurs docstrings. Le serveur MCP ne connaît pas l'agent, il ne fait que relayer les appels
for fonction in outils.OUTILS:
    mcp.tool()(fonction)

if __name__ == "__main__":
    import rag_core  # noqa: F401  (précharge CLIP et la base avant la première requête)
    mcp.run(transport="streamable-http")
