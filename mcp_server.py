"""Serveur MCP : expose les outils de outils.py à tout client MCP (agent, Claude Desktop, IDE…).
Lancer : uv run python mcp_server.py   ->   http://127.0.0.1:8000/mcp"""
import os

from mcp.server.fastmcp import FastMCP

import outils

mcp = FastMCP("RAG multimodal", host=os.getenv("MCP_HOST", "127.0.0.1"),
              port=int(os.getenv("MCP_PORT", "8000")))

for fonction in outils.OUTILS:
    mcp.tool()(fonction)

if __name__ == "__main__":
    import rag_core  # noqa: F401  (précharge CLIP et la base avant la première requête)
    mcp.run(transport="streamable-http")
