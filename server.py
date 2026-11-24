# LE FICHIER server.py est le serveur web principal de l'application RAG multimodale. Il utilise FastAPI pour gérer les requêtes HTTP et les Server-Sent Events (SSE) pour envoyer des mises à jour en temps réel au navigateur. Il définit plusieurs routes pour servir la page HTML principale, vérifier l'état de santé du serveur, servir les images indexées et gérer les requêtes de chat. Le serveur peut fonctionner en mode direct (pipeline RAG complet) ou en mode agent (local ou MCP) pour répondre aux questions des utilisateurs
"""Backend web : FastAPI + Server-Sent Events. Remplace Streamlit.
Lancer : uv run uvicorn server:app --port 8080"""
import asyncio
import json
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

import agent as ag
import rag_core
from retrieval import collection
from chemins import resoudre_image

# On définit la racine du projet et on crée une instance de FastAPI avec un titre. On initialise également un dictionnaire vide pour stocker les agents créés en fonction du mode (local ou MCP).
ROOT = Path(__file__).resolve().parent
app = FastAPI(title="RAG multimodal")
_agents: dict = {}

# On définit un modèle de données Pydantic pour représenter les requêtes de chat. Il contient la question posée par l'utilisateur, le mode de fonctionnement (direct, local ou MCP), le nombre de sources à récupérer (k) et l'historique des échanges précédents.
class ChatIn(BaseModel):
    question: str
    mode: str = "local"      # "direct" | "local" | "mcp"
    k: int = 6
    history: list[dict] = []

# async def get_agent(mode: str): vas definir une fonction asynchrone qui prend en entrée un mode (local ou MCP) et renvoie un agent correspondant. Si l'agent n'existe pas encore dans le dictionnaire _agents, il est créé en utilisant les outils appropriés (locaux ou MCP) et stocké dans le dictionnaire pour une utilisation future
async def get_agent(mode: str):
    if mode not in _agents:
        tools = ag.local_tools() if mode == "local" else await ag.mcp_tools()
        _agents[mode] = ag.build_agent(tools)
    return _agents[mode]

# ici on définit une fonction publique qui prend en entrée un dictionnaire représentant une source et renvoie un dictionnaire contenant les mêmes informations, mais sans le chemin disque de l'image. Si la source est de type "image", on ajoute une clé "image_url" contenant l'URL de l'image à partir de son ID. Cette fonction est utilisée pour préparer les sources avant de les envoyer au navigateur.
def public(source: dict) -> dict:
    """Prépare une source pour le navigateur : pas de chemin disque, une URL d'image à la place."""
    s = {k: v for k, v in source.items() if k != "source"}
    if s["type"] == "image":
        s["image_url"] = f"/api/image?id={s['id']}"
    return s


def sse(event: str, data: dict) -> str:
    return f"data: {json.dumps({'event': event, **data}, ensure_ascii=False, default=float)}\n\n"

# Alors là on définit les routes FastAPI pour le serveur web. La route "/" sert la page HTML principale, la route "/api/health" renvoie le nombre de documents indexés dans la collection ChromaDB, la route "/api/image" sert les images indexées par ID, et la route "/api/chat" gère les requêtes de chat en utilisant soit le mode direct (pipeline RAG complet), soit le mode local ou MCP (agent distant). La route "/api/chat" utilise des Server-Sent Events pour envoyer les résultats au fur et à mesure qu'ils sont générés.
@app.get("/")
def index():
    return FileResponse(ROOT / "static" / "index.html")


@app.get("/api/health")
def health():
    return {"indexed": collection.count()}


@app.get("/api/image")
def image(id: str):
    """Sert uniquement des images indexées (recherche par id), jamais un chemin libre."""
    res = collection.get(ids=[id], include=["metadatas"])
    if not res["ids"] or res["metadatas"][0].get("type") != "image":
        raise HTTPException(404, "Image inconnue")
    path = resoudre_image(res["metadatas"][0]["source"]) 
    if not path.is_file():
        raise HTTPException(404, "Fichier introuvable")
    return FileResponse(path)

# 
@app.post("/api/chat")
async def chat(body: ChatIn):
    question = body.question.strip()
    if not question:
        raise HTTPException(400, "Question vide")
    queue: asyncio.Queue = asyncio.Queue()

    async def work():
        try:
            if body.mode == "direct":
                res = await asyncio.to_thread(rag_core.repondre, question, body.k)
                await queue.put(("sources", {"sources": [public(s) for s in res["sources"]]}))
                await queue.put(("answer", {"text": res["reponse"], "warnings": res["avertissements"]}))
            else:
                agent = await get_agent(body.mode)

                def on_step(kind, data):
                    if kind == "sources":
                        queue.put_nowait(("sources", {"sources": [public(s) for s in data]}))
                    else:  # "call" | "result"
                        queue.put_nowait((kind, data))

                answer, _, _ = await ag.run_agent(agent, question, body.history[-6:], on_step)
                await queue.put(("answer", {"text": answer}))
        except Exception as exc:
            await queue.put(("error", {"message": f"{type(exc).__name__}: {exc}"}))
        finally:
            await queue.put(None)

    async def stream():
        task = asyncio.create_task(work())
        while (item := await queue.get()) is not None:
            yield sse(*item)
        await task

    return StreamingResponse(stream(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})