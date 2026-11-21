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

ROOT = Path(__file__).resolve().parent
app = FastAPI(title="RAG multimodal")
_agents: dict = {}


class ChatIn(BaseModel):
    question: str
    mode: str = "local"      # "direct" | "local" | "mcp"
    k: int = 6
    history: list[dict] = []


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
    path = Path(res["metadatas"][0]["source"])
    if not path.is_file():
        raise HTTPException(404, "Fichier introuvable")
    return FileResponse(path)


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