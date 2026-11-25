# Ce fichier contient le backend web FastAPI + SSE pour l'application RAG multimodal.
"""Backend web : FastAPI + Server-Sent Events.
Lancer : uv run uvicorn server:app --port 8080

Réglages (variables d'environnement) :
  AGENT_TOOLS        "local" (défaut) : l'agent exécute ses outils dans ce processus
                     "mcp"            : l'agent les appelle sur le serveur MCP (mcp_server.py)
  REQUEST_TIMEOUT_S  durée maximale d'une réponse, en secondes (défaut : 120)
"""
import asyncio
import json
import logging
import os
import time
from pathlib import Path
from typing import Literal

import httpx
from fastapi import FastAPI, HTTPException, Response
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field

import agent as ag
import metrics
import rag_core
from chemins import resoudre_image
from retrieval import collection

# variables globales pour le serveur FastAPI : agent, timeout, logger
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
log = logging.getLogger("rag.server")
ROOT = Path(__file__).resolve().parent

# Réglages de l'agent et du serveur FastAPI
AGENT_TOOLS = os.getenv("AGENT_TOOLS", "local").lower()
if AGENT_TOOLS not in ("local", "mcp"):
    raise RuntimeError(f"AGENT_TOOLS doit valoir 'local' ou 'mcp' (reçu : {AGENT_TOOLS!r}).")
TIMEOUT_S = float(os.getenv("REQUEST_TIMEOUT_S", 120))

app = FastAPI(title="RAG multimodal")
_agent = None
_agent_lock = asyncio.Lock()


# Pydantic models pour la validation des entrées et sorties JSON
class ChatIn(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    mode: Literal["direct", "agent"] = "agent"
    k: int = Field(default=6, ge=1, le=15)
    history: list[dict] = Field(default_factory=list, max_length=20)

# Alors là on ne peut pas faire mieux que de copier-coller la docstring de rag_core.py, qui est la source unique de vérité pour l'agent
def historique_propre(history: list[dict]) -> list[dict]:
    """Le navigateur n'est pas fiable : on ne garde que des messages user/assistant en texte,
    jamais de rôle system ou tool injecté par le client."""
    propres = [{"role": m["role"], "content": str(m.get("content", ""))[:4000]}
               for m in history if m.get("role") in ("user", "assistant")]
    return propres[-6:]

# la fonction public() prépare une source pour le navigateur : pas de chemin disque, une URL d'image à la place
def public(source: dict) -> dict:
    """Prépare une source pour le navigateur : pas de chemin disque, une URL d'image à la place."""
    s = {k: v for k, v in source.items() if k != "source"}
    if s["type"] == "image":
        s["image_url"] = f"/api/image?id={s['id']}"
    return s


def sse(event: str, data: dict) -> str:
    return f"data: {json.dumps({'event': event, **data}, ensure_ascii=False, default=float)}\n\n"

# Ici la fonction cause_racine descend dans les ExceptionGroup (anyio/MCP) jusqu'à l'erreur d'origine, pour ne pas afficher une pile de 20 lignes d'ExceptionGroup
def cause_racine(exc: BaseException) -> BaseException:
    """Descend dans les ExceptionGroup (anyio/MCP) jusqu'à l'erreur d'origine."""
    while isinstance(exc, BaseExceptionGroup) and exc.exceptions:
        exc = exc.exceptions[0]
    return exc


def message_erreur(exc: BaseException) -> str:
    racine = cause_racine(exc)
    if AGENT_TOOLS == "mcp" and isinstance(racine, (httpx.ConnectError, ConnectionError)):
        return ("Serveur MCP injoignable. Lancez-le avec : uv run python mcp_server.py "
                f"(URL attendue : {ag.MCP_URL})")
    return f"{type(racine).__name__}: {racine}"

# Asynchronous function to check if the MCP server is reachable
async def mcp_joignable() -> bool:
    """Vrai si le serveur MCP répond (n'importe quel code HTTP prouve qu'il écoute)."""
    try:
        async with httpx.AsyncClient(timeout=1.5) as client:
            await client.get(ag.MCP_URL)
        return True
    except httpx.HTTPError:
        return False

# on a un agent global, construit à la première requête. En cas d'échec (MCP éteint), rien n'est mis en cache
async def get_agent():
    """Construit l'agent à la première requête. En cas d'échec (MCP éteint), rien n'est mis en cache."""
    global _agent
    async with _agent_lock:
        if _agent is None:
            tools = ag.local_tools() if AGENT_TOOLS == "local" else await ag.mcp_tools()
            _agent = ag.build_agent(tools)
        return _agent

# FastAPI endpoints
@app.get("/")
def index():
    return FileResponse(ROOT / "static" / "index.html")

# check le 
@app.get("/api/health")
async def health():
    return {"indexed": await asyncio.to_thread(collection.count), "agent_tools": AGENT_TOOLS,
            "mcp_ok": await mcp_joignable() if AGENT_TOOLS == "mcp" else None}


@app.get("/metrics")
def prometheus():
    return Response(metrics.exporter(), media_type=metrics.CONTENT_TYPE)


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


@app.post("/api/chat")
async def chat(body: ChatIn):
    question = body.question.strip()
    if not question:
        raise HTTPException(400, "Question vide")
    queue: asyncio.Queue = asyncio.Queue()
    suivi = metrics.Suivi()

    async def executer():
        if body.mode == "direct":
            res = await asyncio.to_thread(rag_core.repondre, question, body.k)
            suivi.evenement("sources", res["sources"])
            await queue.put(("sources", {"sources": [public(s) for s in res["sources"]]}))
            await queue.put(("answer", {"text": res["reponse"], "warnings": res["avertissements"]}))
            return
        agent = await get_agent()

        def on_step(kind, data):
            suivi.evenement(kind, data)
            if kind == "sources":
                queue.put_nowait(("sources", {"sources": [public(s) for s in data]}))
            else:  # "call" | "result" | "usage"
                queue.put_nowait((kind, data))

        answer, _, _ = await ag.run_agent(agent, question, historique_propre(body.history), on_step)
        await queue.put(("answer", {"text": answer}))

    async def work():
        debut, issue = time.perf_counter(), "ok"
        metrics.EN_COURS.inc()
        try:
            await asyncio.wait_for(executer(), timeout=TIMEOUT_S)
        except TimeoutError:
            issue = "délai dépassé"
            metrics.ERREURS.labels("TimeoutError").inc()
            await queue.put(("error", {"message": f"Délai dépassé ({TIMEOUT_S:.0f} s). "
                                                  "Réessayez ou simplifiez la question."}))
        except Exception as exc:
            issue = "erreur"
            metrics.ERREURS.labels(type(cause_racine(exc)).__name__).inc()
            await queue.put(("error", {"message": message_erreur(exc)}))
        finally:
            metrics.fin_requete(body.mode, issue, time.perf_counter() - debut, suivi.sources)
            log.info("chat mode=%s outils=%s issue=%s durée=%.1fs",
                     body.mode, AGENT_TOOLS, issue, time.perf_counter() - debut)
            await queue.put(None)

    async def stream():
        task = asyncio.create_task(work())
        try:
            while (item := await queue.get()) is not None:
                yield sse(*item)
        finally:
            task.cancel()  # client parti ou flux terminé : ne pas laisser un agent tourner dans le vide

    return StreamingResponse(stream(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
