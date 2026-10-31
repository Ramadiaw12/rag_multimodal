# Query + Generation
import base64, io
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from PIL import Image

VLM = ChatOllama(model="qwen2.5vl:7b", base_url="http://localhost:11434", temperature=0.1)

SYSTEM = ("Réponds UNIQUEMENT à partir des passages [T#] et des images [I#] fournis. "
          "Lis attentivement les graphiques et schémas. Cite tes sources ([T1], [I2]). "
          "Si l'information manque, dis-le. Réponds dans la langue de la question.")


def retrieve(query: str, k_text: int = 4, k_images: int = 3) -> list[dict]:
    """À BRANCHER sur votre RAG existant. Doit retourner une liste de dicts :
    {"kind": "text"|"image", "source": str, "page": int, "score": float,
     "text": str (si text), "image_path": str (si image)}"""
    raise NotImplementedError


def _b64(path: str, side: int = 768) -> str:
    img = Image.open(path).convert("RGB")
    img.thumbnail((side, side))
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode()


def build_messages(question: str, evidence: list[dict]):
    texts = [e for e in evidence if e["kind"] == "text"]
    imgs = [e for e in evidence if e["kind"] == "image"]
    ctx = "\n\n".join(f"[T{i}] ({e['source']}, p.{e['page']})\n{e['text']}" for i, e in enumerate(texts, 1))
    legend = ", ".join(f"[I{i}] ({e['source']}, p.{e['page']})" for i, e in enumerate(imgs, 1))
    content = [{"type": "text",
                "text": f"PASSAGES:\n{ctx or '(aucun)'}\n\nIMAGES (dans l'ordre): {legend or '(aucune)'}\n\nQUESTION: {question}"}]
    content += [{"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{_b64(e['image_path'])}"}}
                for e in imgs]
    return [SystemMessage(SYSTEM), HumanMessage(content=content)]


def stream_answer(question: str, evidence: list[dict]):
    for chunk in VLM.stream(build_messages(question, evidence)):
        yield chunk.content


def ask(question: str, k_text: int = 4, k_images: int = 3) -> dict:
    evidence = retrieve(question, k_text, k_images)
    answer = VLM.invoke(build_messages(question, evidence)).content
    return {"answer": answer, "evidence": evidence}