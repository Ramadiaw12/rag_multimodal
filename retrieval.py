"""Partie requête de l'index : mêmes modèle, normalisation et collection que l'ingestion du notebook."""
import os
from pathlib import Path

import chromadb
import open_clip
import torch
from PIL import Image

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "ViT-B-32"
PRETRAINED = "laion2b_s34b_b79k"

# Chemin absolu : fonctionne quel que soit le dossier d'où l'on lance le serveur.
CHROMA_PATH = Path(os.getenv("CHROMA_PATH", Path(__file__).resolve().parent / "chroma_pdf"))
COLLECTION_NAME = "pdf_clip_vit_b32"

if not CHROMA_PATH.exists():
    raise FileNotFoundError(
        f"Base ChromaDB introuvable : {CHROMA_PATH}. "
        "Définissez CHROMA_PATH ou placez le dossier 'chroma_pdf' à côté de retrieval.py."
    )

model, _, preprocess = open_clip.create_model_and_transforms(MODEL_NAME, pretrained=PRETRAINED)
model = model.to(DEVICE).eval()
tokenizer = open_clip.get_tokenizer(MODEL_NAME)

client = chromadb.PersistentClient(path=str(CHROMA_PATH))
collection = client.get_or_create_collection(
    name=COLLECTION_NAME,
    embedding_function=None,  # les embeddings sont calculés par notre code
    metadata={"hnsw:space": "cosine"},
)


@torch.inference_mode()
def encoder_textes(textes):
    tokens = tokenizer(textes).to(DEVICE)
    vecteurs = model.encode_text(tokens).float()
    vecteurs = vecteurs / vecteurs.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    return vecteurs.cpu().tolist()


@torch.inference_mode()
def encoder_image(chemin):
    with Image.open(chemin) as image:
        tenseur = preprocess(image.convert("RGB"))
    tenseur = tenseur.unsqueeze(0).to(DEVICE)
    vecteur = model.encode_image(tenseur).float()
    vecteur = vecteur / vecteur.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    return vecteur.cpu().tolist()[0]


def rechercher(texte=None, image=None, type_resultat=None, k=5):
    """
    Fournir soit un texte, soit un chemin d'image.
    type_resultat : None, "texte" ou "image".
    """
    if (texte is None) == (image is None):
        raise ValueError("Fournissez soit un texte, soit une image.")
    if type_resultat not in (None, "texte", "image"):
        raise ValueError("type_resultat doit être 'texte', 'image' ou None.")
    if k < 1:
        raise ValueError("k doit être supérieur ou égal à 1.")

    total = collection.count()
    if total == 0:
        raise ValueError("La collection est vide.")

    vecteur = encoder_textes([texte])[0] if texte is not None else encoder_image(image)

    options = {}
    if type_resultat is not None:
        options["where"] = {"type": type_resultat}

    return collection.query(
        query_embeddings=[vecteur],
        n_results=min(k, total),
        include=["documents", "metadatas", "distances"],
        **options,
    )