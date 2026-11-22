# Étape 5 : Transformation du Retriever en Outil Agentique (Tool)
# FIchier retrieval.py : contient les fonctions de recherche vectorielle pour le RAG multimodal
# Copyright 2026 Ramatoulaye Diawane
# SPDX-License-Identifier: Apache-2.0
"""Partie requête de l'index : mêmes modèle, normalisation et collection que l'ingestion du notebook."""
import os
from pathlib import Path

import chromadb
import open_clip
import torch
from PIL import Image

# on définit le device pour l'encodage : si CUDA est disponible, on utilise le GPU, sinon on reste sur le CPU. On définit également le nom du modèle OpenCLIP et le pré-entraînement à utiliser
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "ViT-B-32"
PRETRAINED = "laion2b_s34b_b79k"

# Chemin absolu : fonctionne quel que soit le dossier d'où l'on lance le serveur.
CHROMA_PATH = Path(os.getenv("CHROMA_PATH", Path(__file__).resolve().parent / "chroma_pdf"))
COLLECTION_NAME = "pdf_clip_vit_b32"

# alors que l'on peut lancer le serveur depuis n'importe quel dossier, on veut que le chemin de la base vectorielle soit correct. 
# On peut définir la variable d'environnement CHROMA_PATH pour pointer vers le dossier contenant la base ChromaDB. Si elle n'est pas définie, on suppose que le dossier 'chroma_pdf' est à côté de retrieval.py.
if not CHROMA_PATH.exists():
    raise FileNotFoundError(
        f"Base ChromaDB introuvable : {CHROMA_PATH}. "
        "Définissez CHROMA_PATH ou placez le dossier 'chroma_pdf' à côté de retrieval.py."
    )

# On peut maintenant créer le modèle et le tokenizer OpenCLIP, ainsi que la collection ChromaDB pour la recherche vectorielle.
model, _, preprocess = open_clip.create_model_and_transforms(MODEL_NAME, pretrained=PRETRAINED)
model = model.to(DEVICE).eval()
tokenizer = open_clip.get_tokenizer(MODEL_NAME)

# On crée un client ChromaDB pour accéder à la collection contenant les vecteurs encodés des documents PDF. La collection est créée si elle n'existe pas
client = chromadb.PersistentClient(path=str(CHROMA_PATH))
collection = client.get_or_create_collection(
    name=COLLECTION_NAME,
    embedding_function=None,  # les embeddings sont calculés par notre code
    metadata={"hnsw:space": "cosine"},
)

# On définit les fonctions d'encodage de texte et d'image, ainsi que la fonction de recherche vectorielle qui peut prendre soit un texte, soit une image, et renvoyer les k résultats les plus proches dans la collection ChromaDB. La fonction de recherche peut filtrer les résultats par type (texte ou image) si nécessaire
@torch.inference_mode()
def encoder_textes(textes):
    tokens = tokenizer(textes).to(DEVICE)
    vecteurs = model.encode_text(tokens).float()
    vecteurs = vecteurs / vecteurs.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    return vecteurs.cpu().tolist()

# alors là on définit l'encodage d'image avec inference_mode pour ne pas calculer les gradients, puis on définit la fonction de recherche vectorielle qui peut prendre soit un texte, soit un chemin d'image, et renvoyer les k résultats les plus proches dans la collection ChromaDB. La fonction de recherche peut filtrer les résultats par type (texte ou image) si nécessaire
@torch.inference_mode()
def encoder_image(chemin):
    with Image.open(chemin) as image:
        tenseur = preprocess(image.convert("RGB"))
    tenseur = tenseur.unsqueeze(0).to(DEVICE)
    vecteur = model.encode_image(tenseur).float()
    vecteur = vecteur / vecteur.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    return vecteur.cpu().tolist()[0]

# une fonction de recherche vectorielle qui peut prendre soit un texte, soit un chemin d'image, et renvoyer les k résultats les plus proches dans la collection ChromaDB. La fonction de recherche peut filtrer les résultats par type (texte ou image) si nécessaire
def rechercher(texte=None, image=None, type_resultat=None, k=5):
    """
    Fournir soit un texte, soit un chemin d'image.
    type_resultat : None, "texte" ou "image".
    """
    if (texte is None) == (image is None): # Cette condition vérifie que l'un des deux paramètres est fourni, mais pas les deux. Si les deux sont None ou si les deux sont fournis, on lève une exception
        raise ValueError("Fournissez soit un texte, soit une image.")
    if type_resultat not in (None, "texte", "image"):
        raise ValueError("type_resultat doit être 'texte', 'image' ou None.")
    if k < 1:
        raise ValueError("k doit être supérieur ou égal à 1.")

    total = collection.count() # On compte le nombre total de documents dans la collection ChromaDB. Si la collection est vide, on lève une exception pour éviter de faire une requête inutile
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