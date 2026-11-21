"""Résolution des chemins d'images : indépendante de la machine qui a fait l'ingestion.
La base stocke des chemins absolus (ex. /home/moi/projet/...) qui n'existent pas dans un conteneur."""
import os
from pathlib import Path

IMAGES_DIR = Path(os.getenv("IMAGES_DIR", Path(__file__).resolve().parent / "resultats_pdf" / "images"))


def resoudre_image(chemin_stocke) -> Path:
    """Chemin enregistré à l'ingestion s'il existe ici, sinon le même nom de fichier dans IMAGES_DIR."""
    p = Path(chemin_stocke)
    return p if p.is_file() else IMAGES_DIR / p.name