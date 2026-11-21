FROM python:3.12-slim
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy PYTHONUNBUFFERED=1 \
    PATH="/app/.venv/bin:$PATH" HF_HOME=/opt/hf-cache \
    CHROMA_PATH=/data/chroma_pdf IMAGES_DIR=/data/images

WORKDIR /app
# uid 1000 : correspond à l'utilisateur habituel de l'hôte, pour les volumes montés
RUN useradd -m -u 1000 app && mkdir -p /opt/hf-cache /data && chown app /opt/hf-cache /data /app
USER app

# Dépendances d'abord : cette couche est mise en cache tant que pyproject.toml / uv.lock ne changent pas
COPY --chown=app pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

COPY --chown=app *.py ./
COPY --chown=app static ./static

# Poids CLIP intégrés à l'image : démarrage sans téléchargement
RUN python -c "import open_clip; open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')"

EXPOSE 8080
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8080"]