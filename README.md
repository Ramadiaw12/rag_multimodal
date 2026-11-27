# RAG multimodal agentique : texte, schémas et graphiques


**Posez une question à un PDF : un agent lit le texte *et* les figures, puis répond en citant ses sources.**

[![CI](https://github.com/Ramadiaw12/rag_multimodal/actions/workflows/ci.yml/badge.svg)](https://github.com/Ramadiaw12/rag_multimodal/actions/workflows/ci.yml)
![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)

| Couche | Technologies |
|---|---|
| **IA et recherche** | ![OpenCLIP](https://img.shields.io/badge/OpenCLIP-ViT--B--32-8A2BE2) ![PyTorch](https://img.shields.io/badge/PyTorch-CPU-EE4C2C?logo=pytorch&logoColor=white) ![Groq](https://img.shields.io/badge/Groq-gpt--oss--120b%20%7C%20Llama%204%20Scout-F55036) |
| **Agent** | ![LangGraph](https://img.shields.io/badge/LangGraph-agent%20ReAct-1C3C3C) ![LangChain](https://img.shields.io/badge/LangChain-1.x-1C3C3C?logo=langchain&logoColor=white) ![MCP](https://img.shields.io/badge/MCP-Model%20Context%20Protocol-000000) |
| **Données** | ![ChromaDB](https://img.shields.io/badge/ChromaDB-base%20vectorielle-FFC107) ![PyMuPDF](https://img.shields.io/badge/PyMuPDF-extraction%20PDF-0A66C2) |
| **Application** | ![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white) ![FastAPI](https://img.shields.io/badge/FastAPI-SSE-009688?logo=fastapi&logoColor=white) ![HTML5](https://img.shields.io/badge/HTML5-E34F26?logo=html5&logoColor=white) ![CSS3](https://img.shields.io/badge/CSS3-1572B6?logo=css3&logoColor=white) ![JavaScript](https://img.shields.io/badge/JavaScript-F7DF1E?logo=javascript&logoColor=black) |
| **Infra et qualité** | ![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?logo=docker&logoColor=white) ![GitHub Actions](https://img.shields.io/badge/GitHub%20Actions-CI%2FCD-2088FF?logo=githubactions&logoColor=white) ![GHCR](https://img.shields.io/badge/GHCR-image%20publiée-181717?logo=github&logoColor=white) ![pytest](https://img.shields.io/badge/pytest-tests-0A9EDC?logo=pytest&logoColor=white) ![ruff](https://img.shields.io/badge/ruff-lint-D7FF64?logo=ruff&logoColor=black) ![uv](https://img.shields.io/badge/uv-gestionnaire%20de%20paquets-DE5FE9) |


> *Multimodal agentic RAG: a LangGraph ReAct agent queries a CLIP + ChromaDB index of text and images, reads figures with a vision model, and exposes its retriever through an MCP server.*

Posez une question à un PDF et obtenez une réponse sourcée qui s'appuie **à la fois sur le texte et sur les figures** (schémas d'architecture, graphiques, tableaux). L'agent décide lui-même quand chercher, quand lire une image et quand répondre.

## 🖥️ Interface utilisateur – RAG Multimodal

Une interface intuitive et interactive permettant d'explorer une base de connaissances multimodale à travers une expérience conversationnelle.

Les utilisateurs peuvent interroger leurs documents et contenus visuels en langage naturel, effectuer des recherches contextuelles et obtenir des réponses pertinentes générées par un LLM à partir des informations récupérées.

Cette interface constitue le point d'interaction avec une architecture RAG multimodale intégrant l'ingestion documentaire, les embeddings, la recherche sémantique et le traitement des contenus visuels.

**Objectif :** transformer des documents non structurés en connaissances accessibles, exploitables et interrogeables grâce à l'intelligence artificielle.
![RAG Multimodal](./images/interface.png)

## Pourquoi ce projet

Un RAG classique ne voit que le texte : les informations contenues dans un schéma ou un graphique sont perdues. Ce projet traite les figures comme des sources de première classe, puis ajoute une couche agentique : le modèle raisonne, interroge la base, lit les images utiles et rédige lui-même la réponse.

## Ce que fait le système

- **Index multimodal unique** : texte et images sont encodés par le même modèle CLIP (ViT-B-32) dans une collection ChromaDB persistante (distance cosinus). Une requête texte retrouve aussi bien des passages que des figures, et une image peut servir de requête.
- **Agent ReAct sur LangGraph** : deux nœuds, `mind` (le LLM raisonne) et `body` (les outils agissent), reliés par la boucle Raisonnement, Action, Observation.
- **Lecture des images par un modèle vision** : l'agent, modèle texte, appelle l'outil `analyser_image` pour obtenir une description fiable d'une figure avant de répondre.
- **Retriever exposé via MCP** : le RAG tourne dans un serveur MCP séparé, et l'agent n'importe aucun code du RAG. N'importe quel client MCP peut réutiliser les mêmes outils.
- **Interface Streamlit à trois modes** pour comparer les approches : RAG direct, agent avec outils locaux, agent via MCP. Les appels d'outils de l'agent s'affichent en direct.

## Architecture

```mermaid
flowchart LR
    Q([Question]) --> UI[Streamlit]
    UI --> MIND

    subgraph AGENT["Agent LangGraph (ReAct)"]
        MIND["MIND<br/>LLM Groq : raisonne"] -- "action : appel d'outil" --> BODY["BODY<br/>exécute les outils"]
        BODY -- observation --> MIND
    end

    BODY -- "mode local" --> CORE
    BODY -- "mode MCP (HTTP)" --> MCP["Serveur MCP"] --> CORE

    CORE["rag_core<br/>recherche + lecture d'image"] --> DB[("ChromaDB<br/>CLIP ViT-B-32<br/>texte + images")]
    CORE --> VLM["Modèle vision Groq"]
    MIND --> A([Réponse sourcée<br/>pdf, page])
```

| Outil de l'agent | Rôle |
|---|---|
| `rechercher_sources` | Interroge la base vectorielle et renvoie passages et images (id, pdf, page, contenu, similarité). |
| `analyser_image` | Lit une image trouvée par la recherche avec le modèle vision et renvoie une observation en texte. |

Le serveur MCP expose aussi `repondre_question` (pipeline RAG complet, retrieval puis génération) pour les clients qui veulent une réponse toute faite. L'agent de l'application ne l'utilise pas : c'est lui qui rédige.

## Stack

| Couche | Technologie |
|---|---|
| Embeddings multimodaux | OpenCLIP `ViT-B-32` (`laion2b_s34b_b79k`) |
| Base vectorielle | ChromaDB (persistante, cosinus) |
| Extraction PDF | PyMuPDF, découpage avec `langchain-text-splitters` |
| LLM de l'agent et génération | Groq : `openai/gpt-oss-120b` |
| Modèle vision | Groq : `meta-llama/llama-4-scout-17b-16e-instruct` |
| Agent | LangGraph (`StateGraph`, `ToolNode`) |
| Protocole d'outils | MCP (serveur `FastMCP`, transport streamable HTTP, `langchain-mcp-adapters`) |
| Interface | Streamlit |
| Environnement | Python 3.12, uv |

## Choix de conception

- **Un seul espace d'embedding** pour le texte et les images : la recherche est simple, et une requête image fonctionne sans pipeline supplémentaire.
- **L'agent rédige la réponse.** Les outils ne renvoient que des preuves ; il n'y a pas de second LLM qui répondrait à sa place.
- **L'agent n'a pas besoin de voir les images.** Un outil de lecture d'image fait office d'yeux, ce qui permet d'utiliser un excellent modèle texte pour le raisonnement et un modèle vision dédié pour la perception.
- **Séparation agent / RAG par MCP** : le serveur charge CLIP et ChromaDB une seule fois, et l'agent ne dépend que d'une URL.
- **Défense contre l'injection de prompt** : les sources sont traitées comme des données non fiables, les consignes qu'elles contiennent sont ignorées, et `analyser_image` n'accepte que des identifiants d'images indexées, jamais un chemin libre.
- **Garde-fous de l'agent** : nombre de recherches limité dans le prompt, limite de récursion du graphe, message d'échec explicite plutôt qu'une réponse vide.

## Installation

Prérequis : Python 3.12+, [uv](https://docs.astral.sh/uv/) et une clé [Groq](https://console.groq.com).

```bash
git clone https://github.com/Ramadiaw12/rag_multimodal.git
cd rag-multimodal

uv sync
cp .env.example .env        # puis renseignez GROQ_API_KEY
```

### 1. Indexer un PDF (une seule fois)

L'ingestion est dans le notebook `rag_multimodal.ipynb` : extraction du texte et des images, découpage, encodage CLIP, écriture dans `chroma_pdf/`. Placez votre PDF dans `pdfs/` puis exécutez le notebook.

### 2. Lancer l'application

Pour le mode « Agent via MCP », lancez aussi le serveur dans un second terminal :

```bash
Terminal 1 : uv run uvicorn server:app --port 8080

Terminal 2 : uv run python mcp_server.py
```

## Utilisation

| Mode | Ce qui se passe |
|---|---|
| **RAG direct** | Une recherche, une génération. Le plus rapide pour une question simple. |
| **Agent avec outils locaux** | L'agent décide des recherches, lit les images utiles et rédige. Les outils tournent dans le processus Streamlit. |
| **Agent via MCP** | Même agent, mais les outils sont servis par le serveur MCP séparé. |

Exemples de questions : *« What is multi-head attention? »*, *« Que montre le schéma de l'architecture du Transformer ? »*, *« Compare le rôle de l'encodeur et du décodeur. »*




## Structure du projet

```
├── server.py                 # Backend FastAPI : API de chat en streaming (SSE)
├── static/index.html         # Interface web : conversation, trace de l'agent, preuves
├── agent.py                  # Agent LangGraph ReAct (mind / body)
├── outils.py                 # Outils de l'agent, définis une seule fois
├── mcp_server.py             # Serveur MCP exposant ces outils
├── rag_core.py               # Recherche, génération, outils partagés
├── retrieval.py              # CLIP + ChromaDB : encodage et requête
├── context.py                # Filtrage, compression, fenêtre, budget de tokens
├── metrics.py                # Métriques Prometheus
├── chemins.py                # Résolution des chemins d'images
├── rag_multimodal.ipynb      # Ingestion et expérimentations
├── tests/                    # Tests unitaires
├── observability/            # Configuration Prometheus et Grafana
├── Dockerfile
├── docker-compose.yaml       # Application et serveur MCP
├── compose.monitoring.yaml   # Prometheus et Grafana
├── .github/workflows/        # CI et CD
├── pyproject.toml / uv.lock
└── .env.example
```


## Architecture conceptuelle de l'agent

```mermaid
flowchart TD
    START([Start])
    MIND["Mind"]
    BODY["Body<br/>Tools: rechercher_sources, analyser_image"]
    END([End])

    START --> MIND
    BODY --> MIND
    MIND -.-> END
    MIND -. Tools .-> BODY

    classDef default fill:#f2f0ff,stroke:#8b7cf6,color:#333
    classDef first fill:#ffffff,stroke:#8b7cf6,color:#333
    classDef last fill:#bfb6fc,stroke:#8b7cf6,color:#333

    class START first
    class END last
```

## Utiliser PostgreSQL + pgvector à la place de ChromaDB

Le projet stocke ses vecteurs dans ChromaDB par défaut, parce qu'il ne demande aucun serveur. Si vous avez déjà une base PostgreSQL, ou si vous voulez une base de production, vous pouvez la remplacer par **PostgreSQL + [pgvector](https://github.com/pgvector/pgvector)** en changeant une variable d'environnement. Le reste du code (agent, serveur MCP, interface web) ne change pas.

### Quand choisir lequel

| | ChromaDB (défaut) | PostgreSQL + pgvector |
|---|---|---|
| Démarrage | Aucun service, un simple dossier | Un serveur PostgreSQL |
| Données applicatives | Non | Oui : utilisateurs, historique, métadonnées dans la même base |
| Filtres | Filtres sur métadonnées | SQL complet |
| Sauvegarde, réplication | Copie du dossier | Outils PostgreSQL habituels |
| Idéal pour | Prototypes, démos locales | Application qui utilise déjà PostgreSQL, production |

À l'échelle d'un PDF ou de quelques milliers de passages, les deux répondent en quelques millisecondes : le choix tient surtout à l'infrastructure.

### Comment ça s'intègre

```
rag_core.py / agent.py / server.py
            │
        retrieval.py        encode la requête avec CLIP
            │
       vectorstore.py       VECTOR_BACKEND = chroma | pgvector
        ┌───┴──────────┐
   ChromaStore    PgVectorStore
```

`vectorstore.py` est la seule couche qui connaît la base. Les deux backends renvoient leurs résultats au **même format** (`ids`, `documents`, `metadatas`, `distances`), avec la même distance cosinus, donc `recuperer_sources` fonctionne sans modification. Un test vérifie que les deux backends renvoient les mêmes voisins sur les mêmes données (`tests/test_vectorstore.py`).

### Mise en route

**1. Démarrer PostgreSQL avec pgvector**

```bash
docker run -d --name pgvector \
  -e POSTGRES_USER=rag -e POSTGRES_PASSWORD=rag -e POSTGRES_DB=rag \
  -p 127.0.0.1:5432:5432 pgvector/pgvector:pg16
```

Le mot de passe `rag` convient pour un essai local ; changez-le avant toute exposition.

**2. Installer les dépendances**

```bash
uv add "psycopg[binary]" pgvector
```

**3. Configurer l'environnement** (fichier `.env`)

```
VECTOR_BACKEND=pgvector
DATABASE_URL=postgresql://rag:rag@localhost:5432/rag
```

Au premier démarrage, le projet crée lui-même l'extension `vector`, la table et les index.

**4. Charger vos données**, de l'une des deux façons :

- **Migrer l'index ChromaDB existant**, sans recalculer les embeddings :
  ```bash
  uv run python migrate_chroma_to_pgvector.py
  ```
  Le script affiche le nombre d'éléments des deux bases et s'arrête si elles diffèrent.
- **Écrire depuis votre propre ingestion** :
  ```python
  from vectorstore import PgVectorStore

  store = PgVectorStore()
  store.add(
      ids=["doc-p1-0"],
      embeddings=[vecteur_clip_512],   # liste de 512 floats, normalisée
      documents=["texte du passage"],
      metadatas=[{"type": "texte", "pdf": "doc.pdf", "page": 1, "source": ""}],
  )
  ```
  Pour une image, mettez `"type": "image"` et `"source"` avec le chemin du fichier.

**5. Lancer l'application** comme d'habitude :

```bash
uv run uvicorn server:app --port 8080
```

### Avec Docker Compose

```bash
docker compose -f docker-compose.yaml -f compose.pgvector.yaml up --build
```

La surcouche ajoute un service `db` et configure `web` et `mcp` pour l'utiliser. Les images extraites restent à fournir dans `./data/images` : seul l'index passe dans PostgreSQL.

### Ce que le projet crée dans PostgreSQL

```sql
CREATE TABLE chunks (
    id        TEXT PRIMARY KEY,
    type      TEXT NOT NULL,             -- 'texte' ou 'image'
    pdf       TEXT,
    page      INTEGER,
    source    TEXT,                      -- chemin de l'image
    document  TEXT,
    embedding vector(512) NOT NULL
);
CREATE INDEX ON chunks USING hnsw (embedding vector_cosine_ops);
```

Une recherche est une requête SQL ordinaire. L'opérateur `<=>` est la distance cosinus :

```sql
SELECT pdf, page, 1 - (embedding <=> :requete) AS similarite
FROM chunks
WHERE type = 'texte'
ORDER BY embedding <=> :requete
LIMIT 5;
```

Pour inspecter l'index :

```sql
SELECT type, pdf, count(*) FROM chunks GROUP BY type, pdf;
```

### Points d'attention

- **Dimension** : la table est en `vector(512)`, la taille des embeddings CLIP ViT-B-32. Si vous changez de modèle d'embedding, changez `DIM` dans `vectorstore.py` et recréez la table.
- **Métrique** : l'index `vector_cosine_ops` et l'opérateur `<=>` doivent aller ensemble. Un index cosinus n'accélère pas une requête en distance euclidienne (`<->`).
- **Filtre et index HNSW** : pgvector applique le filtre `WHERE type = ...` après le parcours de l'index, donc une requête très sélective peut renvoyer moins de `k` résultats. Le projet règle `hnsw.ef_search` à 100 (variable `HNSW_EF_SEARCH`). Avec pgvector 0.8 ou plus, vous pouvez aussi activer `SET hnsw.iterative_scan = relaxed_order`. À l'échelle de ce projet, le problème ne se pose pas.
- **Connexions** : `PgVectorStore` ouvre une connexion par opération, ce qui suffit pour une démo. Pour du trafic réel, remplacez-la par un pool (`psycopg_pool`).
- **Sécurité** : ne publiez pas le port 5432 sur Internet, utilisez un mot de passe fort et ne le versionnez jamais.

## 📊 Observability & Monitoring

To ensure the reliability and performance of the Multimodal RAG system, **Prometheus** and **Grafana** are integrated for monitoring and observability.

The application exposes its operational metrics through a `/metrics` endpoint in Prometheus format. Prometheus collects and stores these metrics, while Grafana provides interactive dashboards to visualize system performance and monitor application behavior.

### 🔍 Monitored Metrics

The monitoring system tracks several aspects of the RAG pipeline:

| Metric | Description |
|---|---|
| Response latency | Measures response generation time, broken down by processing mode |
| Tool execution duration | Tracks the execution time of individual tools |
| Token usage | Monitors estimated and actual token consumption |
| Context compression | Tracks context compression before sending information to the LLM |
| Error tracking | Monitors application errors, categorized by type |

### 🏗️ Monitoring Architecture

The monitoring workflow is organized as follows:

1. **Application (RAG API):** Exposes application metrics through the `/metrics` endpoint.
2. **Prometheus:** Scrapes the metrics endpoint and stores time-series data.
3. **Grafana:** Connects to Prometheus and visualizes the collected metrics through dashboards.

### 🚀 Start the Monitoring Stack

## Lancer avec Docker

```bash
cp .env.example .env     # puis renseignez GROQ_API_KEY
docker compose up -d --build
```

Pour ajouter la supervision (Prometheus et Grafana) :

```bash
docker compose -f docker-compose.yaml -f compose.monitoring.yaml up -d --build
```

| Service | Adresse | Rôle |
|---|---|---|
| Application | http://localhost:8081 | Interface de conversation |
| Grafana | http://localhost:3000 | Tableau de bord d'exploitation |
| Prometheus | http://localhost:9090 | Collecte des métriques |

Les données (index ChromaDB et images extraites) sont montées depuis `./data`
(`./data/chroma_pdf` et `./data/images`) : elles ne sont pas dans l'image Docker.

## Observabilité

L'application expose ses métriques sur `/metrics` (format Prometheus) : latence des
réponses par mode, durée de chaque outil, tokens estimés et réels, compression du
contexte, erreurs par type. Grafana affiche un tableau de bord prêt à l'emploi.

L'observabilité a deux niveaux. L'utilisateur voit la **trace de l'agent** et le
volume de contexte envoyé au LLM à chaque étape. L'exploitant voit les **métriques
agrégées** dans Grafana, qui reste une interface séparée de l'application.

### 📈 Grafana Dashboard

The Grafana dashboard provides a centralized view of the application's operational performance, including response latency, tool execution times, token consumption, context compression, and error distribution.

![Grafana Monitoring Dashboard](images/monitoring.png)

## Dépannage

| Symptôme | Cause et solution |
|---|---|
| Grafana n'affiche aucune donnée | Vous interrogez probablement un autre serveur que le conteneur (par exemple un `uvicorn` lancé à la main). Vérifiez avec `ss -ltnp \| grep :8080` : ne lancez pas `uvicorn` en même temps que Docker |
| Erreur `NotFoundError` dans la réponse | Un nom de modèle Groq n'est plus disponible pour votre clé. Comparez les modèles de `.env` avec la liste de `https://api.groq.com/openai/v1/models` |
| « Serveur MCP injoignable » | Le service `mcp` n'est pas démarré. Avec Docker : `docker compose ps`. En local : `uv run python mcp_server.py` |
| Le `.env` modifié n'a aucun effet | Les conteneurs lisent `.env` à leur création : `docker compose up -d --force-recreate` |## Dépannage

| Symptôme | Cause et solution |
|---|---|
| Grafana n'affiche aucune donnée | Vous interrogez probablement un autre serveur que le conteneur (par exemple un `uvicorn` lancé à la main). Vérifiez avec `ss -ltnp \| grep :8080` : ne lancez pas `uvicorn` en même temps que Docker |
| Erreur `NotFoundError` dans la réponse | Un nom de modèle Groq n'est plus disponible pour votre clé. Comparez les modèles de `.env` avec la liste de `https://api.groq.com/openai/v1/models` |
| « Serveur MCP injoignable » | Le service `mcp` n'est pas démarré. Avec Docker : `docker compose ps`. En local : `uv run python mcp_server.py` |
| Le `.env` modifié n'a aucun effet | Les conteneurs lisent `.env` à leur création : `docker compose up -d --force-recreate` |

### 🎯 Benefits

- Improved visibility into RAG pipeline performance.
- Identification of latency bottlenecks.
- Monitoring of LLM token consumption.
- Detection and analysis of application errors.
- Better understanding of resource utilization and system behavior.

This observability layer helps make the Multimodal RAG system easier to troubleshoot, evaluate, and operate.


## Future Improvements

- Integrate a vision-language model for direct image understanding.
- Improve image and text cross-modal retrieval
- Implement hybrid search combining semantic and keyword retrieval
- Improve source attribution and answer evaluation
- Add a web interface for document uploads and conversational interaction
- Mémoire de conversation pour l'agent
- 

## Learning Objectives

This project explores several important concepts in modern AI engineering:

- Retrieval-Augmented Generation (RAG)
- Multimodal document processing
- Vector databases and semantic search
- Embedding models
- LLM integration and inference
- Agentic workflows and tool calling
- Context management and source attribution



## Author : 
**RD** : [LinkedIn](https://linkedin.com/in/votre-profil) · [GitHub](https://github.com/VOTRE_COMPTE) · superrama86@gmail.com


Developed as a hands-on exploration of document intelligence, multimodal retrieval, and agentic AI systems.

---

*Building intelligent systems that connect unstructured data, retrieval pipelines, and LLM-powered applications.*

## 📄 License

Copyright 2026 Ramatoulaye Diawane

This project is licensed under the Apache License 2.0.
You are free to use, modify, and distribute this software in accordance
with the terms of the license.

See the [LICENSE](LICENSE) file for details.