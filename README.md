# Multimodal RAG — Intelligent PDF Assistant

An intelligent document retrieval system that combines **Retrieval-Augmented Generation (RAG)**, multimodal document indexing, and an AI agent to retrieve and explain information from PDF documents.

The project allows users to query indexed PDF documents, retrieve relevant text passages, find associated figures, and generate contextual answers using an LLM.

## Project Overview

Traditional RAG systems mainly focus on textual information. However, technical and scientific documents often contain valuable information in figures, diagrams, charts, and illustrations.

This project explores a multimodal approach to document retrieval by handling both textual and visual content extracted from PDF documents.

The system combines:
- Document ingestion and processing.
- Text and image indexing.
- Vector-based semantic search.
- Contextual answer generation.
- An AI agent capable of invoking retrieval tools.

## Key Features

### 1. PDF Processing
- Load and process PDF documents.
- Extract textual content.
- Retrieve visual elements and figures.
- Preserve document references and page metadata.

### 2. Multimodal Indexing
- Organize textual and visual document content.
- Generate embeddings for semantic retrieval.
- Store indexed content and metadata in ChromaDB.
- Associate retrieved elements with their original PDF and page.

### 3. Semantic Search
- Retrieve relevant passages based on user questions.
- Search for textual or visual results.
- Configure the number of retrieved results (`k`).
- Display document references and metadata.

### 4. AI-Powered Answer Generation
- Use an LLM to generate contextual responses.
- Ground answers in retrieved document content.
- Reduce unsupported or fabricated information.
- Include document references in generated answers.

### 5. Agentic RAG with LangChain
The project integrates an AI agent using LangChain's `create_agent`.

The agent can:
- Understand the user's request.
- Decide when to invoke the document retrieval tool.
- Call `consulter_pdf` to search indexed documents.
- Reformulate queries when retrieval results are insufficient.
- Generate answers based on retrieved information.

### 6. Visual Results
- Retrieve indexed figures from ChromaDB.
- Recover original image files using their metadata.
- Display retrieved figures directly in the Jupyter Notebook.
- Associate each image with its source PDF and page number.

## Architecture

The project follows a retrieval and generation pipeline:

```text
                 USER QUERY
                      |
                      v
             INTERACTION FUNCTION
                      |
                      v
                LANGCHAIN AGENT
                      |
                      v
              consulter_pdf TOOL
                      |
                      v
             DOCUMENT RETRIEVAL
                      |
              +-------+-------+
              |               |
              v               v
         TEXT SEARCH      IMAGE SEARCH
              |               |
              v               v
          CHROMADB        CHROMADB
              |               |
              +-------+-------+
                      |
                      v
              RETRIEVED SOURCES
                      |
                      v
               GROQ LLM
                      |
                      v
              GENERATED ANSWER
                      |
                      v
          ANSWER + SOURCES + FIGURES
```

## Technology Stack

| Technology | Role |
|---|---|
| Python | Main programming language |
| LangChain | Agent orchestration and tool calling |
| Groq | LLM inference |
| GPT-OSS-120B | Text generation and agent reasoning |
| ChromaDB | Vector storage and retrieval |
| Embedding models | Semantic representation of content |
| PyMuPDF / PDF processing tools | PDF content extraction, depending on the ingestion implementation |
| Pillow (PIL) | Image loading and processing |
| Jupyter Notebook | Development and experimentation |
| UV | Python dependency and environment management |

## Project Structure

The project is organized around document ingestion, indexing, retrieval, and agent interaction.

```text
rag_multimodal/
│
├── .venv/
├── .env
├── .gitignore
│
├── notebooks/
│   └── multimodal_rag.ipynb
│
├── documents/
│   └── pdfs/
│
├── images/
│
├── chroma_db/
│
├── README.md
└── pyproject.toml
```

*Adapt this structure to the actual organization of your repository.*

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/Ramadiaw12/rag_multimodal.git
cd rag_multimodal
```

### 2. Create the virtual environment

```bash
uv sync
```

If you are creating the environment manually:

```bash
uv venv
source .venv/bin/activate
```

### 3. Configure environment variables

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key
```

Never commit your API key or expose it in your source code.

### 4. Install dependencies

If dependencies are not already declared in `pyproject.toml`, install the required packages with UV, according to the modules used in your implementation.

## Usage

### 1. Start the notebook

```bash
uv run jupyter notebook
```

Open the project notebook and execute the cells in order.

### 2. Index PDF documents

Add your PDF documents to the ingestion workflow.

The processing pipeline extracts and prepares document content for indexing.

### 3. Query the RAG system

Example:

```python
sortie = rag(
    question=(
        "What is the main contribution of the "
        "Transformer architecture introduced in "
        "Attention Is All You Need?"
    ),
    type_resultat="texte",
    k=8,
    montrer_resultats=True
)
```

### 4. Interact with the AI agent

Example:

```python
sortie = interroger_agent(
    texte=(
        "Explique les composants de cette architecture "
        "en t'appuyant sur les passages de "
        "Attention Is All You Need."
    ),
    type_resultat="texte",
    k=8
)
```

The agent invokes the document retrieval tool and generates an answer using the retrieved information.

## Retrieval Modes

The system supports three retrieval modes:

| Mode | Description |
|---|---|
| `texte` | Retrieve textual passages |
| `image` | Retrieve visual document elements |
| `tous` | Retrieve both textual and visual elements |

The number of retrieved results is configurable through the `k` parameter.

## Current Limitations

- The configured GPT-OSS-120B model is used for text-based generation and agent reasoning, not direct image understanding.
- User-provided images are not currently analyzed directly by the Groq agent.
- Retrieved document figures can nevertheless be loaded and displayed separately using their ChromaDB identifiers and metadata.
- Direct visual understanding can be added through a dedicated vision-capable model.

## Future Improvements

- Integrate a vision-language model for direct image understanding.
- Improve image and text cross-modal retrieval.
- Add document-level metadata filtering.
- Implement hybrid search combining semantic and keyword retrieval.
- Improve source attribution and answer evaluation.
- Add a web interface for document uploads and conversational interaction.
- Deploy the application through a containerized infrastructure.

## Learning Objectives

This project explores several important concepts in modern AI engineering:

- Retrieval-Augmented Generation (RAG).
- Multimodal document processing.
- Vector databases and semantic search.
- Embedding models.
- LLM integration and inference.
- Agentic workflows and tool calling.
- Context management and source attribution.

## Author : **RD**

**AI Infrastructure & RAG Engineering Project**

Developed as a hands-on exploration of document intelligence, multimodal retrieval, and agentic AI systems.

---

*Building intelligent systems that connect unstructured data, retrieval pipelines, and LLM-powered applications.*