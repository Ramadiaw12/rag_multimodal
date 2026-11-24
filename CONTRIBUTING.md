# Contributing to RAG multimodal

Thanks for your interest in contributing! This project is a multimodal RAG assistant
(text + images) with a LangGraph agent and an MCP server, built so that every answer
comes with verifiable sources. Contributions of all sizes are welcome: bug reports,
documentation, tests, new adapters, and features.

By participating, you agree to follow our [Code of Conduct](CODE_OF_CONDUCT.md).

## Ways to contribute

- **Report a bug**: open an issue using the bug template.
- **Suggest an idea**: open an issue using the feature template *before* writing a lot of code.
- **Improve the docs**: fixes to the README, examples, and comments are always useful.
- **Pick an issue**: look for the labels `good first issue` and `help wanted`.
- **Review**: reading and testing other people's pull requests helps a lot.

## Project layout

```
server.py          FastAPI backend: streaming chat (SSE), image serving
agent.py           LangGraph ReAct agent (mind / body) and its tools
context.py         Context engineering: filtering, compression, window, token budget
mcp_server.py      MCP server exposing the retriever
rag_core.py        Retrieval, generation, shared tools
retrieval.py       CLIP + ChromaDB: encoding and querying
chemins.py         Machine-independent image path resolution
static/index.html  Web UI (chat, agent trace, evidence board)
tests/             Unit tests
```

## Getting started

### Prerequisites

- Python (see `pyproject.toml` for the supported version)
- [uv](https://docs.astral.sh/uv/)
- Docker and Docker Compose (optional, for the full stack)

### Setup

```bash
git clone https://github.com/Ramadiaw12/rag_multimodal.git
cd rag_multimodal
uv sync
cp .env.example .env     # then fill in your values; never commit .env
```

### Run the application

```bash
# with Docker (web app + MCP server)
docker compose up --build

# or locally
uv run uvicorn server:app --reload --port 8080
```

Then open <http://127.0.0.1:8080>.

## Development workflow

1. **Fork** the repository and create a branch from `main`:
   `git checkout -b feat/short-description` (or `fix/...`, `docs/...`).
2. Make your change. Keep it **small and focused**: one purpose per pull request.
3. **Add or update tests** for any behavior you change.
4. Run the checks locally (the same ones as our CI):
   ```bash
   uv run ruff check .       # lint
   uv run pytest             # tests
   docker build .            # image builds
   ```
5. Commit with a clear message, for example:
   `fix(context): respect token budget when compressing history`
6. Push your branch and open a **pull request** against `main`.

## Pull request checklist

- [ ] The change is focused and described clearly (what and why)
- [ ] Tests added or updated, and `uv run pytest` passes
- [ ] Lint passes
- [ ] No secrets, API keys, or private documents in the diff
- [ ] Documentation updated if behavior or configuration changed
- [ ] Screenshots or a short GIF for UI changes

A maintainer will review your pull request. Please be patient and open to feedback:
review is about making the project better together.

## Coding guidelines

- Follow the existing style; the linter is the reference.
- Prefer small, readable functions with type hints.
- Keep modules independent: retrieval, agent logic, and the web layer should not depend on each other more than necessary.
- Never hard-code paths, models, or credentials: use configuration and `.env`.
- Answers must stay **traceable**: do not change the agent in a way that removes citations or the refusal behavior when sources are missing.

## Tests

- Tests live in `tests/` and run with `uv run pytest`.
- Test behavior, not implementation details.
- Tests must not require network access or a running LLM; mock external calls.

## Adding new components

The project is designed so components can be swapped. If you add an adapter
(for example another vector store or another embedding model):

1. Keep it behind the existing interface (do not change the callers).
2. Make it selectable through configuration.
3. Add tests and a short section in the documentation.

## Documents and data

Only add documents you have the right to share (public domain or permissive
license). Do not commit private, confidential, or proprietary manuals.

## Reporting security issues

Please **do not** open a public issue for security vulnerabilities.
See [SECURITY.md](SECURITY.md) for how to report them privately.

## Licensing

By contributing, you agree that your contributions will be licensed under the
same license as the project (see [LICENSE](LICENSE)).

## Questions?

Open a discussion or an issue. No question is too small.