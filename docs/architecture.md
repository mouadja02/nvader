# Architecture

> NVADER v0.1.0 — current state as of Week 2 (knowledge layer complete).

## High-Level Overview

```
┌─────────────────────────────────────────────────────────┐
│                     CLI (nvader)                        │
│              typer + rich · cli.py                      │
├─────────────────────────────────────────────────────────┤
│                      Config                             │
│         pydantic models · config.py                     │
├──────────┬──────────┬──────────┬────────────────────────┤
│  agents/ │  tools/  │ memory/  │       core/            │
├──────────┴──────────┴──────────┤  documents, state,     │
│       ingestion/               │  tracing (planned)     │
│       retrieval/               ├────────────────────────┤
│       evaluation/              │       nvidia/          │
│       guardrails/              │  NIM, NeMo Guardrails  │
│       api/                     │  (planned)             │
├────────────────────────────────┴────────────────────────┤
│                    data/                                │
│        raw/ · processed/ · vector_store/ · traces/      │
└─────────────────────────────────────────────────────────┘
```

## Package Structure

```
src/nvidia_agentic_research_engineer/
├── __init__.py
├── cli.py              # Typer app — entry point (nvader)
├── config.py           # AppConfig + ProjectTOML (Pydantic)
├── agents/             # Agent definitions & orchestrator (planned)
├── api/                # REST / FastAPI surface (planned)
├── core/               # Domain models & cross-cutting concerns
│   └── documents.py    # Document + DocumentChunk models
├── evaluation/         # Eval harnesses, metrics, reports (planned)
├── guardrails/         # Input/output guardrails, NeMo adapter (planned)
├── ingestion/          # Document loaders, chunking, indexing
│   ├── loaders.py      # load_text_file, load_markdown_file
│   └── chunking.py     # chunk_text, chunk_document, chunk_documents
├── memory/             # Conversation & long-term memory stores (planned)
├── nvidia/             # NVIDIA platform integrations (NIM, etc.) (planned)
├── retrieval/          # Vector search & RAG retrieval logic
│   ├── models.py       # SearchResult, RetrievalQuery, RetrievalConfig
│   └── embeddings.py   # EmbedderProtocol, HashEmbedder, Embedder, get_embedder
└── tools/              # Tool definitions & registry (planned)
```

## Implemented Components

### CLI (`cli.py`)

Entry point registered as `nvader` in `pyproject.toml`. Built with **Typer** + **Rich**.

| Command          | Description                              |
|------------------|------------------------------------------|
| `nvader info`    | Show project identity and configuration  |
| `nvader roadmap` | Print the 8-week certification roadmap   |

### Configuration (`config.py`)

Two Pydantic models:

- **`ProjectTOML`** — reads project metadata (name, version, authors) from `pyproject.toml` at import time.
- **`AppConfig`** — runtime settings: data directories, `default_top_k`, `require_citations`, `max_retries`.

### Core Models (`core/documents.py`)

- **`DocumentType`** — supported source types (`text`, `markdown`, `html`, `pdf`, `url`, `repo`, `paper`, `image`).
- **`Document`** — SHA-256 content-derived ID, UTC timestamp, typed metadata, `short_preview()` for display truncation.
- **`DocumentChunk`** — slice of a `Document` with stable `id`, `chunk_index`, character offsets (`start_char`, `end_char`), and inherited metadata.

### Ingestion (`ingestion/`)

- **`loaders.py`** — `load_text_file(path)` and `load_markdown_file(path)`: read files into typed `Document` objects.
- **`chunking.py`**:
  - `chunk_text(text, *, chunk_size, chunk_overlap)` — low-level splitter, returns `(text, start, end)` tuples.
  - `chunk_document(document, ...)` — converts a `Document` into a list of `DocumentChunk` instances, preserving source and metadata.
  - `chunk_documents(documents, ...)` — batch variant over a sequence of documents.

### Retrieval (`retrieval/`)

- **`models.py`**:
  - `SearchResult` — frozen Pydantic model: `chunk_id`, `document_id`, `text`, `score`, optional `source`/`metadata`.
  - `RetrievalQuery` — query string + `top_k` (1–100) + optional filters.
  - `RetrievalConfig` — embedding model, index name, `top_k`, similarity threshold, hybrid-search flag.

- **`embeddings.py`**:
  - `EmbedderProtocol` — `@runtime_checkable` Protocol; any class with `embed_texts` + `embed_query` satisfies it.
  - `HashEmbedder` — fully offline, SHA-256-based, 384-dim deterministic embedder (stable across processes; all dimensions carry independent signal).
  - `Embedder` — NVIDIA NIM OpenAI-compatible endpoint wrapper; lazy-imports `openai` to stay testable without the package installed.
  - `get_embedder(model, api_key)` — factory: returns `Embedder` when `NVIDIA_API_KEY` is set, else `HashEmbedder`.

## Planned Modules (Stubbed)

All modules below exist as empty packages, mapped to NVIDIA certification domains.

| Module         | Purpose                                        | Certification Domain       |
|----------------|------------------------------------------------|----------------------------|
| `agents/`      | Base agent, ReAct loop, orchestrator           | Agent Architecture & Dev   |
| `tools/`       | Tool registry, built-in tools                  | Agent Architecture & Dev   |
| `memory/`      | Conversation buffer, long-term memory          | Cognition & Memory         |
| `evaluation/`  | Eval harnesses, metrics, report generation     | Evaluation                 |
| `guardrails/`  | Input/output validation, NeMo Guardrails       | Safety & Ethics            |
| `nvidia/`      | NIM endpoints, NeMo platform integration       | NVIDIA Platform            |
| `api/`         | REST API surface for the agent                 | Deployment                 |

## Ingestion and Chunking Pipeline

Current flow:

1. Load raw files (`.txt`, `.md`) from local resources.
2. Normalize them into typed `Document` objects (SHA-256 ID, UTC timestamp).
3. Split documents into `DocumentChunk` objects with character offsets.
4. Preserve source, metadata, and stable IDs for downstream retrieval.
5. Chunks are ready for embedding via `HashEmbedder` (offline) or `Embedder` (NVIDIA NIM).

```mermaid
flowchart TD
    A["📁 files / docs / repos"] --> B

    B["loaders.py\nload_text_file · load_markdown_file"]
    B --> C

    C["Document\nid · type · content · source · metadata"]
    C --> D

    D["chunking.py\nchunk_document"]
    D --> E

    E["DocumentChunk[ ]\nid · chunk_index · start_char · end_char"]
    E --> F

    F["embeddings.py\nHashEmbedder · Embedder"]
    F --> G

    G["list[list[float]]\n384-dim vectors"]
    G -.->|"planned"| H

    H["🗄️ Vector Store / Retriever"]
```

## Data Layout

```
data/
├── raw/            # Original ingested documents
├── processed/      # Chunked / transformed documents
├── vector_store/   # Embedding index (e.g. FAISS, Milvus)
└── traces/         # Agent execution traces for observability
```

## Dependencies

| Package        | Role                                     |
|----------------|------------------------------------------|
| `typer ≥0.12`  | CLI framework                            |
| `pydantic ≥2.7`| Config & domain model validation         |
| `rich ≥13.7`   | Terminal formatting & panels             |
| `python-dotenv`| `.env` file loading                      |
| `toml`         | `pyproject.toml` parsing                 |
| `openai`       | NVIDIA NIM embedding API (optional)      |

Dev: `pytest ≥8.0`, `ruff ≥0.5`.

## Test Coverage

| Test file                      | Covers                                         | Tests |
|-------------------------------|------------------------------------------------|-------|
| `test_documents.py`           | `Document`, `DocumentType`, `short_preview`    | —     |
| `test_ingestion_loaders.py`   | `load_text_file`, `load_markdown_file`         | —     |
| `test_chunking.py`            | `chunk_text`, `chunk_document`, `chunk_documents` | —  |
| `test_retrieval_embeddings.py`| `HashEmbedder`, `Embedder`, `EmbedderProtocol`, `SearchResult`, `RetrievalQuery`, `RetrievalConfig` | 45 |
| `test_cli.py`                 | `nvader info`, `nvader roadmap`                | —     |

## Build Roadmap Alignment

The architecture is being built incrementally over 8 weeks:

1. **Week 1** — Foundation & architecture skeleton ← *done*
2. **Week 2** — RAG & knowledge integration (`ingestion/`, `retrieval/`) ← *done*
3. **Week 3** — ReAct agent & tool orchestration (`agents/`, `tools/`) ← *up next*
4. **Week 4** — Planning & memory (`memory/`, `core/state.py`)
5. **Week 5** — Multi-agent workflows (`agents/orchestrator.py`)
6. **Week 6** — Evaluation & tuning (`evaluation/`, `evals/`)
7. **Week 7** — NVIDIA platform & deployment (`nvidia/`, `api/`)
8. **Week 8** — Monitoring, safety, HITL, final exam prep (`guardrails/`, tracing)
