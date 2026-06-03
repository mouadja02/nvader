# Architecture

> NVADER v0.1.0 — current state as of project Day 1 (foundation scaffold).

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
├── agents/             # Agent definitions & orchestrator
├── api/                # REST / FastAPI surface (planned)
├── core/               # Domain models & cross-cutting concerns
│   └── documents.py    # Document model with type enum & preview
├── evaluation/         # Eval harnesses, metrics, reports
├── guardrails/         # Input/output guardrails, NeMo adapter
├── ingestion/          # Document loaders, chunking, indexing
├── memory/             # Conversation & long-term memory stores
├── nvidia/             # NVIDIA platform integrations (NIM, etc.)
├── retrieval/          # Vector search & RAG retrieval logic
└── tools/              # Tool definitions & registry
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

- **`DocumentType`** — supported source types (text, markdown, HTML, PDF, URL, repo, paper, image).
- **`Document`** — auto-ID via UUID, timestamped, with `short_preview()` for display truncation.

## Planned Modules (Stubbed)

All modules below exist as empty packages, mapped to NVIDIA certification domains.

| Module         | Purpose                                        | Certification Domain       |
|----------------|------------------------------------------------|----------------------------|
| `agents/`      | Base agent, ReAct loop, orchestrator           | Agent Architecture & Dev   |
| `tools/`       | Tool registry, built-in tools                  | Agent Architecture & Dev   |
| `memory/`      | Conversation buffer, long-term memory          | Cognition & Memory         |
| `ingestion/`   | Doc loaders, chunking, embedding, indexing     | Knowledge & Data           |
| `retrieval/`   | Vector search, hybrid retrieval, RAG pipeline  | Knowledge & Data           |
| `evaluation/`  | Eval harnesses, metrics, report generation     | Evaluation                 |
| `guardrails/`  | Input/output validation, NeMo Guardrails       | Safety & Ethics            |
| `nvidia/`      | NIM endpoints, NeMo platform integration       | NVIDIA Platform            |
| `api/`         | REST API surface for the agent                 | Deployment                 |

## Data Layout

```
data/
├── raw/            # Original ingested documents
├── processed/      # Chunked / transformed documents
├── vector_store/   # Embedding index (e.g. FAISS, Milvus)
└── traces/         # Agent execution traces for observability
```

## Dependencies

| Package        | Role                          |
|----------------|-------------------------------|
| `typer ≥0.12`  | CLI framework                 |
| `pydantic ≥2.7`| Config & domain model validation |
| `rich ≥13.7`   | Terminal formatting & panels  |
| `python-dotenv`| `.env` file loading           |

Dev: `pytest ≥8.0`, `ruff ≥0.5`.

## Build Roadmap Alignment

The architecture is being built incrementally over 8 weeks:

1. **Week 1** — Foundation & architecture skeleton ← *current*
2. **Week 2** — RAG & knowledge integration (`ingestion/`, `retrieval/`)
3. **Week 3** — ReAct agent & tool orchestration (`agents/`, `tools/`)
4. **Week 4** — Planning & memory (`memory/`, `core/state.py`)
5. **Week 5** — Multi-agent workflows (`agents/orchestrator.py`)
6. **Week 6** — Evaluation & tuning (`evaluation/`, `evals/`)
7. **Week 7** — NVIDIA platform & deployment (`nvidia/`, `api/`)
8. **Week 8** — Monitoring, safety, HITL, final exam prep (`guardrails/`, tracing)
