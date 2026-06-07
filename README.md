# NVIDIA Agentic Research Engineer
![Status](https://img.shields.io/badge/status-active_build-green)
![CI](https://github.com/mouadja02/nvader/actions/workflows/ci.yml/badge.svg)
![Python](https://img.shields.io/badge/python-3.11+-blue)
![Focus](https://img.shields.io/badge/focus-agentic_AI-76B900)
  
A production-style Agentic AI Research and Engineering Assistant built while preparing for the NVIDIA-Certified Professional: Agentic AI certification.

## Purpose

This project demonstrates practical agentic AI engineering:

- Resource ingestion for papers, docs, GitHub repos, and certification material
- RAG over technical knowledge
- Tool-using ReAct-style agents
- Short-term and long-term memory
- Multi-agent workflows
- Evaluation and benchmarking
- Guardrails, safety, and source attribution
- Logs, traces, monitoring, and reliability metrics
- NVIDIA-oriented integration paths: NIM, NeMo Guardrails, NeMo Agent Toolkit, Triton, TensorRT-LLM

## Project Identity

This is not a chatbot. It is an engineering assistant for AI builders.

It helps turn technical resources into grounded answers, implementation plans, code insights, benchmark reports, and documentation.

## Certification Coverage

1. Agent Architecture and Design
2. Agent Development
3. Evaluation and Tuning
4. Deployment and Scaling
5. Cognition, Planning, and Memory
6. Knowledge Integration and Data Handling
7. NVIDIA Platform Implementation
8. Run, Monitor, and Maintain
9. Safety, Ethics, and Compliance
10. Human-AI Interaction and Oversight

## Quickstart

**Prerequisites:** Python 3.11+, `make` (Linux/macOS built-in; Windows: install via `choco install make` or use Git Bash).

```bash
# 1. Clone the repo
git clone https://github.com/mouadja02/nvader.git
cd nvader

# 2. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate        # Linux / macOS
# .venv\Scripts\activate          # Windows (PowerShell)

# 3. Install the package with dev dependencies
make install

# 4. Verify everything works
make test
make lint

# 5. Try the CLI
nvader info
nvader roadmap
```

## Ingestion Pipeline

Current ingestion capabilities:

- Load `.txt` and `.md` files into typed `Document` objects.
- SHA-256 content-derived stable IDs (reproducible across runs).
- Split documents into `DocumentChunk` objects with character offsets.
- Preserve source, metadata, and chunk index for downstream retrieval.
- Offline deterministic embeddings via `HashEmbedder` (SHA-256, 384-dim).
- NVIDIA NIM embedding endpoint via `Embedder` (OpenAI-compatible API, requires `NVIDIA_API_KEY`).

## Developer Workflow

All commands run through the `.venv` Python automatically via the Makefile:

| Command        | What it does                                      |
|----------------|---------------------------------------------------|
| `make install` | Install package in editable mode with dev deps    |
| `make test`    | Run tests with pytest                             |
| `make lint`    | Run Ruff linter checks                            |
| `make format`  | Auto-format code with Ruff                        |
| `make check`   | Run lint + tests together                         |
| `make clean`   | Remove `.pytest_cache` and `__pycache__` dirs     |

## Current Status

### Implemented

**Core (`src/nvidia_agentic_research_engineer/core/`)**
- `Document` Pydantic model — SHA-256 content-derived ID, UTC timestamp, typed metadata
- `DocumentType` enum covering `text`, `markdown`, `html`, `pdf`, `url`, `repo`, `paper`, `image`
- `DocumentChunk` — slice of a Document with stable ID, `chunk_index`, and character offsets
- `short_preview()` helper for truncated content display

**Ingestion (`src/nvidia_agentic_research_engineer/ingestion/`)**
- `load_text_file(path)` — reads a plain-text file into a `Document`
- `load_markdown_file(path)` — reads a Markdown file into a `Document`
- `chunk_text(text, *, chunk_size, chunk_overlap)` — low-level text splitter with offset tracking
- `chunk_document(document, ...)` — converts a `Document` into `DocumentChunk` list
- `chunk_documents(documents, ...)` — batch chunking over a sequence of documents

**Retrieval (`src/nvidia_agentic_research_engineer/retrieval/`)**
- `EmbedderProtocol` — `@runtime_checkable` Protocol for any embedder implementation
- `HashEmbedder` — offline SHA-256-based deterministic embedder (384 dims, PYTHONHASHSEED-stable)
- `Embedder` — NVIDIA NIM OpenAI-compatible embedding endpoint wrapper; lazy-imports `openai`
- `get_embedder(model, api_key)` — factory returning `Embedder` if `NVIDIA_API_KEY` is set, else `HashEmbedder`
- `SearchResult` — frozen Pydantic model for retrieval results
- `RetrievalQuery` — typed query model with `top_k` bounds validation
- `RetrievalConfig` — embedding model, index, threshold, and hybrid-search configuration

**CLI (`nvader`)**
- `nvader info` — displays project identity and active config
- `nvader roadmap` — prints the 8-week certification build plan

**Config (`src/nvidia_agentic_research_engineer/config.py`)**
- `AppConfig` — typed settings for data directories, retrieval, citations, and retries
- `ProjectTOML` — reads name, version, and authors from `pyproject.toml`

**Tests (`tests/`)**
- `test_documents.py` — Document creation, type validation, preview truncation
- `test_ingestion_loaders.py` — text and Markdown file loading
- `test_chunking.py` — chunk_text, chunk_document, chunk_documents
- `test_retrieval_embeddings.py` — 45 tests: HashEmbedder (SHA-256 stability, dimensions, determinism), Embedder (mock API, error handling), EmbedderProtocol conformance, SearchResult/RetrievalQuery/RetrievalConfig models
- `test_cli.py` — `info` and `roadmap` CLI command assertions

### In Progress / Up Next

- ReAct-style tool-using agent (`agents/`, `tools/`)
- Short-term and long-term memory modules (`memory/`)
- Multi-agent workflows
- Evaluation and benchmarking harness (`evaluation/`)
- NIM / NeMo Guardrails / Triton integration (`nvidia/`, `guardrails/`)
- Vector store adapter and end-to-end RAG pipeline