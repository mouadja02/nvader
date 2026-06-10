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

**Prerequisites:** Python 3.11+, `make` (Linux/macOS built-in; Windows: use `make.ps1`).

```bash
# 1. Clone the repo
git clone https://github.com/mouadja02/nvader.git
cd nvader

# 2. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate        # Linux / macOS
# .venv\Scripts\activate          # Windows (PowerShell)

# 3. Install the package with dev dependencies
make install          # Linux/macOS
.\make.ps1 install    # Windows PowerShell

# 4. Verify everything works
make test
make lint

# 5. Try the CLI
nvader info
nvader roadmap
```

## Ingestion Pipeline

Current ingestion capabilities:

- Load `.txt`, `.md`, and `.pdf` files into typed `Document` objects.
- **PDF processing**: PDFs are automatically converted to Markdown using [MarkItDown](https://github.com/microsoft/markitdown) (with optional LLM-based OCR via the `markitdown-ocr` plugin). The converted `.md` is cached alongside the original PDF for fast re-use.
- SHA-256 content-derived stable IDs (reproducible across runs).
- Split documents into `DocumentChunk` objects with character offsets.
- Preserve source, metadata, and chunk index for downstream retrieval.
- Offline deterministic embeddings via `HashEmbedder` (feature hashing, 384-dim).
- **NVIDIA NIM embeddings** via `Embedder` (OpenAI-compatible API, requires `NVIDIA_API_KEY`). When the key is present `get_embedder()` automatically selects the NIM endpoint; otherwise falls back to `HashEmbedder` for fully offline use.

## Local Retrieval Demo

```bash
nvader search examples "agentic AI certification" --top-k 3
```

This builds an offline in-memory retrieval index from all supported files (`.txt`, `.md`, `.pdf`) under the given path using deterministic hash embeddings, then returns source-attributed matching chunks ranked by cosine similarity.

Key options:

| Flag | Default | Description |
|---|---|---|
| `--top-k` / `-k` | `5` | Number of results to return |
| `--chunk-size` | `500` | Characters per chunk |
| `--chunk-overlap` | `100` | Overlap between consecutive chunks |
| `--reconvert` | off | Re-run PDF→Markdown conversion even if a cached `.md` exists |

```bash
# Use NVIDIA NIM embeddings (set your key first)
export NVIDIA_API_KEY=nvapi-...
nvader search docs "multi-agent orchestration" --top-k 5

# Force re-conversion of PDFs (e.g. after updating the source PDF)
nvader search data/raw "deployment" --top-k 3 --reconvert
```

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

On Windows PowerShell, replace `make <target>` with `.\make.ps1 <target>`.

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
- `HashEmbedder` — offline feature-hashing deterministic embedder (384 dims, SHA-256-based hashing trick for vocabulary-aware similarity)
- `Embedder` — NVIDIA NIM OpenAI-compatible embedding endpoint wrapper (`nvidia/nv-embed-v1`); lazy-imports `openai`
- `get_embedder(model, api_key)` — factory returning `Embedder` if `NVIDIA_API_KEY` is set, else `HashEmbedder`
- `SearchResult` — frozen Pydantic model for retrieval results
- `RetrievalQuery` — typed query model with `top_k` bounds validation
- `RetrievalConfig` — embedding model, index, threshold, and hybrid-search configuration
- `InMemoryVectorStore` — in-memory vector index with cosine similarity ranking
- `RetrievalResult` — frozen result with `rank`, `score`, `chunk`, source, and character offsets
- `cosine_similarity(a, b)` — dependency-free helper; safe on zero-norm vectors

**Pipeline (`src/nvidia_agentic_research_engineer/retrieval/pipeline.py`)**
- `supported_files(path)` — discovers `.txt`, `.md`, `.pdf` files from a file or directory
- `build_vector_store_from_path(path, ...)` — loads, chunks, embeds, and indexes all supported files into an `InMemoryVectorStore`
- `search_path(path, query, ...)` — end-to-end search: index a path and return ranked `RetrievalResult` list

**PDF Processing (`src/nvidia_agentic_research_engineer/tools/convert2md.py`)**
- Converts PDF files to Markdown using [MarkItDown](https://github.com/microsoft/markitdown) with NVIDIA NIM LLM client (`nvidia/nemotron-nano-12b-v2-vl`) for LLM-assisted OCR
- Converted `.md` is **cached** alongside the original PDF — subsequent calls reuse it without re-converting
- Pass `--reconvert` on the CLI (or `reconvert=True` in code) to force re-conversion

**CLI (`nvader`)**
- `nvader info` — displays project identity and active config
- `nvader roadmap` — prints the 8-week certification build plan
- `nvader search <path> <query>` — indexes all supported files under `<path>` and returns top-k source-attributed results
  - `--top-k` / `-k`: number of results (default 5)
  - `--chunk-size`: characters per chunk (default 500)
  - `--chunk-overlap`: overlap between chunks (default 100)
  - `--reconvert`: force PDF re-conversion ignoring cache

**Config (`src/nvidia_agentic_research_engineer/config.py`)**
- `AppConfig` — typed settings for data directories, retrieval, citations, and retries
- `ProjectTOML` — reads name, version, and authors from `pyproject.toml`

**Tests (`tests/`)**
- `test_documents.py` — Document creation, type validation, preview truncation
- `test_ingestion_loaders.py` — text and Markdown file loading
- `test_chunking.py` — `chunk_text`, `chunk_document`, `chunk_documents`
- `test_retrieval_embeddings.py` — 45 tests: HashEmbedder, Embedder (mock API), EmbedderProtocol, SearchResult/RetrievalQuery/RetrievalConfig
- `test_retrieval_vector_store.py` — `cosine_similarity`, `InMemoryVectorStore`, `RetrievalResult`
- `test_retrieval_pipeline.py` — 24 tests: `supported_files`, `build_vector_store_from_path` (incl. PDF caching, `--reconvert`), `search_path` relevance ranking
- `test_cli.py` — `info`, `roadmap`, `search` CLI command assertions

### In Progress / Up Next

- ReAct-style tool-using agent (`agents/`, `tools/`)
- Short-term and long-term memory modules (`memory/`)
- Multi-agent workflows
- Evaluation and benchmarking harness (`evaluation/`)
- NIM / NeMo Guardrails / Triton integration (`nvidia/`, `guardrails/`)
- ReAct-style tool-using agent (`agents/`, `tools/`)
- Short-term and long-term memory modules (`memory/`)
- Multi-agent workflows
- Evaluation and benchmarking harness (`evaluation/`)
- NIM / NeMo Guardrails / Triton integration (`nvidia/`, `guardrails/`)
- End-to-end RAG answer synthesis (`retrieve_context()` + `ask` command)

## Current RAG Capability

```
.txt / .md file
    -> Document
    -> DocumentChunk
    -> HashEmbedder (offline) | Embedder (NVIDIA NIM, nvidia/nv-embed-v1)
    -> InMemoryVectorStore
    -> top-k RetrievalResult with source attribution

.pdf file
    -> MarkItDown (NVIDIA NIM LLM OCR, nvidia/nemotron-nano-12b-v2-vl)
    -> cached .md file
    -> Document -> DocumentChunk
    -> HashEmbedder | Embedder
    -> InMemoryVectorStore
    -> top-k RetrievalResult with source attribution
```