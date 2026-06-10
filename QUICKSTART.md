# NVADER — Quick Start

> Get from zero to a working local retrieval search in under 5 minutes.

## 1. Prerequisites

- Python 3.11+
- Git

## 2. Clone & Install

```bash
git clone https://github.com/mouadja02/nvader.git
cd nvader

python -m venv .venv
source .venv/bin/activate         # Linux / macOS
# .venv\Scripts\Activate.ps1      # Windows PowerShell
```

**Linux / macOS**
```bash
make install
```

**Windows PowerShell**
```powershell
.\make.ps1 install
```

This installs the package in editable mode along with all dev dependencies, and builds `markitdown[pdf]` for PDF support.

## 3. Verify

```bash
make test    # or: .\make.ps1 test
```

All tests should pass (98+ currently).

## 4. Check the CLI

```bash
nvader info
nvader roadmap
```

## 5. Local Retrieval Demo

Search across the bundled example files with offline hash embeddings (no API key needed):

```bash
nvader search examples "agentic AI certification" --top-k 5
```

Expected output: a ranked table showing the most relevant chunks from `examples/nvt-sg.md` (the NVIDIA Agentic AI exam study guide).

Tune the chunking strategy:

```bash
nvader search examples "multi-agent orchestration" --top-k 3 --chunk-size 200 --chunk-overlap 40
```

## 6. PDF Support

PDFs are automatically converted to Markdown via [MarkItDown](https://github.com/microsoft/markitdown) and cached alongside the source file:

```bash
# First run — converts the PDF and saves examples/nvt-sg.md
nvader search examples "RAG pipeline" --top-k 5

# Subsequent runs — reuses the cached .md (no conversion cost)
nvader search examples "RAG pipeline" --top-k 5

# Force re-conversion (e.g. after the source PDF is updated)
nvader search examples "RAG pipeline" --top-k 5 --reconvert
```

## 7. NVIDIA NIM Embeddings

Set your NVIDIA API key to use semantic NIM embeddings instead of the offline hash embedder:

```bash
export NVIDIA_API_KEY=nvapi-...     # Linux / macOS
$env:NVIDIA_API_KEY = "nvapi-..."   # Windows PowerShell
```

Then run any `nvader search` command — `get_embedder()` automatically selects `Embedder` (NIM `nvidia/nv-embed-v1`) when the key is present.

```bash
nvader search examples "agentic AI certification" --top-k 5
```

## 8. Developer Commands

| Command | Description |
|---|---|
| `make install` / `.\make.ps1 install` | Install with dev deps |
| `make test` / `.\make.ps1 test` | Run pytest |
| `make lint` / `.\make.ps1 lint` | Run Ruff linter |
| `make format` / `.\make.ps1 format` | Auto-format with Ruff |
| `make check` / `.\make.ps1 check` | Lint + test |
| `make clean` / `.\make.ps1 clean` | Remove caches |

## Next Steps

- See [docs/architecture.md](docs/architecture.md) for the full system design.
- See [README.md](README.md) for the complete feature list and roadmap.
