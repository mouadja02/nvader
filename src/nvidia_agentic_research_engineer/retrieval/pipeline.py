from pathlib import Path

import typer
from rich.console import Console

from nvidia_agentic_research_engineer.ingestion.loaders import load_file
from nvidia_agentic_research_engineer.ingestion.chunking import chunk_document
from nvidia_agentic_research_engineer.retrieval.embeddings import get_embedder, HashEmbedder
from nvidia_agentic_research_engineer.retrieval.vector_store import InMemoryVectorStore, RetrievalResult
from nvidia_agentic_research_engineer.tools.convert2md import pdf_to_markdown 

def supported_files(path: Path) -> list[Path]:
    supported_suffixes = {".txt", ".md", ".pdf"}
    if not path.exists():
        raise FileNotFoundError(f"Path does not exist: {path}")
    if path.is_file():
        files = [path] if path.suffix.lower() in supported_suffixes else []
    else:
        files = sorted(p for p in path.rglob("*") if p.is_file() and p.suffix.lower() in supported_suffixes)
    if not files:
        raise ValueError(f"No supported files found at: {path}")
    return files
    
def build_vector_store_from_path(path: Path,
                                 *,
                                 chunk_size: int = typer.Option(800, "--chunk-size", help="Characters per chunk"),
                                 chunk_overlap: int = typer.Option(100, "--chunk-overlap", help="Overlap between chunks"),
                                 embedder: HashEmbedder | None = None,
                                 store: InMemoryVectorStore | None = None,
                                 reconvert: bool = False,
                                 ) -> InMemoryVectorStore:
    """Load, chunk, embed, and index documents from a path."""
    files = supported_files(path)
    if embedder is None:
        embedder = HashEmbedder()
    if store is None:
        store = InMemoryVectorStore(embedder=embedder)

    console = Console()
    for file in files:
        console.print(f"  [dim]Processing[/dim] {file.name}", end="")
        if file.suffix.lower() == ".pdf":
            md_path = file.with_suffix(".md")
            if reconvert or not md_path.exists():
                console.print(" [dim](converting PDF → md)[/dim]", end="")
                md_content = pdf_to_markdown(str(file))
                md_path.write_text(md_content, encoding="utf-8")
            load_path = md_path
        else:
            load_path = file
        document = load_file(load_path)
        chunks = chunk_document(document, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        store.add_chunks(chunks)
        console.print(f" [dim]→ {len(chunks)} chunks[/dim]")

    return store

def search_path(path: Path,
                query: str,
                top_k: int = 5,
                chunk_size: int = 500,
                chunk_overlap: int = 100,
                reconvert: bool = False,
                ) -> list[RetrievalResult]:
    """Search for a query across documents in a path."""
    embedder = get_embedder()
    store = build_vector_store_from_path(
        path,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        embedder=embedder,
        reconvert=reconvert,
    )
    return store.search(query, top_k=top_k)