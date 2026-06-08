from pathlib import Path
import os
import typer
from dotenv import load_dotenv
from rich.console import Console, Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.padding import Padding

from nvidia_agentic_research_engineer.config import AppConfig, ProjectTOML
from nvidia_agentic_research_engineer.ingestion.chunking import chunk_document
from nvidia_agentic_research_engineer.ingestion.loaders import load_markdown_file, load_text_file
from nvidia_agentic_research_engineer.retrieval import InMemoryVectorStore, get_embedder

load_dotenv()  # Load environment variables from .env file

app = typer.Typer(name="NVADER", help="NVIDIA Agentic Research Engineer CLI")
console = Console()

author = ", ".join(ProjectTOML().authors_names) if ProjectTOML().authors_names else None
version = ProjectTOML().version if ProjectTOML().version else None

# Display welcome banner (CLAUDE CODE STYLE)
def banner(console, author, version) -> None:
    art = [
    "███╗   ██╗ ██╗   ██╗  █████╗  ██████╗  ███████╗ ██████╗",
    "████╗  ██║ ██║   ██║ ██╔══██╗ ██╔══██╗ ██╔════╝ ██╔══██╗",
    "██╔██╗ ██║ ██║   ██║ ███████║ ██║  ██║ █████╗   ██████╔╝",
    "██║╚██╗██║ ╚██╗ ██╔╝ ██╔══██║ ██║  ██║ ██╔══╝   ██╔══██╗",
    "██║ ╚████║  ╚████╔╝  ██║  ██║ ██████╔╝ ███████╗ ██║  ██║",
    "╚═╝  ╚═══╝   ╚═══╝   ╚═╝  ╚═╝ ╚═════╝  ╚══════╝ ╚═╝  ╚═╝",
    ]
    # Pick a different color (solid or gradient) for each run from a predefined set
    color = "76b900"
    lines = Text()
    for i, row in enumerate(art):
        lines.append(row + "\n", style=color)
    tagline = Text("Your specialized data-engineering CLI agent", style="muted")
    meta = Text()
    if version:
        meta.append(f"v{version}", style="accent")
    if author:
        meta.append(f"  ·  {author}", style="muted")
    body = Group(lines, tagline, meta) if (version or author) else Group(lines, tagline)
    console.print(Padding(body, (1, 2, 0, 2)))


@app.command()
def info() -> None:
    # Show project identity and current configuration
    config = AppConfig()
    project_toml = ProjectTOML()
    banner(console, author=project_toml.authors_names[0] if project_toml.authors_names else None, version=project_toml.version)
    console.print(
        Panel.fit(
            f"[bold green]{config.project_name}[/bold green] ({project_toml.name})\n\n"
            "Production-style Agentic AI Research and Engineering Assistant.\n"
            "Built for NVIDIA Agentic AI certification preparation.\n\n"
            f"Data dir: {config.data_dir}\n"
            f"Require citations: {config.require_citations}",
            title="Project Info",
        )
    )


@app.command()
def roadmap() -> None:
    # Print the two-month certification build roadmap
    console.print("[bold]Two-Month Roadmap[/bold]")
    console.print("Week 1: Foundation and architecture")
    console.print("Week 2: RAG and knowledge integration")
    console.print("Week 3: ReAct and tool orchestration")
    console.print("Week 4: Planning and memory")
    console.print("Week 5: Multi-agent workflows")
    console.print("Week 6: Evaluation and tuning")
    console.print("Week 7: NVIDIA platform and deployment")
    console.print("Week 8: Monitoring, safety, HITL, final exam prep")


_LOADERS = {
    ".md": load_markdown_file,
    ".markdown": load_markdown_file,
    ".txt": load_text_file,
}


@app.command()
def search(
    file: Path = typer.Argument(..., exists=True, readable=True, help="File to index and search"),
    query: str = typer.Argument(..., help="Search query"),
    top_k: int = typer.Option(5, "--top-k", "-k", min=1, help="Number of results to return"),
    chunk_size: int = typer.Option(500, "--chunk-size", help="Characters per chunk"),
    chunk_overlap: int = typer.Option(100, "--chunk-overlap", help="Overlap between chunks"),
) -> None:
    """Load a file, chunk it, and search with a query."""
    suffix = file.suffix.lower()
    loader = _LOADERS.get(suffix)
    if loader is None:
        console.print(f"[red]Unsupported file type:[/red] {suffix}. Supported: {', '.join(_LOADERS)}")
        raise typer.Exit(code=1)

    with console.status(f"Loading [bold]{file}[/bold]..."):
        document = loader(file)

    chunks = chunk_document(document, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    if not chunks:
        console.print("[yellow]No chunks produced — file may be empty.[/yellow]")
        raise typer.Exit()

    with console.status(f"Embedding {len(chunks)} chunk(s)..."):
        store = InMemoryVectorStore(embedder=get_embedder(api_key=os.environ.get("NVIDIA_API_KEY")))
        store.add_chunks(chunks)

    results = store.search(query, top_k=top_k)
    if not results:
        console.print("[yellow]No results found.[/yellow]")
        raise typer.Exit()

    table = Table(title=f'Search results for "{query}"', show_lines=True)
    table.add_column("Rank", style="bold cyan", justify="right", no_wrap=True)
    table.add_column("Score", justify="right", no_wrap=True)
    table.add_column("Source", style="dim", overflow="fold")
    table.add_column("Offsets", justify="right", no_wrap=True)
    table.add_column("Preview")

    for r in results:
        preview = r.chunk.text[:120].replace("\n", " ").strip()
        if len(r.chunk.text) > 120:
            preview += "…"
        offsets = (
            f"{r.chunk.start_char}–{r.chunk.end_char}"
            if r.chunk.start_char is not None and r.chunk.end_char is not None
            else "—"
        )
        table.add_row(
            str(r.rank),
            f"{r.score:.4f}",
            r.chunk.source or "—",
            offsets,
            preview,
        )

    console.print(table)


if __name__ == "__main__":
    app()