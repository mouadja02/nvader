from pathlib import Path
import json
import typer
from dotenv import load_dotenv
from rich.console import Console, Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.padding import Padding

from nvidia_agentic_research_engineer.config import AppConfig, ProjectTOML
from nvidia_agentic_research_engineer.retrieval.pipeline import search_path, build_vector_store_from_path
from nvidia_agentic_research_engineer.tools.registry import Tool, ToolParameter, ToolRegistry
from nvidia_agentic_research_engineer.agents.react import ReActAgent
from nvidia_agentic_research_engineer.tools.tracing import write_agent_trace

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


@app.command()
def search(
    path: Path = typer.Argument(..., exists=True, readable=True, help="File to index and search"),
    query: str = typer.Argument(..., help="Search query"),
    top_k: int = typer.Option(5, "--top-k", "-k", min=1, help="Number of results to return"),
    chunk_size: int = typer.Option(200, "--chunk-size", help="Characters per chunk"),
    chunk_overlap: int = typer.Option(20, "--chunk-overlap", help="Overlap between chunks"),
    reconvert: bool = typer.Option(False, "--reconvert", help="Re-convert PDFs even if a cached .md already exists"),
    output: Path | None = typer.Option(None, "--output", "-o", help="Write results as JSON to this file path"),
) -> None:
    """Alias for search-on-file."""
    if not query.strip():
        console.print("[red]Error: Query cannot be empty.[/red]")
        raise typer.Exit(code=1)
    
    results = []
    # Extract all files in directory if a directory is provided
    if path.is_dir():
        for file in path.rglob("*"):
            if file.is_file():
                results.extend(search_path(file, query, top_k, chunk_size, chunk_overlap, reconvert))
    else:
        results.extend(search_path(path, query, top_k, chunk_size, chunk_overlap, reconvert))

    # sort results by score and rank across all files
    results.sort(key=lambda r: r.score, reverse=True)
    # Rebuild the rank based on the sorted order (frozen model requires model_copy)
    results = [r.model_copy(update={"rank": i + 1}) for i, r in enumerate(results)]
    # get top_k results across all files
    results = results[:top_k]
    
    table = Table(title=f'Top {top_k} results for "{query}"', show_lines=True)
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
            f"score={r.score:.4f}",
            r.chunk.source or "—",
            offsets,
            preview,
        )

    console.print(table)

    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        payload = [
            {
                "rank": r.rank,
                "score": r.score,
                "source": r.chunk.source,
                "start_char": r.chunk.start_char,
                "end_char": r.chunk.end_char,
                "chunk_id": r.chunk.id,
                "document_id": r.chunk.document_id,
                "chunk_index": r.chunk.chunk_index,
                "text": r.chunk.text,
                "metadata": r.chunk.metadata,
            }
            for r in results
        ]
        output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        console.print(f"[dim]Results saved to {output}[/dim]")


@app.command()
def agent(
    question: str = typer.Argument(..., help="The question to ask the agent"),
    knowledge_path: Path = typer.Option(
        Path("examples"),
        "--knowledge-path",
        "-p",
        help="Path to the knowledge base files",
    ),
    max_steps: int = typer.Option(5, "--max-steps", "-s", min=1, help="Maximum reasoning steps"),
    top_k: int = typer.Option(5, "--top-k", "-k", min=1, help="Number of search results per query"),
    chunk_size: int = typer.Option(200, "--chunk-size", help="Characters per chunk"),
    chunk_overlap: int = typer.Option(20, "--chunk-overlap", help="Overlap between chunks"),
    trace_output: Path | None = typer.Option(None, "--trace-output", "-t", help="Write the full agent run trace as JSON"),
) -> None:
    """Run the ReAct agent to answer a question using the knowledge base."""
    if not question.strip():
        console.print("[red]Error: Question cannot be empty.[/red]")
        raise typer.Exit(code=1)

    if not knowledge_path.exists():
        console.print(f"[red]Error: Knowledge path '{knowledge_path}' does not exist.[/red]")
        raise typer.Exit(code=1)

    # Build vector store from knowledge path
    console.print(f"[bold]Indexing knowledge base:[/bold] {knowledge_path}")
    store = build_vector_store_from_path(
        knowledge_path,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    console.print("[green]✓[/green] Knowledge base ready\n")

    # Create search tool wrapping the vector store
    def search_knowledge_base(query: str) -> str:
        results = store.search(query, top_k=top_k)
        if not results:
            return "No relevant results found."
        parts = []
        for r in results:
            source = r.chunk.source or "unknown"
            preview = r.chunk.text[:300].replace("\n", " ").strip()
            parts.append(f"[score={r.score:.4f} source={source}] {preview}")
        return "\n---\n".join(parts)

    registry = ToolRegistry()
    registry.register_tool(
        Tool(
            name="search_knowledge_base",
            description="Search the ingested knowledge base for relevant passages. Use this to find information about topics in the documents.",
            parameters=[
                ToolParameter(
                    name="query",
                    type="str",
                    description="The search query string",
                    required=True,
                )
            ],
            func=search_knowledge_base,
        )
    )

    # Instantiate and run the agent with live progress
    react_agent = ReActAgent(name="nvader-react", registry=registry)
    step_counter = {"n": 0}

    def _on_step(event: str, step) -> None:
        if event == "thinking":
            step_counter["n"] += 1
            console.print(f"  [bold cyan]Step {step_counter['n']}[/bold cyan] [dim]Thinking… (calling LLM)[/dim]")
        elif event == "tool_call":
            query_preview = ""
            if step and step.action_input:
                q = step.action_input.get("query", "")
                if q:
                    query_preview = f' query="{q[:60]}"'
            console.print(f"           [yellow]→ Tool:[/yellow] {step.action}{query_preview}")
        elif event == "tool_result":
            obs = (step.observation or "")[:80].replace("\n", " ")
            console.print(f"           [green]← Result:[/green] {obs}{'…' if len(step.observation or '') > 80 else ''}")
        elif event == "error":
            console.print(f"           [red]✗ Error:[/red] {step.error}")
        elif event == "finished":
            console.print("           [green]✓ Final answer reached[/green]")

    console.print(f"[bold]Agent reasoning[/bold] (max {max_steps} steps):")
    run = react_agent.execute(question, max_steps=max_steps, on_step=_on_step)
    console.print()

    # Display run trace
    trace_table = Table(title="Agent Run Trace", show_lines=True)
    trace_table.add_column("Step", style="bold cyan", justify="right", no_wrap=True)
    trace_table.add_column("State", no_wrap=True)
    trace_table.add_column("Thought", overflow="fold")
    trace_table.add_column("Action", no_wrap=True)
    trace_table.add_column("Observation / Error", overflow="fold")

    for s in run.steps:
        thought_preview = (s.thought or "—")[:200]
        obs_preview = (s.observation or s.error or "—")[:200]
        trace_table.add_row(
            str(s.step_number),
            s.state.value,
            thought_preview,
            s.action or "—",
            obs_preview,
        )

    console.print(trace_table)


    if trace_output is not None:
        write_agent_trace(run, trace_output)
        console.print(f"[dim]Trace saved to {trace_output}[/dim]")

    # Display final answer
    status_style = "bold green" if run.success else "bold red"
    status_text = "SUCCESS" if run.success else "FAILED"
    console.print(
        Panel(
            run.final_answer or "No final answer produced.",
            title=f"Final Answer [{status_text}]",
            border_style=status_style,
        )
    )



if __name__ == "__main__":
    app()