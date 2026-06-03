import typer
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from nvidia_agentic_research_engineer.config import AppConfig, ProjectTOML

app = typer.Typer(name="NVADER", help="NVIDIA Agentic Research Engineer CLI")
console = Console()

author = ", ".join(ProjectTOML().authors_names) if ProjectTOML().authors_names else None
version = ProjectTOML().version if ProjectTOML().version else None

"""
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
"""

@app.command()
def info() -> None:
    # Show project identity and current configuration
    config = AppConfig()
    project_toml = ProjectTOML()
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


if __name__ == "__main__":
    app()