from pathlib import Path

from typer.testing import CliRunner

from nvidia_agentic_research_engineer.cli import app

runner = CliRunner()

TEST_FILES_DIR = Path(__file__).parent / "test_files"


def test_info_command() -> None:
    result = runner.invoke(app, ["info"])

    assert result.exit_code == 0
    assert "nvidia-agentic-research-engineer" in result.stdout
    assert "Agentic AI certification" in result.stdout


def test_roadmap_command() -> None:
    result = runner.invoke(app, ["roadmap"])

    assert result.exit_code == 0
    assert "Week 1" in result.stdout
    assert "RAG" in result.stdout
    assert "NVIDIA platform" in result.stdout


def test_search_command() -> None:
    query = "sample text"
    result = runner.invoke(
        app,
        ["search", str(TEST_FILES_DIR), query, "--top-k", "3", "--chunk-size", "50", "--chunk-overlap", "10"],
    )

    assert result.exit_code == 0, result.output
    assert "Top" in result.output
    assert query in result.output
    assert "test." in result.output or ".md" in result.output or ".txt" in result.output  # source filename (test.txt or test.md)
    assert "score=" in result.output