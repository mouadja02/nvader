from pathlib import Path
import json

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


def test_search_command_json_output(tmp_path) -> None:
    query = "sample text"
    out_file = tmp_path / "results.json"
    result = runner.invoke(
        app,
        [
            "search", str(TEST_FILES_DIR), query,
            "--top-k", "3", "--chunk-size", "50", "--chunk-overlap", "10",
            "--output", str(out_file),
        ],
    )

    assert result.exit_code == 0, result.output
    assert out_file.exists()

    data = json.loads(out_file.read_text(encoding="utf-8"))
    assert isinstance(data, list)
    assert len(data) <= 3

    first = data[0]
    assert first["rank"] == 1
    assert "score" in first
    assert "text" in first
    assert "source" in first
    assert "chunk_id" in first
    assert "document_id" in first
    assert isinstance(first["metadata"], dict)

    # ranks must be contiguous starting at 1
    assert [r["rank"] for r in data] == list(range(1, len(data) + 1))
    # scores must be descending
    scores = [r["score"] for r in data]
    assert scores == sorted(scores, reverse=True)