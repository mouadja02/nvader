from typer.testing import CliRunner

from nvidia_agentic_research_engineer.cli import app

runner = CliRunner()


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