from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from nvidia_agentic_research_engineer.retrieval.embeddings import HashEmbedder
from nvidia_agentic_research_engineer.retrieval.pipeline import (
    build_vector_store_from_path,
    search_path,
    supported_files,
)
from nvidia_agentic_research_engineer.retrieval.vector_store import InMemoryVectorStore

# ---------------------------------------------------------------------------
# Fixtures / shared content
# ---------------------------------------------------------------------------

AGENTIC_AI_CONTENT = """\
NVIDIA-Certified Professional: Agentic AI Exam Study Guide

This guide covers the agentic AI certification exam, recommended training,
and suggested reading to prepare for the exam.

Job Description: The agentic AI professional builds, evaluates, and deploys
agentic systems using frameworks like LangGraph and AutoGen.

Certification Topics:
- Agentic AI fundamentals
- RAG and vector databases
- LLM tool use and function calling
- Multi-agent orchestration
- Guardrails and safety
- NVIDIA-Certified Professional | Agentic AI Exam Study Guide
"""

UNRELATED_CONTENT = """\
A recipe for chocolate chip cookies.

Ingredients: flour, sugar, butter, eggs, chocolate chips.
Mix dry ingredients, cream butter and sugar, combine, fold in chips.
Bake at 375°F for 10-12 minutes until golden brown.
Let cool on a wire rack before serving.
"""


# ---------------------------------------------------------------------------
# supported_files
# ---------------------------------------------------------------------------

class TestSupportedFiles:
    def test_single_txt_file(self, tmp_path):
        f = tmp_path / "doc.txt"
        f.write_text("hello")
        assert supported_files(f) == [f]

    def test_single_md_file(self, tmp_path):
        f = tmp_path / "doc.md"
        f.write_text("hello")
        assert supported_files(f) == [f]

    def test_single_pdf_file(self, tmp_path):
        f = tmp_path / "doc.pdf"
        f.write_bytes(b"%PDF-1.4 fake content")
        assert supported_files(f) == [f]

    def test_single_unsupported_file_raises(self, tmp_path):
        f = tmp_path / "data.csv"
        f.write_text("a,b,c")
        with pytest.raises(ValueError, match="No supported files found"):
            supported_files(f)

    def test_directory_collects_supported_files_only(self, tmp_path):
        (tmp_path / "a.txt").write_text("text")
        (tmp_path / "b.md").write_text("md")
        (tmp_path / "c.pdf").write_bytes(b"%PDF fake")
        (tmp_path / "skip.csv").write_text("csv")
        result = supported_files(tmp_path)
        names = {p.name for p in result}
        assert names == {"a.txt", "b.md", "c.pdf"}
        assert "skip.csv" not in {p.name for p in result}

    def test_directory_results_are_sorted(self, tmp_path):
        (tmp_path / "z.md").write_text("z")
        (tmp_path / "a.md").write_text("a")
        (tmp_path / "m.txt").write_text("m")
        result = supported_files(tmp_path)
        assert result == sorted(result)

    def test_nonexistent_path_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            supported_files(tmp_path / "missing.txt")

    def test_empty_directory_raises(self, tmp_path):
        (tmp_path / "only.csv").write_text("data")
        with pytest.raises(ValueError, match="No supported files found"):
            supported_files(tmp_path)

    def test_nested_directory_recurses(self, tmp_path):
        sub = tmp_path / "sub"
        sub.mkdir()
        f = sub / "deep.md"
        f.write_text("deep content")
        result = supported_files(tmp_path)
        assert f in result


# ---------------------------------------------------------------------------
# build_vector_store_from_path
# ---------------------------------------------------------------------------

class TestBuildVectorStoreFromPath:
    def test_builds_store_from_txt(self, tmp_path):
        (tmp_path / "doc.txt").write_text("Some document content.", encoding="utf-8")
        store = build_vector_store_from_path(tmp_path, chunk_size=50, chunk_overlap=10)
        assert len(store._items) > 0

    def test_builds_store_from_md(self, tmp_path):
        (tmp_path / "doc.md").write_text("# Title\n\nSome markdown content.", encoding="utf-8")
        store = build_vector_store_from_path(tmp_path, chunk_size=50, chunk_overlap=10)
        assert len(store._items) > 0

    def test_builds_store_from_multiple_files(self, tmp_path):
        (tmp_path / "a.md").write_text(AGENTIC_AI_CONTENT, encoding="utf-8")
        (tmp_path / "b.txt").write_text(UNRELATED_CONTENT, encoding="utf-8")
        store = build_vector_store_from_path(tmp_path, chunk_size=100, chunk_overlap=20)
        assert len(store._items) > 0

    def test_chunk_count_matches_expected(self, tmp_path):
        # 300 chars / chunk_size 100 with overlap 0 → 3 chunks
        content = "x" * 300
        (tmp_path / "doc.txt").write_text(content, encoding="utf-8")
        store = build_vector_store_from_path(tmp_path, chunk_size=100, chunk_overlap=0)
        assert len(store._items) == 3

    def test_custom_embedder_and_store_are_reused(self, tmp_path):
        (tmp_path / "doc.md").write_text("test content", encoding="utf-8")
        embedder = HashEmbedder()
        existing_store = InMemoryVectorStore(embedder=embedder)
        returned = build_vector_store_from_path(
            tmp_path, chunk_size=50, chunk_overlap=10, embedder=embedder, store=existing_store
        )
        assert returned is existing_store

    def test_pdf_conversion_is_called(self, tmp_path):
        pdf_file = tmp_path / "doc.pdf"
        pdf_file.write_bytes(b"%PDF-1.4 fake")
        with patch(
            "nvidia_agentic_research_engineer.retrieval.pipeline.pdf_to_markdown",
            return_value=AGENTIC_AI_CONTENT,
        ) as mock_convert:
            store = build_vector_store_from_path(tmp_path, chunk_size=100, chunk_overlap=20)
        mock_convert.assert_called_once_with(str(pdf_file))
        assert len(store._items) > 0

    def test_pdf_converted_md_is_persisted(self, tmp_path):
        (tmp_path / "doc.pdf").write_bytes(b"%PDF-1.4 fake")
        with patch(
            "nvidia_agentic_research_engineer.retrieval.pipeline.pdf_to_markdown",
            return_value=AGENTIC_AI_CONTENT,
        ):
            build_vector_store_from_path(tmp_path, chunk_size=100, chunk_overlap=20)
        assert (tmp_path / "doc.md").exists()

    def test_pdf_no_reconvert_reuses_existing_md(self, tmp_path):
        pdf_file = tmp_path / "doc.pdf"
        pdf_file.write_bytes(b"%PDF-1.4 fake")
        cached_md = tmp_path / "doc.md"
        cached_md.write_text(AGENTIC_AI_CONTENT, encoding="utf-8")
        with patch(
            "nvidia_agentic_research_engineer.retrieval.pipeline.pdf_to_markdown",
        ) as mock_convert:
            build_vector_store_from_path(tmp_path, chunk_size=100, chunk_overlap=20, reconvert=False)
        mock_convert.assert_not_called()

    def test_pdf_reconvert_overwrites_existing_md(self, tmp_path):
        pdf_file = tmp_path / "doc.pdf"
        pdf_file.write_bytes(b"%PDF-1.4 fake")
        (tmp_path / "doc.md").write_text("old content", encoding="utf-8")
        with patch(
            "nvidia_agentic_research_engineer.retrieval.pipeline.pdf_to_markdown",
            return_value=AGENTIC_AI_CONTENT,
        ) as mock_convert:
            build_vector_store_from_path(tmp_path, chunk_size=100, chunk_overlap=20, reconvert=True)
        mock_convert.assert_called_once_with(str(pdf_file))
        assert (tmp_path / "doc.md").read_text(encoding="utf-8") == AGENTIC_AI_CONTENT


# ---------------------------------------------------------------------------
# search_path — relevance tests mirroring:
#   nvader search examples "agentic AI certification" --top-k 10
#              --chunk-size 100 --chunk-overlap 20
# Expected: most top results come from the file with agentic AI content
#           (mirrors examples\nvt-sg.md dominating the real run)
# ---------------------------------------------------------------------------

class TestSearchPath:
    def test_search_returns_results(self, tmp_path):
        (tmp_path / "guide.md").write_text(AGENTIC_AI_CONTENT, encoding="utf-8")
        results = search_path(tmp_path, "agentic AI certification", top_k=5,
                              chunk_size=100, chunk_overlap=20)
        assert len(results) > 0

    def test_search_results_are_ranked_ascending(self, tmp_path):
        (tmp_path / "guide.md").write_text(AGENTIC_AI_CONTENT, encoding="utf-8")
        results = search_path(tmp_path, "agentic AI certification", top_k=5,
                              chunk_size=100, chunk_overlap=20)
        ranks = [r.rank for r in results]
        assert ranks == sorted(ranks)

    def test_search_scores_descending(self, tmp_path):
        (tmp_path / "guide.md").write_text(AGENTIC_AI_CONTENT, encoding="utf-8")
        results = search_path(tmp_path, "agentic AI certification", top_k=5,
                              chunk_size=100, chunk_overlap=20)
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_top_k_is_respected(self, tmp_path):
        (tmp_path / "guide.md").write_text(AGENTIC_AI_CONTENT * 5, encoding="utf-8")
        results = search_path(tmp_path, "agentic AI certification", top_k=3,
                              chunk_size=100, chunk_overlap=20)
        assert len(results) <= 3

    def test_relevant_source_dominates_top_results(self, tmp_path):
        """nvt-sg.md (agentic AI content) should dominate over unrelated file."""
        (tmp_path / "nvt-sg.md").write_text(AGENTIC_AI_CONTENT * 4, encoding="utf-8")
        (tmp_path / "cookies.md").write_text(UNRELATED_CONTENT * 4, encoding="utf-8")
        results = search_path(tmp_path, "agentic AI certification", top_k=10,
                              chunk_size=100, chunk_overlap=20)
        top_5_sources = [Path(r.chunk.source).name for r in results[:5]]
        assert top_5_sources.count("nvt-sg.md") >= 3

    def test_rank_1_comes_from_relevant_source(self, tmp_path):
        (tmp_path / "nvt-sg.md").write_text(AGENTIC_AI_CONTENT * 4, encoding="utf-8")
        (tmp_path / "cookies.md").write_text(UNRELATED_CONTENT * 4, encoding="utf-8")
        results = search_path(tmp_path, "agentic AI certification", top_k=10,
                              chunk_size=100, chunk_overlap=20)
        assert results[0].rank == 1
        assert Path(results[0].chunk.source).name == "nvt-sg.md"

    def test_pdf_source_dominates_results(self, tmp_path):
        """PDF converted via markitdown should dominate when it has the relevant content."""
        pdf_file = tmp_path / "nvt-sg.pdf"
        pdf_file.write_bytes(b"%PDF-1.4 fake")
        (tmp_path / "unrelated.md").write_text(UNRELATED_CONTENT * 4, encoding="utf-8")
        with patch(
            "nvidia_agentic_research_engineer.retrieval.pipeline.pdf_to_markdown",
            return_value=AGENTIC_AI_CONTENT * 4,
        ):
            results = search_path(tmp_path, "agentic AI certification", top_k=10,
                                  chunk_size=100, chunk_overlap=20)
        # source is the temp .md which shares the stem with the pdf
        top_5_stems = [Path(r.chunk.source).stem for r in results[:5]]
        assert top_5_stems.count("nvt-sg") >= 3
