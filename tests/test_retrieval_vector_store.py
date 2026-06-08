from __future__ import annotations

import pytest
from pydantic import ValidationError

from nvidia_agentic_research_engineer.core.documents import DocumentChunk
from nvidia_agentic_research_engineer.retrieval.embeddings import HashEmbedder
from nvidia_agentic_research_engineer.retrieval.vector_store import (
    InMemoryVectorStore,
    RetrievalResult,
    cosine_similarity,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_chunk(
    chunk_id: str,
    text: str,
    document_id: str = "doc-1",
    source: str | None = "https://example.com/paper.pdf",
    chunk_index: int = 0,
    start_char: int | None = None,
    end_char: int | None = None,
    metadata: dict | None = None,
) -> DocumentChunk:
    return DocumentChunk(
        id=chunk_id,
        document_id=document_id,
        text=text,
        chunk_index=chunk_index,
        source=source,
        start_char=start_char,
        end_char=end_char,
        metadata=metadata or {},
    )


# ---------------------------------------------------------------------------
# cosine_similarity
# ---------------------------------------------------------------------------

class TestCosineSimilarity:
    def test_cosine_similarity_identical_vectors(self):
        v = [1.0, 2.0, 3.0]
        result = cosine_similarity(v, v)
        assert abs(result - 1.0) < 1e-9

    def test_cosine_similarity_opposite_vectors(self):
        a = [1.0, 0.0, 0.0]
        b = [-1.0, 0.0, 0.0]
        result = cosine_similarity(a, b)
        assert abs(result - (-1.0)) < 1e-9

    def test_cosine_similarity_orthogonal_vectors(self):
        a = [1.0, 0.0]
        b = [0.0, 1.0]
        result = cosine_similarity(a, b)
        assert abs(result) < 1e-9

    def test_cosine_similarity_zero_vector_returns_zero(self):
        a = [0.0, 0.0, 0.0]
        b = [1.0, 2.0, 3.0]
        assert cosine_similarity(a, b) == 0.0
        assert cosine_similarity(b, a) == 0.0

    def test_cosine_similarity_both_zero_vectors_returns_zero(self):
        a = [0.0, 0.0]
        b = [0.0, 0.0]
        assert cosine_similarity(a, b) == 0.0

    def test_cosine_similarity_mismatched_lengths_raises(self):
        with pytest.raises(ValueError, match="length mismatch"):
            cosine_similarity([1.0, 2.0], [1.0, 2.0, 3.0])

    def test_cosine_similarity_single_element(self):
        assert abs(cosine_similarity([5.0], [5.0]) - 1.0) < 1e-9


# ---------------------------------------------------------------------------
# InMemoryVectorStore
# ---------------------------------------------------------------------------

class TestInMemoryVectorStore:
    def setup_method(self):
        self.embedder = HashEmbedder()
        self.store = InMemoryVectorStore(embedder=self.embedder)

    def test_vector_store_returns_empty_results_when_no_chunks(self):
        results = self.store.search("anything")
        assert results == []

    def test_vector_store_returns_ranked_results_with_scores(self):
        chunks = [
            make_chunk("c1", "NVIDIA GPU architecture overview"),
            make_chunk("c2", "Python programming fundamentals"),
            make_chunk("c3", "agentic AI reasoning with LLMs"),
        ]
        self.store.add_chunks(chunks)

        results = self.store.search("GPU NVIDIA architecture", top_k=3)

        assert len(results) == 3
        assert all(isinstance(r, RetrievalResult) for r in results)
        # Ranks are 1-based and sequential
        assert [r.rank for r in results] == [1, 2, 3]
        # Results are in descending score order
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)
        # All scores are floats in [-1, 1]
        for r in results:
            assert isinstance(r.score, float)
            assert -1.0 <= r.score <= 1.0

    def test_vector_store_top_k_limits_returned_results(self):
        chunks = [make_chunk(f"c{i}", f"chunk text number {i}") for i in range(10)]
        self.store.add_chunks(chunks)

        results = self.store.search("chunk text", top_k=3)
        assert len(results) == 3

    def test_vector_store_top_k_larger_than_store_returns_all(self):
        chunks = [make_chunk("c1", "only one chunk")]
        self.store.add_chunks(chunks)

        results = self.store.search("one chunk", top_k=50)
        assert len(results) == 1

    def test_vector_store_rejects_invalid_top_k(self):
        with pytest.raises(ValueError):
            self.store.search("query", top_k=0)
        with pytest.raises(ValueError):
            self.store.search("query", top_k=-1)

    def test_retrieval_result_preserves_source_metadata_and_offsets(self):
        """Attribution data (source, metadata, char offsets) must survive retrieval."""
        chunk = make_chunk(
            chunk_id="c-attr",
            text="Transformer attention mechanisms explained",
            document_id="paper-42",
            source="https://arxiv.org/abs/1706.03762",
            chunk_index=3,
            start_char=512,
            end_char=1024,
            metadata={"page": 4, "section": "3.1", "authors": ["Vaswani et al."]},
        )
        self.store.add_chunks([chunk])

        results = self.store.search("attention mechanism transformer", top_k=1)

        assert len(results) == 1
        result = results[0]
        assert result.rank == 1
        assert result.chunk.id == "c-attr"
        assert result.chunk.document_id == "paper-42"
        assert result.chunk.source == "https://arxiv.org/abs/1706.03762"
        assert result.chunk.chunk_index == 3
        assert result.chunk.start_char == 512
        assert result.chunk.end_char == 1024
        assert result.chunk.metadata["page"] == 4
        assert result.chunk.metadata["section"] == "3.1"
        assert "Vaswani et al." in result.chunk.metadata["authors"]

    def test_add_chunks_accumulates_across_calls(self):
        self.store.add_chunks([make_chunk("c1", "first batch")])
        self.store.add_chunks([make_chunk("c2", "second batch")])

        results = self.store.search("batch", top_k=10)
        assert len(results) == 2

    def test_retrieval_result_is_immutable(self):
        chunks = [make_chunk("c1", "immutability test")]
        self.store.add_chunks(chunks)
        result = self.store.search("test")[0]

        with pytest.raises((AttributeError, TypeError, ValidationError)):
            result.rank = 99  # type: ignore[misc]
