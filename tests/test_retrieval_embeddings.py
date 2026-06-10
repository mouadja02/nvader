from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest
from pydantic import ValidationError

from nvidia_agentic_research_engineer.retrieval.embeddings import (
    EmbedderProtocol,
    HashEmbedder,
    Embedder,
    get_embedder,
)
from nvidia_agentic_research_engineer.retrieval.models import (
    RetrievalConfig,
    RetrievalQuery,
    SearchResult,
)


# ---------------------------------------------------------------------------
# HashEmbedder
# ---------------------------------------------------------------------------

class TestHashEmbedder:
    def setup_method(self):
        self.embedder = HashEmbedder()

    def test_embed_texts_returns_list_of_embeddings(self):
        result = self.embedder.embed_texts(["hello", "world"])
        assert isinstance(result, list)
        assert len(result) == 2

    def test_embed_texts_correct_dimension(self):
        result = self.embedder.embed_texts(["test text"])
        assert len(result[0]) == HashEmbedder.EMBEDDING_DIM

    def test_embed_texts_values_in_range(self):
        result = self.embedder.embed_texts(["some text"])
        for val in result[0]:
            assert -1.0 <= val <= 1.0

    def test_embed_texts_returns_floats(self):
        result = self.embedder.embed_texts(["some text"])
        assert all(isinstance(v, float) for v in result[0])

    def test_embed_texts_deterministic(self):
        text = "deterministic test"
        result1 = self.embedder.embed_texts([text])
        result2 = self.embedder.embed_texts([text])
        assert result1 == result2

    def test_embed_texts_different_texts_differ(self):
        result = self.embedder.embed_texts(["text one", "text two"])
        assert result[0] != result[1]

    def test_embed_texts_empty_list(self):
        result = self.embedder.embed_texts([])
        assert result == []

    def test_embed_query_returns_single_embedding(self):
        result = self.embedder.embed_query("my query")
        assert isinstance(result, list)
        assert len(result) == HashEmbedder.EMBEDDING_DIM

    def test_embed_query_consistent_with_embed_texts(self):
        query = "consistency check"
        via_query = self.embedder.embed_query(query)
        via_texts = self.embedder.embed_texts([query])[0]
        assert via_query == via_texts

    def test_hash_embed_texts_multiple_items(self):
        texts = ["a", "b", "c"]
        result = self.embedder.hash_embed_texts(texts)
        assert len(result) == 3
        for emb in result:
            assert len(emb) == HashEmbedder.EMBEDDING_DIM

    def test_embed_texts_stable_across_instances(self):
        """Embeddings must be identical from two independent HashEmbedder instances."""
        text = "cross-instance stability"
        assert HashEmbedder().embed_texts([text]) == HashEmbedder().embed_texts([text])

    def test_embed_texts_uses_sha256_not_python_hash(self):
        """Word hashing must use SHA-256 so embeddings are PYTHONHASHSEED-stable.

        Verify that the word 'probe' lands on the bucket determined by SHA-256.
        """
        import hashlib

        word = "probe"
        h = int(hashlib.sha256(word.encode()).hexdigest(), 16)
        expected_idx = h % HashEmbedder.EMBEDDING_DIM
        result = self.embedder.embed_texts([word])
        # The expected bucket should be non-zero (the only word)
        assert result[0][expected_idx] != 0.0

    def test_all_dimensions_carry_signal(self):
        """Longer texts should populate dimensions across the full vector width."""
        text = ("agentic AI certification exam study guide "
                "vector database retrieval augmented generation "
                "multi agent orchestration guardrails safety")
        result = self.embedder.embed_texts([text])
        nonzero = sum(1 for v in result[0] if v != 0.0)
        assert nonzero > 5, "Embedding has too few active dimensions"


# ---------------------------------------------------------------------------
# Embedder
# ---------------------------------------------------------------------------

class TestEmbedder:
    def _make_mock_response(self, embeddings: list[list[float]]):
        response = MagicMock()
        response.data = [MagicMock(embedding=emb) for emb in embeddings]
        return response

    def test_init_raises_without_openai(self):
        with patch.dict("sys.modules", {"openai": None}):
            with pytest.raises(ImportError, match="openai package is required"):
                Embedder(api_key="test-key")

    def test_init_uses_env_api_key(self):
        with patch.dict(os.environ, {"NVIDIA_API_KEY": "env-key"}):
            with patch("openai.OpenAI") as mock_openai:
                Embedder()
                mock_openai.assert_called_once()
                call_kwargs = mock_openai.call_args.kwargs
                assert call_kwargs["api_key"] == "env-key"

    def test_init_prefers_explicit_api_key_over_env(self):
        with patch.dict(os.environ, {"NVIDIA_API_KEY": "env-key"}):
            with patch("openai.OpenAI") as mock_openai:
                Embedder(api_key="explicit-key")
                call_kwargs = mock_openai.call_args.kwargs
                assert call_kwargs["api_key"] == "explicit-key"

    def test_init_custom_base_url(self):
        custom_url = "https://custom.api.example.com/v1"
        with patch("openai.OpenAI") as mock_openai:
            Embedder(api_key="key", base_url=custom_url)
            call_kwargs = mock_openai.call_args.kwargs
            assert call_kwargs["base_url"] == custom_url

    def test_init_default_base_url(self):
        with patch("openai.OpenAI") as mock_openai:
            Embedder(api_key="key")
            call_kwargs = mock_openai.call_args.kwargs
            assert call_kwargs["base_url"] == Embedder.BASE_URL

    def test_embed_texts_calls_api_correctly(self):
        with patch("openai.OpenAI") as mock_openai:
            mock_client = MagicMock()
            mock_openai.return_value = mock_client
            mock_client.embeddings.create.return_value = self._make_mock_response(
                [[0.1, 0.2], [0.3, 0.4]]
            )
            embedder = Embedder(api_key="key")
            result = embedder.embed_texts(["doc1", "doc2"])

            mock_client.embeddings.create.assert_called_once_with(
                input=["doc1", "doc2"],
                model=embedder._model,
                encoding_format="float",
            )
            assert result == [[0.1, 0.2], [0.3, 0.4]]

    def test_embed_query_calls_api_with_query_input_type(self):
        with patch("openai.OpenAI") as mock_openai:
            mock_client = MagicMock()
            mock_openai.return_value = mock_client
            mock_client.embeddings.create.return_value = self._make_mock_response([[0.5, 0.6]])
            embedder = Embedder(api_key="key")
            result = embedder.embed_query("search query")

            mock_client.embeddings.create.assert_called_once_with(
                input=["search query"],
                model=embedder._model,
                encoding_format="float",
                extra_body={"input_type": "query"},
            )
            assert result == [0.5, 0.6]

    def test_hash_embed_texts_delegates_to_hash_embedder(self):
        with patch("openai.OpenAI"):
            embedder = Embedder(api_key="key")
            result = embedder.hash_embed_texts(["offline text"])
            assert len(result) == 1
            assert len(result[0]) == HashEmbedder.EMBEDDING_DIM


# ---------------------------------------------------------------------------
# get_embedder factory
# ---------------------------------------------------------------------------

class TestGetEmbedder:
    def test_returns_hash_embedder_without_key(self):
        with patch.dict(os.environ, {}, clear=True):  # clear=True removes NVIDIA_API_KEY
            embedder = get_embedder()
        assert isinstance(embedder, HashEmbedder)

    def test_returns_nvidia_embedder_with_explicit_key(self):
        with patch("openai.OpenAI"):
            embedder = get_embedder(api_key="test-key")
        assert isinstance(embedder, Embedder)

    def test_returns_nvidia_embedder_with_env_key(self):
        with patch.dict(os.environ, {"NVIDIA_API_KEY": "env-key"}):
            with patch("openai.OpenAI"):
                embedder = get_embedder()
        assert isinstance(embedder, Embedder)

    def test_custom_model_passed_to_nvidia_embedder(self):
        with patch("openai.OpenAI"):
            embedder = get_embedder(model="nvidia/custom-model", api_key="key")
        assert isinstance(embedder, Embedder)
        assert embedder._model == "nvidia/custom-model"

    def test_default_model_used_when_none(self):
        with patch("openai.OpenAI"):
            embedder = get_embedder(api_key="key")
        assert embedder._model == "nvidia/nv-embed-v1"


# ---------------------------------------------------------------------------
# EmbedderProtocol conformance
# ---------------------------------------------------------------------------


class TestEmbedderProtocol:
    def test_hash_embedder_satisfies_protocol(self):
        assert isinstance(HashEmbedder(), EmbedderProtocol)

    def test_nvidia_embedder_satisfies_protocol(self):
        with patch("openai.OpenAI"):
            embedder = Embedder(api_key="key")
        assert isinstance(embedder, EmbedderProtocol)


# ---------------------------------------------------------------------------
# SearchResult model
# ---------------------------------------------------------------------------

class TestSearchResult:
    def test_valid_construction(self):
        result = SearchResult(
            chunk_id="c1",
            document_id="d1",
            text="some text",
            score=0.95,
        )
        assert result.chunk_id == "c1"
        assert result.document_id == "d1"
        assert result.text == "some text"
        assert result.score == 0.95
        assert result.source is None
        assert result.metadata is None

    def test_with_optional_fields(self):
        result = SearchResult(
            chunk_id="c2",
            document_id="d2",
            text="text",
            score=0.7,
            source="http://example.com",
            metadata={"page": 3},
        )
        assert result.source == "http://example.com"
        assert result.metadata == {"page": 3}

    def test_frozen_model_raises_on_assignment(self):
        result = SearchResult(chunk_id="c1", document_id="d1", text="t", score=0.5)
        with pytest.raises(Exception):
            result.score = 0.9  # type: ignore[misc]

    def test_missing_required_fields_raises(self):
        with pytest.raises(ValidationError):
            SearchResult(chunk_id="c1", document_id="d1")  # type: ignore[call-arg]


# ---------------------------------------------------------------------------
# RetrievalQuery model
# ---------------------------------------------------------------------------

class TestRetrievalQuery:
    def test_defaults(self):
        q = RetrievalQuery(query="what is AI?")
        assert q.query == "what is AI?"
        assert q.top_k == 5
        assert q.filters is None

    def test_custom_top_k(self):
        q = RetrievalQuery(query="test", top_k=10)
        assert q.top_k == 10

    def test_top_k_lower_bound(self):
        with pytest.raises(ValidationError):
            RetrievalQuery(query="test", top_k=0)

    def test_top_k_upper_bound(self):
        with pytest.raises(ValidationError):
            RetrievalQuery(query="test", top_k=101)

    def test_top_k_boundary_values(self):
        assert RetrievalQuery(query="q", top_k=1).top_k == 1
        assert RetrievalQuery(query="q", top_k=100).top_k == 100

    def test_with_filters(self):
        q = RetrievalQuery(query="test", filters={"category": "science"})
        assert q.filters == {"category": "science"}


# ---------------------------------------------------------------------------
# RetrievalConfig model
# ---------------------------------------------------------------------------

class TestRetrievalConfig:
    def test_defaults(self):
        cfg = RetrievalConfig()
        assert cfg.embedding_model == "nvidia/nv-embed-v1"
        assert cfg.index_name == "default"
        assert cfg.top_k == 5
        assert cfg.similarity_threshold == 0.0
        assert cfg.use_hybrid_search is False

    def test_custom_values(self):
        cfg = RetrievalConfig(
            embedding_model="custom/model",
            index_name="my-index",
            top_k=20,
            similarity_threshold=0.75,
            use_hybrid_search=True,
        )
        assert cfg.embedding_model == "custom/model"
        assert cfg.index_name == "my-index"
        assert cfg.top_k == 20
        assert cfg.similarity_threshold == 0.75
        assert cfg.use_hybrid_search is True

    def test_top_k_lower_bound(self):
        with pytest.raises(ValidationError):
            RetrievalConfig(top_k=0)

    def test_top_k_upper_bound(self):
        with pytest.raises(ValidationError):
            RetrievalConfig(top_k=101)

    def test_similarity_threshold_lower_bound(self):
        with pytest.raises(ValidationError):
            RetrievalConfig(similarity_threshold=-0.1)

    def test_similarity_threshold_upper_bound(self):
        with pytest.raises(ValidationError):
            RetrievalConfig(similarity_threshold=1.1)

    def test_similarity_threshold_boundary_values(self):
        assert RetrievalConfig(similarity_threshold=0.0).similarity_threshold == 0.0
        assert RetrievalConfig(similarity_threshold=1.0).similarity_threshold == 1.0
