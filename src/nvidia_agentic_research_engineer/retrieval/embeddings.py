from __future__ import annotations

import hashlib
import os
from typing import Protocol, runtime_checkable

@runtime_checkable
class EmbedderProtocol(Protocol):
    # Protocol that all embedder implementations must satisfy
    def embed_texts(self, texts: list[str]) -> list[list[float]]: ...
    def embed_query(self, query: str) -> list[float]: ...


class HashEmbedder:
    """Deterministic embedder using feature hashing — useful for testing and offline use.

    Uses a bag-of-words approach with the hashing trick: each lowercased word
    is hashed to a dimension index, producing sparse-like vectors where texts
    sharing vocabulary have high cosine similarity.  Fully deterministic across
    processes and Python versions (relies on SHA-256, not built-in ``hash()``).
    """
    EMBEDDING_DIM = 384

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        # Return deterministic embeddings for a list of texts
        return self.hash_embed_texts(texts)

    def embed_query(self, query: str) -> list[float]:
        # Return a deterministic embedding for a single query string
        return self.hash_embed_texts([query])[0]

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """Lowercase and split on non-alphanumeric characters."""
        import re
        return re.findall(r"[a-z0-9]+", text.lower())

    def hash_embed_texts(self, texts: list[str]) -> list[list[float]]:
        # Generate deterministic fixed-dimension embeddings via feature hashing.
        # Each word is hashed to a bucket; texts sharing words yield similar vectors.

        from math import sqrt as _sqrt

        embeddings = []
        for text in texts:
            vec = [0.0] * self.EMBEDDING_DIM
            for word in self._tokenize(text):
                h = int(hashlib.sha256(word.encode()).hexdigest(), 16)
                idx = h % self.EMBEDDING_DIM
                sign = 1.0 if (h // self.EMBEDDING_DIM) % 2 == 0 else -1.0
                vec[idx] += sign
            # L2-normalise so cosine similarity is just the dot product.
            norm = _sqrt(sum(v * v for v in vec))
            if norm > 0:
                vec = [v / norm for v in vec]
            embeddings.append(vec)
        return embeddings


# Module-level singleton so Embedder.hash_embed_texts avoids repeated allocation.
_HASH_EMBEDDER = HashEmbedder()


class Embedder:
    # Embedder backed by the NVIDIA NIM embedding endpoint (OpenAI-compatible API)

    BASE_URL = "https://integrate.api.nvidia.com/v1"

    def __init__(
        self,
        model: str = "nvidia/nv-embed-v1",
        api_key: str | None = None,
        base_url: str | None = None,
    ) -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError(
                "openai package is required for Embedder. "
                "Install it with: pip install openai"
            ) from exc

        self._model = model
        resolved_key = api_key or os.environ.get("NVIDIA_API_KEY", "")
        self._client = OpenAI(api_key=resolved_key, base_url=base_url or self.BASE_URL)

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of documents via the NVIDIA NIM embedding API."""
        response = self._client.embeddings.create(
            input=texts,
            model=self._model,
            encoding_format="float",
        )
        return [item.embedding for item in response.data]

    def embed_query(self, query: str) -> list[float]:
        """Embed a single query, applying the query instruction prefix for NV-Embed."""
        response = self._client.embeddings.create(
            input=[query],
            model=self._model,
            encoding_format="float",
            extra_body={"input_type": "query"},
        )
        return response.data[0].embedding

    def hash_embed_texts(self, texts: list[str]) -> list[list[float]]:
        # Deterministic hash-based fallback (offline / testing)
        return _HASH_EMBEDDER.hash_embed_texts(texts)


def get_embedder(model: str | None = None, api_key: str | None = None) -> EmbedderProtocol:
    # Return an Embedder when an API key is available, else a HashEmbedder
    resolved_key = api_key or os.environ.get("NVIDIA_API_KEY", "")
    if resolved_key:
        return Embedder(model=model or "nvidia/nv-embed-v1", api_key=resolved_key)
    return HashEmbedder()
