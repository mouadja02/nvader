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
    """Deterministic embedder using SHA-256 hashing — useful for testing and offline use.

    Unlike Python's built-in ``hash()``, SHA-256 is stable across processes and
    Python versions regardless of ``PYTHONHASHSEED``, ensuring truly reproducible
    embeddings.  Each dimension also carries independent signal because the vector
    is built from successive SHA-256 blocks rather than bit-shifted bytes of a
    single integer.
    """
    EMBEDDING_DIM = 384

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        # Return deterministic embeddings for a list of texts
        return self.hash_embed_texts(texts)

    def embed_query(self, query: str) -> list[float]:
        # Return a deterministic embedding for a single query string
        return self.hash_embed_texts([query])[0]

    def hash_embed_texts(self, texts: list[str]) -> list[list[float]]:
        # Generate deterministic fixed-dimension embeddings via SHA-256.
        # Values are normalised to the range [-1.0, 1.0]

        embeddings = []
        for text in texts:
            # Seed from SHA-256 of the input text (32 bytes, process-stable).
            seed = hashlib.sha256(text.encode()).digest()
            # Expand seed to EMBEDDING_DIM bytes using successive SHA-256 blocks.
            raw: list[int] = []
            counter = 0
            while len(raw) < self.EMBEDDING_DIM:
                block = hashlib.sha256(seed + counter.to_bytes(4, "little")).digest()
                raw.extend(block)
                counter += 1
            # Normalise bytes [0, 255] → [-1.0, ~0.992].
            normalized = [float(b) / 128.0 - 1.0 for b in raw[: self.EMBEDDING_DIM]]
            embeddings.append(normalized)
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
