from __future__ import annotations

from math import sqrt

from pydantic import BaseModel, ConfigDict

from nvidia_agentic_research_engineer.core.documents import DocumentChunk
from nvidia_agentic_research_engineer.retrieval.embeddings import EmbedderProtocol


class RetrievalResult(BaseModel):
    model_config = ConfigDict(frozen=True)
    chunk: DocumentChunk
    score: float
    rank: int


def cosine_similarity(a: list[float], b: list[float]) -> float:
    if len(a) != len(b):
        raise ValueError(
            f"Vector length mismatch: {len(a)} vs {len(b)}"
        )
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sqrt(sum(x * x for x in a))
    norm_b = sqrt(sum(y * y for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


class InMemoryVectorStore:
    def __init__(self, embedder: EmbedderProtocol) -> None:
        self.embedder = embedder
        self._items: list[tuple[DocumentChunk, list[float]]] = []

    def add_chunks(self, chunks: list[DocumentChunk]) -> None:
        embeddings = self.embedder.embed_texts([chunk.text for chunk in chunks])
        for chunk, embedding in zip(chunks, embeddings):
            self._items.append((chunk, embedding))

    def search(self, query: str, top_k: int = 5) -> list[RetrievalResult]:
        if top_k <= 0:
            raise ValueError(f"top_k must be greater than 0, got {top_k}")
        if not self._items:
            return []
        query_embedding = self.embedder.embed_query(query)
        scored = [
            (chunk, cosine_similarity(query_embedding, embedding))
            for chunk, embedding in self._items
        ]
        scored.sort(key=lambda x: x[1], reverse=True)
        
        return [
            RetrievalResult(chunk=chunk, score=score, rank=rank)
            for rank, (chunk, score) in enumerate(scored[:top_k], start=1)
        ]
