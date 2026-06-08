from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class SearchResult(BaseModel):
    model_config = ConfigDict(frozen=True)
    chunk_id: str
    document_id: str
    text: str
    score: float
    source: str | None = None
    metadata: dict[str, Any] | None = None


class RetrievalQuery(BaseModel):
    query: str
    top_k: int = Field(default=5, ge=1, le=100)
    filters: dict[str, Any] | None = None


class RetrievalConfig(BaseModel):
    embedding_model: str = "nvidia/nv-embed-v1"
    index_name: str = "default"
    top_k: int = Field(default=5, ge=1, le=100)
    similarity_threshold: float = Field(default=0.0, ge=0.0, le=1.0)
    use_hybrid_search: bool = False


__all__ = ["SearchResult", "RetrievalQuery", "RetrievalConfig"]
