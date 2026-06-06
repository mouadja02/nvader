import hashlib
import json
from datetime import datetime, timezone
from enum import Enum
from typing import Any
from pydantic import BaseModel, Field, model_validator

class DocumentType(str, Enum):
    TEXT = "text"
    MARKDOWN = "markdown"
    HTML = "html"
    PDF = "pdf"
    URL = "url"
    REPO = "repo"
    PAPER = "paper"
    IMAGE = "image"

class Document(BaseModel):
    id: str = Field(default="")
    title: str
    content: Any
    source: str
    document_type: DocumentType
    metadata: dict = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @model_validator(mode="before")
    @classmethod
    def derive_id(cls, values: dict) -> dict:
        if isinstance(values, dict) and not values.get("id"):
            hash_input = json.dumps(
                {
                    "title": str(values.get("title")),
                    "content": str(values.get("content")),
                    "source": str(values.get("source")),
                    "document_type": str(values.get("document_type")),
                    "metadata": values.get("metadata", {}),
                },
                sort_keys=True,
            )
            values["id"] = hashlib.sha256(hash_input.encode()).hexdigest()
        return values

    def short_preview(self, max_chars: int = 240) -> str:
        cleaned = " ".join(str(self.content).split())
        return cleaned[:max_chars] + ("..." if len(cleaned) > max_chars else "")
    
class DocumentChunk(BaseModel):
    id: str
    document_id: str
    text: str
    chunk_index: int
    source: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    start_char: int | None = None
    end_char: int | None = None