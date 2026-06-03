from datetime import datetime, timezone
from enum import Enum
from typing import Any
from uuid import uuid4
from pydantic import BaseModel, Field

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
    id: str = Field(default_factory=lambda: str(uuid4()))
    title: str
    content: Any
    source: str
    document_type: DocumentType
    metadata: dict = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    def short_preview(self, max_chars: int = 240) -> str:
        cleaned = " ".join(str(self.content).split())
        return cleaned[:max_chars] + ("..." if len(cleaned) > max_chars else "")