from nvidia_agentic_research_engineer.core.documents import Document, DocumentType


def test_document_creation() -> None:
    doc = Document(
        title="NVIDIA Agentic AI Study Guide",
        content="Agent architecture and design includes ReAct, memory, and orchestration.",
        source="official-study-guide",
        document_type=DocumentType.PDF,
    )

    assert doc.id
    assert doc.title == "NVIDIA Agentic AI Study Guide"
    assert doc.document_type == DocumentType.PDF
    assert "ReAct" in doc.content


def test_document_preview_truncates() -> None:
    doc = Document(
        title="Long Doc",
        content="word " * 100,
        source="test",
        document_type=DocumentType.TEXT,
    )

    preview = doc.short_preview(max_chars=20)

    assert len(preview) <= 23
    assert preview.endswith("...")