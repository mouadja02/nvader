import pytest
from nvidia_agentic_research_engineer.core.documents import Document, DocumentChunk, DocumentType
from nvidia_agentic_research_engineer.ingestion.chunking import chunk_text, chunk_document, chunk_documents


def _make_document(content: str, source: str = "test_source", metadata: dict | None = None, doc_id: str | None = None) -> Document:
    kwargs = dict(
        title="Test Doc",
        content=content,
        source=source,
        document_type=DocumentType.TEXT,
        metadata=metadata or {},
    )
    if doc_id is not None:
        kwargs["id"] = doc_id
    return Document(**kwargs)


# ---------------------------------------------------------------------------
# chunk_text
# ---------------------------------------------------------------------------

def test_chunk_text_returns_empty_for_empty_text():
    result = chunk_text("")
    assert result == []


def test_chunk_text_splits_long_text_with_overlap():
    # Build a text that is exactly 3 * chunk_size characters so we can reason
    # about boundaries precisely.
    chunk_size = 10
    overlap = 3
    # "0123456789" repeated three times → 30 chars
    text = "0123456789" * 3

    chunks = chunk_text(text, chunk_size=chunk_size, chunk_overlap=overlap)

    # With step = chunk_size - overlap = 7, starts are 0, 7, 14, 21, 28
    assert len(chunks) == 5

    # Every chunk must be at most chunk_size characters; only chunks whose
    # window fits entirely in the text can be exactly chunk_size.
    for chunk_text_val, start, end in chunks:
        assert len(chunk_text_val) <= chunk_size
    # The first chunk always fits fully, so it must be exactly chunk_size.
    assert len(chunks[0][0]) == chunk_size

    # Verify overlap: the trailing chars of chunk[i] that overlap with chunk[i+1]
    # must equal the leading chars of chunk[i+1].  When the next chunk is shorter
    # than 'overlap' (can happen at the end of the text), the actual shared region
    # is min(overlap, len(next_chunk)) characters.
    for i in range(len(chunks) - 1):
        actual_overlap = min(overlap, len(chunks[i + 1][0]))
        trailing = chunks[i][0][-actual_overlap:]
        leading = chunks[i + 1][0][:actual_overlap]
        assert trailing == leading, (
            f"Overlap mismatch at boundary {i}/{i+1}: {trailing!r} != {leading!r}"
        )

    # Verify start/end indices are consistent with the returned text
    for chunk_text_val, start, end in chunks:
        assert text[start:end] == chunk_text_val


def test_chunk_text_rejects_invalid_parameters():
    text = "some text"

    with pytest.raises(ValueError):
        chunk_text(text, chunk_size=0)

    with pytest.raises(ValueError):
        chunk_text(text, chunk_size=-5)

    with pytest.raises(ValueError):
        chunk_text(text, chunk_size=10, chunk_overlap=-1)

    with pytest.raises(ValueError):
        # overlap >= chunk_size should be rejected
        chunk_text(text, chunk_size=10, chunk_overlap=10)

    with pytest.raises(ValueError):
        chunk_text(text, chunk_size=10, chunk_overlap=15)


# ---------------------------------------------------------------------------
# chunk_document
# ---------------------------------------------------------------------------

def test_chunk_document_preserves_source_and_metadata():
    meta = {"author": "alice", "version": 2}
    doc = _make_document("A" * 500, source="my_source", metadata=meta)

    chunks = chunk_document(doc, chunk_size=200, chunk_overlap=20)

    assert len(chunks) > 1
    for chunk in chunks:
        assert chunk.source == "my_source"
        assert chunk.metadata == meta
        # metadata must be a copy, not the same object
        assert chunk.metadata is not doc.metadata


def test_chunk_document_generates_stable_ids():
    """Same document + same text + same settings must produce the same chunk IDs."""
    doc = _make_document("B" * 300, doc_id="fixed-doc-id")

    chunks_first = chunk_document(doc, chunk_size=100, chunk_overlap=10)
    chunks_second = chunk_document(doc, chunk_size=100, chunk_overlap=10)

    ids_first = [c.id for c in chunks_first]
    ids_second = [c.id for c in chunks_second]

    assert ids_first == ids_second
    assert all(cid.startswith("fixed-doc-id") for cid in ids_first)

    # IDs must be unique within a single chunking run
    assert len(ids_first) == len(set(ids_first))


# ---------------------------------------------------------------------------
# chunk_documents
# ---------------------------------------------------------------------------

def test_chunk_documents_chunks_multiple_documents():
    doc_a = _make_document("A" * 250, source="source_a", doc_id="doc-a")
    doc_b = _make_document("B" * 250, source="source_b", doc_id="doc-b")

    all_chunks = chunk_documents([doc_a, doc_b], chunk_size=100, chunk_overlap=10)

    # Both documents must be represented
    doc_ids = {c.document_id for c in all_chunks}
    assert "doc-a" in doc_ids
    assert "doc-b" in doc_ids

    # Each chunk must reference a valid source
    sources = {c.source for c in all_chunks}
    assert sources == {"source_a", "source_b"}

    # Empty sequence must not crash and must return an empty list
    assert chunk_documents([]) == []
