from typing import Sequence
from nvidia_agentic_research_engineer.core.documents import Document, DocumentChunk

def chunk_text(text: str, * , chunk_size: int = 1000, chunk_overlap: int = 150) -> list[tuple[str, int, int]]:
    """
    Splits the input text into chunks of specified size with optional overlap.
    
    Args:
        text: The input text to be chunked.
        chunk_size: The maximum number of characters in each chunk.
        chunk_overlap: The number of overlapping characters between consecutive chunks.
        
    Returns:
        A list of tuples containing the chunk text, start character index, and end character index.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap cannot be negative")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be less than chunk_size")
    chunks = []
    start = 0
    text_length = len(text)
    
    if text_length == 0:
        return chunks
    
    while start < text_length:
        end = min(start + chunk_size, text_length)
        text_chunk = text[start:end]
        chunks.append((text_chunk, start, end))
        
        # Move the start index for the next chunk
        start += (chunk_size - chunk_overlap)
    
    return chunks

def chunk_document(document: Document, * , chunk_size: int = 1000, chunk_overlap: int = 150) -> list[DocumentChunk]:
    """
    Chunks the content of a Document into smaller DocumentChunk instances.
    
    Args:
        document: The Document to be chunked.
        chunk_size: The maximum number of characters in each chunk.
        chunk_overlap: The number of overlapping characters between consecutive chunks.
        
    Returns:
        A list of DocumentChunk instances representing the chunks of the original document.
    """
    text = str(document.content)
    raw_chunks = chunk_text(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    
    document_chunks = []
    for index, (text_chunk, start_char, end_char) in enumerate(raw_chunks):
        chunk = DocumentChunk(
            id=f"{document.id}_chunk_{index}",
            document_id=document.id,
            text=text_chunk,
            chunk_index=index,
            source=document.source,
            metadata=document.metadata.copy(),
            start_char=start_char,
            end_char=end_char
        )
        document_chunks.append(chunk)
    
    return document_chunks


def chunk_documents(documents: Sequence[Document], * , chunk_size: int = 1000, chunk_overlap: int = 150) -> list[DocumentChunk]:
    """
    Chunks a sequence of Documents into a list of DocumentChunk instances.
    
    Args:
        documents: A sequence of Document instances to be chunked.
        chunk_size: The maximum number of characters in each chunk.
        chunk_overlap: The number of overlapping characters between consecutive chunks.
        
    Returns:
        A list of DocumentChunk instances representing the chunks of the original documents.

    """
    all_chunks = []
    for document in documents:
        chunks = chunk_document(document, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        all_chunks.extend(chunks)
    return all_chunks


