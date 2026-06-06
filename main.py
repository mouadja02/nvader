from nvidia_agentic_research_engineer.ingestion.loaders import load_text_file, load_markdown_file
from nvidia_agentic_research_engineer.ingestion.chunking import chunk_text, chunk_document, chunk_documents
from nvidia_agentic_research_engineer.core.documents import Document, DocumentChunk



doc1 = load_text_file("./examples/sample.txt")
doc2 = load_markdown_file("./examples/sample.md")

print("Original Documents:")
print(doc1)
print(doc2)

chunks_doc1 = chunk_document(doc1, chunk_size=100, chunk_overlap=10)
chunks_doc2 = chunk_document(doc2, chunk_size=100, chunk_overlap=10)

print("\nChunks for Document 1:")
for chunk in chunks_doc1:
    print(chunk)
print("-"*40)
print("\nChunks for Document 2:")
for chunk in chunks_doc2:
    print(chunk)