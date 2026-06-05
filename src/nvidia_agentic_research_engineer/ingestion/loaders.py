from pathlib import Path
from nvidia_agentic_research_engineer.core.documents import Document


def load_text_file(file_path: Path) -> Document:
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
    return Document(
        title=file_path.stem,
        content=content,
        source=str(file_path),
        document_type="text",
    )

def load_markdown_file(file_path: Path) -> Document:
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
    return Document(
        title=file_path.stem,
        content=content,
        source=str(file_path),
        document_type="markdown",
    )
