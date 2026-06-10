from pathlib import Path
from nvidia_agentic_research_engineer.core.documents import Document


def load_text_file(file_path: Path) -> Document:
    file_path = Path(file_path)
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
    return Document(
        title=file_path.stem,
        content=content,
        source=str(file_path),
        document_type="text",
    )

def load_markdown_file(file_path: Path) -> Document:
    file_path = Path(file_path)
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
    return Document(
        title=file_path.stem,
        content=content,
        source=str(file_path),
        document_type="markdown",
    )

def load_file(file_path: Path) -> Document:
    if file_path.suffix.lower() == ".txt":
        return load_text_file(file_path)
    elif file_path.suffix.lower() == ".md":
        return load_markdown_file(file_path)
    else:
        raise ValueError(f"Unsupported file type: {file_path.suffix}")