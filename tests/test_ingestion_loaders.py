from pathlib import Path
from nvidia_agentic_research_engineer.ingestion.loaders import load_text_file, load_markdown_file

TEST_DIR = Path(__file__).parent / "test_files"

def test_load_text_file():
    test_file = TEST_DIR / "test.txt"
    doc = load_text_file(test_file)
    assert doc.title == "test"
    assert doc.content == "This is a sample text file for testing."
    assert doc.source == str(test_file)
    assert doc.document_type == "text"

def test_load_markdown_file():
    test_file = TEST_DIR / "test.md"
    doc = load_markdown_file(test_file)
    assert doc.title == "test"
    assert doc.content == "# Sample Markdown\nThis is a sample markdown file for testing."
    assert doc.source == str(test_file)
    assert doc.document_type == "markdown"