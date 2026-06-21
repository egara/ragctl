from pathlib import Path
from unittest.mock import patch

from src.loader import discover_files, load_file, load_files, split_documents


def test_split_documents_empty():
    chunks = split_documents([], chunk_size=500, chunk_overlap=100)
    assert chunks == []


def test_split_documents_single():
    from langchain_core.documents import Document

    doc = Document(page_content="Hello world. " * 100)
    chunks = split_documents([doc], chunk_size=100, chunk_overlap=20)
    assert len(chunks) > 1


def test_discover_files(tmp_path):
    (tmp_path / "sub").mkdir()
    (tmp_path / "doc1.pdf").touch()
    (tmp_path / "doc2.txt").touch()
    (tmp_path / "doc3.md").touch()
    (tmp_path / "sub" / "nested.txt").touch()
    (tmp_path / "notes.txt~").touch()

    files = discover_files(tmp_path)
    paths = [str(f.relative_to(tmp_path)) for f in files]

    assert "doc1.pdf" in paths
    assert "doc2.txt" in paths
    assert "doc3.md" in paths
    assert "sub/nested.txt" in paths
    assert "notes.txt~" not in paths


def test_discover_files_nonexistent():
    import pytest
    with pytest.raises(FileNotFoundError):
        discover_files("/nonexistent/path")


def test_load_file_txt(tmp_path):
    fp = tmp_path / "hello.txt"
    fp.write_text("Hello, world!", encoding="utf-8")
    docs = load_file(fp)
    assert len(docs) == 1
    assert "Hello" in docs[0].page_content


def test_load_file_md(tmp_path):
    fp = tmp_path / "readme.md"
    fp.write_text("# Title\n\nContent here.", encoding="utf-8")
    docs = load_file(fp)
    assert len(docs) == 1
    assert "# Title" in docs[0].page_content
