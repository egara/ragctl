import logging
import re
from pathlib import Path
from typing import List, Union

from langchain_community.document_loaders import PyMuPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md"}

# Control characters that PostgreSQL JSONB cannot store (excludes \n \r \t)
_REMOVE_RE = re.compile(r"[\u0000-\u0008\u000b\u000c\u000e-\u001f\u007f]")


def sanitize_text(text: str) -> str:
    """Removes control characters that PostgreSQL JSONB cannot store."""
    return _REMOVE_RE.sub("", text)


def _sanitize_doc(doc: Document) -> Document:
    """Sanitizes the page content and metadata of a Document."""
    doc.page_content = sanitize_text(doc.page_content)
    doc.metadata = {
        k: sanitize_text(v) if isinstance(v, str) else v
        for k, v in doc.metadata.items()
    }
    return doc


def discover_files(directory: Union[str, Path]) -> List[Path]:
    """
    Scans a directory recursively and returns a list of supported files.

    Args:
        directory: The path to the directory to scan.

    Returns:
        A list of Path objects pointing to supported files.
    """
    directory = Path(directory).expanduser().resolve()
    if not directory.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")

    files: List[Path] = []
    for ext in SUPPORTED_EXTENSIONS:
        files.extend(directory.rglob(f"*{ext}"))
    return sorted(files)


def load_file(file_path: Path) -> List[Document]:
    """
    Loads a single file into a list of Documents.

    Args:
        file_path: The path to the file to load.

    Returns:
        A list of Document objects.
    """
    ext = file_path.suffix.lower()
    if ext == ".pdf":
        loader = PyMuPDFLoader(str(file_path))
    else:
        loader = TextLoader(str(file_path), encoding="utf-8")
    docs = loader.load()
    return [_sanitize_doc(d) for d in docs]


def load_files(file_paths: List[Path]) -> List[Document]:
    """
    Loads multiple files and combines them into a single list of Documents.

    Args:
        file_paths: A list of paths to files to load.

    Returns:
        A combined list of Document objects from all files.
    """
    all_docs: List[Document] = []
    total = len(file_paths)
    for i, fp in enumerate(file_paths):
        try:
            docs = load_file(fp)
            all_docs.extend(docs)
            logger.info(f"  [{i+1}/{total}] {fp.name} ({len(docs)} page(s))")
        except Exception as e:
            logger.error(f"  [{i+1}/{total}] {fp.name} — ERROR: {e}")
    return all_docs


def split_documents(
    documents: List[Document],
    chunk_size: int = 500,
    chunk_overlap: int = 100,
) -> List[Document]:
    """
    Splits a list of Documents into smaller chunks.

    Args:
        documents: The list of Documents to split.
        chunk_size: The maximum size of each chunk.
        chunk_overlap: The overlap between adjacent chunks.

    Returns:
        A list of split Document chunks.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", " ", ""],
        length_function=len,
    )
    return splitter.split_documents(documents)
