# localrag

A portable Retrieval-Augmented Generation (RAG) system that ingests **PDF, TXT, and Markdown** files, stores embeddings in **PostgreSQL with pgvector**, and uses **FastFlowLM** as the local LLM backend.

## Architecture

```
PDF / TXT / MD files → discover_files() → file listing → load per-type → Text Splitter
                                                                              ↓
                                                              sentence-transformers → pgvector (PostgreSQL)
                                                                                            ↓
User question ───────────────────────────────────────────────────────────────────→ Similarity Search
                                                                                            ↓
                                                                       FastFlowLM (OllamaLLM) ← Prompt + Context
                                                                                            ↓
                                                                                        Answer + Sources
```

## Supported Formats

| Format | Loader | Notes |
|---|---|---|
| `.pdf` | `PyMuPDFLoader` | High-quality layout-preserving text extraction |
| `.txt` | `TextLoader` (UTF-8) | Plain text |
| `.md` | `TextLoader` (UTF-8) | Markdown, kept as-is |

## Prerequisites

- [Nix](https://nixos.org/download) (with flakes enabled) — or Python 3.10+ manually
- [Docker](https://docs.docker.com/engine/install/) + Docker Compose v2
- [FastFlowLM](https://github.com/FastFlowLM/FastFlowLM/releases)

## Quick Start

### 1. Start PostgreSQL with pgvector

```bash
docker compose up -d
```

This starts PostgreSQL 16 with the pgvector extension on port **5433**.

### 2. Start FastFlowLM

```bash
flm serve llama3.2:3b
```

FastFlowLM exposes an Ollama-compatible REST API at `http://127.0.0.1:52625`.

### 3. Enter the development environment

```bash
nix develop
```

This creates a Python virtual environment, installs all dependencies, and sets up environment variables.

### 4. Ingest files

Place your PDF, TXT, and MD files in `data/pdfs/` (or any directory). The ingestion discovers files recursively, shows the full list, and asks for confirmation:

```bash
ragctl ingest --dir data/pdfs/
```

Example output:

```
Scanning /home/user/localrag/data/pdfs for supported files...

Found 4 file(s):
[ PDF] docs/report.pdf (245,000 bytes)
[ TXT] notes/summary.txt (1,200 bytes)
[ MD]  README.md (3,400 bytes)
[ MD]  docs/changelog.md (8,900 bytes)

Proceed with ingestion? [y/N]: y

Loading files...
  [1/4] report.pdf (3 page(s))
  [2/4] summary.txt (1 page(s))
  [3/4] README.md (1 page(s))
  [4/4] changelog.md (1 page(s))
   Loaded 6 document(s) total
Splitting documents...
   Created 24 chunk(s)
Loading embedding model...
Storing in pgvector...
Ingestion complete!
```

Use `-y` to skip the confirmation prompt:

```bash
ragctl ingest --dir docs/ -y
```

### 5. Ask questions

```bash
# Basic query (similarity search, top 5 chunks)
ragctl query "What does this document say about X?"

# Query with MMR (diverse coverage across documents)
ragctl query "What does this document say about X?" --search-type mmr

# Query with more context (top 10 chunks)
ragctl query "What does this document say about X?" --top-k 10
```

### 6. Manage ingested documents

```bash
# List all ingested sources
ragctl list

# Delete a specific file by name
ragctl delete report.pdf

# Delete all ingested documents (with confirmation)
ragctl delete --all

# Skip confirmation prompt
ragctl delete --all --yes
```

## Full CLI Reference

```
Usage: ragctl [OPTIONS] COMMAND [ARGS]...

Commands:
  ingest     Discover PDF, TXT, MD files recursively, then embed and store.
  query      Ask a question against the ingested documents.
  list       List all ingested source files.
  delete     Delete ingested documents by source filename.
```

### Query options

| Option | Default | Description |
|---|---|---|
| `--top-k` | `5` | Number of chunks to retrieve (more = more context for the LLM) |
| `--search-type` | `similarity` | `similarity` — exact semantic match; `mmr` — diverse coverage across documents |
| `--collection` | `documents` | Which collection to query |

### Ingest options

| Option | Default | Description |
|---|---|---|
| `--dir` | `data/pdfs/` | Directory to scan recursively |
| `-y` / `--yes` | `false` | Skip confirmation prompt |

## Configuration

All configuration is managed via environment variables or a `.env` file (copy from `.env.example`):

| Variable | Default | Description |
|---|---|---|
| `POSTGRES_HOST` | `localhost` | PostgreSQL host |
| `POSTGRES_PORT` | `5433` | PostgreSQL port |
| `POSTGRES_USER` | `rag_user` | Database user |
| `POSTGRES_PASSWORD` | `rag_password` | Database password |
| `POSTGRES_DB` | `rag_db` | Database name |
| `EMBEDDING_MODEL` | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` | Embedding model (multilingual) |
| `FASTFLOWLM_BASE_URL` | `http://127.0.0.1:52625` | FastFlowLM API endpoint |
| `LLM_MODEL` | `llama3.2:3b` | LLM model name (e.g. qwen3.5:9b) |
| `CHUNK_SIZE` | `1000` | Text chunk size (characters) |
| `CHUNK_OVERLAP` | `200` | Chunk overlap (characters) |

## Project Structure

```
localrag/
├── docker-compose.yml       # PostgreSQL + pgvector
├── flake.nix                # Nix development shell
├── requirements.txt         # Python dependencies
├── pyproject.toml           # Project metadata
├── .env.example             # Configuration template
├── README.md
├── src/
│   ├── config.py            # Pydantic settings
│   ├── loader.py            # File discovery + per-type loading + text splitting
│   ├── embeddings.py        # HuggingFace & Ollama embedding models
│   ├── vector_store.py      # pgvector read/write/delete operations
│   ├── rag.py               # RetrievalQA chain with FastFlowLM
│   └── cli.py               # CLI entry point (ingest / query / list / delete)
├── data/
│   └── pdfs/                # Place your PDFs, TXTs, MDs here
└── tests/
    └── test_loader.py
```

## How It Works

### Ingestion Pipeline

1. **Discover** — `discover_files()` scans the directory recursively for `.pdf`, `.txt`, and `.md` files, then displays the list with types and sizes before proceeding.
2. **Load** — Each file is loaded with the appropriate loader (`PyMuPDFLoader` for PDF, `TextLoader` for TXT/MD).
3. **Split** — `RecursiveCharacterTextSplitter` breaks documents into overlapping chunks (default 1000 chars with 200 overlap).
4. **Embed** — `sentence-transformers/all-MiniLM-L6-v2` converts chunks into 384-dimension vectors.
5. **Store** — Vectors and text are stored in PostgreSQL via the pgvector extension.

### Query Pipeline

1. **Embed** — The user question is embedded with the same model.
2. **Retrieve** — `PGVector` performs a similarity search and returns the top-k most relevant chunks.
3. **Generate** — A prompt with the retrieved context is sent to FastFlowLM (via `OllamaLLM` pointing at its API).
4. **Respond** — The LLM answer is printed along with source document references.

## Improving Retrieval Quality

If query results are poor, try these adjustments from most to least impactful:

| Tweak | What to change | Effect |
|---|---|---|
| **Increase `top-k`** | `--top-k 10` in query | More context for the LLM |
| **Use MMR** | `--search-type mmr` | Diverse coverage — avoids 3 near-identical chunks |
| **Increase `CHUNK_SIZE`** | Set `CHUNK_SIZE=2500` in `.env` | Bigger chunks = more coherent context per item |
| **Better embedding model** | Set `EMBEDDING_MODEL=BAAI/bge-small-en-v1.5` | Higher retrieval accuracy (requires re-ingest) |
| **Better LLM** | Set `LLM_MODEL=llama3.2:3b` or a larger FastFlowLM model | Stronger reasoning over the retrieved context |

After changing chunk size or embedding model, **re-ingest** your documents:

```bash
ragctl delete --all --yes
ragctl ingest --dir data/pdfs/
```

## Running Without Nix

If you don't use Nix:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

## License

MIT
