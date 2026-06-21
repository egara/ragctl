from pathlib import Path

import click

from src.config import settings
from src.embeddings import get_embedding_function
from src.loader import discover_files, load_files, split_documents
from src.rag import build_qa_chain, get_llm
from src.vector_store import (
    create_vector_store,
    delete_by_source,
    get_vector_store,
    list_sources,
)


@click.group()
def cli():
    """localrag — RAG with PDF ingestion, pgvector, and FastFlowLM."""


@cli.command()
@click.option(
    "--dir",
    default="data/pdfs",
    show_default=True,
    help="Directory to scan recursively for PDF, TXT, and MD files.",
)
@click.option(
    "--collection",
    default="documents",
    show_default=True,
    help="PGVector collection name.",
)
@click.option(
    "--yes",
    "-y",
    "skip_confirm",
    is_flag=True,
    default=False,
    help="Skip the file listing confirmation.",
)
def ingest(dir: str, collection: str, skip_confirm: bool):
    """Discover PDF, TXT, MD files recursively, then embed and store."""
    source_dir = Path(dir).expanduser().resolve()

    click.echo(f"Scanning {source_dir} for supported files...")
    files = discover_files(source_dir)

    if not files:
        click.echo("No PDF, TXT, or MD files found.")
        return

    click.echo(f"\nFound {len(files)} file(s):")
    for f in files:
        rel = f.relative_to(source_dir)
        size = f.stat().st_size
        click.echo(f"  [{f.suffix[1:].upper():>4}] {rel} ({size:,} bytes)")

    if not skip_confirm:
        click.confirm("\nProceed with ingestion?", abort=True)

    click.echo("\nLoading files...")
    docs = load_files(files)
    click.echo(f"   Loaded {len(docs)} document(s) total")

    click.echo("Splitting documents...")
    chunks = split_documents(docs, settings.chunk_size, settings.chunk_overlap)
    click.echo(f"   Created {len(chunks)} chunk(s)")

    click.echo("Loading embedding model...")
    embeddings = get_embedding_function(settings.embedding_model)

    click.echo("Storing in pgvector...")
    create_vector_store(chunks, embeddings, settings, collection_name=collection)
    click.echo("Ingestion complete!")


@cli.command()
@click.argument("query")
@click.option(
    "--collection",
    default="documents",
    show_default=True,
    help="PGVector collection name.",
)
@click.option(
    "--top-k",
    default=5,
    show_default=True,
    help="Number of source documents to retrieve.",
)
@click.option(
    "--search-type",
    default="similarity",
    show_default=True,
    type=click.Choice(["similarity", "mmr"]),
    help="Retrieval strategy: similarity (exact) or mmr (diverse coverage).",
)
def query(query: str, collection: str, top_k: int, search_type: str):
    """Ask a question against the ingested documents."""
    click.echo("Loading embedding model...")
    embeddings = get_embedding_function(settings.embedding_model)

    click.echo("Connecting to vector store...")
    vector_store = get_vector_store(embeddings, settings, collection_name=collection)

    search_kwargs: dict = {"k": top_k}
    if search_type == "mmr":
        search_kwargs["fetch_k"] = top_k * 3
        search_kwargs["lambda_mult"] = 0.7

    retriever = vector_store.as_retriever(
        search_type=search_type,
        search_kwargs=search_kwargs,
    )

    click.echo("Initializing LLM (FastFlowLM)...")
    llm = get_llm(settings)
    qa_chain = build_qa_chain(llm, retriever)

    click.echo(f"\n{query}\n")
    result = qa_chain.invoke(query)

    click.echo("=" * 80)
    click.echo("Answer:")
    click.echo(result["result"])
    click.echo("=" * 80)

    click.echo("\nSources:")
    for i, doc in enumerate(result["source_documents"]):
        text = doc.page_content[:300].replace("\n", " ")
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", "?")
        click.echo(f"  {i + 1}. [{Path(source).name} p.{page}]")
        click.echo(f"     {text}...")


@cli.command()
@click.option(
    "--collection",
    default="documents",
    show_default=True,
    help="PGVector collection name.",
)
def list(collection: str):
    """List all ingested source files."""
    sources = list_sources(settings, collection_name=collection)
    if not sources:
        click.echo("No ingested documents found.")
        return
    click.echo(f"Ingested sources ({len(sources)}):")
    for src in sorted(sources):
        size = Path(src).stat().st_size if Path(src).exists() else 0
        name = Path(src).name
        status = "exists" if Path(src).exists() else "missing"
        click.echo(f"  - {name} ({status}, {size:,} bytes)")
        click.echo(f"    full path: {src}")


@cli.command()
@click.argument("source", required=False, default=None)
@click.option(
    "--collection",
    default="documents",
    show_default=True,
    help="PGVector collection name.",
)
@click.option(
    "--all",
    "delete_all",
    is_flag=True,
    default=False,
    help="Delete ALL ingested documents.",
)
@click.option(
    "--yes",
    "-y",
    "skip_confirm",
    is_flag=True,
    default=False,
    help="Skip confirmation prompt.",
)
def delete(source: str | None, collection: str, delete_all: bool, skip_confirm: bool):
    """Delete ingested documents by source filename.

    SOURCE can be a full path or just the filename (e.g., "report.pdf").
    If --all is used, SOURCE is ignored and all documents are deleted.
    """
    if not source and not delete_all:
        click.echo("Error: provide a source filename or use --all.", err=True)
        raise click.UsageError("Missing SOURCE argument. Use --all to delete everything.")

    if delete_all:
        sources = list_sources(settings, collection_name=collection)
        if not sources:
            click.echo("No ingested documents to delete.")
            return
        click.echo(f"Found {len(sources)} ingested source(s):")
        for s in sources:
            click.echo(f"  - {Path(s).name}")
        if not skip_confirm:
            click.confirm("Delete ALL documents?", abort=True)
        total = 0
        for src in sources:
            total += delete_by_source(src, settings, collection_name=collection)
        click.echo(f"Deleted {total} chunk(s) from {len(sources)} source(s).")
        return

    all_sources = list_sources(settings, collection_name=collection)

    matches = [s for s in all_sources if source in s]
    if not matches:
        click.echo(f"No ingested document matches '{source}'.")
        click.echo("Available sources:")
        for s in all_sources:
            click.echo(f"  - {Path(s).name}")
        return

    for match in matches:
        if not skip_confirm:
            click.confirm(f"Delete '{Path(match).name}'?", abort=True)
        count = delete_by_source(match, settings, collection_name=collection)
        click.echo(f"Deleted {count} chunk(s) from {Path(match).name}")


if __name__ == "__main__":
    cli()
