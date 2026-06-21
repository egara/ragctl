from typing import List, Optional

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_postgres import PGVector
from sqlalchemy import create_engine, text

from src.config import Settings


def _get_connection_string(settings: Settings) -> str:
    """Helper method to get the PostgreSQL connection string."""
    return settings.postgres_connection_string


def create_vector_store(
    documents: List[Document],
    embeddings: Embeddings,
    settings: Settings,
    collection_name: str = "documents",
) -> PGVector:
    """
    Creates a new PGVector instance and stores the provided documents.

    Args:
        documents: The list of Documents to store.
        embeddings: The Embeddings instance to use.
        settings: Application settings.
        collection_name: The name of the collection.

    Returns:
        The created PGVector instance.
    """
    connection = _get_connection_string(settings)
    vector_store = PGVector.from_documents(
        documents=documents,
        embedding=embeddings,
        connection=connection,
        collection_name=collection_name,
        use_jsonb=True,
    )
    return vector_store


def get_vector_store(
    embeddings: Embeddings,
    settings: Settings,
    collection_name: str = "documents",
) -> PGVector:
    """
    Retrieves an existing PGVector instance.

    Args:
        embeddings: The Embeddings instance to use.
        settings: Application settings.
        collection_name: The name of the collection.

    Returns:
        The PGVector instance.
    """
    connection = _get_connection_string(settings)
    return PGVector(
        embeddings=embeddings,
        connection=connection,
        collection_name=collection_name,
        use_jsonb=True,
    )


from sqlalchemy.exc import ProgrammingError

def list_sources(
    settings: Settings,
    collection_name: str = "documents",
) -> List[str]:
    """
    Lists all unique document sources ingested in the database.

    Args:
        settings: Application settings.
        collection_name: The name of the collection.

    Returns:
        A list of source filenames/paths.
    """
    engine = create_engine(settings.postgres_connection_string)
    query = text("""
        SELECT DISTINCT e.cmetadata->>'source' AS source
        FROM langchain_pg_embedding e
        JOIN langchain_pg_collection c ON e.collection_id = c.uuid
        WHERE c.name = :collection_name
        ORDER BY source
    """)
    try:
        with engine.connect() as conn:
            rows = conn.execute(query, {"collection_name": collection_name}).fetchall()
        return [row[0] for row in rows if row[0]]
    except ProgrammingError:
        return []


def delete_by_source(
    source: str,
    settings: Settings,
    collection_name: str = "documents",
) -> int:
    """
    Deletes all chunks associated with a specific document source.

    Args:
        source: The source filename or path to delete.
        settings: Application settings.
        collection_name: The name of the collection.

    Returns:
        The number of chunks deleted.
    """
    engine = create_engine(settings.postgres_connection_string)
    count_query = text("""
        SELECT COUNT(*) AS cnt
        FROM langchain_pg_embedding e
        JOIN langchain_pg_collection c ON e.collection_id = c.uuid
        WHERE c.name = :collection_name
          AND e.cmetadata->>'source' = :source
    """)
    delete_query = text("""
        DELETE FROM langchain_pg_embedding e
        USING langchain_pg_collection c
        WHERE e.collection_id = c.uuid
          AND c.name = :collection_name
          AND e.cmetadata->>'source' = :source
    """)
    try:
        with engine.begin() as conn:
            count = conn.execute(count_query, {"collection_name": collection_name, "source": source}).scalar()
            conn.execute(delete_query, {"collection_name": collection_name, "source": source})
        return count
    except ProgrammingError:
        return 0

