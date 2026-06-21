from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application settings, loaded from environment variables or a .env file.
    """
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # PostgreSQL / pgvector
    postgres_host: str = "localhost"
    postgres_port: int = 5433
    postgres_user: str = "rag_user"
    postgres_password: str = "rag_password"
    postgres_db: str = "rag_db"

    # Embeddings
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    # FastFlowLM
    fastflowlm_base_url: str = "http://127.0.0.1:52625"
    llm_model: str = "llama3.2:3b"

    # Chunking
    chunk_size: int = 1500
    chunk_overlap: int = 300

    @property
    def postgres_connection_string(self) -> str:
        """Returns the SQLAlchemy connection string for PostgreSQL."""
        return (
            f"postgresql+psycopg2://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )


settings = Settings()

