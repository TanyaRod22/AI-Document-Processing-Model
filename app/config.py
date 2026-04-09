"""Application configuration from environment."""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime settings loaded from environment and optional `.env` file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    openai_api_key: str = ""
    vector_store_dir: Path = Path("./data/vector_store")

    embedding_model: str = "text-embedding-3-small"
    chat_model: str = "gpt-4o-mini"

    chunk_size_tokens: int = 500
    chunk_overlap_tokens: int = 50
    query_top_k: int = 5
    embedding_batch_size: int = 100

    # Comma-separated browser origins (required when allow_credentials=True; do not use "*").
    cors_origins: str = (
        "http://127.0.0.1:5173,http://localhost:5173,"
        "https://aidocumind.netlify.app"
    )


def get_settings() -> Settings:
    """Return settings instance (callers may cache as needed)."""
    return Settings()
