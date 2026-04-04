"""Request and response models for the public API."""

from __future__ import annotations

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    """Semantic search over ingested chunks."""

    query: str = Field(..., min_length=1, description="Natural language search query.")


class MatchedChunk(BaseModel):
    """A chunk returned from vector similarity search."""

    document_id: str
    chunk_index: int
    text: str
    score: float = Field(..., description="Similarity score (higher is more similar).")


class QueryResponse(BaseModel):
    """Search results with scores."""

    matches: list[MatchedChunk]


class AskRequest(BaseModel):
    """RAG question against ingested documents."""

    query: str = Field(..., min_length=1, description="Question to answer from context.")


class AskResponse(BaseModel):
    """LLM answer grounded in retrieved chunks."""

    answer: str
    sources: list[MatchedChunk] = Field(
        default_factory=list,
        description="Chunks used as context (same shape as /query).",
    )


class UploadResponse(BaseModel):
    """Outcome of a document ingest operation."""

    document_id: str
    filename: str
    chunks_created: int
    message: str = "Document processed successfully."


class DeleteDocumentResponse(BaseModel):
    """Outcome of removing one document's vectors from the index."""

    document_id: str
    vectors_removed: int = Field(..., description="Number of chunk vectors deleted.")
    message: str = "Document removed from the index."
