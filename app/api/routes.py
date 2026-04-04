"""FastAPI route handlers for upload, search, and RAG."""

from __future__ import annotations

import asyncio
import logging
import uuid
from typing import Annotated

from fastapi import APIRouter, File, HTTPException, Request, UploadFile

from app.models.schemas import (
    AskRequest,
    AskResponse,
    DeleteDocumentResponse,
    MatchedChunk,
    QueryRequest,
    QueryResponse,
    UploadResponse,
)
from app.services.document_processor import DocumentProcessingError, extract_and_chunk
from app.services.embedding_service import EmbeddingServiceError
from app.services.rag_service import RagServiceError

logger = logging.getLogger(__name__)

router = APIRouter()


def _services(request: Request):
    return (
        request.app.state.settings,
        request.app.state.embedding_service,
        request.app.state.vector_store,
        request.app.state.rag_service,
    )


@router.post("/upload", response_model=UploadResponse)
async def upload_document(
    request: Request,
    file: Annotated[UploadFile, File(..., description="PDF or TXT document")],
) -> UploadResponse:
    """
    Ingest a document: extract text, chunk, embed, and append to the FAISS index.
    """
    settings, embed_svc, store, _rag = _services(request)

    if not embed_svc.is_configured:
        raise HTTPException(status_code=503, detail="OPENAI_API_KEY is not configured.")

    filename = file.filename or "upload"
    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Empty file.")

    try:
        chunks = extract_and_chunk(
            filename,
            content,
            chunk_size_tokens=settings.chunk_size_tokens,
            overlap_tokens=settings.chunk_overlap_tokens,
        )
    except DocumentProcessingError as e:
        logger.info("Document processing failed for %s: %s", filename, e)
        raise HTTPException(status_code=400, detail=str(e)) from e

    if not chunks:
        raise HTTPException(status_code=400, detail="No chunks produced from document (empty after chunking).")

    document_id = str(uuid.uuid4())
    try:
        vectors = await asyncio.to_thread(embed_svc.embed_texts, chunks, False)
    except EmbeddingServiceError as e:
        logger.exception("Embedding failed during upload")
        raise HTTPException(status_code=502, detail=str(e)) from e

    metadatas = [
        {
            "document_id": document_id,
            "chunk_index": i,
            "text": chunk,
            "filename": filename,
        }
        for i, chunk in enumerate(chunks)
    ]
    store.add(vectors, metadatas)
    store.persist()

    logger.info("Uploaded document_id=%s filename=%s chunks=%s", document_id, filename, len(chunks))
    return UploadResponse(
        document_id=document_id,
        filename=filename,
        chunks_created=len(chunks),
    )


@router.post("/query", response_model=QueryResponse)
def semantic_query(request: Request, body: QueryRequest) -> QueryResponse:
    """
    Embed the query and return the top similar chunks with cosine similarity scores.
    """
    settings, embed_svc, store, _rag = _services(request)

    if not embed_svc.is_configured:
        raise HTTPException(status_code=503, detail="OPENAI_API_KEY is not configured.")
    if store.total_vectors == 0:
        raise HTTPException(status_code=400, detail="No documents ingested yet. Upload a document first.")

    try:
        qvec = embed_svc.embed_query(body.query.strip())
    except EmbeddingServiceError as e:
        raise HTTPException(status_code=502, detail=str(e)) from e

    hits = store.search(qvec, k=settings.query_top_k)
    matches = [
        MatchedChunk(
            document_id=str(m.get("document_id", "")),
            chunk_index=int(m.get("chunk_index", 0)),
            text=str(m.get("text", "")),
            score=score,
        )
        for score, m in hits
    ]
    logger.info("Query matched %s chunks", len(matches))
    return QueryResponse(matches=matches)


@router.post("/ask", response_model=AskResponse)
def ask_question(request: Request, body: AskRequest) -> AskResponse:
    """
    Retrieve relevant chunks and answer the question with an OpenAI chat model (RAG).
    """
    _settings, _embed_svc, _store, rag = _services(request)

    try:
        return rag.ask(body.query.strip())
    except RagServiceError as e:
        detail = str(e)
        if "No documents" in detail or "ingested" in detail.lower():
            raise HTTPException(status_code=400, detail=detail) from e
        if "OPENAI_API_KEY" in detail:
            raise HTTPException(status_code=503, detail=detail) from e
        raise HTTPException(status_code=502, detail=detail) from e


@router.delete("/documents/{document_id}", response_model=DeleteDocumentResponse)
def delete_document(request: Request, document_id: str) -> DeleteDocumentResponse:
    """Remove all vectors for ``document_id`` from the FAISS index and persist."""
    _settings, _embed_svc, store, _rag = _services(request)
    removed = store.remove_document(document_id)
    if removed == 0:
        raise HTTPException(status_code=404, detail=f"No document found with id {document_id!r}.")
    store.persist()
    logger.info("Deleted document_id=%s vectors_removed=%s", document_id, removed)
    return DeleteDocumentResponse(document_id=document_id, vectors_removed=removed)


@router.get("/health")
def health() -> dict[str, str]:
    """Liveness probe."""
    return {"status": "ok"}
