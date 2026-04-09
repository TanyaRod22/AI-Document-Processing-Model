"""
AI document processing microservice — FastAPI entrypoint.

Run locally: ``uvicorn main:app --reload``
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI

from app.api.routes import router
from app.config import get_settings
from app.services.embedding_service import EmbeddingService, TEXT_EMBEDDING_3_SMALL_DIM
from app.services.rag_service import RagService
from app.services.vector_store import VectorStore

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize shared clients and vector store; persist index on shutdown."""
    settings = get_settings()
    app.state.settings = settings

    api_key = settings.openai_api_key.strip()
    openai_client = OpenAI(api_key=api_key) if api_key else None

    embed_svc = EmbeddingService(
        api_key=api_key,
        model=settings.embedding_model,
        batch_size=settings.embedding_batch_size,
    )
    store = VectorStore(
        dimension=TEXT_EMBEDDING_3_SMALL_DIM,
        persist_dir=settings.vector_store_dir,
    )
    rag = RagService(
        client=openai_client,
        chat_model=settings.chat_model,
        embedding_service=embed_svc,
        vector_store=store,
        top_k=settings.query_top_k,
    )

    app.state.embedding_service = embed_svc
    app.state.vector_store = store
    app.state.rag_service = rag

    logger.info("Application started; vector_store_dir=%s", settings.vector_store_dir)

    yield

    try:
        store.persist()
        logger.info("Vector store persisted on shutdown.")
    except Exception as e:  # noqa: BLE001
        logger.warning("Could not persist vector store on shutdown: %s", e)


app = FastAPI(
    title="AI Document Processing",
    description="Upload PDF/TXT, chunk, embed with OpenAI, search with FAISS, optional RAG.",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)

@app.get("/")
def root() -> dict[str, str]:
    """Service info."""
    return {
        "service": "ai-document-processing",
        "docs": "/docs",
        "upload": "POST /upload",
        "query": "POST /query",
        "ask": "POST /ask",
        "delete_document": "DELETE /documents/{document_id}",
    }
