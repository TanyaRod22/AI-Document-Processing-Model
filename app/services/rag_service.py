"""Retrieval-augmented generation using OpenAI chat completions."""

from __future__ import annotations

import logging

from openai import APIError, OpenAI

from app.models.schemas import AskResponse, MatchedChunk
from app.services.embedding_service import EmbeddingService, EmbeddingServiceError
from app.services.vector_store import VectorStore

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are a careful assistant. Answer using only the context provided below. "
    "If the context does not contain enough information, say you do not know "
    "rather than guessing. Cite ideas by referring to the context implicitly; "
    "do not invent sources."
)


class RagServiceError(Exception):
    """Raised when RAG pipeline fails (retrieval or generation)."""


class RagService:
    """Retrieve top chunks and answer with an OpenAI chat model."""

    def __init__(
        self,
        client: OpenAI | None,
        chat_model: str,
        embedding_service: EmbeddingService,
        vector_store: VectorStore,
        top_k: int = 5,
    ) -> None:
        self._client = client
        self._chat_model = chat_model
        self._embed = embedding_service
        self._store = vector_store
        self._top_k = top_k

    def ask(self, question: str) -> AskResponse:
        """
        Run retrieval over the vector store and generate an answer.

        Raises:
            RagServiceError: Missing API client, embedding failure, or chat API error.
        """
        if not self._client:
            raise RagServiceError("OPENAI_API_KEY is not set.")
        if self._store.total_vectors == 0:
            raise RagServiceError("No documents have been ingested yet.")

        try:
            qvec = self._embed.embed_query(question)
        except EmbeddingServiceError as e:
            raise RagServiceError(str(e)) from e

        hits = self._store.search(qvec, k=self._top_k)
        if not hits:
            raise RagServiceError("No relevant chunks found.")

        context_blocks: list[str] = []
        sources: list[MatchedChunk] = []
        for rank, (score, meta) in enumerate(hits, start=1):
            text = str(meta.get("text", ""))
            context_blocks.append(f"[{rank}] {text}")
            sources.append(
                MatchedChunk(
                    document_id=str(meta.get("document_id", "")),
                    chunk_index=int(meta.get("chunk_index", 0)),
                    text=text,
                    score=score,
                )
            )

        context = "\n\n".join(context_blocks)
        user_content = (
            "Context:\n"
            f"{context}\n\n"
            f"Question: {question}\n\n"
            "Answer the question using only the context above."
        )

        try:
            completion = self._client.chat.completions.create(
                model=self._chat_model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_content},
                ],
                temperature=0.2,
            )
        except APIError as e:
            logger.exception("OpenAI chat completion failed")
            raise RagServiceError(str(e)) from e
        except Exception as e:  # noqa: BLE001
            logger.exception("Unexpected error during chat completion")
            raise RagServiceError(str(e)) from e

        choice = completion.choices[0]
        answer = (choice.message.content or "").strip()
        if not answer:
            raise RagServiceError("Model returned an empty answer.")

        logger.info("RAG answer generated for query length=%s", len(question))
        return AskResponse(answer=answer, sources=sources)
