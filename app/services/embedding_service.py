"""OpenAI embedding generation with batching and optional query cache."""

from __future__ import annotations

import logging
from collections import OrderedDict
from typing import Sequence

import numpy as np
from openai import APIError, OpenAI

logger = logging.getLogger(__name__)

# Default output dimension for text-embedding-3-small when not using dimensions param
TEXT_EMBEDDING_3_SMALL_DIM = 1536


class EmbeddingServiceError(Exception):
    """Raised when the embedding provider returns an error."""


class EmbeddingService:
    """Batch embeddings via OpenAI; optional LRU cache for single-query strings."""

    def __init__(
        self,
        api_key: str,
        model: str = "text-embedding-3-small",
        batch_size: int = 100,
        query_cache_max: int = 256,
    ) -> None:
        self._client = OpenAI(api_key=api_key) if api_key else None
        self._model = model
        self._batch_size = max(1, batch_size)
        self._cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self._cache_max = query_cache_max

    @property
    def is_configured(self) -> bool:
        return self._client is not None

    def _cache_get(self, key: str) -> np.ndarray | None:
        if key not in self._cache:
            return None
        self._cache.move_to_end(key)
        return self._cache[key]

    def _cache_set(self, key: str, value: np.ndarray) -> None:
        self._cache[key] = value
        self._cache.move_to_end(key)
        while len(self._cache) > self._cache_max:
            self._cache.popitem(last=False)

    def embed_texts(self, texts: Sequence[str], use_cache: bool = False) -> np.ndarray:
        """
        Embed a sequence of strings into a single float32 matrix ``(n, dim)``.

        Args:
            texts: Non-empty list of strings to embed.
            use_cache: If True, resolve each text via LRU cache when possible
                (intended for single-query or repeated strings).

        Returns:
            Stacked embedding vectors, L2-normalized for cosine / inner-product search.

        Raises:
            EmbeddingServiceError: Client not configured or API failure.
        """
        if not self._client:
            raise EmbeddingServiceError("OPENAI_API_KEY is not set.")
        if not texts:
            return np.zeros((0, TEXT_EMBEDDING_3_SMALL_DIM), dtype=np.float32)

        by_index: dict[int, np.ndarray] = {}
        pending_indices: list[int] = []
        pending_texts: list[str] = []

        for i, t in enumerate(texts):
            if use_cache:
                cached = self._cache_get(t)
                if cached is not None:
                    by_index[i] = cached
                    continue
            pending_indices.append(i)
            pending_texts.append(t)

        for start in range(0, len(pending_texts), self._batch_size):
            batch = pending_texts[start : start + self._batch_size]
            batch_idx = pending_indices[start : start + self._batch_size]
            try:
                resp = self._client.embeddings.create(model=self._model, input=list(batch))
            except APIError as e:
                logger.exception("OpenAI embeddings API error")
                raise EmbeddingServiceError(str(e)) from e
            except Exception as e:  # noqa: BLE001
                logger.exception("Unexpected error calling embeddings API")
                raise EmbeddingServiceError(str(e)) from e

            data_sorted = sorted(resp.data, key=lambda d: d.index)
            for j, item in enumerate(data_sorted):
                arr = np.array(item.embedding, dtype=np.float32)
                norm = np.linalg.norm(arr)
                if norm > 0:
                    arr = arr / norm
                orig_i = batch_idx[j]
                by_index[orig_i] = arr
                if use_cache:
                    self._cache_set(batch[j], arr.copy())

        ordered_rows = [by_index[i] for i in range(len(texts))]
        out = np.vstack(ordered_rows).astype(np.float32)
        logger.debug("Embedded %s texts -> shape %s", len(texts), out.shape)
        return out

    def embed_query(self, query: str) -> np.ndarray:
        """Embed a single search query with caching."""
        m = self.embed_texts([query], use_cache=True)
        return m[0]
