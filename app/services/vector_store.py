"""FAISS-backed vector storage with JSON metadata and disk persistence."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import faiss
import numpy as np

logger = logging.getLogger(__name__)

METADATA_FILENAME = "metadata.json"
INDEX_FILENAME = "index.faiss"


class VectorStore:
    """
    Inner-product index on L2-normalized embeddings (equivalent to cosine similarity).

    Vectors are stored with accompanying metadata records (document id, chunk index, text).
    """

    def __init__(self, dimension: int, persist_dir: Path | str) -> None:
        self.dimension = dimension
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self._index_path = self.persist_dir / INDEX_FILENAME
        self._meta_path = self.persist_dir / METADATA_FILENAME
        self._index = faiss.IndexFlatIP(dimension)
        self._records: list[dict[str, Any]] = []
        self._load_from_disk()

    @property
    def total_vectors(self) -> int:
        return int(self._index.ntotal)

    def _load_from_disk(self) -> None:
        if not self._index_path.exists() or not self._meta_path.exists():
            logger.info("No existing vector store at %s; starting empty.", self.persist_dir)
            return
        try:
            loaded = faiss.read_index(str(self._index_path))
            if loaded.d != self.dimension:
                logger.warning(
                    "Persisted index dim %s != expected %s; ignoring persisted index.",
                    loaded.d,
                    self.dimension,
                )
                return
            with open(self._meta_path, encoding="utf-8") as f:
                payload = json.load(f)
            self._records = payload.get("records", [])
            self._index = loaded
            if len(self._records) != self._index.ntotal:
                logger.warning(
                    "Metadata count %s != FAISS ntotal %s; ignoring persisted store.",
                    len(self._records),
                    self._index.ntotal,
                )
                self._index = faiss.IndexFlatIP(self.dimension)
                self._records = []
                return
            logger.info(
                "Loaded vector store: %s vectors from %s",
                self._index.ntotal,
                self.persist_dir,
            )
        except Exception as e:  # noqa: BLE001
            logger.warning("Failed to load vector store (%s); starting empty.", e)
            self._index = faiss.IndexFlatIP(self.dimension)
            self._records = []

    def persist(self) -> None:
        """Write FAISS index and metadata JSON to ``persist_dir``."""
        faiss.write_index(self._index, str(self._index_path))
        with open(self._meta_path, "w", encoding="utf-8") as f:
            json.dump({"records": self._records}, f, ensure_ascii=False)
        logger.debug("Persisted %s vectors to %s", self._index.ntotal, self.persist_dir)

    def add(
        self,
        embeddings: np.ndarray,
        metadatas: list[dict[str, Any]],
    ) -> None:
        """
        Append normalized vectors and metadata rows.

        Args:
            embeddings: Shape ``(n, dimension)``, float32; should be L2-normalized.
            metadatas: One dict per row (e.g. document_id, chunk_index, text).
        """
        if embeddings.ndim != 2 or embeddings.shape[1] != self.dimension:
            raise ValueError(
                f"embeddings must be (n, {self.dimension}), got {embeddings.shape}"
            )
        if len(metadatas) != embeddings.shape[0]:
            raise ValueError("metadatas length must match number of embedding rows")
        x = embeddings.astype(np.float32, copy=False)
        faiss.normalize_L2(x)
        self._index.add(x)
        self._records.extend(metadatas)
        logger.info("Added %s vectors; index size=%s", x.shape[0], self._index.ntotal)

    def search(
        self,
        query_embedding: np.ndarray,
        k: int = 5,
    ) -> list[tuple[float, dict[str, Any]]]:
        """
        Return up to ``k`` best matches as ``(score, metadata)`` sorted by descending score.

        ``query_embedding`` should be a 1-D array of shape ``(dimension,)``; it is normalized.
        """
        if self._index.ntotal == 0:
            return []
        k = min(k, self._index.ntotal)
        q = np.asarray(query_embedding, dtype=np.float32).reshape(1, -1)
        if q.shape[1] != self.dimension:
            raise ValueError(f"query dim {q.shape[1]} != index dim {self.dimension}")
        faiss.normalize_L2(q)
        scores, indices = self._index.search(q, k)
        row_scores = scores[0]
        row_idx = indices[0]
        out: list[tuple[float, dict[str, Any]]] = []
        for score, idx in zip(row_scores, row_idx, strict=True):
            if idx < 0:
                continue
            out.append((float(score), dict(self._records[idx])))
        return out

    def remove_document(self, document_id: str) -> int:
        """
        Drop all vectors whose metadata ``document_id`` matches.

        Rebuilds the FAISS index from reconstructed rows (``IndexFlatIP`` has no native delete).

        Returns:
            Number of vectors removed (0 if no matching document).
        """
        if not document_id.strip() or not self._records:
            return 0
        target = document_id.strip()
        keep_indices = [
            i
            for i, r in enumerate(self._records)
            if str(r.get("document_id", "")) != target
        ]
        removed = len(self._records) - len(keep_indices)
        if removed == 0:
            return 0

        new_index = faiss.IndexFlatIP(self.dimension)
        new_records: list[dict[str, Any]] = []
        if keep_indices:
            rows: list[np.ndarray] = []
            for i in keep_indices:
                new_records.append(dict(self._records[i]))
                rows.append(self._index.reconstruct(int(i)))
            x = np.vstack(rows).astype(np.float32, copy=False)
            faiss.normalize_L2(x)
            new_index.add(x)

        self._index = new_index
        self._records = new_records
        logger.info(
            "Removed document_id=%s (%s vectors); index size=%s",
            target,
            removed,
            self._index.ntotal,
        )
        return removed
