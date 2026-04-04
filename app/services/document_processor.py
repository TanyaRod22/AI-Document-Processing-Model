"""Extract text from uploads and chunk for embedding."""

from __future__ import annotations

import io
import logging
from typing import BinaryIO

import pdfplumber
import tiktoken

logger = logging.getLogger(__name__)

ALLOWED_EXTENSIONS = {".pdf", ".txt"}
ENCODING_NAME = "cl100k_base"


class DocumentProcessingError(Exception):
    """Raised when a document cannot be read or produces no usable text."""


def _get_encoder() -> tiktoken.Encoding:
    return tiktoken.get_encoding(ENCODING_NAME)


def extract_text_from_pdf(file_obj: BinaryIO) -> str:
    """
    Extract plain text from a PDF stream.

    Concatenates text per page; skips pages that fail individually.
    """
    parts: list[str] = []
    try:
        file_obj.seek(0)
        with pdfplumber.open(file_obj) as pdf:
            for i, page in enumerate(pdf.pages):
                try:
                    t = page.extract_text()
                    if t and t.strip():
                        parts.append(t.strip())
                except Exception as e:  # noqa: BLE001 — per-page resilience
                    logger.warning("PDF page %s extract failed: %s", i, e)
    except Exception as e:  # noqa: BLE001
        raise DocumentProcessingError(f"Could not open or parse PDF: {e}") from e

    return "\n\n".join(parts).strip()


def extract_text_from_txt(file_obj: BinaryIO) -> str:
    """Read UTF-8 text from a text file stream."""
    try:
        file_obj.seek(0)
        raw = file_obj.read()
        if isinstance(raw, bytes):
            return raw.decode("utf-8", errors="replace").strip()
        return str(raw).strip()
    except Exception as e:  # noqa: BLE001
        raise DocumentProcessingError(f"Could not read text file: {e}") from e


def extract_text(filename: str, file_obj: BinaryIO) -> str:
    """
    Route extraction by file extension.

    Args:
        filename: Original filename (used for extension check).
        file_obj: Readable binary stream.

    Returns:
        Normalized plain text.

    Raises:
        DocumentProcessingError: Unsupported type or empty content.
    """
    lower = filename.lower()
    ext = ""
    if "." in lower:
        ext = lower[lower.rindex(".") :]

    if ext not in ALLOWED_EXTENSIONS:
        raise DocumentProcessingError(
            f"Unsupported file type {ext or '(none)'}. Allowed: {sorted(ALLOWED_EXTENSIONS)}"
        )

    if ext == ".pdf":
        text = extract_text_from_pdf(file_obj)
    else:
        text = extract_text_from_txt(file_obj)

    if not text:
        raise DocumentProcessingError("Document contains no extractable text.")

    logger.info("Extracted %s characters from %s", len(text), filename)
    return text


def chunk_text(
    text: str,
    chunk_size_tokens: int = 500,
    overlap_tokens: int = 50,
) -> list[str]:
    """
    Split text into overlapping token windows.

    Uses fixed token counts with stride ``chunk_size - overlap`` to preserve
    local context across chunk boundaries.
    """
    if chunk_size_tokens <= 0:
        raise ValueError("chunk_size_tokens must be positive")
    if overlap_tokens < 0 or overlap_tokens >= chunk_size_tokens:
        raise ValueError("overlap_tokens must be in [0, chunk_size_tokens)")

    enc = _get_encoder()
    ids = enc.encode(text)
    if not ids:
        return []

    stride = chunk_size_tokens - overlap_tokens
    chunks: list[str] = []
    start = 0
    while start < len(ids):
        window = ids[start : start + chunk_size_tokens]
        chunk_str = enc.decode(window)
        if chunk_str.strip():
            chunks.append(chunk_str.strip())
        if start + chunk_size_tokens >= len(ids):
            break
        start += stride

    logger.info("Created %s chunks (size=%s overlap=%s)", len(chunks), chunk_size_tokens, overlap_tokens)
    return chunks


def extract_and_chunk(
    filename: str,
    content: bytes,
    chunk_size_tokens: int,
    overlap_tokens: int,
) -> list[str]:
    """
    Full pipeline: bytes -> text -> chunks.

    Args:
        filename: Original upload name.
        content: Raw file bytes.
        chunk_size_tokens: Target chunk size in tokens.
        overlap_tokens: Token overlap between consecutive chunks.

    Returns:
        List of chunk strings.
    """
    buf = io.BytesIO(content)
    text = extract_text(filename, buf)
    return chunk_text(text, chunk_size_tokens, overlap_tokens)
