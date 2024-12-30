"""Sentence-transformer embedding pipeline.

Lazy-loaded `all-MiniLM-L6-v2` (384-dim). Used by the router exemplar lookup
(Phase 1.4).
"""
from __future__ import annotations

import asyncio
from threading import Lock
from typing import TYPE_CHECKING

import structlog

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

logger = structlog.get_logger(__name__)

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384

_model: "SentenceTransformer | None" = None
_load_lock = Lock()


def _load_model() -> "SentenceTransformer":
    global _model
    with _load_lock:
        if _model is None:
            from sentence_transformers import SentenceTransformer

            logger.info("embeddings.loading", model=MODEL_NAME)
            _model = SentenceTransformer(MODEL_NAME)
            logger.info("embeddings.loaded", model=MODEL_NAME, dim=EMBEDDING_DIM)
    return _model


def _encode_sync(text: str) -> list[float]:
    model = _load_model()
    vec = model.encode(text, normalize_embeddings=True, show_progress_bar=False)
    return vec.tolist()


async def embed(text: str) -> list[float]:
    """Embed a single string. Returns a 384-dim list normalized for cosine similarity."""
    return await asyncio.to_thread(_encode_sync, text)


async def warmup() -> None:
    """Load + run a throwaway embed so the first real request doesn't pay cold-start."""
    await asyncio.to_thread(_encode_sync, "warmup")
    logger.info("embeddings.warmed")
