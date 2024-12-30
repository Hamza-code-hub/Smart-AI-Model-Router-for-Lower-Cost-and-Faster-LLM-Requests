"""Auto-seed router_exemplars from high-confidence Haiku (tier-3) classifications.

When the cascade router falls through to tier 3 and Haiku returns a category
with confidence >= SEED_CONFIDENCE_THRESHOLD, we embed the query and insert
a row tagged ``source='haiku-seeded'`` so future semantically-similar queries
hit the embedding tier at ~10-20ms instead of paying the full Haiku call.

Hard cap at MAX_EXEMPLARS rows. When the table is full, new seeded inserts
are silently dropped — see future_work.md section 3 for the LRU eviction
strategy that lifts that ceiling.

Fail-soft top to bottom: never raises, only logs.
"""
from __future__ import annotations

import hashlib

import structlog

from gateway.db import acquire
from gateway.embeddings import embed

logger = structlog.get_logger(__name__)

MAX_EXEMPLARS = 100_000
SEED_CONFIDENCE_THRESHOLD = 0.9
SEEDED_SOURCE = "haiku-seeded"


def _text_hash(query: str) -> str:
    return hashlib.sha256(query.encode("utf-8")).hexdigest()


async def seed_exemplar(query: str, category: str, confidence: float) -> None:
    """Fire-and-forget seeder. Schedule via ``background_tasks.add_task``.

    Skips silently when the query/category are empty, when the table has
    reached MAX_EXEMPLARS, when an identical text_hash already exists
    (curated rows are protected — ON CONFLICT DO NOTHING), or when any I/O
    step fails.
    """
    query = (query or "").strip()
    if not query or not category:
        return
    if confidence < SEED_CONFIDENCE_THRESHOLD:
        return

    try:
        vec = await embed(query)
    except Exception as exc:
        logger.warning("router.seed.embed_failed", error=str(exc))
        return
    vec_literal = "[" + ",".join(f"{x:.8f}" for x in vec) + "]"
    h = _text_hash(query)

    try:
        async with acquire() as conn:
            count = await conn.fetchval("SELECT COUNT(*) FROM router_exemplars")
            if count is not None and int(count) >= MAX_EXEMPLARS:
                logger.info("router.seed.cap_reached", count=int(count))
                return
            await conn.execute(
                """
                INSERT INTO router_exemplars (
                    query_text, category, embedding, text_hash,
                    source, source_confidence
                ) VALUES ($1, $2, $3::vector, $4, $5, $6)
                ON CONFLICT (text_hash) DO NOTHING
                """,
                query, category, vec_literal, h,
                SEEDED_SOURCE, confidence,
            )
    except Exception as exc:
        logger.warning("router.seed.persist_failed", error=str(exc))
