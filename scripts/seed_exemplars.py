"""Seed the router_exemplars table from config/router_exemplars.yaml.

Idempotent: keyed on sha256(query_text). Re-running with no YAML edits is a
no-op. Editing a query's text creates a new row (hash changes); rows whose
hash no longer appears in the YAML are deleted at the end of the run so the
DB reflects the YAML exactly.

Usage:
    uv run python scripts/seed_exemplars.py
"""
from __future__ import annotations

import asyncio
import hashlib
import sys
from pathlib import Path

# Make `gateway.*` importable when running this script directly
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import yaml  # noqa: E402

from gateway.db import acquire, close_db, init_db  # noqa: E402
from gateway.embeddings import embed, warmup  # noqa: E402

EXEMPLAR_YAML = ROOT / "config" / "router_exemplars.yaml"

VALID_CATEGORIES = {"fast-qa", "code", "deep-reasoning", "long-context"}


def _text_hash(query: str) -> str:
    return hashlib.sha256(query.encode("utf-8")).hexdigest()


def _load_yaml() -> list[dict]:
    with open(EXEMPLAR_YAML, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    exemplars = data.get("exemplars") or []
    if not exemplars:
        raise RuntimeError(f"No exemplars found in {EXEMPLAR_YAML}")
    for row in exemplars:
        if row["category"] not in VALID_CATEGORIES:
            raise ValueError(
                f"Invalid category {row['category']!r} for query: {row['query'][:60]!r}"
            )
    return exemplars


async def _existing_hashes(conn) -> set[str]:
    rows = await conn.fetch("SELECT text_hash FROM router_exemplars")
    return {r["text_hash"] for r in rows}


async def _upsert_one(conn, row: dict) -> str:
    """Returns 'inserted', 'skipped', or 'error'."""
    query = row["query"].strip()
    h = _text_hash(query)
    category = row["category"]
    notes = row.get("notes")

    existing = await conn.fetchrow(
        "SELECT id FROM router_exemplars WHERE text_hash = $1", h
    )
    if existing:
        return "skipped"

    vec = await embed(query)
    # pgvector accepts the literal '[0.1,0.2,...]' string form
    vec_literal = "[" + ",".join(f"{x:.8f}" for x in vec) + "]"

    await conn.execute(
        """
        INSERT INTO router_exemplars (query_text, category, embedding, text_hash, notes)
        VALUES ($1, $2, $3::vector, $4, $5)
        """,
        query,
        category,
        vec_literal,
        h,
        notes,
    )
    return "inserted"


async def main() -> None:
    rows = _load_yaml()
    yaml_hashes = {_text_hash(r["query"].strip()) for r in rows}

    print(f"Loaded {len(rows)} exemplars from {EXEMPLAR_YAML.name}")
    by_cat: dict[str, int] = {}
    for r in rows:
        by_cat[r["category"]] = by_cat.get(r["category"], 0) + 1
    for cat in sorted(by_cat):
        print(f"  {cat:<15} {by_cat[cat]}")

    print("Warming embedding model…")
    await warmup()

    await init_db()
    inserted = 0
    skipped = 0
    deleted = 0
    try:
        async with acquire() as conn:
            existing = await _existing_hashes(conn)
            stale = existing - yaml_hashes
            if stale:
                async with conn.transaction():
                    result = await conn.execute(
                        "DELETE FROM router_exemplars WHERE text_hash = ANY($1::text[])",
                        list(stale),
                    )
                # asyncpg returns "DELETE N"
                deleted = int(result.split()[-1]) if result.startswith("DELETE") else len(stale)

            for row in rows:
                status = await _upsert_one(conn, row)
                if status == "inserted":
                    inserted += 1
                elif status == "skipped":
                    skipped += 1

            total = await conn.fetchval("SELECT COUNT(*) FROM router_exemplars")
    finally:
        await close_db()

    print()
    print(f"Inserted : {inserted}")
    print(f"Skipped  : {skipped} (already present, hash matched)")
    print(f"Deleted  : {deleted} (stale rows not in YAML)")
    print(f"Total in router_exemplars: {total}")


if __name__ == "__main__":
    asyncio.run(main())
