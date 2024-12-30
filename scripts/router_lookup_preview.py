"""Diagnostic preview of the cascade router's tier-2 (embedding) lookup.

Takes a query string, embeds it via the same pipeline the router will use
(Phase 1.4), and prints the top-K nearest exemplars from router_exemplars
with cosine similarity + category. Does NOT apply the polarity-flip check
or threshold gating — that's the router's job in Phase 1.4. This is purely
for eyeballing whether the corpus is positioned reasonably.

Usage:
    docker exec llm-router-gateway-1 python scripts/router_lookup_preview.py "your query here"
    docker exec llm-router-gateway-1 python scripts/router_lookup_preview.py -k 10 "your query"
"""
from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from gateway.db import acquire, close_db, init_db  # noqa: E402
from gateway.embeddings import embed, warmup  # noqa: E402


async def lookup(query: str, k: int) -> None:
    await warmup()
    await init_db()
    try:
        vec = await embed(query)
        vec_literal = "[" + ",".join(f"{x:.8f}" for x in vec) + "]"

        async with acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT
                    category,
                    query_text,
                    notes,
                    1 - (embedding <=> $1::vector) AS similarity
                FROM router_exemplars
                ORDER BY embedding <=> $1::vector
                LIMIT $2
                """,
                vec_literal,
                k,
            )
    finally:
        await close_db()

    print()
    print(f"Query: {query!r}")
    print(f"Top {k} nearest exemplars (tier-2 preview, no polarity check):")
    print()
    print(f"{'sim':>6}  {'category':<15}  query")
    print(f"{'-'*6}  {'-'*15}  {'-'*60}")
    for r in rows:
        sim = r["similarity"]
        cat = r["category"]
        q = r["query_text"].replace("\n", " ⏎ ")
        if len(q) > 80:
            q = q[:77] + "..."
        print(f"{sim:6.3f}  {cat:<15}  {q}")

    # Aggregate vote: which category dominates the top-k?
    votes: dict[str, float] = {}
    for r in rows:
        votes[r["category"]] = votes.get(r["category"], 0.0) + r["similarity"]
    print()
    print("Sum-of-similarity by category in top-k:")
    for cat in sorted(votes, key=lambda c: -votes[c]):
        print(f"  {cat:<15} {votes[cat]:.3f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query", help="Query string to look up")
    parser.add_argument("-k", type=int, default=5, help="How many neighbours to show (default 5)")
    args = parser.parse_args()
    asyncio.run(lookup(args.query, args.k))


if __name__ == "__main__":
    main()
