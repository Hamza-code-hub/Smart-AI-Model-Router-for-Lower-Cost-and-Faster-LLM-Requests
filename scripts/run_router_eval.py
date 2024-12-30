"""Phase 1.3 — router eval runner.

Loads `eval/router_eval.yaml`, runs each query through the full cascade
(`gateway.routing.classifier.classify`), and writes a baseline result file +
prints a Markdown summary the user can paste into history.md or a PR.

Deterministic for tier-1 and tier-2 routes (regex + pgvector NN are
deterministic). Tier-3 routes depend on Haiku output — `temperature=0.0` plus
the strict-JSON parser keep most runs identical, but a query that already
sits on the LLM's decision boundary may flip categories across runs.

Usage (inside the gateway container — host Postgres collides with Docker's
5432, see history.md Phase 1.2):

    docker exec llm-router-gateway-1 python scripts/run_router_eval.py

Optional flags:
    --threshold 0.85          embedding-tier cosine floor
    --classifier-model fast-qa  tier-3 virtual model (resolved via routes.yaml)
    --output eval/results/...  override default JSON path
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import statistics
import sys
import time
from collections import defaultdict
from datetime import date, datetime, timezone
from pathlib import Path

# Make `gateway.*` importable when running this script directly.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import yaml  # noqa: E402

from gateway.db import close_db, init_db  # noqa: E402
from gateway.embeddings import warmup as warmup_embeddings  # noqa: E402
from gateway.models import ChatCompletionRequest, ChatMessage  # noqa: E402
from gateway.providers import close_providers, init_providers  # noqa: E402
from gateway.routing import init_resolver  # noqa: E402
from gateway.routing.classifier import (  # noqa: E402
    DEFAULT_CLASSIFIER_VIRTUAL_MODEL,
    DEFAULT_EMBEDDING_THRESHOLD,
    Decision,
    classify,
)

EVAL_YAML = ROOT / "eval" / "router_eval.yaml"
EXEMPLAR_YAML = ROOT / "config" / "router_exemplars.yaml"
RESULTS_DIR = ROOT / "eval" / "results"

CATEGORIES = ("fast-qa", "code", "deep-reasoning", "long-context")
TIERS = ("heuristic", "embedding", "llm")


# ---------------------------------------------------------------------------
# Loading + leakage check
# ---------------------------------------------------------------------------

def _load_rows(path: Path, key: str) -> list[dict]:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    rows = data.get(key) or []
    if not rows:
        raise RuntimeError(f"No '{key}' in {path}")
    return rows


def _text_hash(query: str) -> str:
    return hashlib.sha256(query.strip().encode("utf-8")).hexdigest()


def _check_overlap() -> None:
    """Refuse to run if any eval query hash collides with an exemplar hash."""
    exemplars = _load_rows(EXEMPLAR_YAML, "exemplars")
    eval_rows = _load_rows(EVAL_YAML, "queries")
    exemplar_hashes = {_text_hash(r["query"]) for r in exemplars}
    collisions = [
        r["query"][:80] for r in eval_rows if _text_hash(r["query"]) in exemplar_hashes
    ]
    if collisions:
        raise SystemExit(
            "Data leakage — these eval queries collide with exemplars by sha256:\n"
            + "\n".join(f"  - {q!r}" for q in collisions)
        )


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

async def _classify_one(
    query: str,
    *,
    threshold: float,
    classifier_model: str,
) -> tuple[Decision, float]:
    request = ChatCompletionRequest(
        model="auto",
        messages=[ChatMessage(role="user", content=query)],
    )
    t0 = time.perf_counter()
    decision = await classify(
        request,
        embedding_threshold=threshold,
        classifier_virtual_model=classifier_model,
    )
    elapsed_ms = (time.perf_counter() - t0) * 1000
    return decision, elapsed_ms


async def _run_eval(
    rows: list[dict],
    *,
    threshold: float,
    classifier_model: str,
) -> list[dict]:
    results: list[dict] = []
    for i, row in enumerate(rows, 1):
        query = row["query"]
        expected = row["expected_category"]
        decision, ms = await _classify_one(
            query, threshold=threshold, classifier_model=classifier_model
        )
        correct = decision.category == expected
        results.append({
            "idx": i,
            "query": query,
            "expected": expected,
            "predicted": decision.category,
            "tier_fired": decision.tier_fired,
            "similarity": decision.similarity,
            "confidence": decision.confidence,
            "matched_exemplar": decision.matched_exemplar,
            "latency_ms": round(ms, 3),
            "correct": correct,
            "notes": row.get("notes"),
        })
        mark = "✓" if correct else "✗"
        print(
            f"  [{i:>2}/{len(rows)}] {mark} tier={decision.tier_fired:<9} "
            f"expected={expected:<14} predicted={decision.category:<14} "
            f"{ms:>6.1f}ms  {query[:60]!r}"
        )
    return results


# ---------------------------------------------------------------------------
# Aggregation + reporting
# ---------------------------------------------------------------------------

def _summarize(results: list[dict]) -> dict:
    n = len(results)
    correct_total = sum(1 for r in results if r["correct"])

    per_tier_fired: dict[str, int] = defaultdict(int)
    per_tier_correct: dict[str, int] = defaultdict(int)
    per_tier_latencies: dict[str, list[float]] = defaultdict(list)
    for r in results:
        per_tier_fired[r["tier_fired"]] += 1
        per_tier_latencies[r["tier_fired"]].append(r["latency_ms"])
        if r["correct"]:
            per_tier_correct[r["tier_fired"]] += 1

    per_category: dict[str, dict] = {
        c: {"n": 0, "correct": 0, "misroute_to": defaultdict(int)} for c in CATEGORIES
    }
    for r in results:
        cat = r["expected"]
        per_category[cat]["n"] += 1
        if r["correct"]:
            per_category[cat]["correct"] += 1
        else:
            per_category[cat]["misroute_to"][r["predicted"]] += 1
    for cat in per_category:
        per_category[cat]["misroute_to"] = dict(per_category[cat]["misroute_to"])

    all_latencies = [r["latency_ms"] for r in results]
    p50 = statistics.median(all_latencies)
    p95 = _percentile(all_latencies, 0.95) if all_latencies else 0.0
    p99 = _percentile(all_latencies, 0.99) if all_latencies else 0.0

    return {
        "n": n,
        "overall_accuracy": round(correct_total / n, 4) if n else 0.0,
        "per_tier": {
            tier: {
                "fired": per_tier_fired.get(tier, 0),
                "correct": per_tier_correct.get(tier, 0),
                "hit_rate": (
                    round(per_tier_correct.get(tier, 0) / per_tier_fired[tier], 4)
                    if per_tier_fired.get(tier)
                    else None
                ),
                "p50_ms": round(statistics.median(per_tier_latencies[tier]), 2)
                    if per_tier_latencies.get(tier) else None,
            }
            for tier in TIERS
        },
        "per_category": {
            cat: {
                "n": data["n"],
                "correct": data["correct"],
                "accuracy": round(data["correct"] / data["n"], 4) if data["n"] else None,
                "misroute_to": data["misroute_to"],
            }
            for cat, data in per_category.items()
        },
        "latency_ms": {
            "p50": round(p50, 2),
            "p95": round(p95, 2),
            "p99": round(p99, 2),
            "max": round(max(all_latencies), 2) if all_latencies else 0.0,
        },
    }


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    k = (len(s) - 1) * pct
    f = int(k)
    c = min(f + 1, len(s) - 1)
    if f == c:
        return s[f]
    return s[f] + (s[c] - s[f]) * (k - f)


def _markdown_report(summary: dict, results: list[dict], meta: dict) -> str:
    lines: list[str] = []
    lines.append(f"## Router eval baseline — {meta['date']}")
    lines.append("")
    lines.append(
        f"- Eval set: `{meta['eval_set']}` ({summary['n']} queries)"
    )
    lines.append(
        f"- Embedding threshold: `{meta['embedding_threshold']}` · "
        f"Tier-3 classifier: `{meta['classifier_virtual_model']}`"
    )
    lines.append(f"- **Overall accuracy: {summary['overall_accuracy']:.1%}**")
    lines.append("")

    lines.append("### Per-tier breakdown")
    lines.append("")
    lines.append("| Tier | Fired | Correct | Hit rate | p50 latency |")
    lines.append("|---|---:|---:|---:|---:|")
    for tier in TIERS:
        d = summary["per_tier"][tier]
        hit = f"{d['hit_rate']:.1%}" if d["hit_rate"] is not None else "—"
        p50 = f"{d['p50_ms']:.1f}ms" if d["p50_ms"] is not None else "—"
        lines.append(f"| {tier} | {d['fired']} | {d['correct']} | {hit} | {p50} |")
    lines.append("")

    lines.append("### Per-category breakdown")
    lines.append("")
    lines.append("| Category | n | Correct | Accuracy | Misroute → |")
    lines.append("|---|---:|---:|---:|---|")
    for cat in CATEGORIES:
        d = summary["per_category"][cat]
        acc = f"{d['accuracy']:.1%}" if d["accuracy"] is not None else "—"
        misroute = (
            ", ".join(f"{k}: {v}" for k, v in d["misroute_to"].items())
            or "—"
        )
        lines.append(f"| {cat} | {d['n']} | {d['correct']} | {acc} | {misroute} |")
    lines.append("")

    lines.append("### Latency")
    lines.append("")
    lat = summary["latency_ms"]
    lines.append(
        f"- p50: **{lat['p50']:.1f}ms** · p95: **{lat['p95']:.1f}ms** · "
        f"p99: **{lat['p99']:.1f}ms** · max: {lat['max']:.1f}ms"
    )
    lines.append("- Budget per spec v3.3: p95 < 100ms")
    lines.append("")

    misses = [r for r in results if not r["correct"]]
    if misses:
        lines.append("### Misroutes (first 15)")
        lines.append("")
        lines.append("| # | Tier | Expected | Predicted | Sim/Conf | Query |")
        lines.append("|---:|---|---|---|---|---|")
        for r in misses[:15]:
            sim = r.get("similarity")
            conf = r.get("confidence")
            score = (
                f"sim={sim:.2f}" if sim is not None
                else f"conf={conf:.2f}" if conf is not None
                else "—"
            )
            q = r["query"].replace("\n", " ⏎ ")[:80]
            lines.append(
                f"| {r['idx']} | {r['tier_fired']} | {r['expected']} | "
                f"{r['predicted']} | {score} | `{q}` |"
            )
        lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threshold", type=float, default=DEFAULT_EMBEDDING_THRESHOLD)
    parser.add_argument("--classifier-model", default=DEFAULT_CLASSIFIER_VIRTUAL_MODEL)
    parser.add_argument("--output", default=None,
                        help="JSON output path; defaults to eval/results/{date}_router_baseline.json")
    args = parser.parse_args()

    _check_overlap()
    rows = _load_rows(EVAL_YAML, "queries")

    print(f"Loaded {len(rows)} eval queries from {EVAL_YAML.name}")
    print(f"Threshold: {args.threshold}  Tier-3 model: {args.classifier_model}")
    print("Warming embedding model + bringing up infra…")

    await init_db()
    init_providers()
    init_resolver()
    await warmup_embeddings()

    try:
        print()
        results = await _run_eval(
            rows,
            threshold=args.threshold,
            classifier_model=args.classifier_model,
        )
    finally:
        await close_providers()
        await close_db()

    summary = _summarize(results)
    run_meta = {
        "date": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "eval_set": str(EVAL_YAML.relative_to(ROOT)),
        "embedding_threshold": args.threshold,
        "classifier_virtual_model": args.classifier_model,
        "git_branch": _read_git_branch(),
    }

    output_path = Path(args.output) if args.output else (
        RESULTS_DIR / f"{date.today().isoformat()}_router_baseline.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(
            {"run_metadata": run_meta, "summary": summary, "results": results},
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    print()
    print(f"JSON written → {output_path}")
    print()
    print(_markdown_report(summary, results, run_meta))


def _read_git_branch() -> str | None:
    head = ROOT / ".git" / "HEAD"
    if not head.exists():
        return None
    try:
        line = head.read_text(encoding="utf-8").strip()
        if line.startswith("ref: refs/heads/"):
            return line[len("ref: refs/heads/"):]
        return line[:12]
    except OSError:
        return None


if __name__ == "__main__":
    asyncio.run(main())
