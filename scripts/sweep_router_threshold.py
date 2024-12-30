"""Phase 1.6 follow-up — tier-2 cosine-similarity threshold sweep.

Reuses `scripts/run_router_eval.py` end-to-end. For each threshold in the sweep
list, runs the full eval through `classify()`, then aggregates with the same
`_summarize` so the numbers are apples-to-apples with the baseline JSON.

Writes one combined JSON to `eval/results/{YYYY-MM-DD}_router_threshold_sweep.json`
and prints a Markdown comparison table.

Usage (inside the gateway container — host Postgres collides with Docker 5432):

    docker exec llm-router-gateway-1 python scripts/sweep_router_threshold.py
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from datetime import date, datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from gateway.db import close_db, init_db  # noqa: E402
from gateway.embeddings import warmup as warmup_embeddings  # noqa: E402
from gateway.providers import close_providers, init_providers  # noqa: E402
from gateway.routing import init_resolver  # noqa: E402
from gateway.routing.classifier import DEFAULT_CLASSIFIER_VIRTUAL_MODEL  # noqa: E402

from run_router_eval import (  # noqa: E402
    EVAL_YAML,
    RESULTS_DIR,
    _check_overlap,
    _load_rows,
    _run_eval,
    _summarize,
)

SWEEP_THRESHOLDS = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.85]


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--classifier-model", default=DEFAULT_CLASSIFIER_VIRTUAL_MODEL)
    parser.add_argument(
        "--thresholds",
        default=",".join(str(t) for t in SWEEP_THRESHOLDS),
        help="comma-separated list of cosine thresholds to sweep",
    )
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    thresholds = [float(t) for t in args.thresholds.split(",")]

    _check_overlap()
    rows = _load_rows(EVAL_YAML, "queries")
    print(f"Loaded {len(rows)} eval queries · sweeping {len(thresholds)} thresholds")
    print(f"Classifier model: {args.classifier_model}")

    await init_db()
    init_providers()
    init_resolver()
    await warmup_embeddings()

    sweep_results: list[dict] = []
    try:
        for threshold in thresholds:
            print()
            print(f"=== threshold = {threshold} ===")
            results = await _run_eval(
                rows,
                threshold=threshold,
                classifier_model=args.classifier_model,
            )
            summary = _summarize(results)
            sweep_results.append({
                "threshold": threshold,
                "summary": summary,
            })
    finally:
        await close_providers()
        await close_db()

    run_meta = {
        "date": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "eval_set": str(EVAL_YAML.relative_to(ROOT)),
        "classifier_virtual_model": args.classifier_model,
        "thresholds_swept": thresholds,
    }

    output_path = Path(args.output) if args.output else (
        RESULTS_DIR / f"{date.today().isoformat()}_router_threshold_sweep.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps({"run_metadata": run_meta, "sweep": sweep_results}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print()
    print(f"JSON written → {output_path}")
    print()
    print(_markdown_table(sweep_results))


def _markdown_table(sweep_results: list[dict]) -> str:
    lines: list[str] = []
    lines.append("## Threshold sweep")
    lines.append("")
    lines.append(
        "| Threshold | Tier-1 fired | Tier-2 fired | Tier-3 fired | Tier-2 share | Accuracy | "
        "fast-qa | code | deep-r. | long-ctx | p50 ms | p95 ms |"
    )
    lines.append(
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
    )
    for row in sweep_results:
        s = row["summary"]
        n = s["n"] or 1
        t1 = s["per_tier"]["heuristic"]["fired"]
        t2 = s["per_tier"]["embedding"]["fired"]
        t3 = s["per_tier"]["llm"]["fired"]
        t2_share = f"{(t2 / n):.0%}"
        acc = f"{s['overall_accuracy']:.1%}"
        cats = s["per_category"]
        cat_acc = lambda c: (
            f"{cats[c]['accuracy']:.0%}" if cats[c]["accuracy"] is not None else "—"
        )
        p50 = f"{s['latency_ms']['p50']:.0f}"
        p95 = f"{s['latency_ms']['p95']:.0f}"
        lines.append(
            f"| {row['threshold']:.2f} | {t1} | {t2} | {t3} | {t2_share} | {acc} | "
            f"{cat_acc('fast-qa')} | {cat_acc('code')} | {cat_acc('deep-reasoning')} | "
            f"{cat_acc('long-context')} | {p50} | {p95} |"
        )
    return "\n".join(lines)


if __name__ == "__main__":
    asyncio.run(main())
