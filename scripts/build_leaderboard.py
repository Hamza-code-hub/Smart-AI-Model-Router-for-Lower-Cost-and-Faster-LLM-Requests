"""Phase 3.1 — public leaderboard generator.

Reads ``eval/results/*.json`` (the Phase 2.4 runner output, optionally
enriched with Phase 2.5 judge verdicts), and writes ``docs/benchmarks.md``.

What lands in the leaderboard:

* For each ``(workload, target)`` pair that appears in any result file, the
  *latest* run wins on the headline tables. Older runs are preserved in the
  "Past runs" appendix.
* Cost is normalised to ``$ / 1k calls`` so verbose vs terse responses
  don't skew the comparison (per spec v3.4 wording).
* Quality prefers the LLM-as-judge ``mean_overall`` when available, else
  the substring ``quality_check.rate``. The column header marks which.

What is deliberately NOT here:

* No Portkey row — no API access. Spec says skip if unavailable.
* No A/B run aggregation — those are kept as separate rows in the appendix.
* No GitHub-Action cron — spec says to wait for the user to ask.

Run from the repo root:

    python scripts/build_leaderboard.py
    python scripts/build_leaderboard.py --results-dir eval/results --out docs/benchmarks.md

Acceptance notes:

* The leaderboard's credibility is the project's credibility — DO NOT
  cherry-pick workloads or hide losing rows. If LiteLLM beats this gateway
  on a workload, the table says so. See spec v3.4 "Honesty rule".
"""
from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_RESULTS_DIR = Path("eval/results")
DEFAULT_OUTPUT = Path("docs/benchmarks.md")


# ---------------------------------------------------------------------------
# Result loading
# ---------------------------------------------------------------------------


def load_results(results_dir: Path) -> list[dict[str, Any]]:
    """Load every ``*.json`` under ``results_dir`` that has the Phase 2.4
    ``results_by_target`` shape. Old router-baseline JSONs (no targets) are
    silently ignored.
    """
    out: list[dict[str, Any]] = []
    for path in sorted(results_dir.glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            print(f"  ! {path}: {exc}")
            continue
        if not isinstance(data, dict) or "results_by_target" not in data:
            # Router-baseline / threshold-sweep files — not workload runs.
            continue
        meta = data.get("run_metadata") or {}
        if not meta.get("workload"):
            continue
        out.append({
            "path": path,
            "mtime": path.stat().st_mtime,
            "date": meta.get("date") or "",
            "workload": meta.get("workload") or {},
            "results_by_target": data.get("results_by_target") or {},
            "gateway_url": meta.get("gateway_url"),
        })
    return out


# ---------------------------------------------------------------------------
# Aggregation: latest per (workload, target)
# ---------------------------------------------------------------------------


def latest_per_workload_target(
    results: list[dict[str, Any]],
) -> dict[tuple[str, str], dict[str, Any]]:
    """Map (workload_name, target_spec) → newest summary bundle."""
    best: dict[tuple[str, str], dict[str, Any]] = {}
    for r in results:
        wl_name = r["workload"].get("name") or "?"
        for target_spec, bundle in r["results_by_target"].items():
            key = (wl_name, target_spec)
            existing = best.get(key)
            if existing is None or r["mtime"] > existing["mtime"]:
                best[key] = {
                    "mtime": r["mtime"],
                    "date": r["date"],
                    "summary": bundle.get("summary") or {},
                    "workload": r["workload"],
                    "path": r["path"],
                    "target_spec": target_spec,
                    "workload_name": wl_name,
                }
    return best


# ---------------------------------------------------------------------------
# Metric extraction
# ---------------------------------------------------------------------------


def cost_per_1k_calls(summary: dict[str, Any]) -> float | None:
    n = summary.get("n_calls") or 0
    if n <= 0:
        return None
    return (summary.get("total_cost_usd") or 0.0) / n * 1000.0


def quality_pair(summary: dict[str, Any]) -> tuple[float | None, str]:
    """Return (score, source) where source is 'judge' or 'substring' or '—'."""
    judge = summary.get("judge") or {}
    if judge.get("n") and judge.get("mean_overall") is not None:
        return float(judge["mean_overall"]), "judge"
    qc = summary.get("quality_check") or {}
    rate = qc.get("rate")
    if rate is not None:
        return float(rate), "substring"
    return None, "—"


def cache_hit_rate(summary: dict[str, Any]) -> float | None:
    n = summary.get("n_calls") or 0
    cached = summary.get("n_cached") or 0
    if n <= 0:
        return None
    return cached / n


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------


def fmt_usd(x: float | None) -> str:
    if x is None:
        return "—"
    if x == 0:
        return "$0"
    if abs(x) < 0.0001:
        return f"${x:.7f}"
    if abs(x) < 0.01:
        return f"${x:.5f}"
    return f"${x:.4f}"


def fmt_pct(x: float | None) -> str:
    if x is None:
        return "—"
    return f"{x * 100:.1f}%"


def fmt_ms(x: float | None) -> str:
    if x is None:
        return "—"
    return f"{x:.0f} ms"


def fmt_score(x: float | None) -> str:
    if x is None:
        return "—"
    return f"{x:.3f}"


def fmt_count(x: int | None) -> str:
    if x is None:
        return "—"
    return str(x)


# ---------------------------------------------------------------------------
# Markdown emit
# ---------------------------------------------------------------------------


HEADER = """# LLM Gateway — Public Benchmarks

> Auto-generated by `scripts/build_leaderboard.py` from `eval/results/*.json`.
> Do not hand-edit — re-run the script.

Each table reports the most recent run of a given `(workload, target)` pair.
Older runs are preserved in [Past runs](#past-runs) at the bottom.

**Honesty rule.** When a competitor beats this gateway on a workload, that
row stays in the table. The leaderboard's credibility is the project's
credibility.

**Quality column** uses LLM-as-judge (Phase 2.5) when the run was judged
(`judge` source), falling back to substring-match (`substr`). Mixing the two
in one column is a lossy simplification — see the per-workload section for
both numbers when available.

**Cost** is normalised to USD per 1,000 calls so verbose vs terse responses
don't skew the comparison.
"""


def _build_comparison_table(rows: list[dict[str, Any]]) -> list[str]:
    """One row per target. Sorted by cost asc."""
    lines: list[str] = []
    lines.append(
        "| Target | Calls | Cost / 1k | p50 lat | p95 lat | Quality | (src) | Cache hit | Fallback |"
    )
    lines.append(
        "|---|---:|---:|---:|---:|---:|:---:|---:|---:|"
    )
    sorted_rows = sorted(
        rows,
        key=lambda r: cost_per_1k_calls(r["summary"]) if cost_per_1k_calls(r["summary"]) is not None else float("inf"),
    )
    for r in sorted_rows:
        s = r["summary"]
        lat = s.get("latency_ms") or {}
        q_score, q_src = quality_pair(s)
        cph = cache_hit_rate(s)
        lines.append(
            f"| `{r['target_spec']}` "
            f"| {fmt_count(s.get('n_calls'))} "
            f"| {fmt_usd(cost_per_1k_calls(s))} "
            f"| {fmt_ms(lat.get('p50'))} "
            f"| {fmt_ms(lat.get('p95'))} "
            f"| {fmt_score(q_score)} "
            f"| {q_src} "
            f"| {fmt_pct(cph)} "
            f"| {fmt_count(s.get('n_fallback_used'))} |"
        )
    return lines


def _build_workload_section(workload_name: str, rows: list[dict[str, Any]]) -> list[str]:
    lines: list[str] = []
    # Anchor for the TOC.
    lines.append(f"## Workload: `{workload_name}`")
    lines.append("")
    # Headline metadata from the most recent run on this workload.
    newest = max(rows, key=lambda r: r["mtime"])
    wl = newest["workload"]
    lines.append(
        f"- Workload version: `{wl.get('version', '?')}` · "
        f"fingerprint `{wl.get('fingerprint', '?')}` · "
        f"{wl.get('n_prompts', '?')} prompts"
    )
    lines.append(
        f"- Most recent run: {newest['date'] or '?'} · "
        f"`{newest['path'].name}`"
    )
    lines.append("")
    lines.extend(_build_comparison_table(rows))
    lines.append("")
    return lines


def _build_past_runs_table(all_results: list[dict[str, Any]]) -> list[str]:
    lines: list[str] = []
    lines.append("## Past runs")
    lines.append("")
    lines.append("Every workload run that ever produced a JSON, newest first.")
    lines.append("")
    lines.append("| Date | Workload | Targets | File |")
    lines.append("|---|---|---|---|")
    by_date = sorted(all_results, key=lambda r: r["mtime"], reverse=True)
    for r in by_date:
        targets = ", ".join(f"`{t}`" for t in r["results_by_target"])
        wl = r["workload"]
        lines.append(
            f"| {r['date'] or '?'} "
            f"| `{wl.get('name', '?')}` v{wl.get('version', '?')} "
            f"| {targets} "
            f"| `{r['path'].name}` |"
        )
    lines.append("")
    return lines


def build_markdown(results: list[dict[str, Any]]) -> str:
    """Top-level Markdown builder."""
    latest = latest_per_workload_target(results)

    by_workload: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for (wl_name, _target), bundle in latest.items():
        by_workload[wl_name].append(bundle)

    lines: list[str] = [HEADER, ""]
    lines.append(
        f"*Last generated: {datetime.now(timezone.utc).isoformat(timespec='seconds')}*"
    )
    lines.append("")
    lines.append(f"*Source files scanned: {len(results)}*")
    lines.append("")

    if not by_workload:
        lines.append("> No workload-shaped result files found in `eval/results/`.")
        return "\n".join(lines)

    # Table of contents.
    lines.append("## Workloads")
    lines.append("")
    for wl_name in sorted(by_workload):
        rows = by_workload[wl_name]
        targets = sorted({r["target_spec"] for r in rows})
        lines.append(
            f"- [`{wl_name}`](#workload-{wl_name.replace('_', '-')}) — "
            f"{len(rows)} target(s): {', '.join(f'`{t}`' for t in targets)}"
        )
    lines.append("")

    # Per-workload sections.
    for wl_name in sorted(by_workload):
        lines.extend(_build_workload_section(wl_name, by_workload[wl_name]))

    # Headline cross-workload table — pick the most-targeted workload(s).
    headline = _build_headline_section(by_workload)
    if headline:
        # Insert headline near the top, after TOC.
        # We'll just append at the end before past runs.
        lines.extend(headline)

    # Documented limitations of the leaderboard methodology.
    lines.extend([
        "## Methodology notes",
        "",
        "- **Latency** is the per-call wall time as measured by the eval runner, "
        "outside the gateway. It includes network hop from the runner to the "
        "gateway and gateway → provider latency. Direct-provider targets skip "
        "the gateway entirely, so their numbers are a floor.",
        "- **Cost** is computed from `config/pricing.yaml` on the resolved "
        "provider model. Cached responses count as $0. Gateway overhead (CPU, "
        "Postgres, embedding tier) is not included — only model spend.",
        "- **Quality** mixes two signals; row-level `(src)` column says which. "
        "`judge` means LLM-as-judge `mean_overall` (Phase 2.5); `substr` means "
        "`quality_check.rate` (any expected_contains substring in the final "
        "response). Substring is a weaker signal for open-ended generation.",
        "- **Cache hit** counts exact-cache hits. Sequential runs against the "
        "same workload will inflate this — re-run with `X-Cache-Skip: true` "
        "for fresh measurements.",
        "- **Fallback** is the count of calls served by the route's secondary "
        "provider (primary returned a retryable error or its circuit was open).",
        "",
        "## Documented limitations",
        "",
        "- The judge prompt is single-revision (`JUDGE_PROMPT_VERSION` in "
        "`src/gateway/judge/prompts.py`). LLM-as-judge has known biases "
        "(length, self-preference, style). Mitigations: rubric explicitly "
        "discourages length bias; judge model is not the same family as the "
        "deep-reasoning route (see `afk_decisions.md` D1); ensemble mode "
        "supported via repeated `--judge`.",
        "- Latency p95 is sensitive to outliers when `n_calls` is small. Treat "
        "tables with fewer than ~20 calls as directional.",
        "- Direct-provider targets bypass the gateway's failover, cache, and "
        "limits stack — they are a *floor*, not an apples-to-apples comparison "
        "for the gateway's all-in latency.",
        "",
    ])

    lines.extend(_build_past_runs_table(results))
    return "\n".join(lines)


def _build_headline_section(
    by_workload: dict[str, list[dict[str, Any]]],
) -> list[str]:
    """Cross-workload one-target-per-row headline. We only emit this if at
    least one workload has both ``gateway:auto`` and a non-gateway target
    — otherwise the comparison isn't meaningful.
    """
    auto_workloads = []
    for wl_name, rows in by_workload.items():
        targets = {r["target_spec"] for r in rows}
        if "gateway:auto" in targets and any(
            not t.startswith("gateway:") for t in targets
        ):
            auto_workloads.append((wl_name, rows))
    if not auto_workloads:
        return []

    lines: list[str] = ["## Cross-workload headline", ""]
    lines.append(
        "For workloads that pit `gateway:auto` against a non-gateway target, "
        "this is the side-by-side. Auto routing is meant to be cheaper *and* "
        "lose ≤5% on quality (spec v3.3 acceptance bar). If a row reads "
        "otherwise, that's the cascade router earning its keep — or not."
    )
    lines.append("")
    lines.append(
        "| Workload | Target | Calls | Cost / 1k | p50 lat | Quality | (src) |"
    )
    lines.append("|---|---|---:|---:|---:|---:|:---:|")
    for wl_name, rows in sorted(auto_workloads, key=lambda x: x[0]):
        # Sort: gateway:auto first, then the rest.
        sorted_rows = sorted(
            rows,
            key=lambda r: (0 if r["target_spec"] == "gateway:auto" else 1, r["target_spec"]),
        )
        for r in sorted_rows:
            s = r["summary"]
            lat = s.get("latency_ms") or {}
            q_score, q_src = quality_pair(s)
            lines.append(
                f"| `{wl_name}` "
                f"| `{r['target_spec']}` "
                f"| {fmt_count(s.get('n_calls'))} "
                f"| {fmt_usd(cost_per_1k_calls(s))} "
                f"| {fmt_ms(lat.get('p50'))} "
                f"| {fmt_score(q_score)} "
                f"| {q_src} |"
            )
    lines.append("")
    return lines


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results-dir", default=str(DEFAULT_RESULTS_DIR),
                   help="directory of eval JSON files (Phase 2.4 shape)")
    p.add_argument("--out", default=str(DEFAULT_OUTPUT),
                   help="output Markdown path")
    args = p.parse_args()

    results_dir = Path(args.results_dir)
    out_path = Path(args.out)

    if not results_dir.exists():
        print(f"results dir not found: {results_dir}")
        return 1

    results = load_results(results_dir)
    md = build_markdown(results)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(md, encoding="utf-8")
    print(f"Wrote {out_path} ({len(md)} chars, {len(results)} runs scanned)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
