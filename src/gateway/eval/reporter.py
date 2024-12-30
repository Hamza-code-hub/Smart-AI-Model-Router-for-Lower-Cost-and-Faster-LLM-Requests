"""Phase 2.4 — Markdown reporter.

Turns a runner JSON payload into a human-readable table. One section per
target. When multiple targets are present, an additional 'Comparison' table
puts them side by side on the headline metrics (cost / latency / quality /
errors) — that's the cross-gateway honesty check the spec asks for.

Designed to be paste-able into history.md or a PR description.
"""
from __future__ import annotations

from typing import Any


def build_markdown_report(payload: dict[str, Any]) -> str:
    meta = payload.get("run_metadata", {})
    results = payload.get("results_by_target", {})
    workload = meta.get("workload", {})

    lines: list[str] = []
    lines.append(f"## Eval — {workload.get('name', '?')} v{workload.get('version', '?')}")
    lines.append("")
    lines.append(f"- Date: `{meta.get('date', '')}`")
    lines.append(
        f"- Workload: `{workload.get('name')}` "
        f"({workload.get('n_prompts')} prompts, fingerprint=`{workload.get('fingerprint')}`)"
    )
    lines.append(f"- Targets: {', '.join(f'`{t}`' for t in meta.get('targets', []))}")
    lines.append("")

    if len(results) > 1:
        lines.extend(_comparison_section(results))
        lines.append("")

    for spec, bundle in results.items():
        lines.extend(_per_target_section(spec, bundle))
        lines.append("")

    return "\n".join(lines)


def _fmt_usd(x: float | None) -> str:
    if x is None:
        return "—"
    if x == 0:
        return "$0"
    if x < 0.0001:
        return f"${x:.7f}"
    return f"${x:.5f}"


def _fmt_pct(x: float | None) -> str:
    if x is None:
        return "—"
    return f"{x * 100:.1f}%"


def _fmt_ms(x: float | None) -> str:
    if x is None:
        return "—"
    return f"{x:.0f}ms"


def _comparison_section(results: dict[str, Any]) -> list[str]:
    lines: list[str] = []
    lines.append("### Comparison")
    lines.append("")
    has_judge = any(
        (b["summary"].get("judge") or {}).get("n", 0) > 0 for b in results.values()
    )
    header = (
        "| Target | Calls | Errors | p50 | p95 | Cost | Quality | Cached | Fallback |"
        + (" Judge |" if has_judge else "")
    )
    sep = (
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|"
        + ("---:|" if has_judge else "")
    )
    lines.append(header)
    lines.append(sep)
    for spec, bundle in results.items():
        s = bundle["summary"]
        lat = s["latency_ms"]
        q = s["quality_check"]
        row = (
            f"| `{spec}` "
            f"| {s['n_calls']} "
            f"| {s['n_errors']} "
            f"| {_fmt_ms(lat['p50'])} "
            f"| {_fmt_ms(lat['p95'])} "
            f"| {_fmt_usd(s['total_cost_usd'])} "
            f"| {_fmt_pct(q['rate'])} ({q['n_pass']}/{q['n_checked']}) "
            f"| {s['n_cached']} "
            f"| {s['n_fallback_used']} |"
        )
        if has_judge:
            jd = s.get("judge") or {}
            mo = jd.get("mean_overall")
            pr = jd.get("pass_rate")
            row += (
                f" {('—' if mo is None else f'{mo:.2f}')} "
                f"({_fmt_pct(pr)}) |"
            )
        lines.append(row)
    return lines


def _per_target_section(spec: str, bundle: dict[str, Any]) -> list[str]:
    s = bundle["summary"]
    lines: list[str] = []
    lines.append(f"### `{spec}`")
    lines.append("")

    tokens = s["tokens"]
    lat = s["latency_ms"]
    lines.append(
        f"- **Calls**: {s['n_calls']} across {s['n_prompts']} prompts "
        f"({s['n_errors']} errors, {s['n_cached']} cached, "
        f"{s['n_fallback_used']} fallback)"
    )
    lines.append(
        f"- **Latency**: p50 {_fmt_ms(lat['p50'])} · p95 {_fmt_ms(lat['p95'])} "
        f"· p99 {_fmt_ms(lat['p99'])} · max {_fmt_ms(lat['max'])}"
    )
    lines.append(
        f"- **Cost**: {_fmt_usd(s['total_cost_usd'])} "
        f"({tokens['prompt']} prompt + {tokens['completion']} completion tokens)"
    )
    q = s["quality_check"]
    if q["n_checked"]:
        lines.append(
            f"- **Quality (substring check)**: "
            f"{_fmt_pct(q['rate'])} ({q['n_pass']}/{q['n_checked']})"
        )
    jd = s.get("judge") or {}
    if jd.get("n"):
        mo = jd.get("mean_overall")
        mc = jd.get("mean_correctness")
        pr = jd.get("pass_rate")
        models = ", ".join(
            f"{m} (n={n})" for m, n in (jd.get("by_judge_model") or {}).items()
        )
        lines.append(
            f"- **LLM-as-judge**: mean overall "
            f"{('—' if mo is None else f'{mo:.3f}')}"
            f" · mean correctness {('—' if mc is None else f'{mc:.3f}')}"
            f" · pass rate {_fmt_pct(pr)}"
            f" · judged {jd['n']}/{s['n_prompts']}"
            f" (errors: {jd['n_errors']})"
            + (f" · judges: {models}" if models else "")
        )
    routing = s.get("routing", {})
    tiers = routing.get("tiers") or {}
    cats = routing.get("categories") or {}
    if tiers or cats:
        tier_str = ", ".join(f"{k}: {v}" for k, v in sorted(tiers.items()))
        cat_str = ", ".join(f"{k}: {v}" for k, v in sorted(cats.items()))
        lines.append(f"- **Routing**: tiers — {tier_str or '—'}; categories — {cat_str or '—'}")
        if routing.get("escalations"):
            lines.append(f"- **Escalations**: {routing['escalations']}")
        cm = routing.get("category_match", {})
        if cm.get("checked"):
            lines.append(
                f"- **Category match vs expected**: "
                f"{_fmt_pct(cm['rate'])} ({cm['match']}/{cm['checked']})"
            )
    lines.append("")

    # Per-prompt mini-table — capped so this doesn't explode for mixed_realistic.
    rows = bundle.get("rows", [])
    if rows:
        lines.append("#### Per-prompt")
        lines.append("")
        has_judge_rows = any(r.get("judge_verdict") for r in rows)
        if has_judge_rows:
            lines.append("| Prompt | Turns | Model | Tier/Cat | Lat (last) | Cost | Quality | Judge |")
            lines.append("|---|---:|---|---|---:|---:|:---:|---:|")
        else:
            lines.append("| Prompt | Turns | Model | Tier/Cat | Lat (last) | Cost | Quality |")
            lines.append("|---|---:|---|---|---:|---:|:---:|")
        for row in rows[:40]:
            last = row["calls"][-1] if row["calls"] else {}
            rd = last.get("routing_decision") or {}
            tier_cat = (
                f"{rd.get('tier', '—')}/{rd.get('category', '—')}"
                if rd
                else "—"
            )
            q = row.get("quality_check", {})
            qmark = (
                "✓" if q.get("pass") is True
                else "✗" if q.get("pass") is False
                else "—"
            )
            base = (
                f"| `{row['id']}` "
                f"| {row['turns']} "
                f"| {last.get('model', '—')} "
                f"| {tier_cat} "
                f"| {_fmt_ms(last.get('latency_ms'))} "
                f"| {_fmt_usd(last.get('cost_usd'))} "
                f"| {qmark} |"
            )
            if has_judge_rows:
                jv = row.get("judge_verdict") or {}
                if not jv:
                    base += " — |"
                elif "error" in jv:
                    base += " err |"
                elif jv.get("mode") == "ensemble":
                    mo = jv.get("mean_overall")
                    base += f" {('—' if mo is None else f'{mo:.2f}')} |"
                else:
                    o = jv.get("overall")
                    base += f" {('—' if o is None else f'{o:.2f}')} |"
            lines.append(base)
        if len(rows) > 40:
            lines.append(f"| … ({len(rows) - 40} more rows in JSON) |")
    return lines
