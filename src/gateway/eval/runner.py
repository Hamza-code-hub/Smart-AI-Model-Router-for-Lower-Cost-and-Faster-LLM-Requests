"""Phase 2.4 — eval runner.

Drive a workload against one or more targets and produce a result file +
Markdown report. Use:

    python -m gateway.eval --workload simple_qa --target gateway:auto
    python -m gateway.eval --workload mixed_realistic \\
        --target gateway:auto --target direct:openai:gpt-4o-mini

The runner handles multi-turn replay (turn N+1 sees turn N's assistant
response), threads a stable X-Conversation-Id per prompt so the sticky
router has a hook to land on, and persists every per-turn ``CallResult``
to the JSON output. The reporter (``reporter.py``) turns that JSON into
the human-readable Markdown summary.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import statistics
from dataclasses import asdict
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

from gateway.eval.reporter import build_markdown_report
from gateway.eval.targets import CallResult, Target, build_target
from gateway.eval.workloads import (
    WORKLOAD_DIR_DEFAULT,
    Prompt,
    Workload,
    load_workload,
    workload_fingerprint,
)


DEFAULT_RESULTS_DIR = Path("eval/results")
DEFAULT_GATEWAY_URL = os.environ.get("EVAL_GATEWAY_URL", "http://localhost:8000")


# ---------------------------------------------------------------------------
# Per-prompt execution
# ---------------------------------------------------------------------------


async def _run_prompt(
    target: Target,
    workload: Workload,
    prompt: Prompt,
    *,
    judge_models: list[str] | None = None,
) -> dict[str, Any]:
    """Replay ``prompt`` against ``target``. Single-turn prompts produce one
    call; multi-turn prompts produce one call per user turn, with the
    assistant reply threaded into the next turn's message list.

    If ``judge_models`` is provided, the FINAL assistant turn is scored by
    each judge (or ensembled if >1) and the result lands on the row as
    ``judge_verdict``.
    """
    messages: list[dict[str, str]] = [dict(m) for m in prompt.seed_messages]
    max_tokens = prompt.max_tokens or workload.default_max_tokens
    temperature = prompt.temperature if prompt.temperature is not None else workload.default_temperature
    conv_id = prompt.conversation_id

    turn_records: list[dict[str, Any]] = []
    for turn_idx, user_text in enumerate(prompt.user_turns, 1):
        messages.append({"role": "user", "content": user_text})
        result = await target.dispatch(
            messages,
            conversation_id=conv_id,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        turn_records.append(_serialize_turn(turn_idx, user_text, result))
        if result.error:
            # Surface the error but stop the conversation here — replaying
            # would just feed garbage into the next turn.
            break
        messages.append({"role": "assistant", "content": result.response_text})

    # Quality signal: any expected_contains substring present in the FINAL
    # assistant message? Cheap, soft.
    quality = _quality_check(prompt, turn_records)

    # Phase 2.5 — judge the final response when requested.
    judge_verdict: dict[str, Any] | None = None
    if judge_models and turn_records and not turn_records[-1].get("error"):
        judge_verdict = await _judge_final_turn(
            prompt=prompt,
            turn_records=turn_records,
            judge_models=judge_models,
        )

    return {
        "id": prompt.id,
        "is_multi_turn": prompt.is_multi_turn,
        "turns": len(turn_records),
        "expected_category": prompt.expected_category,
        "expected_contains": list(prompt.expected_contains),
        "quality_check": quality,
        "judge_verdict": judge_verdict,
        "calls": turn_records,
        "notes": prompt.notes,
    }


async def _judge_final_turn(
    *,
    prompt: Prompt,
    turn_records: list[dict[str, Any]],
    judge_models: list[str],
) -> dict[str, Any] | None:
    """Score the last assistant response. Single judge → its verdict;
    multiple judges → ensemble payload."""
    from gateway.judge import JudgeError, JudgeVerdict, judge_ensemble, judge_response

    last = turn_records[-1]
    response_text = last.get("response_text") or ""
    # Build the query string seen by the model. For single-turn this is
    # prompt.user_turns[0]; for multi-turn, the last user turn is what the
    # judge needs context for, but we also include the running history so
    # the judge can score follow-ups in context.
    if prompt.is_multi_turn:
        running = []
        for i, t in enumerate(turn_records):
            running.append(f"USER (turn {i+1}): {t.get('user_text_preview', '')}")
            if i < len(turn_records) - 1:
                running.append(f"ASSISTANT (turn {i+1}): {t.get('response_text') or ''}")
        running.append(f"USER (turn {len(turn_records)}): {prompt.user_turns[-1]}")
        query = "\n".join(running)
    else:
        query = prompt.user_turns[-1] if prompt.user_turns else ""

    expected_substr = (
        " | ".join(prompt.expected_contains) if prompt.expected_contains else None
    )

    if len(judge_models) == 1:
        verdict = await judge_response(
            query=query,
            response=response_text,
            expected=expected_substr,
            expected_category=prompt.expected_category,
            judge_virtual_model=judge_models[0],
        )
        if isinstance(verdict, JudgeVerdict):
            return {"mode": "single", **verdict.to_dict()}
        if isinstance(verdict, JudgeError):
            return {"mode": "single", "error": verdict.error,
                    "judge_model": verdict.judge_model}
    else:
        return {
            "mode": "ensemble",
            **await judge_ensemble(
                query=query,
                response=response_text,
                expected=expected_substr,
                expected_category=prompt.expected_category,
                judge_virtual_models=judge_models,
            ),
        }
    return None


def _serialize_turn(turn_idx: int, user_text: str, result: CallResult) -> dict[str, Any]:
    d = asdict(result)
    # Trim very long responses to keep the result file readable.
    if d.get("response_text") and len(d["response_text"]) > 800:
        d["response_text"] = d["response_text"][:800] + "…"
    d["turn"] = turn_idx
    d["user_text_preview"] = user_text[:200] + ("…" if len(user_text) > 200 else "")
    return d


def _quality_check(prompt: Prompt, turn_records: list[dict[str, Any]]) -> dict[str, Any]:
    """Pass if ANY of the expected substrings appears in the final response
    (case-insensitive). Soft signal — proper quality scoring is Phase 2.5.
    """
    if not prompt.expected_contains or not turn_records:
        return {"checked": False, "pass": None, "missing": []}
    last = turn_records[-1]
    if last.get("error"):
        return {"checked": True, "pass": False, "missing": list(prompt.expected_contains)}
    text = (last.get("response_text") or "").lower()
    hits = [s for s in prompt.expected_contains if s in text]
    return {"checked": True, "pass": bool(hits), "missing": [s for s in prompt.expected_contains if s not in text]}


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def summarize(rows: list[dict[str, Any]], target_label: str) -> dict[str, Any]:
    """Per-target aggregates: totals, latency percentiles, routing distribution,
    quality-check pass rate.
    """
    calls = [c for r in rows for c in r["calls"]]
    n_prompts = len(rows)
    n_calls = len(calls)
    errors = [c for c in calls if c.get("error")]
    successes = [c for c in calls if not c.get("error")]

    latencies = [c["latency_ms"] for c in calls]
    total_cost = sum(c.get("cost_usd", 0) or 0 for c in calls)
    prompt_tokens = sum(c.get("prompt_tokens", 0) or 0 for c in calls)
    completion_tokens = sum(c.get("completion_tokens", 0) or 0 for c in calls)

    quality_checked = [r for r in rows if r["quality_check"].get("checked")]
    quality_passes = sum(1 for r in quality_checked if r["quality_check"].get("pass"))

    routing_tiers: dict[str, int] = {}
    routing_categories: dict[str, int] = {}
    escalations = 0
    category_match = {"checked": 0, "match": 0}
    for r in rows:
        for c in r["calls"]:
            rd = c.get("routing_decision")
            if not rd:
                continue
            tier = rd.get("tier")
            cat = rd.get("category")
            if tier:
                routing_tiers[tier] = routing_tiers.get(tier, 0) + 1
            if cat:
                routing_categories[cat] = routing_categories.get(cat, 0) + 1
            if rd.get("switched_from"):
                escalations += 1
        # router_correctness: did the final-turn category match the expected?
        if r.get("expected_category") and r["calls"]:
            rd = r["calls"][-1].get("routing_decision")
            if rd and rd.get("category"):
                category_match["checked"] += 1
                if rd["category"] == r["expected_category"]:
                    category_match["match"] += 1

    cached = sum(1 for c in calls if c.get("cached"))
    fallbacks = sum(1 for c in calls if c.get("fallback_used"))

    # Phase 2.5 — judge aggregates.
    judge_summary: dict[str, Any] = {"n": 0, "n_errors": 0,
                                     "mean_overall": None, "mean_correctness": None,
                                     "pass_rate": None, "by_judge_model": {}}
    overalls: list[float] = []
    correctnesses: list[float] = []
    passes = 0
    for r in rows:
        jv = r.get("judge_verdict")
        if not jv:
            continue
        if "error" in jv:
            judge_summary["n_errors"] += 1
            continue
        judge_summary["n"] += 1
        if jv.get("mode") == "ensemble":
            mo = jv.get("mean_overall")
            mc = jv.get("mean_correctness")
            if mo is not None:
                overalls.append(float(mo))
            if mc is not None:
                correctnesses.append(float(mc))
            if jv.get("passed_majority"):
                passes += 1
            for v in jv.get("verdicts", []):
                jm = v.get("judge_model")
                if not jm:
                    continue
                judge_summary["by_judge_model"].setdefault(jm, 0)
                judge_summary["by_judge_model"][jm] += 1
        else:
            overalls.append(float(jv.get("overall", 0.0)))
            correctnesses.append(float(jv.get("correctness", 0.0)))
            if jv.get("passed"):
                passes += 1
            jm = jv.get("judge_model")
            if jm:
                judge_summary["by_judge_model"].setdefault(jm, 0)
                judge_summary["by_judge_model"][jm] += 1
    if overalls:
        judge_summary["mean_overall"] = round(sum(overalls) / len(overalls), 4)
    if correctnesses:
        judge_summary["mean_correctness"] = round(sum(correctnesses) / len(correctnesses), 4)
    if judge_summary["n"]:
        judge_summary["pass_rate"] = round(passes / judge_summary["n"], 4)

    return {
        "target": target_label,
        "n_prompts": n_prompts,
        "n_calls": n_calls,
        "n_errors": len(errors),
        "n_cached": cached,
        "n_fallback_used": fallbacks,
        "total_cost_usd": round(total_cost, 6),
        "tokens": {
            "prompt": prompt_tokens,
            "completion": completion_tokens,
        },
        "latency_ms": {
            "p50": round(_percentile(latencies, 0.50), 2) if latencies else 0,
            "p95": round(_percentile(latencies, 0.95), 2) if latencies else 0,
            "p99": round(_percentile(latencies, 0.99), 2) if latencies else 0,
            "max": round(max(latencies), 2) if latencies else 0,
        },
        "success_rate": round(len(successes) / n_calls, 4) if n_calls else None,
        "quality_check": {
            "n_checked": len(quality_checked),
            "n_pass": quality_passes,
            "rate": round(quality_passes / len(quality_checked), 4)
                if quality_checked else None,
        },
        "judge": judge_summary,
        "routing": {
            "tiers": routing_tiers,
            "categories": routing_categories,
            "escalations": escalations,
            "category_match": {
                "checked": category_match["checked"],
                "match": category_match["match"],
                "rate": (
                    round(category_match["match"] / category_match["checked"], 4)
                    if category_match["checked"]
                    else None
                ),
            },
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


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


async def run_target(
    workload: Workload,
    target: Target,
    *,
    verbose: bool = True,
    judge_models: list[str] | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for i, prompt in enumerate(workload.prompts, 1):
        if verbose:
            preview = prompt.user_turns[0][:60].replace("\n", " ")
            print(
                f"  [{i:>2}/{len(workload)}] {prompt.id} "
                f"({len(prompt.user_turns)}-turn) {preview!r}…",
                flush=True,
            )
        row = await _run_prompt(target, workload, prompt, judge_models=judge_models)
        rows.append(row)
        if verbose:
            last_call = row["calls"][-1] if row["calls"] else {}
            if last_call.get("error"):
                print(f"      ↳ ERROR {last_call['error']}", flush=True)
            else:
                rd = last_call.get("routing_decision") or {}
                tier = rd.get("tier", "—")
                cat = rd.get("category", "—")
                judge_line = ""
                jv = row.get("judge_verdict") or {}
                if jv:
                    overall = jv.get("overall") or jv.get("mean_overall")
                    if overall is not None:
                        judge_line = f" · judge={overall:.2f}"
                print(
                    f"      ↳ {last_call.get('model', '—')} · tier={tier} cat={cat} "
                    f"· {last_call.get('latency_ms', 0):.0f}ms "
                    f"· ${last_call.get('cost_usd', 0):.6f}"
                    f"{judge_line}",
                    flush=True,
                )
    return rows


async def _amain(args: argparse.Namespace) -> None:
    workload_dir = Path(args.workload_dir)
    workload = load_workload(args.workload, workload_dir=workload_dir)
    fingerprint = workload_fingerprint(workload)

    print(
        f"Workload: {workload.name} v{workload.version} "
        f"({len(workload)} prompts, fingerprint={fingerprint})"
    )
    print(f"Targets:  {', '.join(args.target)}")
    print()

    # Direct-target adapters need provider + db init. The judge also uses
    # in-process provider dispatch, so flip the same switch when judging.
    judge_models: list[str] = list(args.judge) if args.judge else []
    needs_provider = any(t.startswith("direct:") for t in args.target) or bool(judge_models)
    if needs_provider:
        from gateway.db import init_db
        from gateway.providers import init_providers
        from gateway.routing import init_resolver
        await init_db()
        init_providers()
        init_resolver()

    targets = [
        build_target(spec, gateway_url=args.gateway_url, api_key=args.api_key)
        for spec in args.target
    ]

    results_by_target: dict[str, dict[str, Any]] = {}
    try:
        for spec, target in zip(args.target, targets):
            print(f"=== {spec} ===")
            rows = await run_target(
                workload, target,
                verbose=not args.quiet,
                judge_models=judge_models or None,
            )
            summary = summarize(rows, target_label=target.label)
            results_by_target[spec] = {
                "label": target.label,
                "summary": summary,
                "rows": rows,
            }
            print()
    finally:
        for t in targets:
            await t.aclose()
        if needs_provider:
            from gateway.db import close_db
            from gateway.providers import close_providers
            await close_providers()
            await close_db()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    target_slug = "_".join(t.replace(":", "-").replace("/", "-") for t in args.target)
    if len(target_slug) > 60:
        target_slug = target_slug[:60]

    output_path = (
        Path(args.output)
        if args.output
        else out_dir / f"{date.today().isoformat()}_{workload.name}_{target_slug}.json"
    )

    run_meta = {
        "date": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "workload": {
            "name": workload.name,
            "version": workload.version,
            "fingerprint": fingerprint,
            "source": str(workload.source_path) if workload.source_path else None,
            "n_prompts": len(workload),
        },
        "targets": args.target,
        "gateway_url": args.gateway_url,
    }

    payload = {
        "run_metadata": run_meta,
        "results_by_target": results_by_target,
    }
    output_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    print(f"JSON written → {output_path}")
    print()
    print(build_markdown_report(payload))


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--workload", required=True, help="workload name (basename of eval/workloads/<name>.yaml)")
    p.add_argument(
        "--target",
        action="append",
        required=True,
        help="target spec (repeat to compare). e.g. gateway:auto, gateway:fast-qa, direct:openai:gpt-4o-mini, litellm:gpt-4o-mini",
    )
    p.add_argument("--gateway-url", default=DEFAULT_GATEWAY_URL)
    p.add_argument("--api-key", default=os.environ.get("EVAL_GATEWAY_API_KEY"))
    p.add_argument("--workload-dir", default=str(WORKLOAD_DIR_DEFAULT))
    p.add_argument("--output-dir", default=str(DEFAULT_RESULTS_DIR))
    p.add_argument("--output", default=None, help="override output JSON path")
    p.add_argument(
        "--judge",
        action="append",
        default=[],
        help=(
            "Phase 2.5 — virtual model to use as a judge on the FINAL turn "
            "of each prompt. Repeat for ensemble averaging "
            "(e.g. --judge claude-opus-4-7 --judge gpt-4o). Verdicts cache "
            "to the `judge_verdicts` Postgres table — re-runs of the same "
            "(query, response) pair under the same judge are free."
        ),
    )
    p.add_argument("--quiet", action="store_true")
    return p


def main() -> None:
    args = _build_parser().parse_args()
    asyncio.run(_amain(args))


if __name__ == "__main__":
    main()
