"""Phase 2.5 — production sampling backstop.

The third spec-listed application of LLM-as-judge: judge 1% of routed
responses and write any flagged misroutes to ``judge_flags``. This is the
catch-net for negation/polarity errors the cascade router missed, drifting
category boundaries, and the long tail of "the cheaper route returned a
wrong-shaped answer but the substring eval was happy".

Wired into ``routes/chat.py`` as a fire-and-forget background task on the
successful non-streaming path. Coin flip is deterministic-per-request to
keep load predictable; the sampler is fail-soft top to bottom — a stuck
judge never affects the live request path.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import os
from typing import Any

import structlog

from gateway.db import acquire
from gateway.judge.judge import (
    DEFAULT_JUDGE_VIRTUAL_MODEL,
    JudgeError,
    JudgeVerdict,
    judge_response,
)

logger = structlog.get_logger(__name__)


def _sample_rate() -> float:
    raw = os.environ.get("JUDGE_SAMPLE_RATE", "0.01")
    try:
        rate = float(raw)
    except ValueError:
        return 0.01
    if rate < 0.0:
        return 0.0
    if rate > 1.0:
        return 1.0
    return rate


def _judge_enabled() -> bool:
    return os.environ.get("JUDGE_SAMPLING_ENABLED", "false").lower() == "true"


def _should_sample(seed: str, rate: float) -> bool:
    """Deterministic per-seed coin. seed = request id / hash of (query+response).

    Deterministic because re-running the same eval shouldn't suddenly judge
    different rows. Also: tests / smoke runs become reproducible.
    """
    if rate <= 0.0:
        return False
    if rate >= 1.0:
        return True
    digest = hashlib.sha256(seed.encode("utf-8", errors="replace")).digest()
    # First 4 bytes → uint32 → ratio in [0, 1).
    bucket = int.from_bytes(digest[:4], "big") / 2**32
    return bucket < rate


async def _maybe_flag(
    verdict: JudgeVerdict | JudgeError,
    *,
    tenant_id: str,
    virtual_model: str | None,
    routed_category: str | None,
    tier_fired: str | None,
    query: str,
    response: str,
    expected_category: str | None,
) -> None:
    """Persist a flag row when the judge marked the response low-quality OR
    when the routed category clearly contradicts the routing context.
    """
    if isinstance(verdict, JudgeError):
        # Don't persist judge failures as flags — they pollute the human
        # review queue. Just log.
        logger.info("judge.sample_error", error=verdict.error)
        return

    flag_reasons: list[str] = []
    if not verdict.passed:
        flag_reasons.append(f"low_quality (overall={verdict.overall:.2f})")
    if verdict.correctness < 0.5:
        flag_reasons.append(f"low_correctness ({verdict.correctness:.2f})")
    if expected_category and routed_category and expected_category != routed_category:
        flag_reasons.append(
            f"category_mismatch (routed={routed_category}, expected={expected_category})"
        )

    if not flag_reasons:
        return

    flag_reason = "; ".join(flag_reasons)
    try:
        async with acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT id FROM judge_verdicts
                WHERE query_hash = $1 AND response_hash = $2
                  AND judge_model = $3 AND prompt_version = $4
                """,
                hashlib.sha256(query.encode("utf-8", errors="replace")).hexdigest(),
                hashlib.sha256(response.encode("utf-8", errors="replace")).hexdigest(),
                verdict.judge_model,
                verdict.prompt_version,
            )
            verdict_id = row["id"] if row else None
            if verdict_id is None:
                logger.warning("judge.flag_missing_verdict_row")
                return
            await conn.execute(
                """
                INSERT INTO judge_flags (
                    verdict_id, tenant_id, virtual_model, routed_category,
                    tier_fired, flag_reason, query_preview, response_preview
                ) VALUES ($1,$2,$3,$4,$5,$6,$7,$8)
                """,
                verdict_id, tenant_id, virtual_model, routed_category,
                tier_fired, flag_reason,
                query[:500],
                response[:500],
            )
    except Exception as exc:
        logger.warning("judge.flag_persist_failed", error=str(exc))


async def sample_and_judge(
    *,
    tenant_id: str,
    virtual_model: str | None,
    routed_category: str | None,
    tier_fired: str | None,
    query: str,
    response: str,
    expected_category: str | None = None,
    judge_virtual_model: str = DEFAULT_JUDGE_VIRTUAL_MODEL,
    sample_rate: float | None = None,
) -> None:
    """1% sampler — call from chat.py as a BackgroundTask on successful
    non-streaming dispatches. Returns immediately if env-flag disabled.

    Fail-soft top to bottom. Never raises.
    """
    if not _judge_enabled():
        return
    rate = sample_rate if sample_rate is not None else _sample_rate()
    seed = query + "::" + response[:200]
    if not _should_sample(seed, rate):
        return

    try:
        verdict = await judge_response(
            query=query,
            response=response,
            expected_category=expected_category,
            judge_virtual_model=judge_virtual_model,
            use_cache=True,
            persist=True,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("judge.sample_call_failed", error=str(exc))
        return

    if isinstance(verdict, JudgeVerdict):
        logger.info(
            "judge.sample",
            virtual_model=virtual_model,
            routed_category=routed_category,
            tier_fired=tier_fired,
            overall=verdict.overall,
            passed=verdict.passed,
            cached=verdict.cached,
            judge_model=verdict.judge_model,
        )

    await _maybe_flag(
        verdict,
        tenant_id=tenant_id,
        virtual_model=virtual_model,
        routed_category=routed_category,
        tier_fired=tier_fired,
        query=query,
        response=response,
        expected_category=expected_category,
    )
