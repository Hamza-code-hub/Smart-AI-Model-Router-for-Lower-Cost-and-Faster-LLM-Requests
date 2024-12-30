"""Phase 2.5 — judge core.

A judge call is conceptually:

    judge_response(query, response, judge_virtual_model="gpt-4o")
    → JudgeVerdict(correctness=…, completeness=…, instruction_following=…,
                   overall=…, passed=…)

The implementation:

1. Hash (query, response) — that plus (judge_model, prompt_version) is the
   verdict cache key. Cache lives in Postgres `judge_verdicts`.
2. Cache hit → return saved verdict.
3. Cache miss → call the judge via the existing virtual-model dispatch
   (so a swap to GPT-5 or local Llama is a routes.yaml edit, not a code
   change). Parse the strict JSON. Persist verdict. Return.

Errors are fail-soft — a JudgeError comes back instead of raising so the
sampling backstop in production never bubbles a 500.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import re
import time
from dataclasses import asdict, dataclass, field
from typing import Any

import structlog

from gateway.db import acquire
from gateway.judge.prompts import (
    JUDGE_PROMPT_VERSION,
    JUDGE_SYSTEM_PROMPT,
    build_judge_user_prompt,
)
from gateway.models import ChatCompletionRequest, ChatMessage
from gateway.providers import get_provider
from gateway.routing import get_resolver

logger = structlog.get_logger(__name__)

# Default judge virtual model. Cross-family on purpose: production traffic
# routes to Anthropic models, so judging with an OpenAI model neutralises the
# self-preference bias documented in research/phase_2_5_llm_judge.md.
# Swappable per call.
DEFAULT_JUDGE_VIRTUAL_MODEL = "gpt-4o"

# Cap judge output. JSON verdict fits in <300 tokens with the short reasoning.
JUDGE_MAX_TOKENS = 400

_JSON_OBJ_RE = re.compile(r"\{.*\}", re.DOTALL)


@dataclass
class JudgeVerdict:
    correctness: float
    completeness: float
    instruction_following: float
    overall: float
    passed: bool
    reasoning: str
    judge_model: str
    prompt_version: str
    judge_latency_ms: int
    cached: bool = False
    raw: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d.pop("raw", None)
        return d


@dataclass
class JudgeError:
    """Returned instead of raising when judging fails (parse error, provider
    outage, etc.). Sampling loops in production keep going on errors.
    """

    error: str
    judge_model: str
    prompt_version: str
    judge_latency_ms: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {"error": self.error, "judge_model": self.judge_model,
                "prompt_version": self.prompt_version,
                "judge_latency_ms": self.judge_latency_ms}


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()


async def _lookup_cached(
    query_hash: str, response_hash: str, judge_model: str, prompt_version: str
) -> JudgeVerdict | None:
    try:
        async with acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT scores, overall_score, passed, reasoning, judge_latency_ms
                FROM judge_verdicts
                WHERE query_hash = $1
                  AND response_hash = $2
                  AND judge_model = $3
                  AND prompt_version = $4
                """,
                query_hash, response_hash, judge_model, prompt_version,
            )
    except Exception as exc:
        logger.warning("judge.cache_lookup_failed", error=str(exc))
        return None
    if row is None:
        return None
    scores = row["scores"]
    if isinstance(scores, str):
        try:
            scores = json.loads(scores)
        except Exception:
            scores = {}
    return JudgeVerdict(
        correctness=float(scores.get("correctness", 0.0)),
        completeness=float(scores.get("completeness", 0.0)),
        instruction_following=float(scores.get("instruction_following", 0.0)),
        overall=float(row["overall_score"]),
        passed=bool(row["passed"]),
        reasoning=row["reasoning"] or "",
        judge_model=judge_model,
        prompt_version=prompt_version,
        judge_latency_ms=int(row["judge_latency_ms"] or 0),
        cached=True,
    )


async def _persist_verdict(
    *,
    query_hash: str,
    response_hash: str,
    judge_model: str,
    prompt_version: str,
    verdict: JudgeVerdict,
) -> str | None:
    scores = {
        "correctness": verdict.correctness,
        "completeness": verdict.completeness,
        "instruction_following": verdict.instruction_following,
    }
    try:
        async with acquire() as conn:
            row = await conn.fetchrow(
                """
                INSERT INTO judge_verdicts (
                    query_hash, response_hash, judge_model, prompt_version,
                    scores, overall_score, passed, reasoning, judge_latency_ms
                ) VALUES ($1,$2,$3,$4,$5::jsonb,$6,$7,$8,$9)
                ON CONFLICT ON CONSTRAINT uniq_judge_verdicts
                  DO UPDATE SET scores = EXCLUDED.scores,
                                overall_score = EXCLUDED.overall_score,
                                passed = EXCLUDED.passed,
                                reasoning = EXCLUDED.reasoning,
                                judge_latency_ms = EXCLUDED.judge_latency_ms
                RETURNING id
                """,
                query_hash, response_hash, judge_model, prompt_version,
                json.dumps(scores), verdict.overall, verdict.passed,
                verdict.reasoning, verdict.judge_latency_ms,
            )
            return str(row["id"]) if row else None
    except Exception as exc:
        logger.warning("judge.cache_store_failed", error=str(exc))
        return None


def _parse_judge_response(text: str) -> dict[str, Any]:
    match = _JSON_OBJ_RE.search(text)
    if match is None:
        raise ValueError("no JSON object in judge response")
    data = json.loads(match.group(0))
    if not isinstance(data, dict):
        raise ValueError("judge response was not a JSON object")
    return data


async def judge_response(
    *,
    query: str,
    response: str,
    expected: str | None = None,
    expected_category: str | None = None,
    judge_virtual_model: str = DEFAULT_JUDGE_VIRTUAL_MODEL,
    use_cache: bool = True,
    persist: bool = True,
) -> JudgeVerdict | JudgeError:
    """Score one (query, response) pair. Fail-soft on error.

    Returns ``JudgeVerdict`` on success or ``JudgeError`` on failure (parse
    error, provider down, cache miss when DB is down — none should bubble).

    ``use_cache`` / ``persist`` are independent so eval runs can choose
    "judge fresh, but also write so other runs benefit".
    """
    q_hash = _hash(query)
    r_hash = _hash(response)

    if use_cache:
        cached = await _lookup_cached(q_hash, r_hash, judge_virtual_model, JUDGE_PROMPT_VERSION)
        if cached is not None:
            return cached

    try:
        route = get_resolver().resolve(judge_virtual_model)
    except KeyError as exc:
        return JudgeError(error=f"unknown judge route: {exc}",
                          judge_model=judge_virtual_model,
                          prompt_version=JUDGE_PROMPT_VERSION)

    provider = get_provider(route.primary.provider)
    real_model = route.primary.model

    judge_req = ChatCompletionRequest(
        model=judge_virtual_model,
        messages=[
            ChatMessage(role="system", content=JUDGE_SYSTEM_PROMPT),
            ChatMessage(role="user",
                        content=build_judge_user_prompt(
                            query, response,
                            expected=expected,
                            expected_category=expected_category,
                        )),
        ],
        # Opus 4-7 rejects temperature as deprecated — leave None for that route.
        temperature=None if "opus" in real_model.lower() else 0.0,
        max_tokens=JUDGE_MAX_TOKENS,
    )

    t0 = time.perf_counter()
    try:
        resp = await provider.complete(judge_req, real_model)
    except Exception as exc:  # noqa: BLE001 — fail-soft
        elapsed = int((time.perf_counter() - t0) * 1000)
        logger.warning("judge.provider_call_failed", error=str(exc),
                       judge_model=judge_virtual_model)
        return JudgeError(error=f"{type(exc).__name__}: {exc}",
                          judge_model=judge_virtual_model,
                          prompt_version=JUDGE_PROMPT_VERSION,
                          judge_latency_ms=elapsed)

    elapsed = int((time.perf_counter() - t0) * 1000)
    text = (resp.choices[0].message.content or "") if resp.choices else ""

    try:
        data = _parse_judge_response(text)
        correctness = float(data.get("correctness", 0.0))
        completeness = float(data.get("completeness", 0.0))
        ifollow = float(data.get("instruction_following", 0.0))
        overall = float(data.get("overall",
                                 0.6 * correctness + 0.25 * completeness + 0.15 * ifollow))
        passed_raw = data.get("passed")
        passed = bool(passed_raw) if passed_raw is not None else overall >= 0.7
        reasoning = str(data.get("reasoning", ""))[:1000]
    except Exception as exc:  # noqa: BLE001
        logger.warning("judge.parse_failed", error=str(exc), raw_text=text[:200])
        return JudgeError(error=f"parse: {exc}",
                          judge_model=judge_virtual_model,
                          prompt_version=JUDGE_PROMPT_VERSION,
                          judge_latency_ms=elapsed)

    verdict = JudgeVerdict(
        correctness=_clamp(correctness),
        completeness=_clamp(completeness),
        instruction_following=_clamp(ifollow),
        overall=_clamp(overall),
        passed=passed,
        reasoning=reasoning,
        judge_model=judge_virtual_model,
        prompt_version=JUDGE_PROMPT_VERSION,
        judge_latency_ms=elapsed,
        cached=False,
        raw={"id": resp.id},
    )

    if persist:
        await _persist_verdict(
            query_hash=q_hash, response_hash=r_hash,
            judge_model=judge_virtual_model,
            prompt_version=JUDGE_PROMPT_VERSION,
            verdict=verdict,
        )

    return verdict


def _clamp(x: float) -> float:
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return x


async def judge_ensemble(
    *,
    query: str,
    response: str,
    expected: str | None = None,
    expected_category: str | None = None,
    judge_virtual_models: list[str],
    use_cache: bool = True,
    persist: bool = True,
) -> dict[str, Any]:
    """Run multiple judges in parallel and average their scores.

    Returns:
        {
          "verdicts": [{judge_model, verdict | error}, ...],
          "mean_overall": float | None,
          "mean_correctness": float | None,
          "passed_majority": bool | None,
        }

    Multi-judge mitigates the spec's documented self-preference bias —
    a single Opus judge tends to score Opus-family responses higher.
    """
    tasks = [
        judge_response(
            query=query, response=response,
            expected=expected, expected_category=expected_category,
            judge_virtual_model=m, use_cache=use_cache, persist=persist,
        )
        for m in judge_virtual_models
    ]
    results = await asyncio.gather(*tasks, return_exceptions=False)

    verdicts: list[JudgeVerdict] = [r for r in results if isinstance(r, JudgeVerdict)]

    by_model = []
    for m, r in zip(judge_virtual_models, results):
        if isinstance(r, JudgeVerdict):
            by_model.append({"judge_model": m, "verdict": r.to_dict()})
        else:
            by_model.append({"judge_model": m, "error": r.to_dict()})

    if not verdicts:
        return {
            "verdicts": by_model,
            "mean_overall": None,
            "mean_correctness": None,
            "passed_majority": None,
        }

    mean_overall = sum(v.overall for v in verdicts) / len(verdicts)
    mean_correctness = sum(v.correctness for v in verdicts) / len(verdicts)
    passed_majority = sum(1 for v in verdicts if v.passed) > (len(verdicts) // 2)
    return {
        "verdicts": by_model,
        "mean_overall": round(mean_overall, 4),
        "mean_correctness": round(mean_correctness, 4),
        "passed_majority": passed_majority,
        "n_judges": len(verdicts),
    }
