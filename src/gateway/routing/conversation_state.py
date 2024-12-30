"""Phase 1.5 — conversation router state + sticky upward escalation.

Per-conversation routing state lives in Redis (HSET `router:conv:{id}`, 24h
TTL). This is separate from v2.5 `conversation_compaction` (Postgres), which
holds *content* state (compaction summary + sticky facts). Router state tracks the
*routing* high-water-mark plus a counter that prevents ping-pong on borderline
turns.

Sticky upward rules (see research/phase_1_5_sticky_routing.md):
  - Quality tiers ordered: fast-qa < code < deep-reasoning
  - Once escalated, stay at the high-water-mark — never downgrade
  - long-context is **orthogonal** — handled per-turn, doesn't touch the floor
  - escalation_count capped at MAX_ESCALATIONS (default 2), +1 when v2.5
    compaction fired on the previous turn (the summary is cheap to re-ingest)
  - Below HISTORY_BIG_THRESHOLD tokens any one-step gap can escalate; above it
    require a two-step gap unless compaction just fired (the "switch tax"
    bound — re-ingesting 20k tokens at Opus pricing dominates the routing win)
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import structlog

from gateway.db import acquire
from gateway.observability import metrics as obs_metrics
from gateway.redis_client import get_redis

logger = structlog.get_logger(__name__)

REDIS_KEY_PREFIX = "router:conv:"
REDIS_TTL_SECONDS = 86_400

MAX_ESCALATIONS = 2
HISTORY_BIG_THRESHOLD = 20_000  # tokens; aligns with v2.5 compaction default

QUALITY_ORDER: dict[str, int] = {
    "fast-qa": 0,
    "code": 1,
    "deep-reasoning": 2,
}
LONG_CONTEXT = "long-context"


@dataclass
class ConvState:
    conv_id: str
    floor_category: str | None
    escalation_count: int
    last_tier_fired: str | None
    last_compaction_seen_at: str | None
    compaction_just_fired: bool


@dataclass
class EscalationOutcome:
    final_category: str
    switched_from: str | None  # set only when this turn moved the floor upward


def _redis_key(conv_id: str) -> str:
    return f"{REDIS_KEY_PREFIX}{conv_id}"


async def _latest_compaction_at(conv_id: str) -> str | None:
    """Read the v2.5 compaction timestamp for this conversation. Fail-soft."""
    try:
        async with acquire() as conn:
            row = await conn.fetchrow(
                "SELECT updated_at FROM conversation_compaction WHERE conversation_id = $1",
                conv_id,
            )
        if row is None:
            return None
        ts = row["updated_at"]
        if isinstance(ts, datetime):
            return ts.isoformat()
        return str(ts)
    except Exception as exc:
        logger.warning(
            "router.conv_state.compaction_lookup_fail",
            error=str(exc),
            conv_id=conv_id,
        )
        return None


async def get_state(conv_id: str) -> ConvState:
    """Load router state for a conversation. Always returns a ConvState — new
    convs come back with `floor_category=None`, `escalation_count=0`."""
    redis = get_redis()
    try:
        raw = await redis.hgetall(_redis_key(conv_id)) or {}
    except Exception as exc:
        logger.warning("router.conv_state.redis_read_fail", error=str(exc), conv_id=conv_id)
        raw = {}

    floor = raw.get("floor_category") or None
    escalation_count = int(raw.get("escalation_count", 0) or 0)
    last_tier = raw.get("last_tier_fired") or None
    last_compaction_seen = raw.get("last_compaction_seen_at") or None

    latest_compaction = await _latest_compaction_at(conv_id)
    compaction_just_fired = bool(
        latest_compaction
        and (last_compaction_seen is None or latest_compaction > last_compaction_seen)
    )

    return ConvState(
        conv_id=conv_id,
        floor_category=floor,
        escalation_count=escalation_count,
        last_tier_fired=last_tier,
        last_compaction_seen_at=latest_compaction or last_compaction_seen,
        compaction_just_fired=compaction_just_fired,
    )


def _gap(from_cat: str, to_cat: str) -> int:
    return QUALITY_ORDER[to_cat] - QUALITY_ORDER[from_cat]


def _escalation_allowed(
    gap: int,
    history_tokens: int,
    compaction_just_fired: bool,
    escalations_used: int,
) -> bool:
    effective_cap = MAX_ESCALATIONS + (1 if compaction_just_fired else 0)
    if escalations_used >= effective_cap:
        return False
    if history_tokens < HISTORY_BIG_THRESHOLD:
        return gap >= 1
    if compaction_just_fired:
        return gap >= 1
    return gap >= 2


def apply_sticky(
    new_category: str,
    history_token_estimate: int,
    state: ConvState,
) -> EscalationOutcome:
    """Pure decision step — no I/O. Apply sticky-upward to the cascade's raw
    category, return the final category to route to + the switch source."""
    # long-context is orthogonal: pass through, leave the quality floor alone.
    if new_category == LONG_CONTEXT:
        return EscalationOutcome(final_category=LONG_CONTEXT, switched_from=None)

    # First quality-tier turn — seed the floor.
    if state.floor_category is None or state.floor_category not in QUALITY_ORDER:
        return EscalationOutcome(final_category=new_category, switched_from=None)

    floor = state.floor_category
    if new_category not in QUALITY_ORDER:
        return EscalationOutcome(final_category=floor, switched_from=None)

    gap = _gap(floor, new_category)
    if gap <= 0:
        # Never downgrade.
        return EscalationOutcome(final_category=floor, switched_from=None)

    if _escalation_allowed(
        gap=gap,
        history_tokens=history_token_estimate,
        compaction_just_fired=state.compaction_just_fired,
        escalations_used=state.escalation_count,
    ):
        return EscalationOutcome(final_category=new_category, switched_from=floor)
    return EscalationOutcome(final_category=floor, switched_from=None)


async def record_decision(
    conv_id: str,
    final_category: str,
    tier_fired: str,
    state: ConvState,
    outcome: EscalationOutcome,
) -> None:
    """Persist the post-decision state. Always called after the provider call
    succeeds — we don't move the floor on a failed request."""
    new_floor = state.floor_category
    if final_category in QUALITY_ORDER:
        if new_floor not in QUALITY_ORDER:
            new_floor = final_category
        elif _gap(new_floor, final_category) > 0:
            new_floor = final_category
    # If this turn was long-context, leave the existing quality floor untouched.
    if new_floor is None:
        new_floor = final_category

    new_count = state.escalation_count + (1 if outcome.switched_from is not None else 0)

    mapping = {
        "floor_category": new_floor,
        "escalation_count": str(new_count),
        "last_tier_fired": tier_fired,
    }
    if state.last_compaction_seen_at:
        mapping["last_compaction_seen_at"] = state.last_compaction_seen_at

    redis = get_redis()
    key = _redis_key(conv_id)
    try:
        await redis.hset(key, mapping=mapping)
        await redis.expire(key, REDIS_TTL_SECONDS)
    except Exception as exc:
        logger.warning("router.conv_state.write_fail", error=str(exc), conv_id=conv_id)

    if outcome.switched_from is not None:
        obs_metrics.record_escalation(
            from_category=outcome.switched_from,
            to_category=final_category,
        )


def estimate_history_tokens(char_total: int) -> int:
    """~4 chars per token — mirrors the v2.5 compaction estimate so the thresholds
    here and there are talking about the same scale."""
    return char_total // 4
