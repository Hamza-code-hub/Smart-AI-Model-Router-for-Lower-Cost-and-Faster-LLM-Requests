from datetime import date, datetime, timezone
from typing import Any

import structlog
from fastapi import APIRouter, Depends, HTTPException, Query

from gateway.auth import get_tenant_limits, require_admin
from gateway.breaker import get_breaker
from gateway.compaction.strategy import delete_state
from gateway.db import acquire

logger = structlog.get_logger(__name__)
router = APIRouter()


@router.get("/usage/{tenant_id}", dependencies=[Depends(require_admin)])
async def get_usage(
    tenant_id: str,
    start_date: date | None = Query(default=None),
    end_date: date | None = Query(default=None),
) -> dict[str, Any]:
    conditions = ["tenant_id = $1"]
    params: list[Any] = [tenant_id]

    if start_date:
        params.append(start_date)
        conditions.append(f"created_at >= ${len(params)}")
    if end_date:
        params.append(end_date)
        conditions.append(f"created_at < ${len(params)} + interval '1 day'")

    where = " AND ".join(conditions)

    async with acquire() as conn:
        totals = await conn.fetchrow(
            f"""
            SELECT
                COUNT(*)            AS total_requests,
                SUM(input_tokens)   AS total_input_tokens,
                SUM(output_tokens)  AS total_output_tokens,
                SUM(cost_usd)       AS total_cost_usd
            FROM requests
            WHERE {where}
            """,
            *params,
        )

        by_model = await conn.fetch(
            f"""
            SELECT
                virtual_model,
                COUNT(*)            AS requests,
                SUM(input_tokens)   AS input_tokens,
                SUM(output_tokens)  AS output_tokens,
                SUM(cost_usd)       AS cost_usd
            FROM requests
            WHERE {where}
            GROUP BY virtual_model
            ORDER BY cost_usd DESC
            """,
            *params,
        )

    return {
        "tenant_id": tenant_id,
        "totals": {
            "requests": totals["total_requests"],
            "input_tokens": totals["total_input_tokens"] or 0,
            "output_tokens": totals["total_output_tokens"] or 0,
            "cost_usd": float(totals["total_cost_usd"] or 0),
        },
        "by_model": [
            {
                "virtual_model": row["virtual_model"],
                "requests": row["requests"],
                "input_tokens": row["input_tokens"],
                "output_tokens": row["output_tokens"],
                "cost_usd": float(row["cost_usd"]),
            }
            for row in by_model
        ],
    }


@router.get("/breakers", dependencies=[Depends(require_admin)])
async def list_breakers() -> dict[str, Any]:
    breaker = get_breaker()
    return {
        "breakers": [
            {"provider": p, "model": m, "state": entry.state.value, "failures": len(entry.failures)}
            for (p, m), entry in breaker._entries.items()  # noqa: SLF001
        ]
    }


@router.delete("/conversations/{conversation_id}", dependencies=[Depends(require_admin)])
async def delete_conversation(conversation_id: str) -> dict[str, Any]:
    deleted = await delete_state(conversation_id)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Conversation {conversation_id!r} not found")
    return {"deleted": True, "conversation_id": conversation_id}


@router.get("/routing/stats", dependencies=[Depends(require_admin)])
async def routing_stats() -> dict[str, Any]:
    """Aggregates over the last 24h of auto-routed requests for cascade telemetry."""
    async with acquire() as conn:
        per_tier = await conn.fetch(
            """
            SELECT tier_fired,
                   COUNT(*)                AS n,
                   AVG(latency_ms)::int    AS mean_latency_ms,
                   PERCENTILE_CONT(0.5)  WITHIN GROUP (ORDER BY latency_ms) AS p50_latency_ms,
                   PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY latency_ms) AS p95_latency_ms
            FROM requests
            WHERE virtual_model = 'auto'
              AND tier_fired IS NOT NULL
              AND created_at >= NOW() - INTERVAL '24 hours'
            GROUP BY tier_fired
            ORDER BY n DESC
            """
        )
        per_category = await conn.fetch(
            """
            SELECT routed_category,
                   COUNT(*) AS n
            FROM requests
            WHERE virtual_model = 'auto'
              AND routed_category IS NOT NULL
              AND created_at >= NOW() - INTERVAL '24 hours'
            GROUP BY routed_category
            ORDER BY n DESC
            """
        )
        totals = await conn.fetchrow(
            """
            SELECT
              COUNT(*)                                                      AS total_auto_requests,
              COUNT(DISTINCT conversation_id) FILTER (WHERE conversation_id IS NOT NULL)
                                                                            AS total_conversations,
              COUNT(DISTINCT conversation_id) FILTER (WHERE model_switched_from IS NOT NULL)
                                                                            AS conversations_with_escalation
            FROM requests
            WHERE virtual_model = 'auto'
              AND created_at >= NOW() - INTERVAL '24 hours'
            """
        )

    total_convs = totals["total_conversations"] or 0
    escalated_convs = totals["conversations_with_escalation"] or 0
    return {
        "window": "last_24h",
        "total_auto_requests": totals["total_auto_requests"] or 0,
        "total_conversations": total_convs,
        "conversations_with_escalation": escalated_convs,
        "escalation_rate": (escalated_convs / total_convs) if total_convs else 0.0,
        "per_tier": [
            {
                "tier": row["tier_fired"],
                "n": row["n"],
                "mean_latency_ms": row["mean_latency_ms"],
                "p50_latency_ms": float(row["p50_latency_ms"]) if row["p50_latency_ms"] is not None else None,
                "p95_latency_ms": float(row["p95_latency_ms"]) if row["p95_latency_ms"] is not None else None,
            }
            for row in per_tier
        ],
        "per_category": [
            {"category": row["routed_category"], "n": row["n"]}
            for row in per_category
        ],
    }


@router.get("/judge/flags", dependencies=[Depends(require_admin)])
async def list_judge_flags(
    limit: int = Query(default=50, ge=1, le=500),
    tenant_id: str | None = Query(default=None),
) -> dict[str, Any]:
    """Phase 2.5 — review queue for the production 1% sampling backstop.

    Each row is one sampled response the judge marked low-quality or whose
    routed category disagrees with an expected category. Ordered newest first.
    """
    params: list[Any] = [limit]
    where = ""
    if tenant_id:
        params.append(tenant_id)
        where = "WHERE f.tenant_id = $2"
    async with acquire() as conn:
        rows = await conn.fetch(
            f"""
            SELECT
                f.id, f.created_at, f.tenant_id, f.virtual_model,
                f.routed_category, f.tier_fired, f.flag_reason,
                f.query_preview, f.response_preview,
                v.judge_model, v.prompt_version, v.overall_score,
                v.passed, v.reasoning, v.scores
            FROM judge_flags f
            JOIN judge_verdicts v ON v.id = f.verdict_id
            {where}
            ORDER BY f.created_at DESC
            LIMIT $1
            """,
            *params,
        )
    return {
        "n": len(rows),
        "flags": [
            {
                "id": str(r["id"]),
                "created_at": r["created_at"].isoformat(),
                "tenant_id": r["tenant_id"],
                "virtual_model": r["virtual_model"],
                "routed_category": r["routed_category"],
                "tier_fired": r["tier_fired"],
                "flag_reason": r["flag_reason"],
                "query_preview": r["query_preview"],
                "response_preview": r["response_preview"],
                "judge_model": r["judge_model"],
                "prompt_version": r["prompt_version"],
                "overall_score": float(r["overall_score"]),
                "passed": r["passed"],
                "reasoning": r["reasoning"],
                "scores": r["scores"],
            }
            for r in rows
        ],
    }


@router.get("/judge/verdicts", dependencies=[Depends(require_admin)])
async def list_judge_verdicts(
    limit: int = Query(default=50, ge=1, le=500),
) -> dict[str, Any]:
    """Most recent judge verdicts (whether flagged or not) for spot inspection."""
    async with acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT id, created_at, judge_model, prompt_version,
                   overall_score, passed, scores, reasoning,
                   judge_latency_ms
            FROM judge_verdicts
            ORDER BY created_at DESC
            LIMIT $1
            """,
            limit,
        )
    return {
        "n": len(rows),
        "verdicts": [
            {
                "id": str(r["id"]),
                "created_at": r["created_at"].isoformat(),
                "judge_model": r["judge_model"],
                "prompt_version": r["prompt_version"],
                "overall_score": float(r["overall_score"]),
                "passed": r["passed"],
                "scores": r["scores"],
                "reasoning": r["reasoning"],
                "judge_latency_ms": r["judge_latency_ms"],
            }
            for r in rows
        ],
    }


@router.get("/limits/{tenant_id}", dependencies=[Depends(require_admin)])
async def get_limits(tenant_id: str) -> dict[str, Any]:
    limits = get_tenant_limits(tenant_id)
    if limits is None:
        raise HTTPException(status_code=404, detail=f"Tenant {tenant_id!r} not found")

    now = datetime.now(timezone.utc)
    day_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    month_start = day_start.replace(day=1)

    async with acquire() as conn:
        row = await conn.fetchrow(
            """
            SELECT
                COALESCE(SUM(CASE WHEN created_at >= $2
                                  THEN input_tokens + output_tokens
                                  ELSE 0 END), 0) AS day_tokens,
                COALESCE(SUM(CASE WHEN created_at >= $3
                                  THEN cost_usd
                                  ELSE 0 END), 0) AS month_cost
            FROM requests
            WHERE tenant_id = $1 AND created_at >= $3
            """,
            tenant_id,
            day_start,
            month_start,
        )

    return {
        "tenant_id": tenant_id,
        "limits": {
            "rate_limit_rpm": limits.rate_limit_rpm,
            "daily_token_budget": limits.daily_token_budget,
            "monthly_cost_cap_usd": limits.monthly_cost_cap_usd,
        },
        "current_period": {
            "day_tokens": int(row["day_tokens"] or 0),
            "month_cost_usd": float(row["month_cost"] or 0),
        },
    }
