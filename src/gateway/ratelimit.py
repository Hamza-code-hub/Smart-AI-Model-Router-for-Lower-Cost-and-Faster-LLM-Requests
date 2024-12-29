"""Per-tenant rate limiting and budget enforcement.

- **Rate limit**: Redis token bucket keyed by tenant. Refilled continuously at
  ``rate_limit_rpm`` requests/minute, capped at the same value (1-minute burst
  capacity). Pure async, single round-trip per check.
- **Daily token budget**: sum of ``input_tokens + output_tokens`` from the
  ``requests`` table since UTC midnight.
- **Monthly cost cap**: sum of ``cost_usd`` from the ``requests`` table since
  the 1st of the month (UTC).

Budget checks query Postgres; for hot tenants they could be cached, but a
hobby gateway doesn't move the needle there.
"""

import time
from dataclasses import dataclass
from datetime import datetime, timezone

import structlog
from fastapi import HTTPException

from gateway.auth import TenantInfo, get_tenant_limits
from gateway.db import acquire
from gateway.redis_client import get_redis

logger = structlog.get_logger(__name__)


@dataclass(frozen=True)
class LimitDecision:
    allowed: bool
    retry_after_seconds: int = 0
    reason: str = ""
    status_code: int = 200


# ---- token bucket -----------------------------------------------------------

_BUCKET_KEY = "ratelimit:tenant:{tenant_id}"


async def _check_rate_limit(tenant_id: str, rpm: int) -> LimitDecision:
    """Atomic token bucket via Redis Lua. One Redis round-trip per check.

    Stores two fields per tenant: current token count and last-refill timestamp.
    Refills proportionally to elapsed time, up to ``rpm`` (1-minute burst).
    """
    if rpm <= 0:
        return LimitDecision(allowed=True)

    redis = get_redis()
    now_ms = int(time.time() * 1000)
    refill_rate_per_ms = rpm / 60_000.0  # tokens per millisecond
    key = _BUCKET_KEY.format(tenant_id=tenant_id)

    lua = """
    local key = KEYS[1]
    local capacity = tonumber(ARGV[1])
    local refill_per_ms = tonumber(ARGV[2])
    local now_ms = tonumber(ARGV[3])

    local bucket = redis.call('HMGET', key, 'tokens', 'last_ms')
    local tokens = tonumber(bucket[1])
    local last_ms = tonumber(bucket[2])

    if tokens == nil then
        tokens = capacity
        last_ms = now_ms
    else
        local delta = math.max(0, now_ms - last_ms)
        tokens = math.min(capacity, tokens + delta * refill_per_ms)
        last_ms = now_ms
    end

    local allowed = 0
    local retry_after_ms = 0
    if tokens >= 1 then
        tokens = tokens - 1
        allowed = 1
    else
        retry_after_ms = math.ceil((1 - tokens) / refill_per_ms)
    end

    redis.call('HMSET', key, 'tokens', tokens, 'last_ms', last_ms)
    redis.call('EXPIRE', key, 120)
    return {allowed, retry_after_ms}
    """

    try:
        allowed_flag, retry_after_ms = await redis.eval(
            lua, 1, key, rpm, refill_rate_per_ms, now_ms
        )
    except Exception as exc:
        # Fail open: rate-limit infra failure shouldn't drop traffic.
        logger.warning("ratelimit.redis_failure", error=str(exc))
        return LimitDecision(allowed=True)

    if int(allowed_flag) == 1:
        return LimitDecision(allowed=True)
    retry_after = max(1, int(retry_after_ms) // 1000 + 1)
    return LimitDecision(
        allowed=False,
        retry_after_seconds=retry_after,
        reason=f"Rate limit {rpm}/min exceeded",
        status_code=429,
    )


# ---- budgets ----------------------------------------------------------------

async def _check_budgets(
    tenant_id: str,
    daily_token_budget: int | None,
    monthly_cost_cap_usd: float | None,
) -> LimitDecision:
    if daily_token_budget is None and monthly_cost_cap_usd is None:
        return LimitDecision(allowed=True)

    now = datetime.now(timezone.utc)
    day_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    month_start = day_start.replace(day=1)

    try:
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
    except Exception as exc:
        logger.warning("ratelimit.budget_query_failed", error=str(exc))
        return LimitDecision(allowed=True)

    day_tokens = int(row["day_tokens"] or 0)
    month_cost = float(row["month_cost"] or 0)

    if daily_token_budget is not None and day_tokens >= daily_token_budget:
        return LimitDecision(
            allowed=False,
            status_code=402,
            reason=f"Daily token budget exceeded ({day_tokens}/{daily_token_budget})",
        )
    if monthly_cost_cap_usd is not None and month_cost >= monthly_cost_cap_usd:
        return LimitDecision(
            allowed=False,
            status_code=402,
            reason=f"Monthly cost cap exceeded (${month_cost:.4f}/${monthly_cost_cap_usd:.2f})",
        )
    return LimitDecision(allowed=True)


# ---- public dependency ------------------------------------------------------

async def enforce_limits(tenant: TenantInfo) -> None:
    """FastAPI dependency: raises 429 / 402 / 503 if a limit is hit, else returns."""
    limits = get_tenant_limits(tenant.tenant_id)
    if limits is None:
        return

    if limits.rate_limit_rpm:
        decision = await _check_rate_limit(tenant.tenant_id, limits.rate_limit_rpm)
        if not decision.allowed:
            headers = {"Retry-After": str(decision.retry_after_seconds)}
            raise HTTPException(
                status_code=decision.status_code,
                detail=decision.reason,
                headers=headers,
            )

    decision = await _check_budgets(
        tenant.tenant_id,
        limits.daily_token_budget,
        limits.monthly_cost_cap_usd,
    )
    if not decision.allowed:
        raise HTTPException(status_code=decision.status_code, detail=decision.reason)
