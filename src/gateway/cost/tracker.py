import uuid
from dataclasses import dataclass
from decimal import Decimal

import structlog

from gateway.cost.pricing import compute_cost
from gateway.db import acquire
from gateway.observability import metrics as obs_metrics

logger = structlog.get_logger(__name__)

_ZERO_UUID = uuid.UUID(int=0)


@dataclass
class RequestRecord:
    virtual_model: str
    provider: str
    model_used: str
    input_tokens: int
    output_tokens: int
    latency_ms: int
    status_code: int
    cached: bool = False
    fallback_used: bool = False
    error_message: str | None = None
    tenant_id: str = "default"
    api_key_id: uuid.UUID = _ZERO_UUID
    start_time: float = 0.0
    # Phase 1.5 sticky upward
    model_switched_from: str | None = None
    # Phase 1.6 auto-routing telemetry
    tier_fired: str | None = None
    routed_category: str | None = None
    conversation_id: str | None = None
    # Phase 2.3 — observability label only, not persisted to Postgres.
    routing_mode: str = "static"


async def log_request(record: RequestRecord) -> None:
    cost = (
        Decimal("0")
        if record.cached
        else compute_cost(record.model_used, record.input_tokens, record.output_tokens)
    )
    try:
        async with acquire() as conn:
            await conn.execute(
                """
                INSERT INTO requests (
                    tenant_id, api_key_id, virtual_model, provider, model_used,
                    input_tokens, output_tokens, cost_usd, latency_ms,
                    cached, fallback_used, status_code, error_message,
                    model_switched_from, tier_fired, routed_category, conversation_id
                ) VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14,$15,$16,$17)
                """,
                record.tenant_id,
                record.api_key_id,
                record.virtual_model,
                record.provider,
                record.model_used,
                record.input_tokens,
                record.output_tokens,
                cost,
                record.latency_ms,
                record.cached,
                record.fallback_used,
                record.status_code,
                record.error_message,
                record.model_switched_from,
                record.tier_fired,
                record.routed_category,
                record.conversation_id,
            )
    except Exception as exc:
        logger.error("tracker.log_failed", error=str(exc))

    obs_metrics.record_request(
        tenant=record.tenant_id,
        virtual_model=record.virtual_model,
        provider=record.provider,
        status_code=record.status_code,
        cached=record.cached,
        fallback=record.fallback_used,
        routing_mode=record.routing_mode,
        duration_seconds=record.latency_ms / 1000.0,
        model=record.model_used,
        input_tokens=record.input_tokens,
        output_tokens=record.output_tokens,
        cost_usd=float(cost),
    )
    if record.status_code >= 400:
        obs_metrics.record_provider_error(provider=record.provider, status_code=record.status_code)
