"""Phase 2.3 — Prometheus metrics.

All metric names follow the `gateway_*` namespace from spec v2.6, plus the
router-specific `gateway_routing_*` series called out in
`research/phase_2_3_observability.md`.

The recorder helpers wrap counter/histogram/gauge increments so the hot path
(`routes/chat.py`, `routing/classifier.py`, `breaker.py`, …) never has to know
which collector or label set to reach for. Fail-open semantics: any metric
write that raises is swallowed in the recorder helper — observability
infrastructure must not be able to drop traffic.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from prometheus_client import (
    CONTENT_TYPE_LATEST,
    REGISTRY,
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)

if TYPE_CHECKING:
    from gateway.breaker import CircuitBreaker

# ---------------------------------------------------------------------------
# Collectors
# ---------------------------------------------------------------------------

# Latency histogram buckets. The gateway sits in front of LLM providers, so
# the interesting range is roughly 10ms (cached / heuristic-tier) to 60s
# (slow Opus generation). Default Prometheus buckets stop at 10s.
_REQUEST_BUCKETS = (
    0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0,
)
# Routing budget is ~100ms p95, so the histogram needs fine resolution there.
_ROUTING_BUCKETS = (
    0.001, 0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.25, 0.5, 1.0, 2.5, 5.0,
)


REQUEST_COUNT = Counter(
    "gateway_requests_total",
    "Chat completion requests handled by the gateway.",
    labelnames=("tenant", "virtual_model", "provider", "status", "cached", "fallback", "routing_mode"),
)

REQUEST_DURATION = Histogram(
    "gateway_request_duration_seconds",
    "End-to-end chat completion request latency.",
    labelnames=("virtual_model", "provider", "model", "routing_mode"),
    buckets=_REQUEST_BUCKETS,
)

CACHE_HITS = Counter(
    "gateway_cache_hits_total",
    "Cache hits by cache type (exact, semantic).",
    labelnames=("cache_type", "virtual_model"),
)

COST_USD = Counter(
    "gateway_cost_usd_total",
    "Cumulative provider cost in USD.",
    labelnames=("tenant", "model"),
)

TOKENS = Counter(
    "gateway_tokens_total",
    "Token counts by direction (input/output).",
    labelnames=("direction", "model"),
)

PROVIDER_ERRORS = Counter(
    "gateway_provider_errors_total",
    "Provider errors classified by HTTP-ish error type.",
    labelnames=("provider", "error_type"),
)

CIRCUIT_BREAKER_STATE = Gauge(
    "gateway_circuit_breaker_state",
    "Circuit breaker state: 0=closed, 1=half_open, 2=open.",
    labelnames=("provider", "model"),
)

# Router-specific (Phase 2.3) -------------------------------------------------

ROUTING_TIER_FIRED = Counter(
    "gateway_routing_tier_fired_total",
    "Cascade tier that produced the routing decision.",
    labelnames=("tier", "routing_mode"),
)

ROUTING_CATEGORY = Counter(
    "gateway_routing_category_total",
    "Routing decisions by final category, labelled with the tier that fired.",
    labelnames=("category", "tier", "routing_mode"),
)

ROUTING_ESCALATIONS = Counter(
    "gateway_routing_escalations_total",
    "Sticky-upward escalations: a turn moved the conversation floor higher.",
    labelnames=("from_category", "to_category"),
)

ROUTING_DECISION_SECONDS = Histogram(
    "gateway_routing_decision_seconds",
    "Latency of classify() — total cascade time.",
    labelnames=("tier",),
    buckets=_ROUTING_BUCKETS,
)

ROUTING_POLARITY_FLIPS = Counter(
    "gateway_routing_polarity_flips_total",
    "Tier-2 embedding hits rejected because the query and matched exemplar had opposite polarity.",
)

MODEL_MISSING_FAILOVER = Counter(
    "gateway_model_missing_failover_total",
    "Failover events triggered by upstream returning 404 model_not_found. "
    "If this is non-zero for a (provider, model) pair, routes.yaml is out of "
    "sync with that provider's catalog.",
    labelnames=("primary_provider", "primary_model", "fallback_provider", "fallback_model"),
)


# ---------------------------------------------------------------------------
# Breaker-state collector — read on scrape rather than written on every event.
# ---------------------------------------------------------------------------

_breaker_ref: "CircuitBreaker | None" = None

_BREAKER_STATE_VALUE = {
    "closed": 0,
    "half_open": 1,
    "open": 2,
}


def attach_breaker(breaker: "CircuitBreaker") -> None:
    global _breaker_ref
    _breaker_ref = breaker


def _refresh_breaker_gauge() -> None:
    if _breaker_ref is None:
        return
    # Snapshot under the breaker's own lock — _entries is not thread-safe.
    with _breaker_ref._lock:  # noqa: SLF001
        snapshot = {(p, m): entry.state.value for (p, m), entry in _breaker_ref._entries.items()}  # noqa: SLF001
    for (provider, model), state_name in snapshot.items():
        CIRCUIT_BREAKER_STATE.labels(provider=provider, model=model).set(
            _BREAKER_STATE_VALUE.get(state_name, 0)
        )


# ---------------------------------------------------------------------------
# Recorder helpers (fail-open)
# ---------------------------------------------------------------------------

def _silent(fn):  # type: ignore[no-untyped-def]
    def wrapper(*args, **kwargs):  # type: ignore[no-untyped-def]
        try:
            return fn(*args, **kwargs)
        except Exception:
            return None
    return wrapper


@_silent
def record_request(
    *,
    tenant: str,
    virtual_model: str,
    provider: str,
    status_code: int,
    cached: bool,
    fallback: bool,
    routing_mode: str,
    duration_seconds: float,
    model: str,
    input_tokens: int,
    output_tokens: int,
    cost_usd: float,
) -> None:
    status_class = f"{status_code // 100}xx"
    REQUEST_COUNT.labels(
        tenant=tenant,
        virtual_model=virtual_model,
        provider=provider,
        status=status_class,
        cached=str(cached).lower(),
        fallback=str(fallback).lower(),
        routing_mode=routing_mode,
    ).inc()
    REQUEST_DURATION.labels(
        virtual_model=virtual_model,
        provider=provider,
        model=model,
        routing_mode=routing_mode,
    ).observe(duration_seconds)
    if input_tokens:
        TOKENS.labels(direction="input", model=model).inc(input_tokens)
    if output_tokens:
        TOKENS.labels(direction="output", model=model).inc(output_tokens)
    if cost_usd:
        COST_USD.labels(tenant=tenant, model=model).inc(cost_usd)


@_silent
def record_cache_hit(*, cache_type: str, virtual_model: str) -> None:
    CACHE_HITS.labels(cache_type=cache_type, virtual_model=virtual_model).inc()


@_silent
def record_provider_error(*, provider: str, status_code: int) -> None:
    if status_code == 408:
        kind = "timeout"
    elif status_code == 429:
        kind = "rate_limit"
    elif 500 <= status_code < 600:
        kind = "server"
    elif 400 <= status_code < 500:
        kind = "client"
    else:
        kind = "other"
    PROVIDER_ERRORS.labels(provider=provider, error_type=kind).inc()


@_silent
def record_routing_decision(*, tier: str, category: str, routing_mode: str, duration_seconds: float) -> None:
    ROUTING_TIER_FIRED.labels(tier=tier, routing_mode=routing_mode).inc()
    ROUTING_CATEGORY.labels(category=category, tier=tier, routing_mode=routing_mode).inc()
    ROUTING_DECISION_SECONDS.labels(tier=tier).observe(duration_seconds)


@_silent
def record_polarity_flip() -> None:
    ROUTING_POLARITY_FLIPS.inc()


@_silent
def record_escalation(*, from_category: str, to_category: str) -> None:
    ROUTING_ESCALATIONS.labels(from_category=from_category, to_category=to_category).inc()


@_silent
def record_model_missing_failover(
    *,
    primary_provider: str,
    primary_model: str,
    fallback_provider: str,
    fallback_model: str,
) -> None:
    MODEL_MISSING_FAILOVER.labels(
        primary_provider=primary_provider,
        primary_model=primary_model,
        fallback_provider=fallback_provider,
        fallback_model=fallback_model,
    ).inc()


# ---------------------------------------------------------------------------
# Exposition
# ---------------------------------------------------------------------------

def render_latest() -> tuple[bytes, str]:
    """Snapshot the registry into the Prometheus text format."""
    _refresh_breaker_gauge()
    return generate_latest(REGISTRY), CONTENT_TYPE_LATEST


__all__ = [
    "CONTENT_TYPE_LATEST",
    "CollectorRegistry",
    "attach_breaker",
    "record_cache_hit",
    "record_escalation",
    "record_model_missing_failover",
    "record_polarity_flip",
    "record_provider_error",
    "record_request",
    "record_routing_decision",
    "render_latest",
]
