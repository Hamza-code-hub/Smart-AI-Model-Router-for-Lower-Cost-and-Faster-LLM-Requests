"""Phase 2.3 — OpenTelemetry tracing.

A thin wrapper that:
  - Initializes a `TracerProvider` only when `OTEL_EXPORTER_OTLP_ENDPOINT`
    is set (so the gateway runs fine without a collector — important for
    local dev and the hobby-project default).
  - Instruments FastAPI + httpx so every inbound request and outbound
    provider call already get spans for free.
  - Exposes `span(name, **attrs)` and `async_span(name, **attrs)` context
    managers for our own custom spans (cache lookup, routing.classify
    per-tier sub-spans, sticky escalation, provider call, compaction,
    fallback).

When tracing isn't initialized the helpers fall through to no-op spans, so
the hot path doesn't need to branch.
"""
from __future__ import annotations

import os
from contextlib import asynccontextmanager, contextmanager
from typing import Any, AsyncIterator, Iterator

import structlog
from opentelemetry import trace
from opentelemetry.trace import Span, Tracer
from opentelemetry.trace.status import Status, StatusCode

logger = structlog.get_logger(__name__)

_TRACER_NAME = "gateway"
_initialized = False


def init(service_name: str = "llm-gateway", service_version: str = "0.1.0") -> None:
    """Initialize the global tracer provider.

    Only sets things up when `OTEL_EXPORTER_OTLP_ENDPOINT` is present — that
    keeps the binary import-safe in environments without a collector. Safe
    to call multiple times (idempotent).
    """
    global _initialized
    if _initialized:
        return

    endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT") or os.getenv(
        "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT"
    )
    if not endpoint:
        logger.info("tracing.disabled", reason="no OTEL_EXPORTER_OTLP_ENDPOINT")
        _initialized = True
        return

    # Lazy-import the SDK so the deps are only resolved when actually used.
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
    from opentelemetry.sdk.resources import SERVICE_NAME, SERVICE_VERSION, Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor

    resource = Resource.create(
        {SERVICE_NAME: service_name, SERVICE_VERSION: service_version}
    )
    provider = TracerProvider(resource=resource)
    provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
    trace.set_tracer_provider(provider)
    _initialized = True
    logger.info("tracing.initialized", endpoint=endpoint)


def instrument_app(app: Any) -> None:
    """Auto-instrument FastAPI + httpx. No-op if tracing was not initialized
    with an exporter (still gives no-op spans, no overhead)."""
    try:
        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

        FastAPIInstrumentor.instrument_app(app)
    except Exception as exc:
        logger.warning("tracing.instrument_fastapi_failed", error=str(exc))

    try:
        from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor

        HTTPXClientInstrumentor().instrument()
    except Exception as exc:
        logger.warning("tracing.instrument_httpx_failed", error=str(exc))


def _tracer() -> Tracer:
    return trace.get_tracer(_TRACER_NAME)


def _apply_attrs(span: Span, attrs: dict[str, Any]) -> None:
    for k, v in attrs.items():
        if v is None:
            continue
        if isinstance(v, (bool, int, float, str)):
            span.set_attribute(k, v)
        else:
            span.set_attribute(k, str(v))


@contextmanager
def span(name: str, **attrs: Any) -> Iterator[Span]:
    with _tracer().start_as_current_span(name) as s:
        _apply_attrs(s, attrs)
        try:
            yield s
        except Exception as exc:
            s.set_status(Status(StatusCode.ERROR, str(exc)))
            s.record_exception(exc)
            raise


@asynccontextmanager
async def async_span(name: str, **attrs: Any) -> AsyncIterator[Span]:
    with _tracer().start_as_current_span(name) as s:
        _apply_attrs(s, attrs)
        try:
            yield s
        except Exception as exc:
            s.set_status(Status(StatusCode.ERROR, str(exc)))
            s.record_exception(exc)
            raise


def set_attrs(**attrs: Any) -> None:
    """Apply attributes to whichever span is currently active. Useful when
    the value (e.g. tokens used, routing tier) is only known partway through."""
    current = trace.get_current_span()
    if current and current.is_recording():
        _apply_attrs(current, attrs)


__all__ = ["async_span", "init", "instrument_app", "set_attrs", "span"]
