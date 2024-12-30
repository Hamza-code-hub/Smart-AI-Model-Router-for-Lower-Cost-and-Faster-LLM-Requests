"""Phase 2.3 — observability surface (metrics + tracing).

Re-exports the two helper modules so callers can `from gateway.observability
import metrics, tracing` and stay decoupled from the wiring.
"""
from gateway.observability import metrics, tracing

__all__ = ["metrics", "tracing"]
