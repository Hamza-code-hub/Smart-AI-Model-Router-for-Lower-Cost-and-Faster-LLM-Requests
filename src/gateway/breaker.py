"""Per-(provider, model) circuit breaker.

Tripped when a target accumulates ``FAILURE_THRESHOLD`` failures within
``FAILURE_WINDOW_SECONDS``. While open, the failover loop skips that target
entirely without making an outbound call. After ``OPEN_DURATION_SECONDS`` the
breaker becomes half-open: the next attempt is allowed through; success
closes it, another failure re-opens it for the full duration.

In-memory, single-node only (matches the spec's v2 assumption). A multi-node
deployment would need Redis-backed state, but a hobby gateway doesn't.
"""

import time
from dataclasses import dataclass, field
from enum import Enum
from threading import Lock

import structlog

logger = structlog.get_logger(__name__)

FAILURE_THRESHOLD = 5
FAILURE_WINDOW_SECONDS = 30.0
OPEN_DURATION_SECONDS = 60.0


class State(str, Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


@dataclass
class _Entry:
    state: State = State.CLOSED
    failures: list[float] = field(default_factory=list)
    opened_at: float = 0.0


class CircuitBreaker:
    def __init__(self) -> None:
        self._entries: dict[tuple[str, str], _Entry] = {}
        self._lock = Lock()

    def _get(self, provider: str, model: str) -> _Entry:
        key = (provider, model)
        entry = self._entries.get(key)
        if entry is None:
            entry = _Entry()
            self._entries[key] = entry
        return entry

    def allow(self, provider: str, model: str) -> bool:
        """Returns True if a call to (provider, model) should be attempted.

        Transitions OPEN → HALF_OPEN once the cooldown has elapsed.
        """
        now = time.monotonic()
        with self._lock:
            entry = self._get(provider, model)
            if entry.state is State.OPEN:
                if now - entry.opened_at >= OPEN_DURATION_SECONDS:
                    entry.state = State.HALF_OPEN
                    logger.info("breaker.half_open", provider=provider, model=model)
                    return True
                return False
            return True

    def record_success(self, provider: str, model: str) -> None:
        with self._lock:
            entry = self._get(provider, model)
            if entry.state is not State.CLOSED:
                logger.info("breaker.closed", provider=provider, model=model)
            entry.state = State.CLOSED
            entry.failures.clear()
            entry.opened_at = 0.0

    def record_failure(self, provider: str, model: str) -> None:
        now = time.monotonic()
        with self._lock:
            entry = self._get(provider, model)
            if entry.state is State.HALF_OPEN:
                # A failure during half-open immediately re-opens the breaker.
                entry.state = State.OPEN
                entry.opened_at = now
                entry.failures = [now]
                logger.warning("breaker.reopened", provider=provider, model=model)
                return

            cutoff = now - FAILURE_WINDOW_SECONDS
            entry.failures = [t for t in entry.failures if t >= cutoff]
            entry.failures.append(now)

            if len(entry.failures) >= FAILURE_THRESHOLD and entry.state is State.CLOSED:
                entry.state = State.OPEN
                entry.opened_at = now
                logger.warning(
                    "breaker.opened",
                    provider=provider,
                    model=model,
                    failures_in_window=len(entry.failures),
                )

    def state_of(self, provider: str, model: str) -> State:
        with self._lock:
            return self._get(provider, model).state


_breaker = CircuitBreaker()


def get_breaker() -> CircuitBreaker:
    return _breaker
