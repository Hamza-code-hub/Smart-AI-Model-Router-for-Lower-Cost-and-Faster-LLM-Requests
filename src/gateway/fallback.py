"""Cross-provider failover.

When a provider call fails with a transient/availability error, retry the
request against a model on a *different* provider, as declared by the route's
``fallback`` block in ``routes.yaml``.
"""

from gateway.providers.base import ProviderError
from gateway.routing.resolver import PrimaryRoute, RouteConfig

# Errors that another provider has a real chance of serving successfully:
# rate limits, timeouts, upstream/server failures, and missing-model 404s.
# 404 is included because a deprecated/retired model on the primary provider
# is exactly what the route's `fallback` block is meant to cover. Deterministic
# client errors (400 bad request, 401/403 auth) still bypass failover since
# they would fail identically on any provider.
FAILOVER_STATUS_CODES = frozenset({404, 408, 425, 429, 500, 502, 503, 504})


def should_failover(exc: ProviderError) -> bool:
    return exc.status_code in FAILOVER_STATUS_CODES


def attempts(route: RouteConfig) -> list[PrimaryRoute]:
    """Ordered list of (provider, model) targets to try: primary, then fallback."""
    targets = [route.primary]
    if route.fallback is not None:
        targets.append(route.fallback)
    return targets
