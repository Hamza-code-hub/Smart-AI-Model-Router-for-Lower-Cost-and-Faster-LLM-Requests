"""Phase 2.4 — eval target adapters.

A *target* is a destination for a single chat-completion request. Targets
implement one method, ``dispatch(messages, conversation_id, max_tokens,
temperature) -> CallResult``, and the runner threads multi-turn replay on
top of that.

Three target families ship out of the box:

- ``gateway:auto`` / ``gateway:<virtual_model>`` — POST to the local gateway
  (or any OpenAI-compatible URL). The auto variant lights up the cascade
  router; the named variant is the sanity baseline (a fixed virtual model
  that bypasses routing).
- ``litellm:<model>`` — POST to a LiteLLM proxy URL.
- ``direct:<provider>:<model>`` — bypass any proxy, call the gateway's
  provider classes (Anthropic / OpenAI / Ollama) in-process. Same code path
  the gateway uses, but no auth / cache / breaker / failover stack — pure
  baseline for cost & latency.

Targets are spec'd by a single string so the CLI stays simple:
``--target gateway:auto``, ``--target direct:openai:gpt-4o-mini``, etc.

All targets normalize their reply into the same ``CallResult`` so the
runner / reporter never branch on which kind of target produced the row.
"""
from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import httpx

from gateway.cost.pricing import compute_cost
from gateway.models import ChatCompletionRequest, ChatMessage


@dataclass
class CallResult:
    """Normalized outcome of a single target.dispatch() call."""

    response_text: str
    prompt_tokens: int
    completion_tokens: int
    latency_ms: float
    cost_usd: float
    model: str | None = None          # provider-side model that actually ran
    cached: bool = False
    fallback_used: bool = False
    routing_decision: dict[str, Any] | None = None
    error: str | None = None
    raw: dict[str, Any] = field(default_factory=dict)


class Target(ABC):
    """Single dispatch surface — one call per turn."""

    name: str
    label: str  # short label used in result filenames / reports

    @abstractmethod
    async def dispatch(
        self,
        messages: list[dict[str, str]],
        *,
        conversation_id: str | None,
        max_tokens: int | None,
        temperature: float | None,
    ) -> CallResult:
        ...

    async def aclose(self) -> None:
        pass


# ---------------------------------------------------------------------------
# Target spec parser
# ---------------------------------------------------------------------------


def build_target(spec: str, *, gateway_url: str, api_key: str | None) -> Target:
    """Resolve a target spec string into a Target instance.

    Spec formats:

    - ``gateway:auto``                  → this gateway, model="auto"
    - ``gateway:<virtual_model>``       → this gateway, fixed virtual model
    - ``litellm:<model>``               → LiteLLM proxy at $LITELLM_URL
    - ``direct:<provider>:<model>``     → in-process provider call
    """
    if ":" not in spec:
        raise ValueError(f"target spec must include ':' — got {spec!r}")
    kind, _, rest = spec.partition(":")
    if kind == "gateway":
        return GatewayTarget(
            virtual_model=rest or "auto",
            base_url=gateway_url,
            api_key=api_key,
            label=f"gateway-{rest or 'auto'}",
        )
    if kind == "litellm":
        return LiteLLMTarget(model=rest, label=f"litellm-{rest}")
    if kind == "direct":
        provider, _, model = rest.partition(":")
        if not provider or not model:
            raise ValueError(
                f"direct target spec must be direct:<provider>:<model>, got {spec!r}"
            )
        return DirectProviderTarget(provider_name=provider, model=model,
                                    label=f"direct-{provider}-{model}")
    raise ValueError(f"unknown target kind {kind!r} in spec {spec!r}")


# ---------------------------------------------------------------------------
# Gateway HTTP target — works for this gateway and any OpenAI-compatible URL
# ---------------------------------------------------------------------------


class GatewayTarget(Target):
    def __init__(
        self,
        *,
        virtual_model: str,
        base_url: str,
        api_key: str | None,
        label: str,
        timeout_seconds: float = 120.0,
    ) -> None:
        self.virtual_model = virtual_model
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.label = label
        self.name = f"gateway[{virtual_model}]"
        self._client = httpx.AsyncClient(timeout=timeout_seconds)

    async def dispatch(
        self,
        messages: list[dict[str, str]],
        *,
        conversation_id: str | None,
        max_tokens: int | None,
        temperature: float | None,
    ) -> CallResult:
        headers: dict[str, str] = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        if conversation_id:
            headers["X-Conversation-Id"] = conversation_id

        body: dict[str, Any] = {
            "model": self.virtual_model,
            "messages": messages,
            "stream": False,
        }
        if max_tokens is not None:
            body["max_tokens"] = max_tokens
        if temperature is not None:
            body["temperature"] = temperature

        t0 = time.perf_counter()
        try:
            resp = await self._client.post(
                f"{self.base_url}/v1/chat/completions",
                json=body,
                headers=headers,
            )
        except httpx.HTTPError as exc:
            elapsed_ms = (time.perf_counter() - t0) * 1000
            return CallResult(
                response_text="",
                prompt_tokens=0,
                completion_tokens=0,
                latency_ms=elapsed_ms,
                cost_usd=0.0,
                error=f"transport: {exc}",
            )
        elapsed_ms = (time.perf_counter() - t0) * 1000

        if resp.status_code >= 400:
            return CallResult(
                response_text="",
                prompt_tokens=0,
                completion_tokens=0,
                latency_ms=elapsed_ms,
                cost_usd=0.0,
                error=f"HTTP {resp.status_code}: {resp.text[:200]}",
            )
        data = resp.json()
        return _parse_openai_response(data, elapsed_ms, fallback_model=self.virtual_model)

    async def aclose(self) -> None:
        await self._client.aclose()


# ---------------------------------------------------------------------------
# LiteLLM HTTP target — same OpenAI shape; configured via constructor URL
# ---------------------------------------------------------------------------


class LiteLLMTarget(Target):
    def __init__(
        self,
        *,
        model: str,
        label: str,
        base_url: str | None = None,
        api_key: str | None = None,
        timeout_seconds: float = 120.0,
    ) -> None:
        import os
        self.model = model
        self.label = label
        self.name = f"litellm[{model}]"
        self.base_url = (base_url or os.environ.get("LITELLM_URL", "")).rstrip("/")
        self.api_key = api_key or os.environ.get("LITELLM_API_KEY", "")
        if not self.base_url:
            raise RuntimeError(
                "LiteLLMTarget needs LITELLM_URL (or base_url=) to be set"
            )
        self._client = httpx.AsyncClient(timeout=timeout_seconds)

    async def dispatch(
        self,
        messages: list[dict[str, str]],
        *,
        conversation_id: str | None,
        max_tokens: int | None,
        temperature: float | None,
    ) -> CallResult:
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        body: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": False,
        }
        if max_tokens is not None:
            body["max_tokens"] = max_tokens
        if temperature is not None:
            body["temperature"] = temperature

        t0 = time.perf_counter()
        try:
            resp = await self._client.post(
                f"{self.base_url}/v1/chat/completions",
                json=body,
                headers=headers,
            )
        except httpx.HTTPError as exc:
            elapsed_ms = (time.perf_counter() - t0) * 1000
            return CallResult(
                response_text="",
                prompt_tokens=0,
                completion_tokens=0,
                latency_ms=elapsed_ms,
                cost_usd=0.0,
                error=f"transport: {exc}",
            )
        elapsed_ms = (time.perf_counter() - t0) * 1000
        if resp.status_code >= 400:
            return CallResult(
                response_text="",
                prompt_tokens=0,
                completion_tokens=0,
                latency_ms=elapsed_ms,
                cost_usd=0.0,
                error=f"HTTP {resp.status_code}: {resp.text[:200]}",
            )
        return _parse_openai_response(resp.json(), elapsed_ms, fallback_model=self.model)

    async def aclose(self) -> None:
        await self._client.aclose()


# ---------------------------------------------------------------------------
# Direct in-process provider call
# ---------------------------------------------------------------------------


class DirectProviderTarget(Target):
    """Bypass the gateway and call its provider classes directly.

    Acts as the cost/latency floor: no auth, no breaker, no failover, no
    cache, no router. Used to answer "how much is the gateway overhead?"
    """

    def __init__(self, *, provider_name: str, model: str, label: str) -> None:
        from gateway.providers import get_provider
        self.provider_name = provider_name
        self.model = model
        self.label = label
        self.name = f"direct[{provider_name}:{model}]"
        self._provider = get_provider(provider_name)

    async def dispatch(
        self,
        messages: list[dict[str, str]],
        *,
        conversation_id: str | None,
        max_tokens: int | None,
        temperature: float | None,
    ) -> CallResult:
        req = ChatCompletionRequest(
            model=self.model,
            messages=[ChatMessage(**m) for m in messages],
            max_tokens=max_tokens,
            temperature=temperature if temperature is not None else 1.0,
            stream=False,
        )
        t0 = time.perf_counter()
        try:
            resp = await self._provider.complete(req, self.model)
        except Exception as exc:  # noqa: BLE001 — surface as result.error
            elapsed_ms = (time.perf_counter() - t0) * 1000
            return CallResult(
                response_text="",
                prompt_tokens=0,
                completion_tokens=0,
                latency_ms=elapsed_ms,
                cost_usd=0.0,
                error=f"{type(exc).__name__}: {exc}",
            )
        elapsed_ms = (time.perf_counter() - t0) * 1000

        text = ""
        if resp.choices:
            content = resp.choices[0].message.content
            text = content if isinstance(content, str) else str(content or "")
        cost = float(
            compute_cost(self.model, resp.usage.prompt_tokens, resp.usage.completion_tokens)
        )
        return CallResult(
            response_text=text,
            prompt_tokens=resp.usage.prompt_tokens,
            completion_tokens=resp.usage.completion_tokens,
            latency_ms=elapsed_ms,
            cost_usd=cost,
            model=self.model,
            cached=False,
            fallback_used=False,
            routing_decision=None,
            raw={"id": resp.id},
        )


# ---------------------------------------------------------------------------
# OpenAI-shape response normalizer (used by gateway + litellm targets)
# ---------------------------------------------------------------------------


def _parse_openai_response(
    data: dict[str, Any],
    elapsed_ms: float,
    *,
    fallback_model: str,
) -> CallResult:
    text = ""
    choices = data.get("choices") or []
    if choices:
        msg = choices[0].get("message") or {}
        content = msg.get("content")
        if isinstance(content, str):
            text = content
        elif isinstance(content, list):
            parts = [c.get("text", "") for c in content if isinstance(c, dict)]
            text = "\n".join(parts)

    usage = data.get("usage") or {}
    pt = int(usage.get("prompt_tokens") or 0)
    ct = int(usage.get("completion_tokens") or 0)

    # Gateway responses carry the resolved provider model in `model`;
    # LiteLLM does the same.
    resolved_model = data.get("model") or fallback_model
    cost = float(compute_cost(resolved_model, pt, ct))

    return CallResult(
        response_text=text,
        prompt_tokens=pt,
        completion_tokens=ct,
        latency_ms=elapsed_ms,
        cost_usd=cost,
        model=resolved_model,
        cached=bool(data.get("cached")),
        fallback_used=bool(data.get("fallback_used")),
        routing_decision=data.get("routing_decision"),
        raw={"id": data.get("id")},
    )
