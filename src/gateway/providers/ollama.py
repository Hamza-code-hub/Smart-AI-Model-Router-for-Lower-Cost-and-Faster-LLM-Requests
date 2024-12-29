"""Phase 2.2 — local Ollama provider.

Talks to Ollama's native `/api/chat` (NDJSON over HTTP), not its
OpenAI-compatible shim. Slightly less abstraction, full access to `options`
(temperature, num_predict, stop) and the `eval_count` / `prompt_eval_count`
usage numbers Ollama returns natively.

Reachable at `OLLAMA_BASE_URL` (default `http://host.docker.internal:11434` —
the gateway container talks to host Ollama). For Linux hosts the docker-compose
gateway service exports `extra_hosts: ["host.docker.internal:host-gateway"]`.
"""
from __future__ import annotations

import json
import time
import uuid
from collections.abc import AsyncIterator
from typing import Any

import httpx
import structlog

from gateway.config import get_settings
from gateway.models import (
    ChatChoice,
    ChatChoiceDelta,
    ChatCompletionChunk,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessage,
    ChatStreamDelta,
    UsageInfo,
)
from gateway.providers.base import Provider, ProviderError

logger = structlog.get_logger(__name__)


def _ollama_messages(request: ChatCompletionRequest) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    for msg in request.messages:
        content = msg.content
        if isinstance(content, list):
            content = "".join(
                block["text"]
                for block in content
                if isinstance(block, dict) and isinstance(block.get("text"), str)
            )
        messages.append({"role": msg.role, "content": content or ""})
    return messages


def _build_payload(request: ChatCompletionRequest, model: str, stream: bool) -> dict[str, Any]:
    options: dict[str, Any] = {}
    if request.temperature is not None:
        options["temperature"] = request.temperature
    if request.top_p is not None:
        options["top_p"] = request.top_p
    if request.max_tokens is not None:
        options["num_predict"] = request.max_tokens
    if request.stop is not None:
        options["stop"] = [request.stop] if isinstance(request.stop, str) else request.stop

    payload: dict[str, Any] = {
        "model": model,
        "messages": _ollama_messages(request),
        "stream": stream,
    }
    if options:
        payload["options"] = options
    return payload


_FINISH_REASON: dict[str, str] = {
    "stop": "stop",
    "length": "length",
    "load": "stop",
}


class OllamaProvider(Provider):
    name = "ollama"

    def __init__(self) -> None:
        settings = get_settings()
        self._client = httpx.AsyncClient(
            base_url=settings.ollama_base_url,
            headers={"content-type": "application/json"},
            timeout=httpx.Timeout(settings.ollama_timeout_seconds, connect=10.0),
        )

    async def complete(self, request: ChatCompletionRequest, model: str) -> ChatCompletionResponse:
        payload = _build_payload(request, model, stream=False)

        try:
            resp = await self._client.post("/api/chat", json=payload)
        except httpx.TimeoutException as exc:
            raise ProviderError(504, "Ollama request timed out", self.name) from exc
        except httpx.RequestError as exc:
            raise ProviderError(502, f"Ollama connection error: {exc}", self.name) from exc

        if resp.status_code != 200:
            try:
                detail = resp.json().get("error", resp.text)
            except Exception:
                detail = resp.text
            raise ProviderError(resp.status_code, str(detail), self.name)

        try:
            data = resp.json()
        except Exception as exc:
            raise ProviderError(502, "Ollama returned malformed JSON", self.name) from exc

        message = data.get("message", {}) or {}
        content_text = message.get("content", "") or ""
        finish_reason = _FINISH_REASON.get(data.get("done_reason", "stop"), "stop")
        input_tokens = int(data.get("prompt_eval_count", 0) or 0)
        output_tokens = int(data.get("eval_count", 0) or 0)

        logger.debug(
            "ollama.complete",
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )

        return ChatCompletionResponse(
            id=f"chatcmpl-{uuid.uuid4().hex}",
            created=int(time.time()),
            model=model,
            choices=[
                ChatChoice(
                    index=0,
                    message=ChatMessage(role="assistant", content=content_text),
                    finish_reason=finish_reason,
                )
            ],
            usage=UsageInfo(
                prompt_tokens=input_tokens,
                completion_tokens=output_tokens,
                total_tokens=input_tokens + output_tokens,
            ),
        )

    async def stream(self, request: ChatCompletionRequest, model: str) -> AsyncIterator[str]:  # type: ignore[override]
        payload = _build_payload(request, model, stream=True)
        completion_id = f"chatcmpl-{uuid.uuid4().hex}"
        created = int(time.time())

        input_tokens = 0
        output_tokens = 0
        finish_reason: str | None = None

        try:
            async with self._client.stream("POST", "/api/chat", json=payload) as resp:
                if resp.status_code != 200:
                    body = await resp.aread()
                    try:
                        detail = json.loads(body).get("error", body.decode())
                    except Exception:
                        detail = body.decode()
                    raise ProviderError(resp.status_code, str(detail), self.name)

                # Ollama streams NDJSON, one JSON object per line.
                async for line in resp.aiter_lines():
                    if not line:
                        continue
                    try:
                        event = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    message = event.get("message") or {}
                    delta_text = message.get("content", "")
                    if delta_text:
                        chunk = ChatCompletionChunk(
                            id=completion_id,
                            created=created,
                            model=model,
                            choices=[
                                ChatStreamDelta(
                                    index=0,
                                    delta=ChatChoiceDelta(content=delta_text),
                                )
                            ],
                        )
                        yield f"data: {chunk.model_dump_json()}\n\n"

                    if event.get("done"):
                        finish_reason = _FINISH_REASON.get(
                            event.get("done_reason", "stop"), "stop"
                        )
                        input_tokens = int(event.get("prompt_eval_count", 0) or 0)
                        output_tokens = int(event.get("eval_count", 0) or 0)

        except httpx.TimeoutException as exc:
            raise ProviderError(504, "Ollama stream timed out", self.name) from exc
        except httpx.RequestError as exc:
            raise ProviderError(502, f"Ollama connection error: {exc}", self.name) from exc

        final = ChatCompletionChunk(
            id=completion_id,
            created=created,
            model=model,
            choices=[
                ChatStreamDelta(
                    index=0,
                    delta=ChatChoiceDelta(),
                    finish_reason=finish_reason,
                )
            ],
            usage=UsageInfo(
                prompt_tokens=input_tokens,
                completion_tokens=output_tokens,
                total_tokens=input_tokens + output_tokens,
            ),
        )
        yield f"data: {final.model_dump_json()}\n\n"
        yield "data: [DONE]\n\n"

    async def aclose(self) -> None:
        await self._client.aclose()
