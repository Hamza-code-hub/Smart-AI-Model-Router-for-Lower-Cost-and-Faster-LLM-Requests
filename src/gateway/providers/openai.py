import json
import time
from collections.abc import AsyncIterator
from typing import Any

import httpx
import structlog

from gateway.config import get_settings
from gateway.models import (
    ChatChoice,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessage,
    UsageInfo,
)
from gateway.providers.base import Provider, ProviderError

logger = structlog.get_logger(__name__)


def _build_payload(request: ChatCompletionRequest, model: str, stream: bool) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": model,
        "messages": [m.model_dump(exclude_none=True) for m in request.messages],
        "stream": stream,
    }
    if stream:
        payload["stream_options"] = {"include_usage": True}
    if request.temperature is not None:
        payload["temperature"] = request.temperature
    if request.max_tokens is not None:
        payload["max_tokens"] = request.max_tokens
    if request.stop is not None:
        payload["stop"] = request.stop
    if request.top_p is not None:
        payload["top_p"] = request.top_p
    return payload


class OpenAIProvider(Provider):
    name = "openai"

    def __init__(self) -> None:
        settings = get_settings()
        self._client = httpx.AsyncClient(
            base_url="https://api.openai.com",
            headers={
                "Authorization": f"Bearer {settings.openai_api_key}",
                "Content-Type": "application/json",
            },
            timeout=httpx.Timeout(60.0, connect=10.0),
        )

    async def complete(self, request: ChatCompletionRequest, model: str) -> ChatCompletionResponse:
        payload = _build_payload(request, model, stream=False)

        try:
            resp = await self._client.post("/v1/chat/completions", json=payload)
        except httpx.TimeoutException as exc:
            raise ProviderError(504, "OpenAI request timed out", self.name) from exc
        except httpx.RequestError as exc:
            raise ProviderError(502, f"OpenAI connection error: {exc}", self.name) from exc

        if resp.status_code != 200:
            try:
                detail = resp.json().get("error", {}).get("message", resp.text)
            except Exception:
                detail = resp.text
            raise ProviderError(resp.status_code, detail, self.name)

        try:
            data = resp.json()
        except Exception as exc:
            raise ProviderError(502, "OpenAI returned malformed JSON", self.name) from exc

        choices = [
            ChatChoice(
                index=c["index"],
                message=ChatMessage(
                    role=c["message"].get("role", "assistant"),
                    content=c["message"].get("content", ""),
                ),
                finish_reason=c.get("finish_reason"),
            )
            for c in data.get("choices", [])
        ]
        usage = data.get("usage", {})

        logger.debug(
            "openai.complete",
            model=model,
            input_tokens=usage.get("prompt_tokens", 0),
            output_tokens=usage.get("completion_tokens", 0),
        )

        return ChatCompletionResponse(
            id=data.get("id", ""),
            created=data.get("created", int(time.time())),
            model=model,
            choices=choices,
            usage=UsageInfo(
                prompt_tokens=usage.get("prompt_tokens", 0),
                completion_tokens=usage.get("completion_tokens", 0),
                total_tokens=usage.get("total_tokens", 0),
            ),
        )

    async def stream(self, request: ChatCompletionRequest, model: str) -> AsyncIterator[str]:  # type: ignore[override]
        payload = _build_payload(request, model, stream=True)

        try:
            async with self._client.stream("POST", "/v1/chat/completions", json=payload) as resp:
                if resp.status_code != 200:
                    body = await resp.aread()
                    try:
                        detail = json.loads(body).get("error", {}).get("message", body.decode())
                    except Exception:
                        detail = body.decode()
                    raise ProviderError(resp.status_code, detail, self.name)

                async for line in resp.aiter_lines():
                    if line.startswith("data:"):
                        # Pass OpenAI SSE lines through as-is — already in correct format
                        yield f"{line}\n\n"
                        if line.strip() == "data: [DONE]":
                            break

        except httpx.TimeoutException as exc:
            raise ProviderError(504, "OpenAI stream timed out", self.name) from exc
        except httpx.RequestError as exc:
            raise ProviderError(502, f"OpenAI connection error: {exc}", self.name) from exc

    async def aclose(self) -> None:
        await self._client.aclose()
