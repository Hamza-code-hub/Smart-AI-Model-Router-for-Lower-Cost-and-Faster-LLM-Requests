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

_STOP_REASON: dict[str, str] = {
    "end_turn": "stop",
    "max_tokens": "length",
    "stop_sequence": "stop",
    "tool_use": "tool_calls",
}


def _build_payload(request: ChatCompletionRequest, model: str, stream: bool) -> dict[str, Any]:
    system_prompt: str | None = None
    messages: list[dict[str, Any]] = []
    for msg in request.messages:
        if msg.role == "system":
            system_prompt = str(msg.content) if msg.content else ""
        else:
            messages.append({"role": msg.role, "content": msg.content or ""})

    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "max_tokens": request.max_tokens or 4096,
        "stream": stream,
    }
    if system_prompt:
        payload["system"] = system_prompt
    if request.temperature is not None:
        payload["temperature"] = request.temperature
    if request.stop:
        payload["stop_sequences"] = (
            [request.stop] if isinstance(request.stop, str) else request.stop
        )
    return payload


class AnthropicProvider(Provider):
    name = "anthropic"

    def __init__(self) -> None:
        settings = get_settings()
        self._client = httpx.AsyncClient(
            base_url="https://api.anthropic.com",
            headers={
                "x-api-key": settings.anthropic_api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            timeout=httpx.Timeout(60.0, connect=10.0),
        )

    async def complete(self, request: ChatCompletionRequest, model: str) -> ChatCompletionResponse:
        payload = _build_payload(request, model, stream=False)

        try:
            resp = await self._client.post("/v1/messages", json=payload)
        except httpx.TimeoutException as exc:
            raise ProviderError(504, "Anthropic request timed out", self.name) from exc
        except httpx.RequestError as exc:
            raise ProviderError(502, f"Anthropic connection error: {exc}", self.name) from exc

        if resp.status_code != 200:
            try:
                detail = resp.json().get("error", {}).get("message", resp.text)
            except Exception:
                detail = resp.text
            raise ProviderError(resp.status_code, detail, self.name)

        try:
            data = resp.json()
        except Exception as exc:
            raise ProviderError(502, "Anthropic returned malformed JSON", self.name) from exc

        content_text = "".join(
            block["text"]
            for block in data.get("content", [])
            if block.get("type") == "text"
        )
        finish_reason = _STOP_REASON.get(data.get("stop_reason", ""), "stop")
        usage = data.get("usage", {})
        input_tokens = usage.get("input_tokens", 0)
        output_tokens = usage.get("output_tokens", 0)

        logger.debug("anthropic.complete", model=model, input_tokens=input_tokens, output_tokens=output_tokens)

        return ChatCompletionResponse(
            id=data.get("id", f"chatcmpl-{uuid.uuid4().hex}"),
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
            async with self._client.stream("POST", "/v1/messages", json=payload) as resp:
                if resp.status_code != 200:
                    body = await resp.aread()
                    try:
                        detail = json.loads(body).get("error", {}).get("message", body.decode())
                    except Exception:
                        detail = body.decode()
                    raise ProviderError(resp.status_code, detail, self.name)

                async for line in resp.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    raw = line[5:].strip()
                    if not raw:
                        continue

                    try:
                        event = json.loads(raw)
                    except json.JSONDecodeError:
                        continue

                    etype = event.get("type")

                    if etype == "message_start":
                        usage = event.get("message", {}).get("usage", {})
                        input_tokens = usage.get("input_tokens", 0)

                    elif etype == "content_block_delta":
                        delta = event.get("delta", {})
                        if delta.get("type") == "text_delta":
                            text = delta.get("text", "")
                            chunk = ChatCompletionChunk(
                                id=completion_id,
                                created=created,
                                model=model,
                                choices=[
                                    ChatStreamDelta(
                                        index=0,
                                        delta=ChatChoiceDelta(content=text),
                                    )
                                ],
                            )
                            yield f"data: {chunk.model_dump_json()}\n\n"

                    elif etype == "message_delta":
                        delta = event.get("delta", {})
                        finish_reason = _STOP_REASON.get(delta.get("stop_reason", ""), "stop")
                        usage = event.get("usage", {})
                        output_tokens = usage.get("output_tokens", 0)

        except httpx.TimeoutException as exc:
            raise ProviderError(504, "Anthropic stream timed out", self.name) from exc
        except httpx.RequestError as exc:
            raise ProviderError(502, f"Anthropic connection error: {exc}", self.name) from exc

        # Final chunk: finish_reason + usage
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
