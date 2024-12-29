import hashlib
import json
import time
from collections.abc import AsyncIterator
from typing import Any

import structlog
from fastapi import APIRouter, BackgroundTasks, Depends, Header, HTTPException
from fastapi.responses import StreamingResponse

from gateway.auth import TenantInfo, require_tenant
from gateway.breaker import get_breaker
from gateway.compaction.strategy import maybe_compact
from gateway.cost.tracker import RequestRecord, log_request
from gateway.fallback import attempts, should_failover
from gateway.judge import sample_and_judge
from gateway.models import ChatCompletionRequest, ChatCompletionResponse
from gateway.observability import metrics as obs_metrics
from gateway.observability import tracing as obs_tracing
from gateway.providers import ProviderError, get_provider
from gateway.ratelimit import enforce_limits
from gateway.routing import get_resolver
from gateway.routing.classifier import classify
from gateway.routing.exemplar_seeder import (
    SEED_CONFIDENCE_THRESHOLD,
    seed_exemplar,
)
from gateway.routing.conversation_state import (
    ConvState,
    EscalationOutcome,
    apply_sticky,
    estimate_history_tokens,
    get_state,
    record_decision,
)

logger = structlog.get_logger(__name__)
router = APIRouter()

AUTO_MODEL = "auto"

# Phase 2.3 — A/B routing. `X-Routing-Mode: ab` flips a deterministic coin
# (per conversation when present, otherwise per last-user-message hash) and
# runs half through the cascade, half through a default static virtual
# model. The `routing_mode` label distinguishes the two arms so the dashboard
# can show cost / quality delta.
AB_STATIC_FALLBACK_MODEL = "fast-qa"


def _ab_coin(seed: str) -> bool:
    """True → use auto cascade. False → substitute the static fallback."""
    digest = hashlib.sha256(seed.encode("utf-8")).digest()
    return digest[0] & 1 == 0


def _resolve_routing_mode(
    requested_model: str,
    header_value: str | None,
    conv_id: str | None,
    messages: list,
) -> tuple[str, str]:
    """Decide (effective_model, routing_mode_label) before classify().

    routing_mode label values:
      - 'static'    — the request named a literal virtual model
      - 'auto'      — the request used model:'auto'
      - 'ab-auto'   — header `X-Routing-Mode: ab` + coin chose the cascade
      - 'ab-static' — header `X-Routing-Mode: ab` + coin chose the fallback
    """
    requested_mode = (header_value or "").strip().lower() or None
    if requested_mode == "ab" and requested_model == AUTO_MODEL:
        seed = conv_id or _flatten_content(messages[-1].content if messages else "")
        if _ab_coin(seed):
            return AUTO_MODEL, "ab-auto"
        return AB_STATIC_FALLBACK_MODEL, "ab-static"
    if requested_model == AUTO_MODEL:
        return AUTO_MODEL, "auto"
    return requested_model, "static"


def _flatten_content(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    parts: list[str] = []
    for block in content:
        if isinstance(block, dict) and isinstance(block.get("text"), str):
            parts.append(block["text"])
    return "".join(parts)


async def _resolve_auto(
    request: ChatCompletionRequest,
    conv_id: str | None,
    *,
    routing_mode: str = "auto",
) -> tuple[ChatCompletionRequest, dict[str, Any], ConvState | None, EscalationOutcome | None, str]:
    """Run the cascade + sticky upward, return (substituted_request, routing_decision_meta,
    conv_state, sticky_outcome, tier_fired)."""
    decision = await classify(request, routing_mode=routing_mode)
    tier_fired = decision.tier_fired

    conv_state: ConvState | None = None
    outcome: EscalationOutcome | None = None
    final_category = decision.category
    if conv_id:
        conv_state = await get_state(conv_id)
        history_chars = sum(len(_flatten_content(m.content)) for m in request.messages)
        outcome = apply_sticky(
            decision.category,
            estimate_history_tokens(history_chars),
            conv_state,
        )
        final_category = outcome.final_category

    routing_decision = {
        "tier": tier_fired,
        "category": final_category,
        "escalation_count": (
            conv_state.escalation_count + (1 if outcome and outcome.switched_from else 0)
            if conv_state
            else 0
        ),
        "switched_from": outcome.switched_from if outcome else None,
        "raw_category": decision.category,
        "similarity": decision.similarity,
        "confidence": decision.confidence,
    }

    substituted = request.model_copy(update={"model": final_category})
    return substituted, routing_decision, conv_state, outcome, tier_fired


def _track_usage(chunk: str, record: RequestRecord) -> None:
    if chunk.startswith("data:") and chunk.strip() != "data: [DONE]":
        raw = chunk[5:].strip()
        try:
            data = json.loads(raw)
            usage = data.get("usage") or {}
            if usage:
                record.input_tokens = usage.get("prompt_tokens", record.input_tokens)
                record.output_tokens = usage.get("completion_tokens", record.output_tokens)
        except Exception:
            pass


def _error_events(message: str) -> list[str]:
    error_event = json.dumps({"error": {"message": message, "type": "provider_error"}})
    return [f"data: {error_event}\n\n", "data: [DONE]\n\n"]


async def _stream_with_failover(
    request: ChatCompletionRequest,
    targets: list,
    record: RequestRecord,
) -> AsyncIterator[str]:
    # Failover only happens *before* the first chunk reaches the client: once
    # bytes are on the wire we are committed to that provider and any later
    # error surfaces as an inline SSE error event.
    breaker = get_breaker()
    committed: AsyncIterator[str] | None = None
    first_chunk: str | None = None
    last_error_msg = "All providers unavailable"
    last_error_code = 503

    for i, target in enumerate(targets):
        if not breaker.allow(target.provider, target.model):
            logger.info("chat.breaker_skip", provider=target.provider, model=target.model)
            last_error_msg = f"{target.provider} circuit open"
            last_error_code = 503
            continue

        source = get_provider(target.provider).stream(request, target.model).__aiter__()
        record.provider = target.provider
        record.model_used = target.model
        record.fallback_used = i > 0
        try:
            first_chunk = await source.__anext__()
        except StopAsyncIteration:
            committed = source
            first_chunk = None
            breaker.record_success(target.provider, target.model)
            break
        except ProviderError as exc:
            if should_failover(exc):
                breaker.record_failure(target.provider, target.model)
            last_error_msg = exc.message
            last_error_code = exc.status_code
            if i < len(targets) - 1 and should_failover(exc):
                next_target = targets[i + 1]
                event = (
                    "chat.failover_model_missing"
                    if exc.status_code == 404
                    else "chat.failover"
                )
                logger.warning(
                    event,
                    from_provider=target.provider,
                    from_model=target.model,
                    to_provider=next_target.provider,
                    to_model=next_target.model,
                    status_code=exc.status_code,
                )
                if exc.status_code == 404:
                    obs_metrics.record_model_missing_failover(
                        primary_provider=target.provider,
                        primary_model=target.model,
                        fallback_provider=next_target.provider,
                        fallback_model=next_target.model,
                    )
                continue
            logger.warning("chat.stream_error", provider=target.provider, status_code=exc.status_code, message=exc.message)
            record.status_code = exc.status_code
            record.error_message = exc.message
            record.latency_ms = int((time.monotonic() - record.start_time) * 1000)
            for event in _error_events(exc.message):
                yield event
            await log_request(record)
            return
        else:
            if i > 0:
                logger.info("chat.failover_succeeded", provider=target.provider, model=target.model)
            breaker.record_success(target.provider, target.model)
            committed = source
            break

    if committed is None and first_chunk is None:
        # Every target was either circuit-open or unrecoverable before yielding.
        record.status_code = last_error_code
        record.error_message = last_error_msg
        record.latency_ms = int((time.monotonic() - record.start_time) * 1000)
        for event in _error_events(last_error_msg):
            yield event
        await log_request(record)
        return

    try:
        if first_chunk is not None:
            _track_usage(first_chunk, record)
            yield first_chunk
        if committed is not None:
            async for chunk in committed:
                _track_usage(chunk, record)
                yield chunk
    except ProviderError as exc:
        logger.warning("chat.stream_error", provider=record.provider, status_code=exc.status_code, message=exc.message)
        if should_failover(exc):
            get_breaker().record_failure(record.provider, record.model_used)
        record.status_code = exc.status_code
        record.error_message = exc.message
        for event in _error_events(exc.message):
            yield event
    finally:
        record.latency_ms = int((time.monotonic() - record.start_time) * 1000)
        await log_request(record)


@router.post("/chat/completions", response_model=ChatCompletionResponse)
async def chat_completions(
    request: ChatCompletionRequest,
    background_tasks: BackgroundTasks,
    tenant: TenantInfo = Depends(require_tenant),
    x_conversation_id: str | None = Header(default=None, alias="X-Conversation-Id"),
    x_compaction_skip: str | None = Header(default=None, alias="X-Compaction-Skip"),
    x_routing_mode: str | None = Header(default=None, alias="X-Routing-Mode"),
) -> StreamingResponse | ChatCompletionResponse:
    resolver = get_resolver()

    requested_model = request.model
    effective_model, routing_mode = _resolve_routing_mode(
        requested_model, x_routing_mode, x_conversation_id, request.messages
    )
    if effective_model != requested_model:
        request = request.model_copy(update={"model": effective_model})

    was_auto = request.model == AUTO_MODEL
    routing_decision: dict[str, Any] | None = None
    conv_state: ConvState | None = None
    sticky_outcome: EscalationOutcome | None = None
    auto_tier_fired: str | None = None
    auto_routed_category: str | None = None

    obs_tracing.set_attrs(
        gateway_tenant=tenant.tenant_id,
        gateway_requested_model=requested_model,
        gateway_routing_mode=routing_mode,
        gateway_conversation_id=x_conversation_id,
    )

    if was_auto:
        try:
            request, routing_decision, conv_state, sticky_outcome, auto_tier_fired = (
                await _resolve_auto(request, x_conversation_id, routing_mode=routing_mode)
            )
        except Exception as exc:
            logger.error("chat.auto_classify_failed", error=str(exc))
            raise HTTPException(status_code=500, detail=f"auto-routing failed: {exc}")
        auto_routed_category = request.model
        logger.info(
            "chat.auto_decision",
            tenant_id=tenant.tenant_id,
            tier=auto_tier_fired,
            category=auto_routed_category,
            switched_from=sticky_outcome.switched_from if sticky_outcome else None,
            conversation_id=x_conversation_id,
            routing_mode=routing_mode,
        )

    try:
        route = resolver.resolve(request.model)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc))

    await enforce_limits(tenant)

    compaction_skipped = (x_compaction_skip or "").lower() == "true"
    if (
        not compaction_skipped
        and x_conversation_id
        and route.compaction is not None
        and route.compaction.enabled
    ):
        request = await maybe_compact(
            request,
            route.compaction,
            conversation_id=x_conversation_id,
            tenant_id=tenant.tenant_id,
            virtual_model=request.model,
        )

    targets = attempts(route)
    start = time.monotonic()

    logger.info(
        "chat.request",
        tenant_id=tenant.tenant_id,
        virtual_model=request.model,
        provider=route.primary.provider,
        model=route.primary.model,
        stream=request.stream,
        fallback=route.fallback.provider if route.fallback else None,
    )

    record = RequestRecord(
        virtual_model=AUTO_MODEL if was_auto else request.model,
        provider=route.primary.provider,
        model_used=route.primary.model,
        input_tokens=0,
        output_tokens=0,
        latency_ms=0,
        status_code=200,
        tenant_id=tenant.tenant_id,
        api_key_id=tenant.api_key_id,
        start_time=start,
        tier_fired=auto_tier_fired,
        routed_category=auto_routed_category,
        conversation_id=x_conversation_id,
        model_switched_from=sticky_outcome.switched_from if sticky_outcome else None,
        routing_mode=routing_mode,
    )

    if request.stream:
        if was_auto and conv_state is not None and sticky_outcome is not None and x_conversation_id:
            background_tasks.add_task(
                record_decision,
                x_conversation_id,
                auto_routed_category or request.model,
                auto_tier_fired or "heuristic",
                conv_state,
                sticky_outcome,
            )
        return StreamingResponse(
            _stream_with_failover(request, targets, record),
            media_type="text/event-stream",
            headers={"X-Accel-Buffering": "no", "Cache-Control": "no-cache"},
        )

    breaker = get_breaker()
    last_exc: ProviderError | None = None
    breaker_blocked_all = True

    for i, target in enumerate(targets):
        if not breaker.allow(target.provider, target.model):
            logger.info("chat.breaker_skip", provider=target.provider, model=target.model)
            continue
        breaker_blocked_all = False

        provider = get_provider(target.provider)
        record.provider = target.provider
        record.model_used = target.model
        record.fallback_used = i > 0
        try:
            response = await provider.complete(request, target.model)
        except ProviderError as exc:
            last_exc = exc
            if should_failover(exc):
                breaker.record_failure(target.provider, target.model)
            if i < len(targets) - 1 and should_failover(exc):
                next_target = targets[i + 1]
                event = (
                    "chat.failover_model_missing"
                    if exc.status_code == 404
                    else "chat.failover"
                )
                logger.warning(
                    event,
                    from_provider=target.provider,
                    from_model=target.model,
                    to_provider=next_target.provider,
                    to_model=next_target.model,
                    status_code=exc.status_code,
                )
                if exc.status_code == 404:
                    obs_metrics.record_model_missing_failover(
                        primary_provider=target.provider,
                        primary_model=target.model,
                        fallback_provider=next_target.provider,
                        fallback_model=next_target.model,
                    )
                continue
            logger.warning("chat.provider_error", provider=exc.provider, status_code=exc.status_code)
            record.status_code = exc.status_code
            record.error_message = exc.message
            record.latency_ms = int((time.monotonic() - record.start_time) * 1000)
            background_tasks.add_task(log_request, record)
            raise HTTPException(status_code=exc.status_code, detail=exc.message)

        if i > 0:
            logger.info("chat.failover_succeeded", provider=target.provider, model=target.model)
        breaker.record_success(target.provider, target.model)

        record.input_tokens = response.usage.prompt_tokens
        record.output_tokens = response.usage.completion_tokens
        record.latency_ms = int((time.monotonic() - record.start_time) * 1000)

        logger.info(
            "chat.response",
            tenant_id=tenant.tenant_id,
            virtual_model=request.model,
            provider=target.provider,
            model=target.model,
            input_tokens=record.input_tokens,
            output_tokens=record.output_tokens,
            latency_ms=record.latency_ms,
        )

        background_tasks.add_task(log_request, record)
        if was_auto and conv_state is not None and sticky_outcome is not None and x_conversation_id:
            background_tasks.add_task(
                record_decision,
                x_conversation_id,
                auto_routed_category or request.model,
                auto_tier_fired or "heuristic",
                conv_state,
                sticky_outcome,
            )
        # Phase 2.5 — production sampling backstop. 1% (configurable via
        # JUDGE_SAMPLE_RATE) of successful responses get scored by the judge
        # and any flagged misroutes land in `judge_flags` for human review.
        # No-op when JUDGE_SAMPLING_ENABLED is unset. Fail-soft top to bottom.
        last_user = next(
            (
                _flatten_content(m.content)
                for m in reversed(request.messages)
                if m.role == "user"
            ),
            "",
        )
        response_text = ""
        if response.choices:
            content = response.choices[0].message.content
            response_text = content if isinstance(content, str) else (str(content) if content else "")
        background_tasks.add_task(
            sample_and_judge,
            tenant_id=tenant.tenant_id,
            virtual_model=request.model,
            routed_category=auto_routed_category,
            tier_fired=auto_tier_fired,
            query=last_user,
            response=response_text,
            expected_category=None,
        )

        # Auto-seed router_exemplars on tier-3 high-confidence decisions so
        # future similar queries hit tier 2 directly. Gated on the LLM tier
        # firing (so heuristic/embedding hits don't re-seed what's already
        # there) and on Haiku's self-reported confidence clearing the bar.
        # See src/gateway/routing/exemplar_seeder.py — fail-soft, hard-capped
        # at 100k rows.
        if (
            routing_decision is not None
            and routing_decision.get("tier") == "llm"
            and (routing_decision.get("confidence") or 0.0) >= SEED_CONFIDENCE_THRESHOLD
            and last_user
            and routing_decision.get("raw_category")
        ):
            background_tasks.add_task(
                seed_exemplar,
                query=last_user,
                category=routing_decision["raw_category"],
                confidence=float(routing_decision["confidence"]),
            )

        if was_auto and routing_decision is not None:
            # Surface the real model + provider that produced this response so
            # the UI can tag the assistant bubble with `claude-haiku-4-5` etc.
            # instead of the virtual `auto`. `target` is the loop variable
            # that succeeded — primary on first iteration, fallback if not.
            routing_decision["model_used"] = target.model
            routing_decision["provider"] = target.provider
            response = response.model_copy(update={"routing_decision": routing_decision})
        return response

    if breaker_blocked_all:
        record.status_code = 503
        record.error_message = "All providers circuit-open"
        record.latency_ms = int((time.monotonic() - record.start_time) * 1000)
        background_tasks.add_task(log_request, record)
        raise HTTPException(status_code=503, detail="All providers unavailable (circuit open)")

    raise HTTPException(status_code=last_exc.status_code if last_exc else 502, detail=str(last_exc))
