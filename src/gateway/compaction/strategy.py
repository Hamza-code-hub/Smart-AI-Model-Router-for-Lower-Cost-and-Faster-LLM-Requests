"""Compaction orchestration.

Given a conversation_id + the incoming request, this:

1. Loads any prior compacted state for the conversation.
2. Estimates token count of the current message history.
3. If the history exceeds the route's ``token_threshold``, slices off the
   "old" turns (everything before the last ``keep_last_n_turns``), sends
   them to the summarizer, and persists the result.
4. Rewrites the outgoing ``request.messages`` to:
     [system summary (from prior state)]
     [verbatim recent turns]
   ...so the downstream provider only sees compressed context plus the live tail.

The request mutation is non-destructive — we always work on a copy.
"""

import json
from typing import Any

import structlog

from gateway.compaction.summarizer import (
    extract_sticky_facts,
    render_summary_message,
    summarize_turns,
)
from gateway.db import acquire
from gateway.models import ChatCompletionRequest, ChatMessage
from gateway.routing.resolver import CompactionRouteConfig

logger = structlog.get_logger(__name__)

# Rough char→token ratio. Real token counting is provider-specific; we use this
# heuristic (~4 chars/token) for the *threshold check* only. The actual
# downstream provider counts billable tokens normally.
_CHARS_PER_TOKEN = 4


def _estimate_tokens(messages: list[ChatMessage]) -> int:
    total = 0
    for msg in messages:
        if isinstance(msg.content, str):
            total += len(msg.content)
        elif isinstance(msg.content, list):
            total += len(json.dumps(msg.content))
    return total // _CHARS_PER_TOKEN


async def _load_state(conversation_id: str) -> dict[str, Any] | None:
    async with acquire() as conn:
        row = await conn.fetchrow(
            "SELECT last_compacted_turn, summary, sticky_facts FROM conversation_compaction WHERE conversation_id = $1",
            conversation_id,
        )
    if row is None:
        return None
    summary = row["summary"]
    if isinstance(summary, str):
        summary = json.loads(summary)
    sticky = row["sticky_facts"]
    if isinstance(sticky, str):
        sticky = json.loads(sticky)
    return {
        "last_compacted_turn": row["last_compacted_turn"],
        "summary": summary,
        "sticky_facts": sticky or [],
    }


async def _save_state(
    conversation_id: str,
    tenant_id: str,
    virtual_model: str,
    last_compacted_turn: int,
    summary: dict[str, Any],
) -> None:
    async with acquire() as conn:
        await conn.execute(
            """
            INSERT INTO conversation_compaction (
                conversation_id, tenant_id, virtual_model,
                last_compacted_turn, summary, sticky_facts, updated_at
            ) VALUES ($1, $2, $3, $4, $5::jsonb, $6::jsonb, NOW())
            ON CONFLICT (conversation_id) DO UPDATE SET
                tenant_id = EXCLUDED.tenant_id,
                virtual_model = EXCLUDED.virtual_model,
                last_compacted_turn = EXCLUDED.last_compacted_turn,
                summary = EXCLUDED.summary,
                sticky_facts = EXCLUDED.sticky_facts,
                updated_at = NOW()
            """,
            conversation_id,
            tenant_id,
            virtual_model,
            last_compacted_turn,
            json.dumps(summary),
            json.dumps(summary.get("sticky_facts", [])),
        )


async def delete_state(conversation_id: str) -> bool:
    async with acquire() as conn:
        result = await conn.execute(
            "DELETE FROM conversation_compaction WHERE conversation_id = $1",
            conversation_id,
        )
    return result.endswith(" 1")


def _split_keep_last(
    messages: list[ChatMessage], keep_last_n: int
) -> tuple[list[ChatMessage], list[ChatMessage]]:
    """Split messages into (compactable_prefix, kept_tail).

    ``keep_last_n`` counts *turns* — a user+assistant pair is one turn. We
    walk back through ``messages`` counting role transitions ``user → assistant``.
    System messages at index 0 are never sliced (treated as always-on context).
    """
    if not messages:
        return [], []

    system_head: list[ChatMessage] = []
    body_start = 0
    if messages[0].role == "system":
        system_head = [messages[0]]
        body_start = 1
    body = messages[body_start:]

    # Walk from the end, count user→assistant cycles.
    kept_count = 0
    cut_index = len(body)
    for i in range(len(body) - 1, -1, -1):
        if body[i].role == "user":
            kept_count += 1
            if kept_count >= keep_last_n:
                cut_index = i
                break

    prefix = body[:cut_index]
    tail = body[cut_index:]
    return system_head + prefix, tail


async def maybe_compact(
    request: ChatCompletionRequest,
    config: CompactionRouteConfig,
    conversation_id: str,
    tenant_id: str,
    virtual_model: str,
) -> ChatCompletionRequest:
    """Returns a (possibly rewritten) request with compacted history if needed."""
    if not config.enabled:
        return request

    state = await _load_state(conversation_id)
    estimated_tokens = _estimate_tokens(request.messages)

    # If we already have prior state, prepend its summary to the messages we
    # consider for "current history" — keeps the threshold check honest after
    # one compaction has already happened.
    if state is None and estimated_tokens < config.token_threshold:
        # Cheap path: nothing to do.
        return request

    compactable, tail = _split_keep_last(request.messages, config.keep_last_n_turns)

    needs_compaction = (
        state is None
        and estimated_tokens >= config.token_threshold
        and len(compactable) > 0
    ) or (
        state is not None
        and estimated_tokens >= config.token_threshold
        and len(compactable) > 0
    )

    summary: dict[str, Any]
    if needs_compaction:
        stitched_sticky = extract_sticky_facts(compactable)
        if state and state.get("sticky_facts"):
            # Union previous sticky facts forward so they never decay.
            seen = set(stitched_sticky)
            for fact in state["sticky_facts"]:
                if fact not in seen:
                    stitched_sticky.append(fact)
                    seen.add(fact)

        summary = await summarize_turns(
            compactable, config.summarizer_virtual_model, stitched_sticky
        )
        await _save_state(
            conversation_id,
            tenant_id,
            virtual_model,
            last_compacted_turn=len(compactable),
            summary=summary,
        )
        logger.info(
            "compaction.compacted",
            conversation_id=conversation_id,
            compacted_turns=len(compactable),
            kept_turns=len(tail),
            estimated_tokens=estimated_tokens,
        )
    elif state is not None:
        summary = state["summary"]
    else:
        return request

    summary_msg = render_summary_message(summary)
    new_messages = [summary_msg, *tail]
    rewritten = request.model_copy(update={"messages": new_messages})
    return rewritten
