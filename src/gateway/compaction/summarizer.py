"""Summarizer for conversation compaction.

Calls a configured virtual model (typically a cheap one — Haiku, gpt-4o-mini)
to compress a span of conversation turns into a structured JSON summary
with four sections: key_facts, decisions, open_questions, sticky_facts.

The output is *not* free-form text — we ask for JSON and validate it before
substituting into the prompt. If the summarizer hallucinates structure or
fails entirely we fall back to a verbatim transcript clipped to N tokens,
which is safe but loses the cost benefit for that one request.
"""

import json
import re
from typing import Any

import structlog

from gateway.models import ChatCompletionRequest, ChatMessage
from gateway.providers import ProviderError, get_provider
from gateway.routing import get_resolver

logger = structlog.get_logger(__name__)

SUMMARIZER_SYSTEM_PROMPT = """You compress conversation history for later context.

Output ONLY a single JSON object, no prose, with these exact keys:
{
  "key_facts": [<string>, ...],          // Stable facts established in the conversation
  "decisions": [<string>, ...],           // Choices the user or assistant committed to
  "open_questions": [<string>, ...],      // Unresolved points the next turn may need
  "sticky_facts": [<string>, ...]         // Identifiers that MUST survive verbatim:
                                          // file paths, version numbers, names, IDs,
                                          // ticket numbers, exact code snippets the
                                          // user provided
}

Be concise. Preserve precise identifiers (numbers, paths, names) exactly.
Do not invent facts the conversation did not state."""


# Sticky-fact regex bank. These are extracted *before* summarization and
# threaded into the prompt and the stored state, so they survive even if the
# summarizer paraphrases or drops them.
_STICKY_PATTERNS = [
    re.compile(r"\bv?\d+\.\d+(?:\.\d+)?(?:[-+][a-zA-Z0-9.]+)?\b"),   # versions
    re.compile(r"[\w./-]+\.(?:py|ts|tsx|js|jsx|go|rs|java|sql|yaml|yml|json|md|toml|sh)\b"),  # file paths
    re.compile(r"\b[A-Z]{2,}-\d+\b"),                                  # JIRA-style IDs
    re.compile(r"\b[A-Za-z_][\w]*::[A-Za-z_][\w]*\b"),                 # qualified names
    re.compile(r"`([^`\n]{1,80})`"),                                   # backtick-quoted spans
]


def extract_sticky_facts(messages: list[ChatMessage]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for msg in messages:
        if not isinstance(msg.content, str):
            continue
        for pat in _STICKY_PATTERNS:
            for match in pat.findall(msg.content):
                text = match if isinstance(match, str) else " ".join(match)
                text = text.strip()
                if 1 < len(text) < 120 and text not in seen:
                    seen.add(text)
                    out.append(text)
    return out


def _validate_summary(raw: str) -> dict[str, list[str]] | None:
    try:
        # Tolerate models that wrap JSON in ```json fences.
        stripped = raw.strip()
        if stripped.startswith("```"):
            stripped = re.sub(r"^```(?:json)?\n?|\n?```$", "", stripped)
        data = json.loads(stripped)
    except Exception:
        return None
    if not isinstance(data, dict):
        return None
    keys = ("key_facts", "decisions", "open_questions", "sticky_facts")
    if not all(k in data and isinstance(data[k], list) for k in keys):
        return None
    return {k: [str(item) for item in data[k]] for k in keys}


async def summarize_turns(
    turns: list[ChatMessage],
    summarizer_virtual_model: str,
    stitched_sticky: list[str],
) -> dict[str, Any]:
    """Calls the summarizer model and returns a structured summary dict.

    On any failure returns a degraded summary built from sticky_facts only —
    the request still proceeds, and the caller treats the result as authoritative
    state for that conversation_id going forward.
    """
    resolver = get_resolver()
    try:
        route = resolver.resolve(summarizer_virtual_model)
    except KeyError:
        logger.warning("compaction.summarizer_missing", model=summarizer_virtual_model)
        return _degraded_summary(stitched_sticky)

    # Render the turns being compacted as a single user message; cheaper than
    # sending the raw role-stream because the summarizer doesn't need to
    # role-play, it just needs to compress.
    transcript = "\n\n".join(
        f"[{m.role}]: {m.content}" for m in turns if isinstance(m.content, str)
    )
    sticky_hint = ""
    if stitched_sticky:
        sticky_hint = (
            "\n\nThese identifiers were extracted heuristically; preserve them verbatim "
            "in your output's sticky_facts list:\n- " + "\n- ".join(stitched_sticky)
        )

    summarizer_request = ChatCompletionRequest(
        model=summarizer_virtual_model,
        messages=[
            ChatMessage(role="system", content=SUMMARIZER_SYSTEM_PROMPT),
            ChatMessage(
                role="user",
                content=f"Summarize this conversation transcript:\n\n{transcript}{sticky_hint}",
            ),
        ],
        temperature=0.0,
        stream=False,
        max_tokens=1024,
    )

    try:
        provider = get_provider(route.primary.provider)
        response = await provider.complete(summarizer_request, route.primary.model)
    except ProviderError as exc:
        logger.warning("compaction.summarizer_failed", error=exc.message, code=exc.status_code)
        return _degraded_summary(stitched_sticky)

    if not response.choices or response.choices[0].message.content is None:
        return _degraded_summary(stitched_sticky)

    raw = response.choices[0].message.content
    if not isinstance(raw, str):
        return _degraded_summary(stitched_sticky)
    parsed = _validate_summary(raw)
    if parsed is None:
        logger.warning("compaction.summary_invalid_json", preview=raw[:200])
        return _degraded_summary(stitched_sticky)

    # Union model-supplied sticky facts with our regex-extracted ones.
    merged = list(dict.fromkeys(stitched_sticky + parsed["sticky_facts"]))
    parsed["sticky_facts"] = merged
    return parsed


def _degraded_summary(sticky_facts: list[str]) -> dict[str, Any]:
    return {
        "key_facts": [],
        "decisions": [],
        "open_questions": [],
        "sticky_facts": sticky_facts,
    }


def render_summary_message(summary: dict[str, Any]) -> ChatMessage:
    """Renders a stored summary back into a single system message for injection."""
    sections = []
    if summary.get("key_facts"):
        sections.append("Key facts:\n- " + "\n- ".join(summary["key_facts"]))
    if summary.get("decisions"):
        sections.append("Decisions:\n- " + "\n- ".join(summary["decisions"]))
    if summary.get("open_questions"):
        sections.append("Open questions:\n- " + "\n- ".join(summary["open_questions"]))
    if summary.get("sticky_facts"):
        sections.append(
            "Sticky facts (preserve verbatim):\n- " + "\n- ".join(summary["sticky_facts"])
        )
    body = "\n\n".join(sections) if sections else "(no prior context)"
    return ChatMessage(
        role="system",
        content=f"<conversation_summary>\n{body}\n</conversation_summary>",
    )
