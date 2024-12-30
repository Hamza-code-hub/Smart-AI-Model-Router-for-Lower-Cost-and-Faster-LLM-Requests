"""Phase 1.4 — cascade router classifier.

Three tiers, first confident answer wins:

  1. Heuristic (regex/structural, ~0ms) — asymmetric. Returns None on uncertainty.
  2. Embedding lookup (~10-20ms) — MiniLM NN against router_exemplars, plus a
     polarity-flip check to dodge MiniLM's negation blind spot.
  3. LLM (Haiku via the existing virtual model, ~200-500ms) — always emits.
     Invalid JSON falls back to deep-reasoning (safe-side default).

Categories emitted (locked — see memory/project_router_phase1_decisions.md):
    fast-qa | code | deep-reasoning | long-context

Threshold and tier-3 classifier model are kwargs so Phase 1.3 eval tuning and
Phase 2.2 Llama swap are both single-line changes.
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from typing import Any

import structlog

from gateway.db import acquire
from gateway.embeddings import embed
from gateway.models import ChatCompletionRequest, ChatMessage
from gateway.observability import metrics as obs_metrics
from gateway.observability import tracing as obs_tracing
from gateway.providers import get_provider
from gateway.routing import get_resolver

logger = structlog.get_logger(__name__)

CATEGORIES: tuple[str, ...] = ("fast-qa", "code", "deep-reasoning", "long-context")

# Tier 2 cosine-similarity floor. Raised to 0.80 when auto-seeding from
# tier-3 (Haiku) high-confidence verdicts went in — see
# src/gateway/routing/exemplar_seeder.py. The lower 0.45 from the curated-only
# era (eval/results/2026-06-17_router_threshold_sweep.json) admitted too wide
# a neighbourhood once machine-seeded rows started filling the table; at 0.80,
# only near-duplicates of an existing exemplar fire tier 2, which is the bar
# the noisier seeded rows can sustain. Cold-start tax: until enough
# high-confidence Haiku verdicts accumulate, most traffic falls through to
# tier 3 — the warm-up trades short-term Haiku spend for long-term tier-2
# hit-rate growth on the queries the system actually sees.
DEFAULT_EMBEDDING_THRESHOLD = 0.80

# Tier-3 classifier virtual model. Resolved through routes.yaml so the swap
# is a routes.yaml edit. Phase 2.2 wired `local-llm` (ollama / qwen2:7b,
# fallback claude-haiku-4-5) as a tier-3 candidate, but the qwen2:7b eval
# (eval/results/2026-06-17_router_baseline_qwen2.json) showed 91.0% overall
# (vs Haiku's 98.5%) with long-context collapsing to 66.7%, and CPU-bound
# inference at ~2s per call (vs Haiku's ~700ms). Default stays at `fast-qa`
# (Haiku) until GPU acceleration or a better-tuned local tier-3 is in place.
DEFAULT_CLASSIFIER_VIRTUAL_MODEL = "fast-qa"

# `>5k tokens of context` ≈ 20k chars at ~4 chars/token.
LONG_CONTEXT_CHAR_THRESHOLD = 20_000

# Tier-3 prompt JSON cap. 64 tokens is plenty for `{"category":"...","confidence":0.xx}`.
LLM_MAX_TOKENS = 64

# ---------------------------------------------------------------------------
# Heuristic patterns
# ---------------------------------------------------------------------------

_CODE_FENCE_RE = re.compile(r"```")
_FILE_PATH_RE = re.compile(
    r"(?:^|[\s(\[/])[\w\-./]*[\w\-]\."
    r"(?:py|pyi|ts|tsx|js|jsx|mjs|go|rs|java|cpp|cc|cxx|c|h|hpp|"
    r"rb|php|swift|kt|scala|cs|sh|bash|zsh|sql|yaml|yml|toml|json|md)"
    r"\b",
    re.IGNORECASE,
)
_STACK_TRACE_RE = re.compile(
    r"(?:File\s+\"[^\"]+\",\s+line\s+\d+"
    r"|at\s+\S+\s*\([^)]+:\d+(?::\d+)?\)"
    r"|Traceback\s*\(most recent call last\)"
    r"|Exception\s+in\s+thread\s+\")",
    re.IGNORECASE,
)
_CODE_VERB_RE = re.compile(
    r"\b(?:write|implement|debug|fix|refactor|rewrite|optimi[sz]e|port)"
    r"(?:\s+\w+){0,6}\s+"
    r"(?:function|script|class|method|module|loop|test|query|regex|"
    r"component|handler|endpoint|api|schema|migration)\b",
    re.IGNORECASE,
)
_FAST_QA_HEAD_RE = re.compile(
    r"^\s*(?:"
    r"what(?:'s|\s+(?:is|are|was|were|does|did))"
    r"|who(?:'s|\s+(?:is|was|are|were))"
    r"|when\s+(?:did|was|is|were)"
    r"|where\s+(?:is|are|was|were)"
    r"|how\s+(?:many|much|tall|deep|long|old|far|fast|big)"
    r"|define\b"
    r"|list\s+the\b"
    r"|translate\b"
    r")\b",
    re.IGNORECASE,
)
_DEEP_REASONING_RE = re.compile(
    r"\b(?:prove|derive|architect|trade[- ]?offs?|"
    r"design\s+(?:a|the|an|my))\b",
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# Negation handling (Tier 2's MiniLM blind spot — see phase_1_4 research doc)
# ---------------------------------------------------------------------------

NEG_TOKENS: frozenset[str] = frozenset({
    "not", "no", "never", "without", "isn't", "doesn't", "shouldn't", "won't",
    "neither", "nor", "don't", "wasn't", "aren't", "weren't", "can't", "cannot",
    "wouldn't", "couldn't", "hadn't", "hasn't", "haven't",
})
NEG_VERBS: frozenset[str] = frozenset({
    "allow", "permit", "summarize", "run", "include", "use", "do", "want",
    "need", "give", "show", "tell", "explain", "write", "answer",
})

_WORD_RE = re.compile(r"[\w']+")


def has_negation_near_verb(text: str) -> bool:
    """True if a negation token sits within a small window of a polarity verb.

    Asymmetric on purpose — bare 'not' doesn't fire (`what's NOT on this list?`
    is a fine fast-qa query). Pairs from the research doc.
    """
    tokens = _WORD_RE.findall(text.lower())
    for i, tok in enumerate(tokens):
        if tok in NEG_TOKENS:
            window = tokens[max(0, i - 2): i + 5]
            if any(w in NEG_VERBS for w in window):
                return True
    return False


# ---------------------------------------------------------------------------
# Decision shape
# ---------------------------------------------------------------------------

@dataclass
class Decision:
    category: str
    tier_fired: str  # "heuristic" | "embedding" | "llm"
    similarity: float | None = None
    confidence: float | None = None
    matched_exemplar: str | None = None  # debug aid for the eval reporter


# ---------------------------------------------------------------------------
# Helpers — message flattening
# ---------------------------------------------------------------------------

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


def _last_user_text(request: ChatCompletionRequest) -> str:
    for msg in reversed(request.messages):
        if msg.role == "user":
            return _flatten_content(msg.content)
    return ""


def _total_context_chars(request: ChatCompletionRequest) -> int:
    return sum(len(_flatten_content(m.content)) for m in request.messages)


# ---------------------------------------------------------------------------
# Tier 1 — heuristic
# ---------------------------------------------------------------------------

def _heuristic_route(
    query: str,
    *,
    total_chars: int = 0,
    max_tokens: int | None = None,
) -> str | None:
    """Confident-positive only. Returns None when nothing matches strongly."""
    # Structural — highest confidence
    if _CODE_FENCE_RE.search(query) or _FILE_PATH_RE.search(query) or _STACK_TRACE_RE.search(query):
        return "code"
    if total_chars > LONG_CONTEXT_CHAR_THRESHOLD:
        return "long-context"

    # Verb/phrase
    if _CODE_VERB_RE.search(query):
        return "code"
    if _DEEP_REASONING_RE.search(query):
        return "deep-reasoning"

    # Length/shape
    if max_tokens is not None and max_tokens > 4000 and len(query) < 200:
        return "deep-reasoning"
    if _FAST_QA_HEAD_RE.match(query) and len(query) < 100:
        return "fast-qa"

    return None


# ---------------------------------------------------------------------------
# Tier 2 — embedding lookup with polarity-flip check
# ---------------------------------------------------------------------------

async def _embedding_route(
    query: str,
    *,
    threshold: float = DEFAULT_EMBEDDING_THRESHOLD,
) -> tuple[str, float, str] | None:
    """Returns (category, similarity, matched_exemplar_text) on a confident hit."""
    vec = await embed(query)
    vec_literal = "[" + ",".join(f"{x:.8f}" for x in vec) + "]"
    async with acquire() as conn:
        row = await conn.fetchrow(
            "SELECT id, query_text, category, "
            "       1 - (embedding <=> $1::vector) AS similarity "
            "FROM router_exemplars "
            "ORDER BY embedding <=> $1::vector "
            "LIMIT 1",
            vec_literal,
        )
        if row is None:
            return None
        similarity = float(row["similarity"])
        if similarity < threshold:
            return None
        if has_negation_near_verb(query) != has_negation_near_verb(row["query_text"]):
            # Same neighbourhood, opposite polarity — let Tier 3 sort it out.
            obs_metrics.record_polarity_flip()
            obs_tracing.set_attrs(routing_polarity_flip=True)
            return None
        # Bump usage stats on the matched row so LRU eviction (future_work.md
        # section 3) can distinguish active exemplars from dead weight. Single
        # indexed UPDATE on the same connection — ~1ms, invisible at the
        # routing budget.
        await conn.execute(
            "UPDATE router_exemplars "
            "SET last_matched_at = NOW(), match_count = match_count + 1 "
            "WHERE id = $1",
            row["id"],
        )
    return (str(row["category"]), similarity, str(row["query_text"]))


# ---------------------------------------------------------------------------
# Tier 3 — LLM-as-router (Haiku via existing virtual model)
# ---------------------------------------------------------------------------

_LLM_SYSTEM_PROMPT = """You are a routing classifier. Pick exactly one category for the user's query and respond with a single JSON object — no prose, no code fences.

Categories:
- fast-qa: short factual lookups, definitions, unit conversions, simple translations, brief TL;DR summaries.
- code: writing, debugging, refactoring, or explaining code. Stack traces, file paths, code fences.
- deep-reasoning: multi-step reasoning, system design, derivations, architecture, tradeoff analysis.
- long-context: queries whose context exceeds ~5k tokens of attached material.

Examples:
  "What is the capital of France?" -> {"category":"fast-qa","confidence":0.99}
  "Refactor this for clarity: def f(x): return x*2" -> {"category":"code","confidence":0.92}
  "Design a sharded queue with at-most-once semantics" -> {"category":"deep-reasoning","confidence":0.95}
  "<5k+ tokens of attached doc>\\nSummarize" -> {"category":"long-context","confidence":0.9}

Output: {"category":"<one of fast-qa|code|deep-reasoning|long-context>","confidence":0.0..1.0}"""

_JSON_OBJ_RE = re.compile(r"\{.*?\}", re.DOTALL)


async def _llm_route(
    query: str,
    *,
    classifier_virtual_model: str = DEFAULT_CLASSIFIER_VIRTUAL_MODEL,
) -> tuple[str, float]:
    """Returns (category, confidence). Invalid JSON / bad category → deep-reasoning."""
    route = get_resolver().resolve(classifier_virtual_model)
    provider = get_provider(route.primary.provider)
    real_model = route.primary.model

    classify_req = ChatCompletionRequest(
        model=classifier_virtual_model,
        messages=[
            ChatMessage(role="system", content=_LLM_SYSTEM_PROMPT),
            ChatMessage(role="user", content=query),
        ],
        temperature=0.0,
        max_tokens=LLM_MAX_TOKENS,
    )

    text = ""
    try:
        resp = await provider.complete(classify_req, real_model)
        text = (resp.choices[0].message.content or "") if resp.choices else ""
        match = _JSON_OBJ_RE.search(text)
        if match is None:
            raise ValueError("no JSON object in response")
        data = json.loads(match.group(0))
        category = data.get("category")
        if category not in CATEGORIES:
            raise ValueError(f"invalid category {category!r}")
        confidence = float(data.get("confidence", 0.0))
        return category, confidence
    except Exception as exc:
        logger.warning(
            "router.llm_route.parse_fail",
            error=str(exc),
            response_text=text[:200],
        )
        return ("deep-reasoning", 0.0)


# ---------------------------------------------------------------------------
# Cascade entry point
# ---------------------------------------------------------------------------

async def classify(
    request: ChatCompletionRequest,
    *,
    embedding_threshold: float = DEFAULT_EMBEDDING_THRESHOLD,
    classifier_virtual_model: str = DEFAULT_CLASSIFIER_VIRTUAL_MODEL,
    routing_mode: str = "auto",
) -> Decision:
    """Cascade: heuristic → embedding → LLM. First confident tier wins.

    `routing_mode` is a label only — does not change the cascade. Phase 2.3
    uses it to break out auto / static / ab telemetry.
    """
    query = _last_user_text(request)
    total_chars = _total_context_chars(request)
    start = time.monotonic()

    async with obs_tracing.async_span(
        "routing.classify",
        routing_mode=routing_mode,
        query_chars=len(query),
        total_context_chars=total_chars,
    ):
        with obs_tracing.span("routing.tier.heuristic") as s:
            cat = _heuristic_route(query, total_chars=total_chars, max_tokens=request.max_tokens)
            if cat is not None:
                s.set_attribute("hit", True)
                s.set_attribute("category", cat)
                duration = time.monotonic() - start
                obs_metrics.record_routing_decision(
                    tier="heuristic", category=cat, routing_mode=routing_mode, duration_seconds=duration
                )
                logger.debug("router.classify", tier="heuristic", category=cat)
                return Decision(category=cat, tier_fired="heuristic")
            s.set_attribute("hit", False)

        async with obs_tracing.async_span("routing.tier.embedding") as s:
            embed_hit = await _embedding_route(query, threshold=embedding_threshold)
            if embed_hit is not None:
                cat, sim, matched = embed_hit
                s.set_attribute("hit", True)
                s.set_attribute("category", cat)
                s.set_attribute("similarity", sim)
                duration = time.monotonic() - start
                obs_metrics.record_routing_decision(
                    tier="embedding", category=cat, routing_mode=routing_mode, duration_seconds=duration
                )
                logger.debug("router.classify", tier="embedding", category=cat, similarity=sim)
                return Decision(
                    category=cat,
                    tier_fired="embedding",
                    similarity=sim,
                    matched_exemplar=matched,
                )
            s.set_attribute("hit", False)

        async with obs_tracing.async_span("routing.tier.llm") as s:
            cat, conf = await _llm_route(query, classifier_virtual_model=classifier_virtual_model)
            s.set_attribute("category", cat)
            s.set_attribute("confidence", conf)
            duration = time.monotonic() - start
            obs_metrics.record_routing_decision(
                tier="llm", category=cat, routing_mode=routing_mode, duration_seconds=duration
            )
            logger.debug("router.classify", tier="llm", category=cat, confidence=conf)
            return Decision(category=cat, tier_fired="llm", confidence=conf)
