"""Phase 2.4 — eval workload loader.

A *workload* is a versioned YAML file in ``eval/workloads/`` that defines a
deterministic sequence of prompts (single-turn or multi-turn) to run against
one or more target gateways. Three variants:

1. **prompts** — explicit per-prompt definitions. Each prompt has
   ``messages`` (single-turn) or ``turns`` (multi-turn).
2. **generator** — programmatic prompt synthesis (e.g. needle-in-haystack
   at a target context size). Keeps the YAML small even when individual
   prompts are large.
3. **mix** — references other workloads with weights, deterministically
   sampled by ``seed``. Used for ``mixed_realistic.yaml``.

The loader is the single source of truth on prompt shape for the runner /
reporter — they consume the dataclasses below, never raw YAML.
"""
from __future__ import annotations

import hashlib
import random
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


WORKLOAD_DIR_DEFAULT = Path("eval/workloads")

VALID_ROLES = {"system", "user", "assistant"}


@dataclass(frozen=True)
class Prompt:
    """A single executable prompt — either single-turn (one user message)
    or multi-turn (replay user turns one at a time, accumulating assistant
    responses on a single conversation id).
    """

    id: str
    # Pre-existing messages prepended to every turn (e.g. system messages).
    seed_messages: tuple[dict[str, str], ...] = ()
    # For single-turn: one entry. For multi-turn: each is a user turn the
    # runner sends sequentially, accumulating the assistant response between.
    user_turns: tuple[str, ...] = ()
    conversation_id: str | None = None
    expected_category: str | None = None
    expected_contains: tuple[str, ...] = ()
    max_tokens: int | None = None
    temperature: float | None = None
    notes: str | None = None

    @property
    def is_multi_turn(self) -> bool:
        return len(self.user_turns) > 1


@dataclass(frozen=True)
class Workload:
    name: str
    version: int
    description: str
    prompts: tuple[Prompt, ...]
    default_max_tokens: int | None = None
    default_temperature: float | None = None
    source_path: Path | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.prompts)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load_workload(name: str, *, workload_dir: Path = WORKLOAD_DIR_DEFAULT) -> Workload:
    """Resolve ``name`` to ``<workload_dir>/<name>.yaml`` and parse it."""
    path = workload_dir / f"{name}.yaml"
    if not path.exists():
        available = sorted(p.stem for p in workload_dir.glob("*.yaml"))
        raise FileNotFoundError(
            f"Workload {name!r} not found at {path}. Available: {available}"
        )
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return _parse_workload(data, path, workload_dir=workload_dir)


def _parse_workload(
    data: dict[str, Any], path: Path, *, workload_dir: Path
) -> Workload:
    name = data.get("name") or path.stem
    version = int(data.get("version", 1))
    description = data.get("description", "")
    default_max_tokens = data.get("default_max_tokens")
    default_temperature = data.get("default_temperature")

    prompts: tuple[Prompt, ...]
    if "prompts" in data:
        prompts = tuple(_parse_prompt(p, idx) for idx, p in enumerate(data["prompts"], 1))
    elif "generator" in data:
        prompts = tuple(_run_generator(data["generator"]))
    elif "mix" in data:
        prompts = tuple(_build_mix(data, workload_dir=workload_dir))
    elif "source_router_eval" in data:
        prompts = tuple(_load_from_router_eval(data["source_router_eval"], path))
    else:
        raise ValueError(
            f"{path}: workload must define 'prompts', 'generator', 'mix', or 'source_router_eval'"
        )

    return Workload(
        name=name,
        version=version,
        description=description,
        prompts=prompts,
        default_max_tokens=default_max_tokens,
        default_temperature=default_temperature,
        source_path=path,
        metadata={k: v for k, v in data.items() if k.startswith("meta_")},
    )


def _parse_prompt(raw: dict[str, Any], idx: int) -> Prompt:
    prompt_id = str(raw.get("id") or f"p{idx:03d}")

    seed_messages: list[dict[str, str]] = []
    user_turns: list[str] = []

    if "messages" in raw and "turns" in raw:
        raise ValueError(f"prompt {prompt_id}: use one of 'messages' or 'turns'")

    if "messages" in raw:
        # Single-turn shape — the final user message is the executable turn,
        # everything before it (system / assistant priming) is seed_messages.
        msgs = list(raw["messages"])
        if not msgs or msgs[-1].get("role") != "user":
            raise ValueError(
                f"prompt {prompt_id}: 'messages' must end with a user role"
            )
        for m in msgs[:-1]:
            role = m.get("role")
            if role not in VALID_ROLES:
                raise ValueError(f"prompt {prompt_id}: invalid role {role!r}")
            seed_messages.append({"role": role, "content": _as_str(m.get("content"))})
        user_turns.append(_as_str(msgs[-1].get("content")))
    elif "turns" in raw:
        # Multi-turn shape — list of user turns (strings).
        turns = list(raw["turns"])
        for t in turns:
            if isinstance(t, str):
                user_turns.append(t)
            elif isinstance(t, dict) and t.get("role") == "user":
                user_turns.append(_as_str(t.get("content")))
            else:
                raise ValueError(
                    f"prompt {prompt_id}: 'turns' entries must be a user string"
                )
        if "system" in raw:
            seed_messages.insert(0, {"role": "system", "content": str(raw["system"])})
    else:
        raise ValueError(f"prompt {prompt_id}: missing 'messages' or 'turns'")

    expected_contains = tuple(
        str(x).lower() for x in (raw.get("expected_contains") or [])
    )

    return Prompt(
        id=prompt_id,
        seed_messages=tuple(seed_messages),
        user_turns=tuple(user_turns),
        conversation_id=raw.get("conversation_id"),
        expected_category=raw.get("expected_category"),
        expected_contains=expected_contains,
        max_tokens=raw.get("max_tokens"),
        temperature=raw.get("temperature"),
        notes=raw.get("notes"),
    )


def _as_str(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    # OpenAI's typed-content arrays land as list[dict] — flatten into text.
    if isinstance(content, list):
        parts: list[str] = []
        for chunk in content:
            if isinstance(chunk, dict) and "text" in chunk:
                parts.append(str(chunk["text"]))
        return "\n".join(parts)
    return str(content)


# ---------------------------------------------------------------------------
# Generators
# ---------------------------------------------------------------------------


def _run_generator(spec: dict[str, Any]) -> Iterable[Prompt]:
    gen_type = spec.get("type")
    if gen_type == "needle_in_haystack":
        return _gen_needle_in_haystack(spec)
    raise ValueError(f"unknown generator type: {gen_type!r}")


def _gen_needle_in_haystack(spec: dict[str, Any]) -> Iterable[Prompt]:
    """Synthesize long-context prompts: a haystack of filler paragraphs with
    a 'needle' sentence at a controlled depth, then ask a question that
    only the needle can answer.

    Deterministic given ``seed``. Sentence count is the budget — we don't
    pretend to count tokens precisely, the runner / target log actual usage.
    """
    target_chars = int(spec.get("haystack_size_chars", 8000))
    num_prompts = int(spec.get("num_prompts", 5))
    seed = int(spec.get("seed", 0))
    depths = spec.get("depths") or [0.25, 0.5, 0.75]

    rng = random.Random(seed)
    needles = [
        ("magic word", "purple-trombone-42"),
        ("secret code", "ZX-99-OMEGA"),
        ("favorite number", "73-and-a-half"),
        ("project codename", "Operation Birchwood"),
        ("treasure room", "the third door on the left in the north tower"),
    ]

    filler_pool = _NEEDLE_FILLER_SENTENCES

    out: list[Prompt] = []
    for i in range(num_prompts):
        topic, fact = needles[i % len(needles)]
        depth = depths[i % len(depths)]
        depth = float(depth)
        needle_sentence = f"Important note: the {topic} is {fact}."
        question = f"What is the {topic}? Reply with only the value, no explanation."

        sentences: list[str] = []
        total_chars = 0
        while total_chars < target_chars:
            s = rng.choice(filler_pool)
            sentences.append(s)
            total_chars += len(s) + 1
        # Insert the needle at the configured depth.
        insert_at = max(0, min(len(sentences), int(len(sentences) * depth)))
        sentences.insert(insert_at, needle_sentence)

        haystack = " ".join(sentences)
        user_msg = (
            f"Below is a long document. Read it carefully, then answer the "
            f"question at the end.\n\n"
            f"---\n{haystack}\n---\n\n"
            f"{question}"
        )
        prompt = Prompt(
            id=f"nh-{i+1:03d}",
            user_turns=(user_msg,),
            expected_category="long-context",
            expected_contains=(fact.lower(),),
            max_tokens=64,
            # Don't pin temperature here — Opus 4-7 rejects it as deprecated.
            notes=f"needle at depth={depth}, haystack≈{len(haystack)} chars",
        )
        out.append(prompt)
    return out


_NEEDLE_FILLER_SENTENCES = (
    "The library was unusually quiet that afternoon.",
    "Outside, the rain had begun to fall against the cobblestones.",
    "A small clock on the desk ticked steadily toward five.",
    "She turned the page and continued reading in silence.",
    "The old map had been folded so many times its creases were nearly translucent.",
    "Somewhere down the hall, a door closed with a soft click.",
    "The cat regarded the stranger with imperial indifference.",
    "There was a faint smell of pine and ink in the room.",
    "He scribbled a note in the margin and underlined it twice.",
    "The river bent sharply to the east before disappearing into the trees.",
    "A row of brass keys hung from hooks beside the kitchen door.",
    "Most of the chairs in the parlor were older than the house itself.",
    "Lanterns swung gently from the rafters of the inn.",
    "The compass needle wavered, then settled firmly on north-northwest.",
    "Voices in the courtyard rose, then fell, then rose again.",
    "A single candle burned on the windowsill, oblivious to the wind.",
    "The wagon wheels creaked over the loose gravel of the lane.",
    "She tucked the letter into her coat pocket and stepped outside.",
    "Three small boats were tied to the dock, knocking gently together.",
    "The bookkeeper's ledger lay open to a page covered in tiny figures.",
)


# ---------------------------------------------------------------------------
# Router-eval bridge
# ---------------------------------------------------------------------------


def _load_from_router_eval(source: str, workload_path: Path) -> Iterable[Prompt]:
    """Unpack rows from ``eval/router_eval.yaml`` into Prompt objects so the
    Phase 1.3 router corpus runs through the Phase 2.4 framework.
    """
    # Resolve relative to repo root (parent of eval/).
    repo_root = workload_path.parent.parent.parent
    source_path = repo_root / source
    if not source_path.exists():
        # Fall back to relative-to-cwd.
        source_path = Path(source)
    if not source_path.exists():
        raise FileNotFoundError(f"router-eval source not found: {source}")
    data = yaml.safe_load(source_path.read_text(encoding="utf-8")) or {}
    rows = data.get("queries") or []
    out: list[Prompt] = []
    for i, row in enumerate(rows, 1):
        query = row.get("query")
        if not query:
            continue
        out.append(
            Prompt(
                id=f"re-{i:03d}",
                user_turns=(query,),
                expected_category=row.get("expected_category"),
                notes=row.get("notes"),
                max_tokens=120,
                temperature=0.0,
            )
        )
    return out


# ---------------------------------------------------------------------------
# Mixing
# ---------------------------------------------------------------------------


def _build_mix(data: dict[str, Any], *, workload_dir: Path) -> Iterable[Prompt]:
    """Sample prompts from referenced workloads at the given weights."""
    mix: list[dict[str, Any]] = data["mix"]
    total = int(data.get("total_prompts", 20))
    seed = int(data.get("seed", 0))
    rng = random.Random(seed)

    # Eagerly load all source workloads.
    sources: list[tuple[Workload, float]] = []
    for entry in mix:
        wl = load_workload(entry["workload"], workload_dir=workload_dir)
        sources.append((wl, float(entry.get("weight", 1.0))))

    weight_sum = sum(w for _, w in sources)
    if weight_sum <= 0:
        raise ValueError("mix weights must sum to > 0")

    out: list[Prompt] = []
    for i in range(total):
        pick = rng.random() * weight_sum
        acc = 0.0
        chosen: Workload | None = None
        for wl, w in sources:
            acc += w
            if pick <= acc:
                chosen = wl
                break
        assert chosen is not None
        source_prompt = rng.choice(chosen.prompts)
        # Rewrite id so duplicates from the same workload are distinguishable.
        new_id = f"mix-{i+1:03d}-{chosen.name}-{source_prompt.id}"
        # Also rewrite conversation_id so multi-turn prompts pulled twice
        # don't collide on the gateway's per-conversation router state.
        new_conv_id = (
            f"{source_prompt.conversation_id}-{i+1}"
            if source_prompt.conversation_id
            else None
        )
        out.append(
            Prompt(
                id=new_id,
                seed_messages=source_prompt.seed_messages,
                user_turns=source_prompt.user_turns,
                conversation_id=new_conv_id,
                expected_category=source_prompt.expected_category,
                expected_contains=source_prompt.expected_contains,
                max_tokens=source_prompt.max_tokens,
                temperature=source_prompt.temperature,
                notes=source_prompt.notes,
            )
        )
    return out


# ---------------------------------------------------------------------------
# Hash for results metadata
# ---------------------------------------------------------------------------


def workload_fingerprint(wl: Workload) -> str:
    """sha256 over (id, user_turns) for every prompt — pins the result file
    to the exact workload contents so diffs across runs are meaningful even
    if the YAML gets edited.
    """
    h = hashlib.sha256()
    h.update(f"{wl.name}@{wl.version}\n".encode("utf-8"))
    for p in wl.prompts:
        h.update(p.id.encode("utf-8"))
        h.update(b"\0")
        for t in p.user_turns:
            h.update(t.encode("utf-8"))
            h.update(b"\0")
    return h.hexdigest()[:16]
