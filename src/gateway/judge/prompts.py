"""Phase 2.5 — judge prompts.

Versioned: every time the system prompt changes, bump ``JUDGE_PROMPT_VERSION``
so cached verdicts don't get mixed across prompt revisions.

Spec v3.2 calls for three quality axes:
- ``correctness``    — is the answer factually right
- ``completeness``   — does it cover what was asked
- ``instruction_following`` — does it obey constraints in the prompt

The judge replies with a single JSON object so parsing is deterministic.
"""

# Bump on every prompt edit. Cached verdicts key on this so old verdicts
# don't shadow new prompts.
JUDGE_PROMPT_VERSION = "2026-06-19.v1"


JUDGE_SYSTEM_PROMPT = """You are an impartial evaluator scoring an LLM's response to a user query.

Score the response on three axes from 0.0 to 1.0:
- correctness: is the response factually right and free of hallucination?
- completeness: does it cover what the user asked for, no obvious gaps?
- instruction_following: does it obey constraints in the prompt (format, length, "no code", etc.)?

Be strict but fair. Reward concise answers — do NOT prefer long-winded responses to short correct ones. 
Score from the perspective of the user, not the model's style.

If an `expected` value is provided, treat that as ground truth: only mark correctness=1.0 if the response 
contains or matches it.

Reply with a SINGLE JSON object — no prose before or after, no code fences:

{
  "correctness": 0.0-1.0,
  "completeness": 0.0-1.0,
  "instruction_following": 0.0-1.0,
  "overall": 0.0-1.0,
  "passed": true|false,
  "reasoning": "one short sentence"
}

`overall` is your weighted aggregate (correctness should dominate). `passed` is true iff overall >= 0.7."""


def build_judge_user_prompt(
    query: str,
    response: str,
    *,
    expected: str | None = None,
    expected_category: str | None = None,
) -> str:
    """Build the user-turn payload for the judge. Trims for context safety."""
    # Trim aggressively — judges don't need a 24k-char haystack to score the
    # final answer, and Opus token cost scales with input.
    q = query if len(query) <= 4000 else query[:4000] + "\n…[truncated]"
    r = response if len(response) <= 4000 else response[:4000] + "\n…[truncated]"

    parts = ["=== USER QUERY ===", q, "", "=== MODEL RESPONSE ===", r]
    if expected:
        parts += ["", "=== EXPECTED CONTAINS (ground truth) ===", expected]
    if expected_category:
        parts += [
            "",
            f"(Routing context — query was expected to be category "
            f"{expected_category!r}; this is informational only, do not let it "
            f"bias correctness/completeness.)",
        ]
    parts += ["", "Now score the response. Reply with the JSON object only."]
    return "\n".join(parts)
