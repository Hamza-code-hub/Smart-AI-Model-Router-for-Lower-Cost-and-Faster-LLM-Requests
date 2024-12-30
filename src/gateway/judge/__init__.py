"""Phase 2.5 — LLM-as-judge.

Public surface:

* ``JudgeVerdict`` / ``JudgeError`` dataclasses
* ``judge_response()`` — score one (query, response) pair through one judge
* ``judge_ensemble()`` — average multiple judges
* ``sample_and_judge()`` — production 1% backstop entry point
"""
from gateway.judge.judge import (
    JUDGE_PROMPT_VERSION,
    JudgeError,
    JudgeVerdict,
    judge_ensemble,
    judge_response,
)
from gateway.judge.sampler import sample_and_judge

__all__ = [
    "JUDGE_PROMPT_VERSION",
    "JudgeError",
    "JudgeVerdict",
    "judge_ensemble",
    "judge_response",
    "sample_and_judge",
]
