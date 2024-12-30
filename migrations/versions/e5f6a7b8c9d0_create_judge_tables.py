"""create_judge_tables

Revision ID: e5f6a7b8c9d0
Revises: c3f4d5e6a7b8
Create Date: 2026-06-19

Phase 2.5 — LLM-as-judge. Two tables:

* ``judge_verdicts`` is a result cache. The same (query_hash, response_hash,
  judge_model, prompt_version) tuple always yields the same verdict, so we
  never pay Opus twice for the identical pair. Used by both the eval runner
  (judge every row) and the production sampling backstop (judge 1%).
* ``judge_flags`` is the production backstop's hit list. Each row points at
  a sampled production request whose response the judge marked low-quality
  or whose category looked wrong — these surface as human-review fodder.

Additive only — see memory/feedback_permissions_mode.md.
"""
from typing import Sequence, Union

from alembic import op

revision: str = "e5f6a7b8c9d0"
down_revision: Union[str, Sequence[str], None] = "c3f4d5e6a7b8"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute(
        """
        CREATE TABLE judge_verdicts (
            id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
            query_hash TEXT NOT NULL,
            response_hash TEXT NOT NULL,
            judge_model TEXT NOT NULL,
            prompt_version TEXT NOT NULL,
            scores JSONB NOT NULL,
            overall_score NUMERIC(4, 3) NOT NULL,
            passed BOOLEAN NOT NULL,
            reasoning TEXT,
            judge_latency_ms INTEGER NOT NULL DEFAULT 0,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            CONSTRAINT uniq_judge_verdicts UNIQUE
                (query_hash, response_hash, judge_model, prompt_version)
        )
        """
    )
    op.execute(
        "CREATE INDEX idx_judge_verdicts_created "
        "ON judge_verdicts (created_at DESC)"
    )

    op.execute(
        """
        CREATE TABLE judge_flags (
            id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
            verdict_id UUID NOT NULL REFERENCES judge_verdicts(id) ON DELETE CASCADE,
            tenant_id TEXT NOT NULL,
            request_id UUID,
            virtual_model TEXT,
            routed_category TEXT,
            tier_fired TEXT,
            flag_reason TEXT NOT NULL,
            query_preview TEXT NOT NULL,
            response_preview TEXT NOT NULL,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        )
        """
    )
    op.execute(
        "CREATE INDEX idx_judge_flags_created "
        "ON judge_flags (created_at DESC)"
    )
    op.execute(
        "CREATE INDEX idx_judge_flags_tenant "
        "ON judge_flags (tenant_id, created_at DESC)"
    )


def downgrade() -> None:
    raise NotImplementedError(
        "Migrations are additive only — see memory/feedback_permissions_mode.md"
    )
