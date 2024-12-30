"""create_conversation_compaction_table

Revision ID: 7c4a1b89d2e3
Revises: fa96a430b391
Create Date: 2026-06-15

v2.5 conversation compaction store. Distinct from Phase 1.5 sticky-routing
state, which lives in Redis under `router:conv:{id}`.

Where the table is used
-----------------------
- src/gateway/compaction/strategy.py — SELECT to load prior summary, UPSERT
  to persist a new one, DELETE for the admin clear endpoint.
- src/gateway/routing/conversation_state.py — SELECT updated_at as the
  "compaction fired on previous turn" signal that relaxes the Phase 1.5
  escalation gap.

Columns
-------
- conversation_id     TEXT         PRIMARY KEY. Caller-supplied
                                   `X-Conversation-Id`; one row per
                                   conversation, idempotent upsert target.
- tenant_id           TEXT NOT NULL Tenant that owns the conversation; leads
                                   the `idx_conversation_compaction_tenant`
                                   index for tenant-scoped admin lookups.
- virtual_model       TEXT NOT NULL Which route compacted this conversation
                                   (e.g. `deep-reasoning`); compaction config
                                   is per-route.
- last_compacted_turn INTEGER      Index of the last message folded into the
                                   summary. Anything past this index is the
                                   verbatim tail.
- summary             JSONB        Structured summary payload
                                   `{key_facts, decisions, open_questions,
                                   sticky_facts}` — rendered into a synthetic
                                   system message on every subsequent turn.
- sticky_facts        JSONB        Verbatim facts (version numbers, file
                                   paths, IDs) preserved across summaries via
                                   forward union-merge so paraphrase cannot
                                   corrupt them. Defaults to `'[]'`.
- updated_at          TIMESTAMPTZ  When the row last changed. Doubles as the
                                   routing signal read by Phase 1.5.

Scenario
--------
Turn 18 of a `deep-reasoning` debugging conversation arrives at ~22k tokens,
above the 20k threshold. `_load_state` reads `last_compacted_turn`, `summary`,
`sticky_facts` for `conv_xyz` — no prior row, so first-time compaction. Haiku
summarizes the oldest 14 turns; sticky facts `["Python 3.11", "0042_users.sql"]`
are extracted. `_save_state` UPSERTs: `tenant_id='acme'` for admin scoping,
`virtual_model='deep-reasoning'` records the route, `last_compacted_turn=14`
marks the verbatim cutoff, `summary` holds the JSONB payload, `sticky_facts`
holds the preserved tokens, `updated_at=NOW()`. On turn 19 the router reads
`updated_at`, detects fresh compaction, and relaxes its escalation gap.
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "7c4a1b89d2e3"
down_revision: Union[str, Sequence[str], None] = "fa96a430b391"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "conversation_compaction",
        sa.Column("conversation_id", sa.Text(), primary_key=True),
        sa.Column("tenant_id", sa.Text(), nullable=False),
        sa.Column("virtual_model", sa.Text(), nullable=False),
        sa.Column("last_compacted_turn", sa.Integer(), nullable=False),
        sa.Column("summary", postgresql.JSONB(), nullable=False),
        sa.Column("sticky_facts", postgresql.JSONB(), nullable=False, server_default="[]"),
        sa.Column(
            "updated_at",
            sa.TIMESTAMP(timezone=True),
            nullable=False,
            server_default=sa.text("NOW()"),
        ),
    )

    op.create_index(
        "idx_conversation_compaction_tenant",
        "conversation_compaction",
        ["tenant_id", sa.text("updated_at DESC")],
    )


def downgrade() -> None:
    op.drop_index("idx_conversation_compaction_tenant", table_name="conversation_compaction")
    op.drop_table("conversation_compaction")
