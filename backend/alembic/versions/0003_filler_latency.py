"""Record when a filler phrase was first heard, apart from the answer.

Revision ID: 0003
Revises: 0002

While a tool runs the line would otherwise be silent for a few seconds, so the
agent says a stopgap ("one moment"). Without a column of its own that would be
logged as the turn's first audio and make slow turns look fast. ``total_ms`` and
``tts_first_audio_ms`` keep meaning "when the answer arrived"; ``filler_ms`` says
when the silence ended, and is NULL for turns that needed no stopgap.

Additive and nullable: existing rows and a rolled-back deploy are both unaffected.
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0003"
down_revision: str | None = "0002"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column("call_turns", sa.Column("filler_ms", sa.Integer(), nullable=True))


def downgrade() -> None:
    op.drop_column("call_turns", "filler_ms")
