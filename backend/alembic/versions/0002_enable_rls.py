"""Enable row level security on every table.

Revision ID: 0002
Revises: 0001

Supabase serves the `public` schema over a REST API authenticated with the
project's anon key -- a key designed to be embedded in browsers. Without RLS,
anyone holding it can read or write every row: customer names and phone numbers
in `bookings`, and full transcripts in `call_turns`.

RLS is enabled with no policies, which is a deny-all for that API. That is the
correct posture here because nothing legitimate uses the anon key: the backend
is the only client, it connects over Postgres as the table owner, and an owner
bypasses RLS unless FORCE ROW LEVEL SECURITY is set. So this closes the REST
surface and changes nothing about how the application behaves.

If a browser ever needs to read this data directly -- a customer-facing booking
page, say -- that is when policies get written, one per table, granting exactly
what that page needs. The owner dashboard in Phase 5 does not count: it talks to
our own API, not to Supabase.
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "0002"
down_revision: str | None = "0001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

# alembic_version is included deliberately. It is not sensitive, but leaving one
# table exposed keeps the project permanently failing Supabase's linter, and a
# warning people learn to ignore is worse than no warning.
TABLES = (
    "businesses",
    "services",
    "staff",
    "availability_rules",
    "bookings",
    "calls",
    "call_turns",
    "alembic_version",
)


def upgrade() -> None:
    for table in TABLES:
        op.execute(f'ALTER TABLE "public"."{table}" ENABLE ROW LEVEL SECURITY')


def downgrade() -> None:
    for table in TABLES:
        op.execute(f'ALTER TABLE "public"."{table}" DISABLE ROW LEVEL SECURITY')
