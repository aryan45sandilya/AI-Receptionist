"""Initial schema: businesses, services, staff, availability_rules, bookings,
calls, call_turns -- plus the constraint that makes double-booking impossible.

Revision ID: 0001
Revises:
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0001"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


# Enum types are created explicitly rather than implicitly by the first table
# that uses one. Left implicit, a type referenced by two tables gets a duplicate
# CREATE TYPE and the migration fails halfway through.
BOOKING_STATUS = postgresql.ENUM(
    "booked", "cancelled", "completed", "no_show",
    name="booking_status", create_type=False,
)
BOOKING_SOURCE = postgresql.ENUM(
    "voice", "whatsapp", "dashboard", name="booking_source", create_type=False
)
CALL_CHANNEL = postgresql.ENUM(
    "web_demo", "twilio_voice", "whatsapp", "terminal",
    name="call_channel", create_type=False,
)
CALL_STATUS = postgresql.ENUM(
    "in_progress", "completed", "failed", "handed_off",
    name="call_status", create_type=False,
)
TURN_ROLE = postgresql.ENUM("caller", "agent", name="turn_role", create_type=False)

ALL_ENUMS = (BOOKING_STATUS, BOOKING_SOURCE, CALL_CHANNEL, CALL_STATUS, TURN_ROLE)


def upgrade() -> None:
    bind = op.get_bind()

    # btree_gist supplies the GiST operator class for '=' on a uuid, which the
    # exclusion constraint at the bottom of this file needs in order to combine
    # an equality test with a range-overlap test in one index.
    op.execute("CREATE EXTENSION IF NOT EXISTS btree_gist")

    for enum_type in ALL_ENUMS:
        enum_type.create(bind, checkfirst=True)

    op.create_table(
        "businesses",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("name", sa.String(200), nullable=False),
        sa.Column("timezone", sa.String(64), nullable=False, server_default="Asia/Kolkata"),
        sa.Column("phone_e164", sa.String(20), nullable=False, unique=True),
        sa.Column("handoff_phone_e164", sa.String(20)),
        sa.Column("whatsapp_phone_id", sa.String(64)),
        sa.Column("greeting", sa.Text, nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False,
                  server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False,
                  server_default=sa.func.now()),
    )

    op.create_table(
        "staff",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("business_id", postgresql.UUID(as_uuid=True),
                  sa.ForeignKey("businesses.id", ondelete="CASCADE"), nullable=False),
        sa.Column("name", sa.String(120), nullable=False),
        sa.Column("active", sa.Boolean, nullable=False, server_default=sa.true()),
    )

    op.create_table(
        "services",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("business_id", postgresql.UUID(as_uuid=True),
                  sa.ForeignKey("businesses.id", ondelete="CASCADE"), nullable=False),
        sa.Column("name", sa.String(120), nullable=False),
        sa.Column("duration_min", sa.Integer, nullable=False),
        sa.Column("price_paise", sa.Integer),
        sa.Column("active", sa.Boolean, nullable=False, server_default=sa.true()),
        sa.UniqueConstraint("business_id", "name", name="services_unique_name_per_business"),
        sa.CheckConstraint("duration_min > 0", name="services_positive_duration"),
    )

    op.create_table(
        "availability_rules",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("business_id", postgresql.UUID(as_uuid=True),
                  sa.ForeignKey("businesses.id", ondelete="CASCADE"), nullable=False),
        # NULL staff_id means the rule covers the business as a whole.
        sa.Column("staff_id", postgresql.UUID(as_uuid=True),
                  sa.ForeignKey("staff.id", ondelete="CASCADE")),
        # Monday=0, matching Python's date.weekday(). Postgres EXTRACT(DOW) is
        # Sunday=0, so SQL comparing against this column must convert.
        sa.Column("weekday", sa.SmallInteger, nullable=False),
        # TIME, not TIMESTAMPTZ: opening hours are wall-clock facts.
        sa.Column("start_time", sa.Time, nullable=False),
        sa.Column("end_time", sa.Time, nullable=False),
        sa.CheckConstraint("weekday BETWEEN 0 AND 6", name="availability_weekday_range"),
        sa.CheckConstraint("end_time > start_time", name="availability_end_after_start"),
    )
    op.create_index("availability_by_business_weekday", "availability_rules",
                    ["business_id", "weekday"])

    op.create_table(
        "calls",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("business_id", postgresql.UUID(as_uuid=True),
                  sa.ForeignKey("businesses.id", ondelete="CASCADE"), nullable=False),
        sa.Column("channel", CALL_CHANNEL, nullable=False),
        sa.Column("from_phone_e164", sa.String(20)),
        sa.Column("twilio_call_sid", sa.String(64), unique=True),
        sa.Column("status", CALL_STATUS, nullable=False, server_default="in_progress"),
        sa.Column("handoff_reason", sa.Text),
        sa.Column("transcript_summary", sa.Text),
        sa.Column("recording_url", sa.Text),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=False,
                  server_default=sa.func.now()),
        sa.Column("ended_at", sa.DateTime(timezone=True)),
        sa.Column("duration_ms", sa.Integer),
    )

    op.create_table(
        "bookings",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("reference", sa.String(8), nullable=False, unique=True),
        sa.Column("business_id", postgresql.UUID(as_uuid=True),
                  sa.ForeignKey("businesses.id", ondelete="CASCADE"), nullable=False),
        sa.Column("service_id", postgresql.UUID(as_uuid=True),
                  sa.ForeignKey("services.id"), nullable=False),
        sa.Column("staff_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("staff.id")),
        sa.Column("call_id", postgresql.UUID(as_uuid=True),
                  sa.ForeignKey("calls.id", ondelete="SET NULL")),
        sa.Column("customer_name", sa.String(120), nullable=False),
        sa.Column("customer_phone_e164", sa.String(20), nullable=False),
        sa.Column("slot_start", sa.DateTime(timezone=True), nullable=False),
        sa.Column("slot_end", sa.DateTime(timezone=True), nullable=False),
        sa.Column("status", BOOKING_STATUS, nullable=False, server_default="booked"),
        sa.Column("source", BOOKING_SOURCE, nullable=False),
        sa.Column("notes", sa.Text),
        sa.Column("google_event_id", sa.String(128)),
        # The resource being booked: a named staff member, or the business
        # itself. Generated, because NULL != NULL would leave two
        # business-level bookings free to overlap under the constraint below.
        sa.Column(
            "resource_id",
            postgresql.UUID(as_uuid=True),
            sa.Computed("COALESCE(staff_id, business_id)", persisted=True),
            nullable=False,
        ),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False,
                  server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False,
                  server_default=sa.func.now()),
        sa.Column("cancelled_at", sa.DateTime(timezone=True)),
        sa.CheckConstraint("slot_end > slot_start", name="bookings_end_after_start"),
    )
    op.create_index("bookings_by_business_slot", "bookings", ["business_id", "slot_start"])
    op.create_index("bookings_by_phone", "bookings",
                    ["business_id", "customer_phone_e164"])

    op.create_table(
        "call_turns",
        sa.Column("id", sa.BigInteger, primary_key=True, autoincrement=True),
        sa.Column("call_id", postgresql.UUID(as_uuid=True),
                  sa.ForeignKey("calls.id", ondelete="CASCADE"), nullable=False),
        sa.Column("turn_index", sa.Integer, nullable=False),
        sa.Column("role", TURN_ROLE, nullable=False),
        sa.Column("text", sa.Text),
        # One anchor plus integer ms offsets; all offsets nullable so a turn
        # that fails partway still records how far it got.
        sa.Column("turn_started_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("stt_final_ms", sa.Integer),
        sa.Column("llm_first_token_ms", sa.Integer),
        sa.Column("llm_done_ms", sa.Integer),
        sa.Column("tts_first_audio_ms", sa.Integer),
        sa.Column("total_ms", sa.Integer),
        sa.Column("barge_in", sa.Boolean, nullable=False, server_default=sa.false()),
        sa.Column("tool_calls", postgresql.JSONB),
        sa.Column("model", sa.String(64)),
        sa.Column("input_tokens", sa.Integer),
        sa.Column("output_tokens", sa.Integer),
        sa.Column("cache_read_tokens", sa.Integer),
        sa.UniqueConstraint("call_id", "turn_index", name="call_turns_unique_index"),
    )

    # ----------------------------------------------------------------------
    # The double-booking defence.
    #
    # Slots are computed from availability_rules, never stored, so there is no
    # row for "6:30pm Tuesday" to SELECT ... FOR UPDATE. This constraint is
    # therefore the serialisation point: two concurrent transactions inserting
    # overlapping bookings for one resource cannot both commit, whatever the
    # application code does. No lock ordering, so no deadlock.
    #
    # Why not UNIQUE (business_id, slot_start)?
    #   - a cancelled booking would keep occupying the key, so the slot could
    #     never be resold -- hence the WHERE clause here;
    #   - it only catches identical start times. A 60-minute booking at 10:00
    #     and a 30-minute one at 10:30 have different slot_start values and
    #     overlap anyway. The range test catches that; an equality test cannot.
    #
    # tstzrange defaults to '[)' bounds, so 10:00-10:30 and 10:30-11:00 are
    # adjacent rather than overlapping -- exactly right for appointments, and
    # matching domain/timeslots.py so both layers agree on what collides.
    # ----------------------------------------------------------------------
    op.execute(
        """
        ALTER TABLE bookings ADD CONSTRAINT bookings_no_overlap
        EXCLUDE USING gist (
            resource_id WITH =,
            tstzrange(slot_start, slot_end) WITH &&
        ) WHERE (status = 'booked')
        """
    )


def downgrade() -> None:
    op.execute("ALTER TABLE bookings DROP CONSTRAINT IF EXISTS bookings_no_overlap")
    op.drop_table("call_turns")
    op.drop_index("bookings_by_phone", table_name="bookings")
    op.drop_index("bookings_by_business_slot", table_name="bookings")
    op.drop_table("bookings")
    op.drop_table("calls")
    op.drop_index("availability_by_business_weekday", table_name="availability_rules")
    op.drop_table("availability_rules")
    op.drop_table("services")
    op.drop_table("staff")
    op.drop_table("businesses")

    bind = op.get_bind()
    for enum_type in ALL_ENUMS:
        enum_type.drop(bind, checkfirst=True)
    # btree_gist is left installed: other things may depend on it, and dropping
    # an extension is not this migration's business.
