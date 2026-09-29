"""SQLAlchemy models for the seven tables.

Read alongside ``backend/alembic/versions/0001_initial.py``, which owns the one
piece of schema that cannot be expressed as a column: the GiST exclusion
constraint that makes double-booking impossible.
"""

from __future__ import annotations

import uuid
from datetime import datetime, time

from sqlalchemy import (
    BigInteger,
    Boolean,
    CheckConstraint,
    Computed,
    DateTime,
    Enum,
    ForeignKey,
    Index,
    Integer,
    SmallInteger,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID as PgUUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

# Postgres enum types. Declared once here so the Python enum and the database
# type can never drift apart.
from .enums import BookingSource, BookingStatus, CallChannel, CallStatus, TurnRole


class Base(DeclarativeBase):
    pass


def _uuid_pk() -> Mapped[uuid.UUID]:
    return mapped_column(PgUUID(as_uuid=True), primary_key=True, default=uuid.uuid4)


def _tstz(**kwargs) -> Mapped[datetime]:
    return mapped_column(DateTime(timezone=True), **kwargs)


class Business(Base):
    __tablename__ = "businesses"

    id: Mapped[uuid.UUID] = _uuid_pk()
    name: Mapped[str] = mapped_column(String(200), nullable=False)
    # An IANA zone name, never a UTC offset: offsets do not survive DST, and
    # "Asia/Kolkata" keeps meaning the right thing if the rules ever change.
    timezone: Mapped[str] = mapped_column(String(64), nullable=False, default="Asia/Kolkata")
    # The number callers dial. This is how an inbound call finds its business.
    phone_e164: Mapped[str] = mapped_column(String(20), unique=True, nullable=False)
    handoff_phone_e164: Mapped[str | None] = mapped_column(String(20))
    whatsapp_phone_id: Mapped[str | None] = mapped_column(String(64))
    greeting: Mapped[str] = mapped_column(Text, nullable=False)

    created_at: Mapped[datetime] = _tstz(server_default=func.now(), nullable=False)
    updated_at: Mapped[datetime] = _tstz(
        server_default=func.now(), onupdate=func.now(), nullable=False
    )

    services: Mapped[list[Service]] = relationship(back_populates="business")
    staff: Mapped[list[Staff]] = relationship(back_populates="business")


class Service(Base):
    __tablename__ = "services"
    __table_args__ = (
        # Two services with the same name would make the agent's choice
        # ambiguous, and it has no way to ask which one is meant.
        UniqueConstraint("business_id", "name", name="services_unique_name_per_business"),
        CheckConstraint("duration_min > 0", name="services_positive_duration"),
    )

    id: Mapped[uuid.UUID] = _uuid_pk()
    business_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("businesses.id", ondelete="CASCADE"), nullable=False
    )
    name: Mapped[str] = mapped_column(String(120), nullable=False)
    duration_min: Mapped[int] = mapped_column(Integer, nullable=False)
    # Money as integer minor units. Floats are the wrong type for currency.
    price_paise: Mapped[int | None] = mapped_column(Integer)
    active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)

    business: Mapped[Business] = relationship(back_populates="services")


class Staff(Base):
    __tablename__ = "staff"

    id: Mapped[uuid.UUID] = _uuid_pk()
    business_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("businesses.id", ondelete="CASCADE"), nullable=False
    )
    name: Mapped[str] = mapped_column(String(120), nullable=False)
    active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)

    business: Mapped[Business] = relationship(back_populates="staff")


class AvailabilityRule(Base):
    """One recurring weekly opening window."""

    __tablename__ = "availability_rules"
    __table_args__ = (
        CheckConstraint("weekday BETWEEN 0 AND 6", name="availability_weekday_range"),
        CheckConstraint("end_time > start_time", name="availability_end_after_start"),
        Index("availability_by_business_weekday", "business_id", "weekday"),
    )

    id: Mapped[uuid.UUID] = _uuid_pk()
    business_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("businesses.id", ondelete="CASCADE"), nullable=False
    )
    # NULL means the rule applies to the business as a whole ("anyone can see you").
    staff_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("staff.id", ondelete="CASCADE")
    )
    # Monday=0, matching Python's date.weekday(). Postgres EXTRACT(DOW) is
    # Sunday=0, so any SQL that compares against this column has to convert.
    # Picking one convention and writing it down beats discovering the
    # off-by-one in production.
    weekday: Mapped[int] = mapped_column(SmallInteger, nullable=False)
    # Wall-clock local time, not an instant. "We open at 9" is a fact about the
    # clock on the wall and must not shift when an offset does.
    start_time: Mapped[time] = mapped_column(nullable=False)
    end_time: Mapped[time] = mapped_column(nullable=False)


class Booking(Base):
    __tablename__ = "bookings"
    __table_args__ = (
        CheckConstraint("slot_end > slot_start", name="bookings_end_after_start"),
        Index("bookings_by_business_slot", "business_id", "slot_start"),
        Index("bookings_by_phone", "business_id", "customer_phone_e164"),
    )

    id: Mapped[uuid.UUID] = _uuid_pk()
    # Short spoken/typed handle: "your booking is XK4P2R". Six characters from
    # an alphabet with no 0/O/1/I, so it survives being read out over a phone.
    reference: Mapped[str] = mapped_column(String(8), unique=True, nullable=False)

    business_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("businesses.id", ondelete="CASCADE"), nullable=False
    )
    service_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("services.id"), nullable=False)
    staff_id: Mapped[uuid.UUID | None] = mapped_column(ForeignKey("staff.id"))
    call_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("calls.id", ondelete="SET NULL")
    )

    customer_name: Mapped[str] = mapped_column(String(120), nullable=False)
    customer_phone_e164: Mapped[str] = mapped_column(String(20), nullable=False)

    slot_start: Mapped[datetime] = _tstz(nullable=False)
    # Stored rather than derived from service.duration_min, so that editing a
    # service's duration cannot retroactively move appointments already agreed
    # with a customer.
    slot_end: Mapped[datetime] = _tstz(nullable=False)

    status: Mapped[BookingStatus] = mapped_column(
        Enum(BookingStatus, name="booking_status", values_callable=lambda e: [m.value for m in e]),
        nullable=False,
        default=BookingStatus.BOOKED,
    )
    source: Mapped[BookingSource] = mapped_column(
        Enum(BookingSource, name="booking_source", values_callable=lambda e: [m.value for m in e]),
        nullable=False,
    )
    notes: Mapped[str | None] = mapped_column(Text)
    google_event_id: Mapped[str | None] = mapped_column(String(128))

    # The resource actually being booked: a named staff member, or the business
    # itself when the caller does not care who they see. Generated, because a
    # NULL staff_id would never collide with another NULL under the exclusion
    # constraint below (NULL is not equal to NULL), leaving business-wide
    # bookings unprotected. One generated column lets one constraint cover both.
    resource_id: Mapped[uuid.UUID] = mapped_column(
        PgUUID(as_uuid=True),
        Computed("COALESCE(staff_id, business_id)", persisted=True),
        nullable=False,
    )

    created_at: Mapped[datetime] = _tstz(server_default=func.now(), nullable=False)
    updated_at: Mapped[datetime] = _tstz(
        server_default=func.now(), onupdate=func.now(), nullable=False
    )
    cancelled_at: Mapped[datetime | None] = _tstz()

    service: Mapped[Service] = relationship(lazy="joined")
    staff: Mapped[Staff | None] = relationship(lazy="joined")


class Call(Base):
    __tablename__ = "calls"

    id: Mapped[uuid.UUID] = _uuid_pk()
    business_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("businesses.id", ondelete="CASCADE"), nullable=False
    )
    channel: Mapped[CallChannel] = mapped_column(
        Enum(CallChannel, name="call_channel", values_callable=lambda e: [m.value for m in e]),
        nullable=False,
    )
    from_phone_e164: Mapped[str | None] = mapped_column(String(20))
    twilio_call_sid: Mapped[str | None] = mapped_column(String(64), unique=True)
    status: Mapped[CallStatus] = mapped_column(
        Enum(CallStatus, name="call_status", values_callable=lambda e: [m.value for m in e]),
        nullable=False,
        default=CallStatus.IN_PROGRESS,
    )
    # Why we escalated, in plain language, so the owner reading the dashboard
    # can tell "asked for a human" from "we could not understand them".
    handoff_reason: Mapped[str | None] = mapped_column(Text)
    transcript_summary: Mapped[str | None] = mapped_column(Text)
    recording_url: Mapped[str | None] = mapped_column(Text)

    started_at: Mapped[datetime] = _tstz(server_default=func.now(), nullable=False)
    ended_at: Mapped[datetime | None] = _tstz()
    duration_ms: Mapped[int | None] = mapped_column(Integer)

    turns: Mapped[list[CallTurn]] = relationship(
        back_populates="call", order_by="CallTurn.turn_index"
    )


class CallTurn(Base):
    """One conversational turn, and the latency budget it consumed."""

    __tablename__ = "call_turns"
    __table_args__ = (
        UniqueConstraint("call_id", "turn_index", name="call_turns_unique_index"),
    )

    # bigserial, not uuid: this is the highest-volume table in the system and a
    # random primary key would only hurt index locality.
    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    call_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("calls.id", ondelete="CASCADE"), nullable=False
    )
    turn_index: Mapped[int] = mapped_column(Integer, nullable=False)
    role: Mapped[TurnRole] = mapped_column(
        Enum(TurnRole, name="turn_role", values_callable=lambda e: [m.value for m in e]),
        nullable=False,
    )
    text: Mapped[str | None] = mapped_column(Text)

    # One anchor timestamp plus integer millisecond offsets, rather than six
    # timestamps. Phase 6's p50/p95 then reduces to percentile_cont over an int
    # column, and the numbers stay readable in the dashboard.
    #
    # Every offset is nullable on purpose: a turn that dies after the LLM but
    # before TTS must still record how far it got. A row with
    # llm_first_token_ms set and tts_first_audio_ms NULL *is* the TTS failure.
    turn_started_at: Mapped[datetime] = _tstz(nullable=False)
    stt_final_ms: Mapped[int | None] = mapped_column(Integer)
    llm_first_token_ms: Mapped[int | None] = mapped_column(Integer)
    llm_done_ms: Mapped[int | None] = mapped_column(Integer)
    tts_first_audio_ms: Mapped[int | None] = mapped_column(Integer)
    # What the caller waited. On a voice channel that is anchor -> first audio
    # they hear: the only latency number that reflects what the call felt like,
    # and NULL when nothing was ever heard (a turn whose voice failed had no
    # response time, not a small one). On a text channel there is no audio, so it
    # is anchor -> reply complete. The two are not comparable: filter by
    # calls.channel before taking percentiles.
    total_ms: Mapped[int | None] = mapped_column(Integer)

    barge_in: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    # Tool name, arguments, result and duration. JSONB rather than a table per
    # tool, because the payload shape differs per tool and nothing queries
    # inside it except ad-hoc debugging.
    tool_calls: Mapped[dict | None] = mapped_column(JSONB)

    # Recorded now so the Phase 7 cost-per-call estimate needs no migration,
    # and so the numbers stay correct across a model change.
    model: Mapped[str | None] = mapped_column(String(64))
    input_tokens: Mapped[int | None] = mapped_column(Integer)
    output_tokens: Mapped[int | None] = mapped_column(Integer)
    cache_read_tokens: Mapped[int | None] = mapped_column(Integer)

    call: Mapped[Call] = relationship(back_populates="turns")
