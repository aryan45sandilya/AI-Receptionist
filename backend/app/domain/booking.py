"""Booking writes: create, cancel, reschedule.

This module owns the transaction, because here the transaction *is* the business
rule. It does not know about HTTP, Twilio, WhatsApp or the LLM -- those are
transports that call in.

The concurrency story in one paragraph: slots are computed, never stored, so
there is no row for "6:30pm Tuesday" to lock. The GiST exclusion constraint
created in migration 0001 is therefore the serialisation point. When two
transactions try to insert overlapping bookings for the same resource, Postgres
makes the second one wait on the first, then rejects it -- so exactly one
survives no matter how the application is scheduled, and there is no lock
ordering to get wrong. ``SlotTakenError`` is that rejection, translated into
something the agent can say out loud.
"""

from __future__ import annotations

import secrets
import uuid
from datetime import datetime

from psycopg import errors as pg_errors
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from ..enums import BookingSource, BookingStatus
from ..models import Booking, Service

# No 0/O/1/I/5/S: this code gets read aloud over a phone line and typed back
# into WhatsApp, so visually confusable characters are removed rather than
# explained.
_REFERENCE_ALPHABET = "23467894ABCDEFGHJKLMNPQRTUVWXYZ"
_REFERENCE_LENGTH = 6

# A collision on a 6-character random reference is vanishingly unlikely, but
# "unlikely" is not "impossible", so retry rather than fail the booking.
_REFERENCE_RETRIES = 5


class BookingError(Exception):
    """Base class, so callers can catch everything this module raises."""


class SlotTakenError(BookingError):
    """Someone else booked the slot first. Offer alternatives, do not retry."""


class BookingNotFoundError(BookingError):
    pass


class ServiceNotFoundError(BookingError):
    pass


def generate_reference() -> str:
    return "".join(secrets.choice(_REFERENCE_ALPHABET) for _ in range(_REFERENCE_LENGTH))


def _is_reference_collision(exc: IntegrityError) -> bool:
    orig = exc.orig
    if not isinstance(orig, pg_errors.UniqueViolation):
        return False
    constraint = getattr(getattr(orig, "diag", None), "constraint_name", "") or ""
    return "reference" in constraint


async def get_service(session: AsyncSession, service_id: uuid.UUID) -> Service:
    service = await session.get(Service, service_id)
    if service is None or not service.active:
        raise ServiceNotFoundError(str(service_id))
    return service


async def create_booking(
    session: AsyncSession,
    *,
    business_id: uuid.UUID,
    service_id: uuid.UUID,
    staff_id: uuid.UUID | None,
    customer_name: str,
    customer_phone_e164: str,
    slot_start: datetime,
    slot_end: datetime,
    source: BookingSource,
    call_id: uuid.UUID | None = None,
    notes: str | None = None,
) -> Booking:
    """Insert a booking, or raise ``SlotTakenError`` if the slot just went.

    The caller commits. Keeping commit control outside this function is what
    lets the concurrency test hold two transactions open at once and observe
    them race, and it lets a single call transaction book and log together.

    Each attempt runs inside a SAVEPOINT. Without one, the first
    ``IntegrityError`` would poison the whole transaction and a reference retry
    would be impossible.
    """
    if slot_end <= slot_start:
        raise ValueError("slot_end must be after slot_start")
    if slot_start.tzinfo is None or slot_end.tzinfo is None:
        raise ValueError("slot times must be timezone-aware")

    last_error: IntegrityError | None = None
    for _ in range(_REFERENCE_RETRIES):
        booking = Booking(
            reference=generate_reference(),
            business_id=business_id,
            service_id=service_id,
            staff_id=staff_id,
            call_id=call_id,
            customer_name=customer_name.strip(),
            customer_phone_e164=customer_phone_e164.strip(),
            slot_start=slot_start,
            slot_end=slot_end,
            status=BookingStatus.BOOKED,
            source=source,
            notes=notes,
        )
        try:
            # session.add() goes inside the SAVEPOINT so the INSERT is covered
            # by it. See reschedule_booking for why the ordering matters.
            async with session.begin_nested():
                session.add(booking)
                await session.flush()
        except IntegrityError as exc:
            # The savepoint rolled back, but the instance can still be sitting
            # in session.new. Left there, it would be flushed again on the next
            # commit and fail a second time, somewhere far from here.
            if booking in session:
                session.expunge(booking)
            if isinstance(exc.orig, pg_errors.ExclusionViolation):
                raise SlotTakenError(
                    f"{slot_start.isoformat()} is no longer free"
                ) from exc
            if _is_reference_collision(exc):
                last_error = exc
                continue
            raise
        else:
            # resource_id is a generated column, so the value Postgres computed
            # is only visible after a refresh. Phase 5's dashboard reads it.
            await session.refresh(booking, ["resource_id"])
            return booking

    raise BookingError("could not allocate a unique booking reference") from last_error


async def find_active_bookings(
    session: AsyncSession,
    *,
    business_id: uuid.UUID,
    customer_phone_e164: str,
    after: datetime | None = None,
) -> list[Booking]:
    """Upcoming, still-live bookings for a phone number, soonest first.

    This is how "cancel my appointment" finds its target without the caller
    having to remember a reference number.
    """
    stmt = (
        select(Booking)
        .where(
            Booking.business_id == business_id,
            Booking.customer_phone_e164 == customer_phone_e164.strip(),
            Booking.status == BookingStatus.BOOKED,
        )
        .order_by(Booking.slot_start)
    )
    if after is not None:
        stmt = stmt.where(Booking.slot_start >= after)
    result = await session.execute(stmt)
    return list(result.unique().scalars().all())


async def get_booking_by_reference(
    session: AsyncSession, *, business_id: uuid.UUID, reference: str
) -> Booking:
    stmt = select(Booking).where(
        Booking.business_id == business_id,
        # References are spoken, so accept whatever case they arrive in.
        Booking.reference == reference.strip().upper(),
    )
    result = await session.execute(stmt)
    booking = result.unique().scalar_one_or_none()
    if booking is None:
        raise BookingNotFoundError(reference)
    return booking


async def cancel_booking(session: AsyncSession, booking: Booking, *, now: datetime) -> Booking:
    """Mark a booking cancelled. Idempotent: cancelling twice is not an error.

    Status becomes 'cancelled' rather than the row being deleted, which both
    frees the slot (the exclusion constraint only covers status='booked') and
    keeps the record for the owner's dashboard.
    """
    if booking.status is BookingStatus.CANCELLED:
        return booking
    booking.status = BookingStatus.CANCELLED
    booking.cancelled_at = now
    await session.flush()
    return booking


async def reschedule_booking(
    session: AsyncSession,
    booking: Booking,
    *,
    slot_start: datetime,
    slot_end: datetime,
    staff_id: uuid.UUID | None = None,
) -> Booking:
    """Move a booking to a new slot, keeping its reference.

    Updated in place rather than cancel-and-rebook so the customer keeps the
    reference they were already told. The exclusion constraint guards the new
    time exactly as it guards an insert, and because the old and new rows are
    the same row, moving a booking can never collide with itself.
    """
    if slot_end <= slot_start:
        raise ValueError("slot_end must be after slot_start")
    if booking.status is not BookingStatus.BOOKED:
        raise BookingError(f"cannot reschedule a {booking.status.value} booking")

    try:
        # The SAVEPOINT is opened *before* the booking is mutated, and this
        # ordering is load-bearing: begin_nested() autoflushes pending changes
        # before it issues the SAVEPOINT. Mutating first would send the UPDATE
        # outside the savepoint, where a constraint violation poisons the whole
        # transaction instead of being contained -- the caller's next commit()
        # then fails with PendingRollbackError, long after the real cause.
        async with session.begin_nested():
            booking.slot_start = slot_start
            booking.slot_end = slot_end
            if staff_id is not None:
                booking.staff_id = staff_id
            await session.flush()
    except IntegrityError as exc:
        # Rolling back the savepoint restores the database but leaves the ORM
        # object expired, still carrying the times it failed to get. Reload it
        # so the caller -- the agent, which catches SlotTakenError and offers an
        # alternative on the same session -- is holding a booking that matches
        # reality. Without this, the next plain attribute read attempts IO from
        # sync code and raises MissingGreenlet somewhere unrelated.
        await session.refresh(booking)
        if isinstance(exc.orig, pg_errors.ExclusionViolation):
            raise SlotTakenError(
                f"{slot_start.isoformat()} is no longer free"
            ) from exc
        raise
    await session.refresh(booking, ["resource_id"])
    return booking
