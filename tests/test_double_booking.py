"""Proof that two callers cannot take the same slot.

The first test is the one that matters: two real transactions, on two real
connections, inserting the same slot at the same time, with a barrier to make
them actually overlap rather than politely queue. Exactly one commits.

Everything after it covers the cases a naive
``UNIQUE (business_id, slot_start)`` would get wrong -- overlapping-but-not-equal
times, cancelled bookings blocking resale, and bookings with no named staff
member.

These require Postgres. See the note in conftest.py about why SQLite is not an
acceptable substitute here.
"""

from __future__ import annotations

import asyncio
from datetime import timedelta

import pytest

from backend.app.domain.booking import (
    SlotTakenError,
    cancel_booking,
    create_booking,
    reschedule_booking,
)
from backend.app.enums import BookingSource, BookingStatus
from tests.conftest import count_booked, next_weekday_at

pytestmark = pytest.mark.db

THIRTY = timedelta(minutes=30)
SIXTY = timedelta(minutes=60)


async def _book(session, seeded, *, start, duration=THIRTY, staff=..., name="Caller"):
    staff_id = seeded.staff_a.id if staff is ... else staff
    return await create_booking(
        session,
        business_id=seeded.business_id,
        service_id=seeded.service_id,
        staff_id=staff_id,
        customer_name=name,
        customer_phone_e164="+919000000001",
        slot_start=start,
        slot_end=start + duration,
        source=BookingSource.VOICE,
    )


# ---------------------------------------------------------------------------
# The concurrency test
# ---------------------------------------------------------------------------


async def test_two_concurrent_bookings_only_one_wins(sessionmaker_for, seeded):
    """Two callers, one slot, at the same moment: one booking, one rejection.

    The barrier is what makes this a real test. Without it, asyncio would likely
    run the first attempt to completion before starting the second, and the
    second would fail against an already-committed row -- true, but not the race
    we care about. With it, both transactions are open and both attempt the
    insert before either commits, so Postgres has to arbitrate: the second
    INSERT blocks on the first's uncommitted row, then fails once it commits.
    """
    slot_start = next_weekday_at(18, 0)
    barrier = asyncio.Barrier(2)
    outcomes: list[str] = []

    async def attempt(name: str) -> str:
        async with sessionmaker_for() as session:
            # Both sessions are open and idle here, so neither has a head start.
            await barrier.wait()
            try:
                booking = await _book(session, seeded, start=slot_start, name=name)
                await session.commit()
                return f"booked:{booking.reference}"
            except SlotTakenError:
                await session.rollback()
                return "rejected"

    outcomes = await asyncio.gather(attempt("Asha"), attempt("Bharat"))

    booked = [o for o in outcomes if o.startswith("booked:")]
    rejected = [o for o in outcomes if o == "rejected"]
    assert len(booked) == 1, f"expected exactly one winner, got {outcomes}"
    assert len(rejected) == 1, f"expected exactly one rejection, got {outcomes}"

    # And the database agrees -- the losing transaction left nothing behind.
    async with sessionmaker_for() as check:
        assert await count_booked(check, seeded.business_id, slot_start) == 1


async def test_many_concurrent_bookings_still_yield_exactly_one(sessionmaker_for, seeded):
    """Five callers instead of two, in case two happened to pass by luck."""
    slot_start = next_weekday_at(18, 30)
    barrier = asyncio.Barrier(5)

    async def attempt(name: str) -> bool:
        async with sessionmaker_for() as session:
            await barrier.wait()
            try:
                await _book(session, seeded, start=slot_start, name=name)
                await session.commit()
                return True
            except SlotTakenError:
                await session.rollback()
                return False

    results = await asyncio.gather(*(attempt(f"Caller {i}") for i in range(5)))
    assert sum(results) == 1, f"expected one winner out of five, got {results}"

    async with sessionmaker_for() as check:
        assert await count_booked(check, seeded.business_id, slot_start) == 1


# ---------------------------------------------------------------------------
# What a plain UNIQUE(business_id, slot_start) would get wrong
# ---------------------------------------------------------------------------


async def test_overlapping_but_different_start_times_collide(sessionmaker_for, seeded):
    """A 60-minute 17:00 booking and a 30-minute 17:30 booking overlap.

    Their slot_start values differ, so a unique constraint on the start time
    would let both through. The range-overlap test catches it.
    """
    start = next_weekday_at(17, 0)
    async with sessionmaker_for() as session:
        await _book(session, seeded, start=start, duration=SIXTY, name="First")
        await session.commit()

    async with sessionmaker_for() as session:
        with pytest.raises(SlotTakenError):
            await _book(session, seeded, start=start + THIRTY, name="Second")
        await session.rollback()


async def test_a_booking_that_swallows_another_collides(sessionmaker_for, seeded):
    """A long booking spanning an existing short one must also be rejected."""
    async with sessionmaker_for() as session:
        await _book(session, seeded, start=next_weekday_at(17, 30), name="Short")
        await session.commit()

    async with sessionmaker_for() as session:
        with pytest.raises(SlotTakenError):
            await _book(
                session, seeded, start=next_weekday_at(17, 0), duration=SIXTY, name="Long"
            )
        await session.rollback()


async def test_back_to_back_bookings_are_allowed(sessionmaker_for, seeded):
    """17:00-17:30 and 17:30-18:00 are adjacent, not overlapping.

    This is the '[)' half-open bound on tstzrange, and it has to match
    domain/timeslots.py or the availability view and the database would disagree
    about what is free.
    """
    start = next_weekday_at(19, 0)
    async with sessionmaker_for() as session:
        await _book(session, seeded, start=start, name="First")
        await _book(session, seeded, start=start + THIRTY, name="Second")
        await session.commit()

    async with sessionmaker_for() as check:
        assert await count_booked(check, seeded.business_id, start) == 1
        assert await count_booked(check, seeded.business_id, start + THIRTY) == 1


async def test_cancelled_booking_frees_the_slot(sessionmaker_for, seeded):
    """The constraint only covers status='booked', so a slot can be resold.

    A plain unique constraint would keep the cancelled row occupying the key and
    the slot could never be booked again -- a bug that would only show up once a
    real customer cancelled.
    """
    from datetime import UTC, datetime

    start = next_weekday_at(12, 0)
    async with sessionmaker_for() as session:
        first = await _book(session, seeded, start=start, name="Cancels")
        await session.commit()
        await cancel_booking(session, first, now=datetime.now(UTC))
        await session.commit()
        assert first.status is BookingStatus.CANCELLED

    async with sessionmaker_for() as session:
        second = await _book(session, seeded, start=start, name="Rebooks")
        await session.commit()
        assert second.status is BookingStatus.BOOKED

    async with sessionmaker_for() as check:
        assert await count_booked(check, seeded.business_id, start) == 1


# ---------------------------------------------------------------------------
# Resources: named staff, and "anyone"
# ---------------------------------------------------------------------------


async def test_two_staff_can_hold_the_same_time(sessionmaker_for, seeded):
    start = next_weekday_at(11, 0)
    async with sessionmaker_for() as session:
        await _book(session, seeded, start=start, staff=seeded.staff_a.id, name="With Rao")
        await _book(session, seeded, start=start, staff=seeded.staff_b.id, name="With Menon")
        await session.commit()

    async with sessionmaker_for() as check:
        assert await count_booked(check, seeded.business_id, start) == 2


async def test_bookings_with_no_named_staff_are_protected(sessionmaker_for, seeded):
    """The reason resource_id is a generated column.

    With staff_id NULL, an exclusion constraint keyed on staff_id would never
    match -- NULL is not equal to NULL -- so two business-level bookings would
    silently overlap. resource_id falls back to business_id and closes that hole.
    """
    start = next_weekday_at(10, 0)
    async with sessionmaker_for() as session:
        first = await _book(session, seeded, start=start, staff=None, name="Anyone")
        await session.commit()
        assert first.staff_id is None
        assert first.resource_id == seeded.business_id

    async with sessionmaker_for() as session:
        with pytest.raises(SlotTakenError):
            await _book(session, seeded, start=start, staff=None, name="Also anyone")
        await session.rollback()


async def test_a_named_staff_booking_does_not_block_the_business_resource(
    sessionmaker_for, seeded
):
    """A known simplification, asserted so it is a decision and not a surprise.

    A booking with Dr Rao and a booking with "anyone" at the same time are
    different resources, so both are allowed -- which can over-book a one-chair
    clinic. Mixing the two modes needs a staff-to-service mapping, which is out
    of scope for Phase 1 and recorded in the README's limitations.
    """
    start = next_weekday_at(9, 30)
    async with sessionmaker_for() as session:
        await _book(session, seeded, start=start, staff=seeded.staff_a.id, name="With Rao")
        await _book(session, seeded, start=start, staff=None, name="Anyone")
        await session.commit()

    async with sessionmaker_for() as check:
        assert await count_booked(check, seeded.business_id, start) == 2


# ---------------------------------------------------------------------------
# Rescheduling
# ---------------------------------------------------------------------------


async def test_reschedule_into_a_taken_slot_leaves_the_original_intact(
    sessionmaker_for, seeded
):
    early = next_weekday_at(17, 0, weeks=2)
    taken = next_weekday_at(18, 0, weeks=2)

    async with sessionmaker_for() as session:
        mine = await _book(session, seeded, start=early, name="Mine")
        await _book(session, seeded, start=taken, name="Someone else")
        await session.commit()

        with pytest.raises(SlotTakenError):
            await reschedule_booking(
                session, mine, slot_start=taken, slot_end=taken + THIRTY
            )
        # The in-memory object was rolled back to its original time, so a failed
        # move cannot silently corrupt the booking the caller already has.
        assert mine.slot_start == early
        await session.commit()

    async with sessionmaker_for() as check:
        assert await count_booked(check, seeded.business_id, early) == 1
        assert await count_booked(check, seeded.business_id, taken) == 1


async def test_session_is_still_usable_after_a_rejected_booking(sessionmaker_for, seeded):
    """A rejection must be contained, not poison the whole transaction.

    Regression test. The savepoint originally opened *after* the booking was
    mutated, and begin_nested() autoflushes before issuing SAVEPOINT -- so the
    failing statement ran outside the savepoint and deactivated the transaction.
    The SlotTakenError still surfaced correctly, which is what made it sneaky:
    the damage only showed up at the caller's next commit(), as a
    PendingRollbackError pointing nowhere useful.

    This matters in production, not just in tests: the agent catches
    SlotTakenError, apologises, and then books an alternative slot on the very
    same session.
    """
    taken = next_weekday_at(18, 0, weeks=5)
    free = next_weekday_at(19, 0, weeks=5)

    async with sessionmaker_for() as session:
        mine = await _book(session, seeded, start=taken, name="First")
        await session.commit()

    async with sessionmaker_for() as session:
        with pytest.raises(SlotTakenError):
            await _book(session, seeded, start=taken, name="Loses the race")

        # Same session, no rollback in between: exactly what the agent does.
        second = await _book(session, seeded, start=free, name="Takes the alternative")
        await session.commit()
        assert second.reference

    async with sessionmaker_for() as check:
        assert await count_booked(check, seeded.business_id, taken) == 1
        assert await count_booked(check, seeded.business_id, free) == 1


async def test_session_is_still_usable_after_a_rejected_reschedule(sessionmaker_for, seeded):
    """The same containment guarantee for the reschedule path."""
    early = next_weekday_at(9, 30, weeks=5)
    taken = next_weekday_at(11, 0, weeks=5)
    free = next_weekday_at(12, 0, weeks=5)

    async with sessionmaker_for() as session:
        mine = await _book(session, seeded, start=early, name="Mover")
        await _book(session, seeded, start=taken, name="Blocker")
        await session.commit()

        with pytest.raises(SlotTakenError):
            await reschedule_booking(
                session, mine, slot_start=taken, slot_end=taken + THIRTY
            )

        # Still usable: move it somewhere that is actually free.
        await reschedule_booking(session, mine, slot_start=free, slot_end=free + THIRTY)
        await session.commit()

    async with sessionmaker_for() as check:
        assert await count_booked(check, seeded.business_id, early) == 0
        assert await count_booked(check, seeded.business_id, free) == 1


async def test_reschedule_to_a_free_slot_keeps_the_reference(sessionmaker_for, seeded):
    start = next_weekday_at(17, 0, weeks=3)
    moved_to = next_weekday_at(19, 30, weeks=3)

    async with sessionmaker_for() as session:
        booking = await _book(session, seeded, start=start, name="Mover")
        await session.commit()
        reference = booking.reference

        await reschedule_booking(
            session, booking, slot_start=moved_to, slot_end=moved_to + THIRTY
        )
        await session.commit()

    async with sessionmaker_for() as check:
        assert await count_booked(check, seeded.business_id, start) == 0
        assert await count_booked(check, seeded.business_id, moved_to) == 1
        # The customer was already told this reference, so it must not change.
        assert booking.reference == reference


async def test_moving_a_booking_does_not_collide_with_itself(sessionmaker_for, seeded):
    """Nudging a booking 15 minutes later overlaps its own old range."""
    start = next_weekday_at(11, 30, weeks=4)
    async with sessionmaker_for() as session:
        booking = await _book(session, seeded, start=start, name="Nudger")
        await session.commit()

        nudged = start + timedelta(minutes=15)
        await reschedule_booking(
            session, booking, slot_start=nudged, slot_end=nudged + THIRTY
        )
        await session.commit()
        assert booking.slot_start == nudged
