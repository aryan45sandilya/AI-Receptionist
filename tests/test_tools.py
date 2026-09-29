"""Tests for the five tools.

No model is called anywhere here. The tools are exercised directly with the
arguments the model would have produced, which is what makes these fast, free
and deterministic -- and means a failure points at the booking logic rather than
at a sampling difference.

The test that matters most is
``test_booking_a_slot_that_was_never_offered_is_refused``: it is the executable
form of the rule that the agent can never invent an appointment time.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from zoneinfo import ZoneInfo

import pytest
import pytest_asyncio

from backend.app.agent.session import AgentSession
from backend.app.agent.tools import ToolContext, execute_tool, tool_schemas
from backend.app.config import get_settings
from backend.app.domain.booking import create_booking
from backend.app.enums import BookingSource, BookingStatus
from tests.conftest import count_booked

pytestmark = pytest.mark.db

IST = ZoneInfo("Asia/Kolkata")
CALLER = "+919876500001"


def a_future_monday_at_ten() -> datetime:
    """A fixed-shape 'now': a Monday at 10:00 local, at least a week out.

    A Monday morning means the clinic's evening surgery (17:00-20:00) is still
    ahead, so "today evening" always has slots, and being a week out keeps these
    tests clear of anything today's other tests booked.
    """
    now_local = datetime.now(IST)
    days_ahead = (0 - now_local.weekday()) % 7
    monday = (now_local + timedelta(days=days_ahead + 7)).replace(
        hour=10, minute=0, second=0, microsecond=0
    )
    return monday.astimezone(UTC)


@pytest_asyncio.fixture
async def bare_ctx(db, seeded) -> ToolContext:
    """A conversation in which the caller has not yet said anything."""
    session = AgentSession(
        business_id=seeded.business_id,
        business_name=seeded.business.name,
        tz=seeded.tz,
        caller_phone=CALLER,
    )
    return ToolContext(
        db=db,
        session=session,
        settings=get_settings(),
        now=a_future_monday_at_ten(),
        source=BookingSource.VOICE,
    )


@pytest_asyncio.fixture
async def ctx(bare_ctx) -> ToolContext:
    """The same conversation, after the caller has given their name.

    That is how a real booking goes, and it is what the name guard requires:
    book_appointment refuses any name the caller never said. Tests about the
    guard itself use ``bare_ctx``.
    """
    bare_ctx.session.note_caller_speech("Hi, this is Asha Menon. Asha again, for a rebooking.")
    return bare_ctx


async def check(ctx, service="Dental cleaning", when="today evening", staff=None) -> str:
    return await execute_tool(
        "check_availability",
        {"service": service, "when": when, "staff_name": staff},
        ctx,
    )


# ---------------------------------------------------------------------------
# check_availability
# ---------------------------------------------------------------------------


async def test_check_availability_returns_real_slots_with_ids(ctx):
    result = await check(ctx)
    assert "Dental cleaning" in result
    assert "[s1]" in result
    # The ids in the text are the ids the session will accept.
    assert set(ctx.session.offered) == {"s1", "s2", "s3"}


async def test_offered_slots_are_far_enough_apart_to_all_be_bookable(ctx):
    """A 30-minute service on a 15-minute grid: offering 5:00, 5:15 and 5:30
    would be offering options the caller cannot also have."""
    from datetime import timedelta

    await check(ctx)
    starts = sorted(slot.start for slot in ctx.session.offered.values())
    assert len(starts) >= 2
    assert all(later - earlier >= timedelta(minutes=30) for earlier, later in zip(starts, starts[1:]))


async def test_offered_slots_are_inside_opening_hours(ctx):
    await check(ctx)
    for slot in ctx.session.offered.values():
        local = slot.start.astimezone(IST)
        assert 17 <= local.hour < 20, local


async def test_offered_count_respects_the_configured_maximum(ctx):
    await check(ctx, when="today evening")
    assert len(ctx.session.offered) == ctx.settings.max_slots_offered


async def test_vague_date_is_resolved_and_labelled(ctx):
    result = await check(ctx, when="day after tomorrow")
    assert "Wednesday" in result  # 'now' is a Monday
    assert ctx.session.offered


async def test_unparseable_date_asks_rather_than_guessing(ctx):
    result = await check(ctx, when="whenever you like")
    assert "Do not guess a date" in result
    # Nothing was offered, so nothing can be booked off the back of it.
    assert ctx.session.offered == {}


async def test_unknown_service_lists_the_real_ones(ctx):
    result = await check(ctx, service="haircut")
    assert "Dental cleaning" in result
    assert "Root canal" in result
    assert ctx.session.offered == {}


async def test_ambiguous_service_asks_which(ctx):
    """"canal" matches only one here, but a partial match that hits several must
    ask instead of picking. 'a' matches both seeded services."""
    result = await execute_tool(
        "check_availability", {"service": "a", "when": "today evening", "staff_name": None}, ctx
    )
    assert "more than one service" in result
    assert ctx.session.offered == {}


async def test_unknown_staff_name_does_not_silently_fall_back(ctx):
    result = await check(ctx, staff="Dr Nobody")
    assert "Nobody here matches" in result
    assert ctx.session.offered == {}


async def test_named_staff_slots_are_attributed(ctx):
    result = await check(ctx, staff="Rao")
    assert "Dr Rao" in result
    assert all(s.staff_name == "Dr Rao" for s in ctx.session.offered.values())


async def test_slots_for_anyone_are_collapsed_to_one_per_time(ctx):
    """Two staff are free at 17:00; the caller should hear one offer, not two."""
    await check(ctx)
    starts = [s.start for s in ctx.session.offered.values()]
    assert len(starts) == len(set(starts))


async def test_a_fully_booked_evening_offers_no_time_at_all(ctx, seeded, db):
    """With every resource busy, the tool must say so and name no time.

    This is the case where a model under pressure to be helpful would invent
    something. The tool gives it nothing to work with and an explicit
    instruction not to improvise.
    """
    evening = a_future_monday_at_ten().astimezone(IST).replace(hour=17, minute=0)
    for member in (seeded.staff_a, seeded.staff_b):
        await create_booking(
            db,
            business_id=seeded.business_id,
            service_id=seeded.service_id,
            staff_id=member.id,
            customer_name=f"Blocker {member.name}",
            customer_phone_e164="+919000000009",
            slot_start=evening.astimezone(UTC),
            slot_end=(evening + timedelta(hours=3)).astimezone(UTC),
            source=BookingSource.VOICE,
        )
    await db.commit()

    result = await check(ctx, when="today evening")
    assert "Nothing free" in result
    assert "Do not offer any time this tool has not returned." in result
    assert ctx.session.offered == {}


async def test_blocking_one_staff_member_still_leaves_the_other_free(ctx, seeded, db):
    """The busy check is per-resource, not per-business."""
    evening = a_future_monday_at_ten().astimezone(IST).replace(hour=17, minute=0)
    await create_booking(
        db,
        business_id=seeded.business_id,
        service_id=seeded.service_id,
        staff_id=seeded.staff_a.id,
        customer_name="Blocks Rao only",
        customer_phone_e164="+919000000009",
        slot_start=evening.astimezone(UTC),
        slot_end=(evening + timedelta(hours=3)).astimezone(UTC),
        source=BookingSource.VOICE,
    )
    await db.commit()

    result = await check(ctx, when="today evening", staff="Menon")
    assert "[s1]" in result
    assert all(s.staff_name == "Dr Menon" for s in ctx.session.offered.values())


# ---------------------------------------------------------------------------
# book_appointment
# ---------------------------------------------------------------------------


async def test_booking_a_slot_that_was_never_offered_is_refused(ctx):
    """The hard rule, as an executable test.

    Even with a plausible-looking id, a slot the availability tool never issued
    cannot be booked. This is why book_appointment takes an id and not a time:
    there is no argument the model could pass that would invent an appointment.
    """
    result = await execute_tool(
        "book_appointment",
        {"slot_id": "s99", "customer_name": "Asha", "customer_phone": None},
        ctx,
    )
    assert "has been offered" in result
    assert "check_availability" in result


async def test_booking_before_any_availability_check_is_refused(ctx):
    result = await execute_tool(
        "book_appointment",
        {"slot_id": "s1", "customer_name": "Asha", "customer_phone": None},
        ctx,
    )
    assert "none yet" in result


async def test_booking_an_offered_slot_succeeds(ctx, sessionmaker_for, seeded):
    await check(ctx)
    slot = ctx.session.offered["s1"]

    result = await execute_tool(
        "book_appointment",
        {"slot_id": "s1", "customer_name": "Asha Menon", "customer_phone": None},
        ctx,
    )
    assert "Booked." in result
    assert slot.label in result

    async with sessionmaker_for() as check_db:
        assert await count_booked(check_db, seeded.business_id, slot.start) == 1


async def test_booking_uses_the_callers_own_number_by_default(ctx, db):
    from sqlalchemy import select

    from backend.app.models import Booking

    await check(ctx)
    await execute_tool(
        "book_appointment",
        {"slot_id": "s1", "customer_name": "Asha", "customer_phone": None},
        ctx,
    )
    row = (await db.execute(select(Booking).where(Booking.customer_name == "Asha"))).unique().scalar_one()
    assert row.customer_phone_e164 == CALLER


async def test_booking_a_slot_taken_in_the_meantime_is_reported_honestly(
    ctx, db, seeded
):
    """Between offering a slot and booking it, someone else took it."""
    await check(ctx)
    slot = ctx.session.offered["s1"]

    # Simulate the other caller winning the race, on the same resource.
    await create_booking(
        db,
        business_id=seeded.business_id,
        service_id=seeded.service_id,
        staff_id=slot.staff_id,
        customer_name="Faster caller",
        customer_phone_e164="+919000000008",
        slot_start=slot.start,
        slot_end=slot.end,
        source=BookingSource.VOICE,
    )
    await db.commit()

    result = await execute_tool(
        "book_appointment",
        {"slot_id": "s1", "customer_name": "Asha", "customer_phone": None},
        ctx,
    )
    assert "taken by someone else" in result
    assert "check_availability again" in result


async def test_booking_without_a_name_asks_for_one(ctx):
    await check(ctx)
    result = await execute_tool(
        "book_appointment",
        {"slot_id": "s1", "customer_name": "   ", "customer_phone": None},
        ctx,
    )
    assert "name" in result.lower()


async def test_booking_under_a_name_the_caller_never_said_is_refused(bare_ctx, sessionmaker_for, seeded):
    """Found in a live call: told "yes, the first one", the model booked under
    "[awaiting]" instead of asking. It is now refused, and nothing is written."""
    await check(bare_ctx)
    slot = bare_ctx.session.offered["s1"]
    bare_ctx.session.note_caller_speech("Yes, the first one works for me.")

    result = await execute_tool(
        "book_appointment",
        {"slot_id": "s1", "customer_name": "[awaiting]", "customer_phone": None},
        bare_ctx,
    )
    assert "May I have your name" in result
    assert "Booked." not in result
    async with sessionmaker_for() as check_db:
        assert await count_booked(check_db, seeded.business_id, slot.start) == 0


async def test_the_refused_booking_can_be_completed_once_the_caller_gives_a_name(bare_ctx):
    """The refusal is a question, not a dead end: the slot is still on offer."""
    await check(bare_ctx)
    bare_ctx.session.note_caller_speech("Yes, the first one works for me.")
    refused = await execute_tool(
        "book_appointment",
        {"slot_id": "s1", "customer_name": "Guest", "customer_phone": None},
        bare_ctx,
    )
    assert "Booked." not in refused

    bare_ctx.session.note_caller_speech("My name is Asha Menon")
    booked = await execute_tool(
        "book_appointment",
        {"slot_id": "s1", "customer_name": "Asha Menon", "customer_phone": None},
        bare_ctx,
    )
    assert "Booked." in booked


async def test_a_refused_name_counts_toward_the_handoff_policy(bare_ctx):
    """An agent that keeps trying to book without a name is not making progress."""
    await check(bare_ctx)
    before = bare_ctx.session.failed_attempts
    await execute_tool(
        "book_appointment",
        {"slot_id": "s1", "customer_name": "Unknown", "customer_phone": None},
        bare_ctx,
    )
    assert bare_ctx.session.failed_attempts == before + 1


# ---------------------------------------------------------------------------
# cancel_appointment
# ---------------------------------------------------------------------------


async def test_cancel_with_nothing_booked_says_so(ctx):
    result = await execute_tool(
        "cancel_appointment", {"booking_reference": None, "customer_phone": None}, ctx
    )
    assert "No upcoming bookings" in result


async def test_cancel_a_single_booking_needs_no_reference(ctx, db):
    await check(ctx)
    await execute_tool(
        "book_appointment",
        {"slot_id": "s1", "customer_name": "Asha", "customer_phone": None},
        ctx,
    )
    result = await execute_tool(
        "cancel_appointment", {"booking_reference": None, "customer_phone": None}, ctx
    )
    assert "Cancelled" in result


async def test_cancel_with_two_bookings_asks_which(ctx):
    await check(ctx)
    # s1 and s3, not s1 and s2: the service is 30 minutes on a 15-minute grid,
    # so 17:00 and 17:15 would overlap and the second booking would be rejected.
    # s1 (17:00) and s3 (17:30) are back to back.
    for slot_id in ("s1", "s3"):
        await execute_tool(
            "book_appointment",
            {"slot_id": slot_id, "customer_name": "Asha", "customer_phone": None},
            ctx,
        )
    result = await execute_tool(
        "cancel_appointment", {"booking_reference": None, "customer_phone": None}, ctx
    )
    assert "2 upcoming bookings" in result
    assert "Ask which one to cancel" in result


async def test_cancel_with_a_bad_reference_does_not_cancel_anything(ctx, sessionmaker_for, seeded):
    await check(ctx)
    slot = ctx.session.offered["s1"]
    await execute_tool(
        "book_appointment",
        {"slot_id": "s1", "customer_name": "Asha", "customer_phone": None},
        ctx,
    )
    result = await execute_tool(
        "cancel_appointment", {"booking_reference": "ZZZZZZ", "customer_phone": None}, ctx
    )
    assert "No booking with reference" in result
    async with sessionmaker_for() as check_db:
        assert await count_booked(check_db, seeded.business_id, slot.start) == 1


async def test_cancelling_frees_the_slot_for_rebooking(ctx, sessionmaker_for, seeded):
    await check(ctx)
    slot = ctx.session.offered["s1"]
    await execute_tool(
        "book_appointment",
        {"slot_id": "s1", "customer_name": "Asha", "customer_phone": None},
        ctx,
    )
    await execute_tool(
        "cancel_appointment", {"booking_reference": None, "customer_phone": None}, ctx
    )
    result = await execute_tool(
        "book_appointment",
        {"slot_id": "s1", "customer_name": "Asha again", "customer_phone": None},
        ctx,
    )
    assert "Booked." in result
    async with sessionmaker_for() as check_db:
        assert await count_booked(check_db, seeded.business_id, slot.start) == 1


# ---------------------------------------------------------------------------
# reschedule_appointment
# ---------------------------------------------------------------------------


async def test_reschedule_without_an_offered_slot_is_refused(ctx):
    result = await execute_tool(
        "reschedule_appointment",
        {"new_slot_id": "s42", "booking_reference": None, "customer_phone": None},
        ctx,
    )
    assert "has been offered" in result


async def test_reschedule_moves_the_booking_and_keeps_the_reference(ctx, db):
    from sqlalchemy import select

    from backend.app.models import Booking

    await check(ctx)
    await execute_tool(
        "book_appointment",
        {"slot_id": "s1", "customer_name": "Asha", "customer_phone": None},
        ctx,
    )
    row = (await db.execute(select(Booking).where(Booking.customer_name == "Asha"))).unique().scalar_one()
    reference = row.reference

    result = await execute_tool(
        "reschedule_appointment",
        {"new_slot_id": "s3", "booking_reference": None, "customer_phone": None},
        ctx,
    )
    assert "Moved" in result
    assert "reference is unchanged" in result
    await db.refresh(row)
    assert row.reference == reference
    assert row.slot_start == ctx.session.offered["s3"].start


# ---------------------------------------------------------------------------
# handoff_to_human
# ---------------------------------------------------------------------------


async def test_handoff_records_reason_and_summary(ctx):
    result = await execute_tool(
        "handoff_to_human",
        {"reason": "caller_asked_for_human", "summary": "Wants to speak to the dentist."},
        ctx,
    )
    assert ctx.session.handoff is not None
    assert ctx.session.handoff.reason == "caller_asked_for_human"
    assert "putting them through" in result


async def test_handoff_mentions_the_transfer_number_when_configured(ctx, seeded):
    result = await execute_tool(
        "handoff_to_human",
        {"reason": "outside_booking_scope", "summary": "Asking about X-ray results."},
        ctx,
    )
    assert seeded.business.handoff_phone_e164 in result


# ---------------------------------------------------------------------------
# Failure tracking, used by the handoff policy
# ---------------------------------------------------------------------------


async def test_repeated_failures_are_counted(ctx):
    assert ctx.session.failed_attempts == 0
    await check(ctx, when="mmm no idea")
    await check(ctx, service="haircut")
    assert ctx.session.failed_attempts == 2


async def test_a_success_resets_the_failure_count(ctx):
    await check(ctx, when="mmm no idea")
    assert ctx.session.failed_attempts == 1
    await check(ctx)
    assert ctx.session.failed_attempts == 0


async def test_unknown_tool_name_is_reported_not_raised(ctx):
    result = await execute_tool("teleport_caller", {}, ctx)
    assert "no tool named" in result


async def test_booked_rows_record_where_they_came_from(ctx, db):
    from sqlalchemy import select

    from backend.app.models import Booking

    await check(ctx)
    await execute_tool(
        "book_appointment",
        {"slot_id": "s1", "customer_name": "Asha", "customer_phone": None},
        ctx,
    )
    row = (await db.execute(select(Booking))).unique().scalars().first()
    assert row.source is BookingSource.VOICE
    assert row.status is BookingStatus.BOOKED
