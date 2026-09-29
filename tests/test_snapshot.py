"""Tests for the per-call snapshot.

The snapshot exists to take database round trips out of the caller's wait, so
the tests are about two things: that it gives the same answers the database
lookups did, and that the line between cached and live is in the right place.

That line is the important part. Services, staff and opening hours are read once
and held. Who is *booked* is never cached: a slot another caller takes while this
conversation is in progress must stop being offered immediately, and a cache that
missed it would offer a time that is already gone.
"""

from __future__ import annotations

from datetime import UTC, timedelta
from zoneinfo import ZoneInfo

import pytest
import pytest_asyncio
from sqlalchemy import event

from backend.app.agent.session import AgentSession
from backend.app.agent.tools import ToolContext, execute_tool
from backend.app.config import get_settings
from backend.app.domain import availability
from backend.app.domain.booking import create_booking
from backend.app.enums import BookingSource
from tests.test_tools import CALLER, a_future_monday_at_ten

pytestmark = pytest.mark.db

IST = ZoneInfo("Asia/Kolkata")


@pytest_asyncio.fixture
async def ctx(db, seeded) -> ToolContext:
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


async def check(ctx, *, when="today evening", staff=None) -> str:
    return await execute_tool(
        "check_availability",
        {"service": "Dental cleaning", "when": when, "staff_name": staff},
        ctx,
    )


class QueryCounter:
    """Counts the SQL statements a block of code sends to the database."""

    def __init__(self, session) -> None:
        self._engine = session.bind.sync_engine
        self.statements: list[str] = []

    def _record(self, conn, cursor, statement, *_args) -> None:
        self.statements.append(statement)

    def __enter__(self) -> QueryCounter:
        event.listen(self._engine, "before_cursor_execute", self._record)
        return self

    def __exit__(self, *_exc) -> None:
        event.remove(self._engine, "before_cursor_execute", self._record)


# --------------------------------------------------------------------------
# Same answers as the lookups it replaces
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    "needle",
    ["Dental cleaning", "DENTAL CLEANING", "  root canal  ", "cleaning", "canal", "a", "haircut", ""],
)
async def test_service_matching_agrees_with_the_database_lookup(db, seeded, needle):
    """Exact wins, otherwise every partial match, sorted -- whichever way it is
    computed. The database version is the reference."""
    snapshot = await availability.load_snapshot(db, seeded.business_id)
    from_db = await availability.resolve_service(db, business_id=seeded.business_id, name=needle)
    assert [s.name for s in snapshot.find_services(needle)] == [s.name for s in from_db]


@pytest.mark.parametrize("needle", ["Rao", "menon", "Dr", "nobody", ""])
async def test_staff_matching_agrees_with_the_database_lookup(db, seeded, needle):
    snapshot = await availability.load_snapshot(db, seeded.business_id)
    from_db = await availability.resolve_staff(db, business_id=seeded.business_id, name=needle)
    assert [s.name for s in snapshot.find_staff(needle)] == [s.name for s in from_db]


async def test_opening_rules_are_scoped_by_staff_member(db, seeded):
    snapshot = await availability.load_snapshot(db, seeded.business_id)
    rao = snapshot.rules_for(seeded.staff_a.id)
    assert rao and all(rule.resource_id == seeded.staff_a.id for rule in rao)
    assert {rule.resource_id for rule in snapshot.rules_for(None)} == {
        seeded.staff_a.id,
        seeded.staff_b.id,
    }


async def test_snapshot_holds_plain_values_not_orm_rows(db, seeded):
    """The reason for the design: ORM objects go stale on rollback."""
    snapshot = await availability.load_snapshot(db, seeded.business_id)
    assert snapshot.business.timezone == "Asia/Kolkata"
    assert snapshot.business.tz == IST
    assert {s.name for s in snapshot.services} == {"Dental cleaning", "Root canal"}
    assert snapshot.services[0].__class__.__name__ == "ServiceInfo"


async def test_an_unknown_business_is_an_error_not_an_empty_snapshot(db):
    import uuid

    with pytest.raises(LookupError):
        await availability.load_snapshot(db, uuid.uuid4())


# --------------------------------------------------------------------------
# What it saves
# --------------------------------------------------------------------------

async def test_check_availability_makes_exactly_one_query_once_loaded(db, ctx):
    """The point of the exercise. Before the snapshot this was about six round
    trips -- over two seconds against a remote database. Now it is one: who is
    already booked."""
    await ctx.get_snapshot()

    with QueryCounter(db) as counter:
        result = await check(ctx)

    assert "[s1]" in result
    assert len(counter.statements) == 1, counter.statements
    assert "FROM bookings" in counter.statements[0]


async def test_the_busy_query_reads_three_columns_not_the_whole_entity(db, ctx):
    """Loading the Booking entity would eager-join services and staff to answer
    a question that only needs to know when a resource is taken."""
    await ctx.get_snapshot()
    with QueryCounter(db) as counter:
        await check(ctx)
    statement = counter.statements[0]
    assert "JOIN" not in statement.upper()
    assert "customer_name" not in statement


async def test_the_snapshot_is_loaded_once_and_reused(db, ctx):
    with QueryCounter(db) as counter:
        first = await ctx.get_snapshot()
        second = await ctx.get_snapshot()
    assert first is second
    # business, services, staff, rules: four, and only once.
    assert len(counter.statements) == 4


# --------------------------------------------------------------------------
# The line between cached and live
# --------------------------------------------------------------------------

async def test_a_booking_made_after_the_snapshot_still_blocks_its_slot(db, ctx, seeded):
    """Hours are cached; bookings are not. Someone else takes 5:00pm while this
    caller is on the line, and it must stop being offered on the very next check."""
    await ctx.get_snapshot()

    first = await check(ctx)
    assert "[s1]" in first
    offered = ctx.session.offered["s1"]

    # A different caller books that exact slot, on the same resource.
    await create_booking(
        db,
        business_id=seeded.business_id,
        service_id=seeded.service_id,
        staff_id=offered.staff_id,
        customer_name="Someone else",
        customer_phone_e164="+919000000007",
        slot_start=offered.start,
        slot_end=offered.end,
        source=BookingSource.VOICE,
    )
    await db.commit()

    ctx.session.offered.clear()
    await check(ctx)
    # The other doctor may still be free at that time, so the time itself can
    # legitimately come back. What must not: that time with that person.
    now_offered = {(slot.start, slot.staff_id) for slot in ctx.session.offered.values()}
    assert (offered.start, offered.staff_id) not in now_offered


async def test_a_cancellation_after_the_snapshot_frees_the_slot_again(db, ctx, seeded):
    """The other direction: a slot released mid-call becomes bookable, because
    bookings are read live every time."""
    from backend.app.domain.booking import cancel_booking

    await ctx.get_snapshot()
    await check(ctx, staff="Rao")
    slot = ctx.session.offered["s1"]

    blocker = await create_booking(
        db,
        business_id=seeded.business_id,
        service_id=seeded.service_id,
        staff_id=slot.staff_id,
        customer_name="Holds it",
        customer_phone_e164="+919000000006",
        slot_start=slot.start,
        slot_end=slot.end,
        source=BookingSource.VOICE,
    )
    await db.commit()

    ctx.session.offered.clear()
    await check(ctx, staff="Rao")
    assert slot.start not in {s.start for s in ctx.session.offered.values()}

    await cancel_booking(db, blocker, now=ctx.now.astimezone(UTC) + timedelta(seconds=1))
    await db.commit()

    ctx.session.offered.clear()
    await check(ctx, staff="Rao")
    assert slot.start in {s.start for s in ctx.session.offered.values()}


async def test_tools_keep_working_after_a_rollback(db, ctx):
    """A failed turn rolls the session back, which expires every ORM object in
    it. Reading one afterwards from async code raises MissingGreenlet. The
    snapshot is plain data precisely so that cannot happen."""
    await ctx.get_snapshot()
    await db.rollback()

    result = await check(ctx)
    assert "[s1]" in result
    result = await execute_tool(
        "handoff_to_human",
        {"reason": "caller_asked_for_human", "summary": "Wants a person."},
        ctx,
    )
    assert "putting them through" in result
