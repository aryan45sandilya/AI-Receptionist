"""Fixtures for the tests that need a real Postgres.

The pure tests (``test_timeslots``, ``test_datetime_parse``) import nothing from
here and need no database. The ones that do are skipped with a clear message when
DATABASE_URL is unset, so a fresh clone still gets a green run for the logic that
can be checked without infrastructure.

These tests must run against Postgres specifically, not SQLite: the behaviour
under test -- a GiST exclusion constraint serialising two concurrent inserts --
does not exist in SQLite. Substituting SQLite here would produce a passing test
that proves nothing about the thing it claims to prove.

Isolation is by data, not by transaction: each test seeds its own business with a
unique phone number and deletes it afterwards, cascading to everything it owns.
The usual trick of wrapping each test in a transaction and rolling back would
make the concurrency test impossible, because that test needs two real
transactions to really commit.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, time, timedelta
from zoneinfo import ZoneInfo

import pytest
import pytest_asyncio
from sqlalchemy import delete, select, text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from pydantic import ValidationError

from backend.app.config import get_settings, uses_transaction_pooling
from backend.app.models import AvailabilityRule, Business, Service, Staff
from backend.app.runtime import LOOP_FACTORY

IST = ZoneInfo("Asia/Kolkata")


@pytest.fixture(scope="session")
def _asyncio_loop_factory():
    """Give pytest-asyncio the same event loop the application uses.

    Overrides pytest-asyncio's own fixture. Without this, tests on Windows get
    the default ProactorEventLoop and every psycopg connection fails with
    "Psycopg cannot use the 'ProactorEventLoop' to run in async mode" -- which
    looks like a database problem and is not one.

    Returns None off Windows, which is exactly what pytest-asyncio defaults to.
    """
    return LOOP_FACTORY


def _database_url() -> str:
    """The same URL the application uses.

    Read through get_settings() rather than os.environ so that .env is picked
    up exactly as it is at runtime. Reading the environment directly made these
    tests skip silently while the app connected fine -- the worst kind of
    disagreement, because it looks like everything passed.
    """
    try:
        url = get_settings().database_url.strip()
    except ValidationError:
        url = ""
    if not url or "PROJECTREF" in url:
        pytest.skip(
            "DATABASE_URL is not configured, so the database tests are skipped. "
            "Set it in .env and run 'alembic upgrade head' first."
        )
    return url


@pytest_asyncio.fixture(scope="session")
async def db_engine():
    """One engine for the whole run.

    Session-scoped because a connection to a hosted database is expensive to
    establish, and the schema checks below only need doing once. Safe now that
    the whole session shares a single event loop.
    """
    url = _database_url()
    engine = create_async_engine(
        url,
        pool_pre_ping=True,
        # Same transaction-pooler handling as the app engine; without it these
        # tests would fail intermittently on a Supabase 6543 URL.
        connect_args={"prepare_threshold": None} if uses_transaction_pooling(url) else {},
    )
    async with engine.connect() as conn:
        present = await conn.scalar(text("SELECT to_regclass('public.bookings')"))
        if present is None:
            await engine.dispose()
            pytest.fail(
                "The 'bookings' table does not exist. Run 'alembic upgrade head' "
                "against DATABASE_URL before running the database tests."
            )
        constraint = await conn.scalar(
            text(
                "SELECT conname FROM pg_constraint WHERE conname = 'bookings_no_overlap'"
            )
        )
        if constraint is None:
            await engine.dispose()
            pytest.fail(
                "The bookings_no_overlap exclusion constraint is missing. The "
                "double-booking test would pass for the wrong reason, so refusing "
                "to run. Re-run 'alembic upgrade head'."
            )
    try:
        yield engine
    finally:
        await engine.dispose()


@pytest_asyncio.fixture(scope="session")
async def sessionmaker_for(db_engine):
    return async_sessionmaker(db_engine, class_=AsyncSession, expire_on_commit=False)


@pytest_asyncio.fixture
async def db(sessionmaker_for):
    async with sessionmaker_for() as session:
        yield session


class Fixture:
    """Ids for a seeded demo business, so tests can stay short."""

    def __init__(self, business, service, long_service, staff_a, staff_b):
        self.business = business
        self.business_id = business.id
        self.service = service
        self.service_id = service.id
        self.long_service = long_service
        self.long_service_id = long_service.id
        self.staff_a = staff_a
        self.staff_b = staff_b
        self.tz = ZoneInfo(business.timezone)


@pytest_asyncio.fixture
async def seeded(sessionmaker_for):
    """One business, two services, two staff, Mon-Sat 09:00-13:00 and 17:00-20:00."""
    unique = uuid.uuid4().hex[:10]
    async with sessionmaker_for() as session:
        business = Business(
            name=f"Test Clinic {unique}",
            timezone="Asia/Kolkata",
            phone_e164=f"+9199{unique[:8]}",
            handoff_phone_e164="+919900000000",
            greeting="Test Clinic, how can I help?",
        )
        session.add(business)
        await session.flush()

        service = Service(
            business_id=business.id, name="Dental cleaning", duration_min=30
        )
        long_service = Service(
            business_id=business.id, name="Root canal", duration_min=60
        )
        staff_a = Staff(business_id=business.id, name="Dr Rao")
        staff_b = Staff(business_id=business.id, name="Dr Menon")
        session.add_all([service, long_service, staff_a, staff_b])
        await session.flush()

        session.add_all(
            [
                AvailabilityRule(
                    business_id=business.id,
                    staff_id=member.id,
                    weekday=weekday,
                    start_time=start,
                    end_time=end,
                )
                for member in (staff_a, staff_b)
                for weekday in range(0, 6)
                for start, end in [(time(9, 0), time(13, 0)), (time(17, 0), time(20, 0))]
            ]
        )
        await session.commit()

        fixture = Fixture(business, service, long_service, staff_a, staff_b)
        business_id = business.id

    try:
        yield fixture
    finally:
        # ON DELETE CASCADE removes services, staff, rules, calls and bookings.
        async with sessionmaker_for() as cleanup:
            await cleanup.execute(delete(Business).where(Business.id == business_id))
            await cleanup.commit()


def next_weekday_at(hour: int, minute: int = 0, *, weekday: int = 1, weeks: int = 1) -> datetime:
    """A future instant on a day the test clinic is open.

    Pushed at least a week out so it is always beyond the booking lead time and
    never collides with a slot another test happens to use today.
    """
    now_local = datetime.now(IST)
    days_ahead = (weekday - now_local.weekday()) % 7
    target = (now_local + timedelta(days=days_ahead + 7 * weeks)).replace(
        hour=hour, minute=minute, second=0, microsecond=0
    )
    return target.astimezone(UTC)


async def count_booked(session: AsyncSession, business_id: uuid.UUID, slot_start: datetime) -> int:
    from backend.app.enums import BookingStatus
    from backend.app.models import Booking

    result = await session.execute(
        select(Booking).where(
            Booking.business_id == business_id,
            Booking.slot_start == slot_start,
            Booking.status == BookingStatus.BOOKED,
        )
    )
    return len(list(result.unique().scalars().all()))
