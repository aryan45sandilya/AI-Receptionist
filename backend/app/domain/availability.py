"""The read side: load schedule rows and hand them to the pure slot engine.

This is the only adapter between the database and ``timeslots.free_slots``. Keeping
it thin is what lets the slot arithmetic itself stay pure and exhaustively
tested without a database.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from zoneinfo import ZoneInfo

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from ..enums import BookingStatus
from ..models import AvailabilityRule, Booking, Business, Service, Staff
from .datetime_parse import DateWindow
from .timeslots import BusyInterval, OpeningRule, Slot, free_slots


async def get_business(session: AsyncSession, business_id: uuid.UUID) -> Business:
    business = await session.get(Business, business_id)
    if business is None:
        raise LookupError(f"no business {business_id}")
    return business


def business_tz(business: Business) -> ZoneInfo:
    return ZoneInfo(business.timezone)


async def resolve_service(
    session: AsyncSession, *, business_id: uuid.UUID, name: str
) -> list[Service]:
    """Match a spoken service name to rows. Returns every plausible match.

    Returning a list rather than one row (or raising) is deliberate: when a
    caller says "cleaning" and the business offers both "Dental cleaning" and
    "Deep cleaning", the honest answer is to ask which, and only the agent can
    do that. Resolving to a single arbitrary row here would make the agent
    confidently book the wrong service.
    """
    needle = name.strip().lower()
    if not needle:
        return []

    base = select(Service).where(Service.business_id == business_id, Service.active.is_(True))

    exact = await session.execute(base.where(func.lower(Service.name) == needle))
    hits = list(exact.scalars().all())
    if hits:
        return hits

    partial = await session.execute(
        base.where(Service.name.ilike(f"%{needle}%")).order_by(Service.name)
    )
    return list(partial.scalars().all())


async def resolve_staff(
    session: AsyncSession, *, business_id: uuid.UUID, name: str
) -> list[Staff]:
    needle = name.strip().lower()
    if not needle:
        return []
    stmt = (
        select(Staff)
        .where(
            Staff.business_id == business_id,
            Staff.active.is_(True),
            Staff.name.ilike(f"%{needle}%"),
        )
        .order_by(Staff.name)
    )
    result = await session.execute(stmt)
    return list(result.scalars().all())


async def list_services(session: AsyncSession, business_id: uuid.UUID) -> list[Service]:
    result = await session.execute(
        select(Service)
        .where(Service.business_id == business_id, Service.active.is_(True))
        .order_by(Service.name)
    )
    return list(result.scalars().all())


async def load_opening_rules(
    session: AsyncSession,
    *,
    business_id: uuid.UUID,
    staff_id: uuid.UUID | None = None,
) -> list[OpeningRule]:
    """Load weekly opening rules as pure dataclasses.

    A rule with a NULL staff_id belongs to the business as a whole, and maps to
    ``resource_id = business_id`` -- the same value the database's generated
    ``resource_id`` column computes for a booking with no named staff member.
    The two representations have to agree or the availability view and the
    exclusion constraint would disagree about what is free.
    """
    stmt = select(AvailabilityRule).where(AvailabilityRule.business_id == business_id)
    if staff_id is not None:
        stmt = stmt.where(AvailabilityRule.staff_id == staff_id)
    result = await session.execute(stmt)

    return [
        OpeningRule(
            resource_id=row.staff_id or row.business_id,
            weekday=row.weekday,
            start_time=row.start_time,
            end_time=row.end_time,
        )
        for row in result.scalars().all()
    ]


async def load_busy_intervals(
    session: AsyncSession,
    *,
    business_id: uuid.UUID,
    window_start: datetime,
    window_end: datetime,
) -> list[BusyInterval]:
    """Live bookings overlapping the window.

    The overlap test is ``slot_start < window_end AND slot_end > window_start``,
    which is the same half-open comparison used everywhere else in the system.
    Only status='booked' rows count, matching the exclusion constraint's WHERE
    clause -- a cancelled booking must free its slot for resale.
    """
    stmt = select(Booking).where(
        Booking.business_id == business_id,
        Booking.status == BookingStatus.BOOKED,
        Booking.slot_start < window_end,
        Booking.slot_end > window_start,
    )
    result = await session.execute(stmt)
    return [
        BusyInterval(resource_id=row.resource_id, start=row.slot_start, end=row.slot_end)
        for row in result.unique().scalars().all()
    ]


async def find_free_slots(
    session: AsyncSession,
    *,
    business: Business,
    service: Service,
    window: DateWindow,
    now: datetime,
    staff_id: uuid.UUID | None = None,
    granularity_min: int,
    min_lead_min: int,
    limit: int | None = None,
) -> list[Slot]:
    rules = await load_opening_rules(
        session, business_id=business.id, staff_id=staff_id
    )
    busy = await load_busy_intervals(
        session,
        business_id=business.id,
        window_start=window.start,
        window_end=window.end,
    )
    return free_slots(
        rules=rules,
        busy=busy,
        duration_min=service.duration_min,
        window=window,
        tz=business_tz(business),
        now=now,
        granularity_min=granularity_min,
        min_lead_min=min_lead_min,
        limit=limit,
    )


def collapse_by_time(slots: list[Slot]) -> list[Slot]:
    """Keep one slot per start time.

    When the caller does not mind who they see, two staff free at 6:30 is one
    offer, not two. Reading out "6:30 with Dr Rao, or 6:30 with Dr Menon" wastes
    the caller's time; the first free resource is assigned silently. Ordering is
    preserved, so whatever ``free_slots`` decided to offer first still leads.
    """
    seen: set[datetime] = set()
    collapsed: list[Slot] = []
    for slot in slots:
        if slot.start in seen:
            continue
        seen.add(slot.start)
        collapsed.append(slot)
    return collapsed
