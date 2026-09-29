"""Compute bookable slots from opening rules minus existing bookings.

Also pure, and for the same reason as ``datetime_parse``: this is the logic the
agent's honesty depends on, so it has to be testable without a database, an API
key, or a network. The database layer's only job is to load rows and hand them
here as the small dataclasses below.

Slots are *computed*, never stored. That is a deliberate choice and it drives
the concurrency design: because no row exists for "6:30pm on Tuesday", there is
nothing to ``SELECT ... FOR UPDATE``, so the serialisation point for two callers
racing for one slot is the exclusion constraint on ``bookings`` rather than a
row lock. See ``domain/booking.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from uuid import UUID
from zoneinfo import ZoneInfo

from .datetime_parse import DateWindow

# Slots start on quarter hours. A 20-minute service therefore offers 6:00, 6:15,
# 6:30 ... rather than 6:00, 6:20, 6:40, which is easier to say out loud and
# easier for a receptionist to read off a register.
DEFAULT_GRANULARITY_MIN = 15

# Do not offer a slot that starts in the next few minutes; the caller is still
# on the phone and cannot teleport.
DEFAULT_MIN_LEAD_MIN = 30


@dataclass(frozen=True)
class OpeningRule:
    """One recurring weekly opening window for one resource."""

    resource_id: UUID  # a staff member, or the business itself for "anyone"
    weekday: int  # Monday=0, matching date.weekday()
    start_time: time  # wall-clock local time
    end_time: time


@dataclass(frozen=True)
class BusyInterval:
    """Time already taken for a resource. Half-open: ``[start, end)``."""

    resource_id: UUID
    start: datetime  # timezone-aware
    end: datetime


@dataclass(frozen=True)
class Slot:
    resource_id: UUID
    start: datetime  # timezone-aware
    end: datetime

    def label(self, tz: ZoneInfo) -> str:
        """Spoken form, e.g. "Tuesday 30 September at 6:30 pm"."""
        local = self.start.astimezone(tz)
        clock = local.strftime("%I:%M %p").lstrip("0").lower()
        return f"{local.strftime('%A')} {local.day} {local.strftime('%B')} at {clock}"


def _dates_in(window: DateWindow, tz: ZoneInfo) -> list[date]:
    first = window.start.astimezone(tz).date()
    last = window.end.astimezone(tz).date()
    return [first + timedelta(days=i) for i in range((last - first).days + 1)]


def _overlaps(start: datetime, end: datetime, busy: BusyInterval) -> bool:
    # Half-open intervals, so a booking ending at 10:30 and one starting at
    # 10:30 do not collide. This matches the tstzrange '[)' bounds used by the
    # exclusion constraint in the database, so the two layers agree.
    return start < busy.end and busy.start < end


def free_slots(
    *,
    rules: list[OpeningRule],
    busy: list[BusyInterval],
    duration_min: int,
    window: DateWindow,
    tz: ZoneInfo,
    now: datetime,
    granularity_min: int = DEFAULT_GRANULARITY_MIN,
    min_lead_min: int = DEFAULT_MIN_LEAD_MIN,
    limit: int | None = None,
) -> list[Slot]:
    """Return bookable slots inside ``window``, soonest first.

    When ``window.preferred_start`` is set (the caller named a clock time) the
    result is ordered by closeness to that time instead, so "6:30 please" yields
    6:30 first and 6:15 / 6:45 next when 6:30 is gone.

    ``rules`` and ``busy`` should already be filtered to the resources the
    caller is willing to see. Callers asking for "anyone" will get one slot per
    resource per time; collapsing those for presentation is the tool layer's
    job, not this function's.
    """
    if duration_min <= 0:
        raise ValueError("duration_min must be positive")
    if granularity_min <= 0:
        raise ValueError("granularity_min must be positive")

    duration = timedelta(minutes=duration_min)
    step = timedelta(minutes=granularity_min)
    earliest = now + timedelta(minutes=min_lead_min)

    busy_by_resource: dict[UUID, list[BusyInterval]] = {}
    for interval in busy:
        busy_by_resource.setdefault(interval.resource_id, []).append(interval)

    found: set[tuple[UUID, datetime]] = set()
    slots: list[Slot] = []

    for day in _dates_in(window, tz):
        for rule in rules:
            if rule.weekday != day.weekday():
                continue

            # Opening hours are wall-clock facts ("we open at 9"), so the local
            # date and time are combined first and only then resolved to an
            # instant. Doing it the other way round breaks across a DST change.
            open_at = datetime.combine(day, rule.start_time, tzinfo=tz)
            close_at = datetime.combine(day, rule.end_time, tzinfo=tz)

            cursor = max(open_at, window.start, earliest)
            # Re-align to the granularity grid in local time after clipping, so
            # slots stay on quarter hours instead of inheriting "now"'s seconds.
            cursor = _align_up(cursor, tz, granularity_min)
            hard_stop = min(close_at, window.end)

            while cursor + duration <= hard_stop:
                key = (rule.resource_id, cursor)
                if key not in found and not any(
                    _overlaps(cursor, cursor + duration, b)
                    for b in busy_by_resource.get(rule.resource_id, ())
                ):
                    found.add(key)
                    slots.append(
                        Slot(resource_id=rule.resource_id, start=cursor, end=cursor + duration)
                    )
                cursor += step

    if window.preferred_start is not None:
        preferred = window.preferred_start
        slots.sort(key=lambda s: (abs(s.start - preferred), s.start))
    else:
        slots.sort(key=lambda s: s.start)

    return slots[:limit] if limit is not None else slots


def _align_up(moment: datetime, tz: ZoneInfo, granularity_min: int) -> datetime:
    """Round a moment up to the next granularity boundary in local wall-clock time."""
    local = moment.astimezone(tz)
    remainder = (local.minute % granularity_min) * 60 + local.second
    if remainder == 0 and local.microsecond == 0:
        return moment
    bump = timedelta(minutes=granularity_min) - timedelta(
        seconds=remainder, microseconds=local.microsecond
    )
    return (local + bump).replace(second=0, microsecond=0)


def spread(slots: list[Slot], *, min_gap: timedelta, limit: int) -> list[Slot]:
    """Pick up to ``limit`` slots that are genuinely different choices.

    On a 15-minute grid a 30-minute service is free at 5:00, 5:15 and 5:30, but
    those are not three options: 5:15 overlaps both neighbours, so offering it
    means offering something the caller cannot also have. Read aloud it is worse
    -- "5:00, 5:15 or 5:30" is a sentence of nearly identical times.

    Goes down the list in the order it arrived (soonest first, or closest to what
    the caller asked for) and keeps a slot only if it starts at least ``min_gap``
    after every slot already kept. Ranking is preserved; only near-duplicates
    are dropped.
    """
    picked: list[Slot] = []
    for slot in slots:
        if all(abs(slot.start - kept.start) >= min_gap for kept in picked):
            picked.append(slot)
            if len(picked) >= limit:
                break
    return picked
