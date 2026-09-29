"""Tests for slot computation.

No database and no API key: opening rules and busy intervals are built as plain
dataclasses, which is the whole point of keeping this logic in ``domain/``.
"""

from __future__ import annotations

from datetime import datetime, time, timedelta
from uuid import UUID
from zoneinfo import ZoneInfo

import pytest

from backend.app.domain.datetime_parse import DateWindow
from backend.app.domain.timeslots import BusyInterval, OpeningRule, Slot, free_slots

IST = ZoneInfo("Asia/Kolkata")
DOCTOR = UUID(int=1)
HYGIENIST = UUID(int=2)

# Monday 28 September 2026, 10:00 local.
NOW = datetime(2026, 9, 28, 10, 0, tzinfo=IST)
MONDAY = NOW.date()
SATURDAY = datetime(2026, 10, 3, 10, 0, tzinfo=IST).date()


def weekday_rules(resource: UUID = DOCTOR, *, weekdays=range(0, 5)) -> list[OpeningRule]:
    """Mon-Fri, 9-13 and 17-21 -- a clinic with a lunch break."""
    return [
        OpeningRule(resource_id=resource, weekday=d, start_time=start, end_time=end)
        for d in weekdays
        for start, end in [(time(9, 0), time(13, 0)), (time(17, 0), time(21, 0))]
    ]


def window(day, start: time, end: time, preferred: time | None = None) -> DateWindow:
    return DateWindow(
        start=datetime.combine(day, start, tzinfo=IST),
        end=datetime.combine(day, end, tzinfo=IST),
        label="test window",
        preferred_start=(
            datetime.combine(day, preferred, tzinfo=IST) if preferred else None
        ),
    )


def local_times(slots: list[Slot]) -> list[time]:
    return [s.start.astimezone(IST).time() for s in slots]


# --------------------------------------------------------------------------
# Opening hours
# --------------------------------------------------------------------------

def test_slots_stay_inside_opening_hours():
    slots = free_slots(
        rules=weekday_rules(),
        busy=[],
        duration_min=30,
        window=window(MONDAY, time(8, 0), time(23, 0)),
        tz=IST,
        now=NOW,
    )
    times = local_times(slots)
    assert times, "expected some slots"
    assert min(times) >= time(10, 30)  # 30-minute booking lead from 10:00
    assert max(times) == time(20, 30)  # last 30-minute slot before a 21:00 close


def test_lunch_break_is_not_bookable():
    slots = free_slots(
        rules=weekday_rules(),
        busy=[],
        duration_min=30,
        window=window(MONDAY, time(8, 0), time(23, 0)),
        tz=IST,
        now=NOW,
    )
    assert not any(time(13, 0) <= t < time(17, 0) for t in local_times(slots))


def test_last_slot_ends_exactly_at_closing_time():
    slots = free_slots(
        rules=weekday_rules(),
        busy=[],
        duration_min=60,
        window=window(MONDAY, time(17, 0), time(23, 0)),
        tz=IST,
        now=NOW,
    )
    assert local_times(slots)[-1] == time(20, 0)
    assert slots[-1].end.astimezone(IST).time() == time(21, 0)


def test_closed_day_yields_nothing():
    slots = free_slots(
        rules=weekday_rules(),  # Mon-Fri only
        busy=[],
        duration_min=30,
        window=window(SATURDAY, time(8, 0), time(23, 0)),
        tz=IST,
        now=NOW,
    )
    assert slots == []


def test_service_longer_than_the_open_window_yields_nothing():
    slots = free_slots(
        rules=[OpeningRule(DOCTOR, MONDAY.weekday(), time(17, 0), time(18, 0))],
        busy=[],
        duration_min=90,
        window=window(MONDAY, time(8, 0), time(23, 0)),
        tz=IST,
        now=NOW,
    )
    assert slots == []


# --------------------------------------------------------------------------
# Existing bookings
# --------------------------------------------------------------------------

def test_busy_interval_removes_overlapping_slots():
    busy = [
        BusyInterval(
            resource_id=DOCTOR,
            start=datetime.combine(MONDAY, time(17, 0), tzinfo=IST),
            end=datetime.combine(MONDAY, time(18, 0), tzinfo=IST),
        )
    ]
    slots = free_slots(
        rules=weekday_rules(),
        busy=busy,
        duration_min=30,
        window=window(MONDAY, time(17, 0), time(19, 0)),
        tz=IST,
        now=NOW,
    )
    times = local_times(slots)
    assert time(17, 0) not in times
    assert time(17, 30) not in times
    assert time(18, 0) in times


def test_half_open_bounds_allow_back_to_back_bookings():
    """A booking ending at 17:30 must not block a slot starting at 17:30.

    This mirrors the tstzrange '[)' bounds on the database exclusion
    constraint, so the in-memory view and the database agree about what
    counts as a collision.
    """
    busy = [
        BusyInterval(
            resource_id=DOCTOR,
            start=datetime.combine(MONDAY, time(17, 0), tzinfo=IST),
            end=datetime.combine(MONDAY, time(17, 30), tzinfo=IST),
        )
    ]
    slots = free_slots(
        rules=weekday_rules(),
        busy=busy,
        duration_min=30,
        window=window(MONDAY, time(17, 0), time(18, 0)),
        tz=IST,
        now=NOW,
    )
    assert local_times(slots) == [time(17, 30)]


def test_a_busy_interval_for_another_resource_is_ignored():
    busy = [
        BusyInterval(
            resource_id=HYGIENIST,
            start=datetime.combine(MONDAY, time(17, 0), tzinfo=IST),
            end=datetime.combine(MONDAY, time(18, 0), tzinfo=IST),
        )
    ]
    slots = free_slots(
        rules=weekday_rules(DOCTOR),
        busy=busy,
        duration_min=30,
        window=window(MONDAY, time(17, 0), time(18, 0)),
        tz=IST,
        now=NOW,
    )
    # 15-minute grid, so 17:15 is a slot too (it ends at 17:45, inside hours).
    assert local_times(slots) == [time(17, 0), time(17, 15), time(17, 30)]


def test_fully_booked_window_yields_nothing():
    busy = [
        BusyInterval(
            resource_id=DOCTOR,
            start=datetime.combine(MONDAY, time(17, 0), tzinfo=IST),
            end=datetime.combine(MONDAY, time(21, 0), tzinfo=IST),
        )
    ]
    slots = free_slots(
        rules=weekday_rules(),
        busy=busy,
        duration_min=30,
        window=window(MONDAY, time(17, 0), time(21, 0)),
        tz=IST,
        now=NOW,
    )
    assert slots == []


# --------------------------------------------------------------------------
# Lead time and grid alignment
# --------------------------------------------------------------------------

def test_no_slot_inside_the_minimum_lead_time():
    """The caller is still on the phone; they cannot arrive in five minutes."""
    slots = free_slots(
        rules=weekday_rules(),
        busy=[],
        duration_min=30,
        window=window(MONDAY, time(9, 0), time(13, 0)),
        tz=IST,
        now=NOW,
        min_lead_min=30,
    )
    assert local_times(slots)[0] == time(10, 30)


def test_slots_snap_back_onto_the_quarter_hour_grid():
    """At 10:07 the lead time lands on 10:37, which is not a time anyone says."""
    slots = free_slots(
        rules=weekday_rules(),
        busy=[],
        duration_min=30,
        window=window(MONDAY, time(9, 0), time(13, 0)),
        tz=IST,
        now=datetime(2026, 9, 28, 10, 7, 23, tzinfo=IST),
        min_lead_min=30,
    )
    assert local_times(slots)[0] == time(10, 45)
    assert all(t.minute % 15 == 0 for t in local_times(slots))


def test_granularity_is_configurable():
    slots = free_slots(
        rules=weekday_rules(),
        busy=[],
        duration_min=30,
        window=window(MONDAY, time(17, 0), time(18, 0)),
        tz=IST,
        now=NOW,
        granularity_min=30,
    )
    assert local_times(slots) == [time(17, 0), time(17, 30)]


# --------------------------------------------------------------------------
# Ordering, resources, limits
# --------------------------------------------------------------------------

def test_default_order_is_soonest_first():
    slots = free_slots(
        rules=weekday_rules(),
        busy=[],
        duration_min=30,
        window=window(MONDAY, time(17, 0), time(19, 0)),
        tz=IST,
        now=NOW,
    )
    assert local_times(slots) == sorted(local_times(slots))


def test_preferred_time_is_offered_first_then_near_misses():
    """"6:30 please" -- if 6:30 is free it comes first; either way the
    alternatives offered are the closest ones, not the earliest ones."""
    busy = [
        BusyInterval(
            resource_id=DOCTOR,
            start=datetime.combine(MONDAY, time(18, 30), tzinfo=IST),
            end=datetime.combine(MONDAY, time(19, 0), tzinfo=IST),
        )
    ]
    slots = free_slots(
        rules=weekday_rules(),
        busy=busy,
        duration_min=30,
        window=window(MONDAY, time(17, 0), time(20, 0), preferred=time(18, 30)),
        tz=IST,
        now=NOW,
    )
    # 18:30 itself is taken, and so is everything overlapping it (18:15, 18:45).
    # 18:00 and 19:00 are both exactly 30 minutes away, and the tie breaks
    # toward the earlier one -- an equally-close earlier appointment is the
    # better thing to offer first.
    assert local_times(slots)[:2] == [time(18, 0), time(19, 0)]


def test_slots_are_returned_for_each_available_resource():
    rules = weekday_rules(DOCTOR) + weekday_rules(HYGIENIST)
    slots = free_slots(
        rules=rules,
        busy=[],
        duration_min=30,
        window=window(MONDAY, time(17, 0), time(17, 30)),
        tz=IST,
        now=NOW,
    )
    assert {s.resource_id for s in slots} == {DOCTOR, HYGIENIST}


def test_limit_truncates():
    slots = free_slots(
        rules=weekday_rules(),
        busy=[],
        duration_min=30,
        window=window(MONDAY, time(17, 0), time(21, 0)),
        tz=IST,
        now=NOW,
        limit=3,
    )
    assert len(slots) == 3


def test_window_clips_opening_hours():
    slots = free_slots(
        rules=weekday_rules(),
        busy=[],
        duration_min=30,
        window=window(MONDAY, time(18, 0), time(19, 0)),
        tz=IST,
        now=NOW,
    )
    # 18:45 would end at 19:15, past the window, so it is not offered.
    assert local_times(slots) == [time(18, 0), time(18, 15), time(18, 30)]


# --------------------------------------------------------------------------
# Labels and input validation
# --------------------------------------------------------------------------

def test_label_is_spoken_form_in_business_timezone():
    slot = Slot(
        resource_id=DOCTOR,
        start=datetime(2026, 9, 29, 13, 0, tzinfo=ZoneInfo("UTC")),  # 18:30 IST
        end=datetime(2026, 9, 29, 13, 30, tzinfo=ZoneInfo("UTC")),
    )
    assert slot.label(IST) == "Tuesday 29 September at 6:30 pm"


@pytest.mark.parametrize("bad", [0, -30])
def test_non_positive_duration_is_rejected(bad):
    with pytest.raises(ValueError):
        free_slots(
            rules=weekday_rules(),
            busy=[],
            duration_min=bad,
            window=window(MONDAY, time(17, 0), time(18, 0)),
            tz=IST,
            now=NOW,
        )


def test_multi_day_window_spans_days():
    slots = free_slots(
        rules=weekday_rules(),
        busy=[],
        duration_min=30,
        window=DateWindow(
            start=datetime(2026, 9, 28, 17, 0, tzinfo=IST),
            end=datetime(2026, 9, 30, 21, 0, tzinfo=IST),
            label="rest of the week",
        ),
        tz=IST,
        now=NOW,
    )
    days = {s.start.astimezone(IST).date() for s in slots}
    assert len(days) == 3
    assert all(s.start < s.end for s in slots)
    assert all(s.end - s.start == timedelta(minutes=30) for s in slots)
