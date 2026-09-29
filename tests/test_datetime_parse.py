"""Tests for vague-date parsing.

Everything is anchored to one fixed "now" so the results never depend on the day
the suite happens to run. NOW is deliberately a Monday, because the interesting
ambiguity ("next Tuesday") turns on where we are in the week.
"""

from __future__ import annotations

from datetime import date, datetime, time
from zoneinfo import ZoneInfo

import pytest

from backend.app.domain.datetime_parse import parse_when

IST = ZoneInfo("Asia/Kolkata")
NOW = datetime(2026, 9, 28, 10, 0, tzinfo=IST)


def test_anchor_is_a_monday():
    """Guards every "next Tuesday" expectation below."""
    assert NOW.date().weekday() == 0
    assert NOW.strftime("%A") == "Monday"


def parse(phrase: str, *, now: datetime = NOW):
    return parse_when(phrase, now_local=now, tz=IST)


# --------------------------------------------------------------------------
# Relative days
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    "phrase,expected",
    [
        ("today", date(2026, 9, 28)),
        ("tomorrow", date(2026, 9, 29)),
        ("tmrw", date(2026, 9, 29)),
        ("day after tomorrow", date(2026, 9, 30)),
        ("in 3 days", date(2026, 10, 1)),
    ],
)
def test_relative_days(phrase, expected):
    window = parse(phrase)
    assert window is not None
    assert window.start.astimezone(IST).date() == expected


def test_day_after_tomorrow_is_not_read_as_tomorrow():
    """"tomorrow" is a substring of the phrase, so order of checks matters."""
    assert parse("day after tomorrow").start.date() == date(2026, 9, 30)
    assert parse("tomorrow").start.date() == date(2026, 9, 29)


# --------------------------------------------------------------------------
# Weekday names, and the "next Tuesday" ambiguity
# --------------------------------------------------------------------------

def test_bare_weekday_is_the_coming_one():
    assert parse("tuesday").start.astimezone(IST).date() == date(2026, 9, 29)


def test_this_weekday_is_the_coming_one():
    assert parse("this tuesday").start.astimezone(IST).date() == date(2026, 9, 29)


def test_next_weekday_is_the_following_week():
    """Our documented convention: weeks run Monday-Sunday, so from Monday the
    28th, "next Tuesday" is the 6th, not the 29th."""
    assert parse("next tuesday").start.astimezone(IST).date() == date(2026, 10, 6)


def test_next_weekday_from_a_sunday_is_only_two_days_out():
    sunday = datetime(2026, 9, 27, 10, 0, tzinfo=IST)
    assert sunday.weekday() == 6
    assert parse("next tuesday", now=sunday).start.astimezone(IST).date() == date(2026, 9, 29)


def test_abbreviated_weekday():
    assert parse("tue evening").start.astimezone(IST).date() == date(2026, 9, 29)


def test_weekday_today_resolves_to_today():
    assert parse("monday").start.astimezone(IST).date() == date(2026, 9, 28)


# --------------------------------------------------------------------------
# Parts of the day
# --------------------------------------------------------------------------

def test_next_tuesday_evening():
    """The worked example from the brief."""
    window = parse("next tuesday evening")
    assert window is not None
    local_start = window.start.astimezone(IST)
    local_end = window.end.astimezone(IST)
    assert local_start.date() == date(2026, 10, 6)
    assert local_start.time() == time(17, 0)
    assert local_end.time() == time(21, 0)
    assert window.preferred_start is None
    assert "Tuesday 6 October" in window.label
    assert "evening" in window.label


def test_tomorrow_morning():
    window = parse("tomorrow morning")
    assert window.start.astimezone(IST).time() == time(8, 0)
    assert window.end.astimezone(IST).time() == time(12, 0)


def test_tonight_implies_evening():
    window = parse("tonight")
    assert window.start.astimezone(IST).date() == date(2026, 9, 28)
    assert window.end.astimezone(IST).time() == time(21, 0)


# --------------------------------------------------------------------------
# Clock times
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    "phrase,expected",
    [
        ("tomorrow at 6:30 pm", time(18, 30)),
        ("tomorrow at 6.30pm", time(18, 30)),
        ("tomorrow at 18:30", time(18, 30)),
        ("tomorrow at 6 pm", time(18, 0)),
        ("tomorrow at 11 am", time(11, 0)),
        ("tomorrow at half past six", time(18, 30)),
        ("tomorrow at quarter past nine", time(9, 15)),
        ("tomorrow at 7 o'clock", time(19, 0)),
        ("tomorrow at six thirty", time(18, 30)),
        ("tomorrow at noon", time(12, 0)),
        ("tomorrow at 6", time(18, 0)),
        ("tomorrow around 4", time(16, 0)),
    ],
)
def test_clock_times(phrase, expected):
    window = parse(phrase)
    assert window is not None
    if expected is None:
        assert window.preferred_start is None
    else:
        assert window.preferred_start.astimezone(IST).time() == expected


def test_bare_hour_without_meridiem_assumes_working_hours():
    """"at 6" means 6pm; "at 9" means 9am. Documented heuristic, and the reason
    the agent is required to read the time back to the caller."""
    assert parse("tomorrow at 6").preferred_start.astimezone(IST).time() == time(18, 0)
    assert parse("tomorrow at 9").preferred_start.astimezone(IST).time() == time(9, 0)


def test_part_of_day_overrides_the_heuristic():
    assert parse("tomorrow at 8 in the evening").preferred_start.astimezone(
        IST
    ).time() == time(20, 0)
    assert parse("tomorrow at 8 in the morning").preferred_start.astimezone(
        IST
    ).time() == time(8, 0)


def test_named_time_gets_a_narrow_window_not_the_whole_day():
    """Alternatives to "6:30 pm" should be near misses, not 9am."""
    window = parse("tomorrow at 6:30 pm")
    assert window.start.astimezone(IST).time() == time(16, 30)
    assert window.end.astimezone(IST).time() == time(20, 30)


def test_bare_time_rolls_to_tomorrow_when_already_past():
    """It is 10:00. "at 9" cannot mean 9am today."""
    window = parse("at 9")
    assert window.preferred_start.astimezone(IST).date() == date(2026, 9, 29)
    assert window.preferred_start.astimezone(IST).time() == time(9, 0)


def test_bare_time_stays_today_when_still_ahead():
    window = parse("at 6")
    assert window.preferred_start.astimezone(IST).date() == date(2026, 9, 28)


# --------------------------------------------------------------------------
# Explicit dates
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    "phrase,expected",
    [
        ("30 september", date(2026, 9, 30)),
        ("30th september", date(2026, 9, 30)),
        ("september 30", date(2026, 9, 30)),
        ("sept 30", date(2026, 9, 30)),
        ("30/9", date(2026, 9, 30)),
        ("30-09", date(2026, 9, 30)),
        ("30/9/2027", date(2027, 9, 30)),
    ],
)
def test_explicit_dates(phrase, expected):
    assert parse(phrase).start.astimezone(IST).date() == expected


def test_day_first_not_month_first():
    """India-facing: 9/10 is 9 October, not 10 September."""
    assert parse("9/10").start.astimezone(IST).date() == date(2026, 10, 9)


def test_past_date_without_a_year_rolls_forward():
    """Asked in September, "3 January" means next January."""
    assert parse("3 january").start.astimezone(IST).date() == date(2027, 1, 3)


def test_impossible_date_is_rejected():
    assert parse("31 february") is None


# --------------------------------------------------------------------------
# Windows, weeks, and refusing to guess
# --------------------------------------------------------------------------

def test_next_week_spans_monday_to_sunday():
    window = parse("next week")
    assert window.start.astimezone(IST).date() == date(2026, 10, 5)
    assert window.end.astimezone(IST).date() == date(2026, 10, 11)


def test_window_never_starts_in_the_past():
    """At 19:00, "today" must not offer this morning."""
    evening = datetime(2026, 9, 28, 19, 0, tzinfo=IST)
    window = parse("today", now=evening)
    assert window.start == evening


def test_today_returns_none_once_the_day_is_over():
    midnight_ish = datetime(2026, 9, 28, 23, 30, tzinfo=IST)
    assert parse("today", now=midnight_ish) is None


@pytest.mark.parametrize("phrase", ["", "   ", "sometime", "whenever suits you", "asdfgh"])
def test_unparseable_phrases_return_none_rather_than_guessing(phrase):
    assert parse(phrase) is None


def test_naive_now_is_rejected():
    with pytest.raises(ValueError):
        parse_when("tomorrow", now_local=datetime(2026, 9, 28, 10, 0), tz=IST)


def test_windows_are_timezone_aware():
    window = parse("tomorrow morning")
    assert window.start.tzinfo is not None
    assert window.end.tzinfo is not None
