"""Turn vague spoken dates into a concrete search window, in business local time.

Two rules shape everything here:

1.  This module is *pure*. ``now_local`` is always passed in, never read from the
    clock, so every case below is reproducible in a test. A parser that calls
    ``datetime.now()`` internally can only be tested on the day you wrote it.

2.  When a phrase cannot be understood we return ``None`` rather than guessing.
    Silently defaulting "sometime next week" to today would put the caller in
    the wrong slot, which is worse than asking them to repeat themselves.

English is genuinely ambiguous about "next Tuesday" (the coming one, or the one
in the following week?). We resolve it one specific way, documented below and
covered by tests, and the system prompt separately requires the agent to read
the resolved date back to the caller before booking. Confirming is the only
honest way to deal with an ambiguity the language itself does not settle.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from zoneinfo import ZoneInfo

# Spoken parts of the day, as wall-clock ranges. A clinic's "evening" is a
# business fact, so these live here rather than being hardcoded mid-function.
PART_OF_DAY: dict[str, tuple[time, time]] = {
    "morning": (time(8, 0), time(12, 0)),
    "afternoon": (time(12, 0), time(17, 0)),
    "evening": (time(17, 0), time(21, 0)),
    "night": (time(19, 0), time(22, 0)),
}

# The widest window we will search when the caller names only a day.
DAY_START = time(8, 0)
DAY_END = time(21, 0)

_WEEKDAYS: dict[str, int] = {
    "monday": 0, "mon": 0,
    "tuesday": 1, "tue": 1, "tues": 1,
    "wednesday": 2, "wed": 2,
    "thursday": 3, "thu": 3, "thur": 3, "thurs": 3,
    "friday": 4, "fri": 4,
    "saturday": 5, "sat": 5,
    "sunday": 6, "sun": 6,
}

_MONTHS: dict[str, int] = {
    "january": 1, "jan": 1, "february": 2, "feb": 2, "march": 3, "mar": 3,
    "april": 4, "apr": 4, "may": 5, "june": 6, "jun": 6, "july": 7, "jul": 7,
    "august": 8, "aug": 8, "september": 9, "sep": 9, "sept": 9,
    "october": 10, "oct": 10, "november": 11, "nov": 11, "december": 12, "dec": 12,
}

# Room to grow for the Tamil-English mixed phrasing in the Phase 6 eval set:
# adding a phrase becomes a dict entry, not a new code path.
_DAY_ALIASES: dict[str, str] = {
    "tmrw": "tomorrow",
    "tmr": "tomorrow",
}

# Speech-to-text returns words, not digits: a caller says "half past six", and
# Deepgram transcribes exactly that. Rewriting number words to digits during
# normalisation means every clock pattern below keeps working on digits only,
# instead of each one needing a word variant.
# Substitution order matters: compounds come first, so "forty five" is rewritten
# to 45 before the bare "five" rule could turn it into "forty 5".
_NUMBER_WORDS: dict[str, int] = {
    "twenty five": 25, "thirty five": 35, "forty five": 45, "fifty five": 55,
    "fifteen": 15, "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50,
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
    "noon": 12, "midday": 12,
}


@dataclass(frozen=True)
class DateWindow:
    """A span of business-local time to search for free slots.

    ``preferred_start`` is set when the caller named an actual clock time. We
    keep it separate from the window so that "6:30 tomorrow" can still return
    6:45 and 7:00 when 6:30 is gone, ordered by closeness to what was asked
    for, instead of failing outright.
    """

    start: datetime
    end: datetime
    label: str
    preferred_start: datetime | None = None

    def __post_init__(self) -> None:
        if self.start.tzinfo is None or self.end.tzinfo is None:
            raise ValueError("DateWindow needs timezone-aware datetimes")
        if self.end <= self.start:
            raise ValueError("DateWindow end must be after start")


def _normalise(phrase: str) -> str:
    text = phrase.strip().lower()
    # A hyphen between digits is a date separator ("30-09"); rewrite it before
    # punctuation is stripped, so the numeric-date pattern still sees it. A
    # hyphen anywhere else is just punctuation.
    text = re.sub(r"(?<=\d)\s*-\s*(?=\d)", "/", text)
    # Keep ':' '.' '/' and the apostrophe ("o'clock"); drop other punctuation.
    text = re.sub(r"[^\w:./' ]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    for alias, canonical in _DAY_ALIASES.items():
        text = re.sub(rf"\b{alias}\b", canonical, text)
    for word, value in _NUMBER_WORDS.items():
        text = re.sub(rf"\b{word}\b", str(value), text)
    return text


def _day_label(d: date) -> str:
    # Built by hand rather than with strftime("%-d"), which is glibc-only and
    # raises on Windows.
    return f"{d.strftime('%A')} {d.day} {d.strftime('%B')}"


def _next_weekday(from_date: date, weekday: int, *, following_week: bool) -> date:
    """Resolve a weekday name to a real date.

    ``following_week=False`` ("this Tuesday", or a bare "Tuesday") means the
    next occurrence, counting today. ``following_week=True`` ("next Tuesday")
    means that weekday inside the *next* calendar week, weeks running Monday to
    Sunday. So on a Sunday, "next Tuesday" is two days away, not nine.
    """
    if following_week:
        next_monday = from_date + timedelta(days=7 - from_date.weekday())
        return next_monday + timedelta(days=weekday)
    days_ahead = (weekday - from_date.weekday()) % 7
    return from_date + timedelta(days=days_ahead)


def _shift_year_if_past(d: date, today: date) -> date | None:
    """"3 January", spoken in December, means next January."""
    if d >= today:
        return d
    try:
        return d.replace(year=d.year + 1)
    except ValueError:  # 29 February in a non-leap year
        return None


def _parse_named_date(text: str, today: date) -> date | None:
    names = "|".join(_MONTHS)
    day_num = month_name = year_s = None

    m = re.search(rf"\b(\d{{1,2}})(?:st|nd|rd|th)? ({names})\b ?(\d{{4}})?", text)
    if m:
        day_num, month_name, year_s = m.group(1), m.group(2), m.group(3)
    else:
        m = re.search(rf"\b({names}) (\d{{1,2}})(?:st|nd|rd|th)?\b ?(\d{{4}})?", text)
        if m:
            day_num, month_name, year_s = m.group(2), m.group(1), m.group(3)

    if not (day_num and month_name):
        return None
    try:
        d = date(int(year_s) if year_s else today.year, _MONTHS[month_name], int(day_num))
    except ValueError:
        return None
    return d if year_s else _shift_year_if_past(d, today)


def _parse_numeric_date(text: str, today: date) -> date | None:
    """Day-first: this is an India-facing product, so 30/9 is 30 September."""
    m = re.search(r"\b(\d{1,2})[/-](\d{1,2})(?:[/-](\d{2,4}))?\b", text)
    if not m:
        return None
    day_num, month_num, year_s = int(m.group(1)), int(m.group(2)), m.group(3)
    year = today.year
    if year_s:
        year = int(year_s) if len(year_s) == 4 else 2000 + int(year_s)
    try:
        d = date(year, month_num, day_num)
    except ValueError:
        return None
    return d if year_s else _shift_year_if_past(d, today)


def _parse_day(text: str, today: date) -> tuple[date, date, str] | None:
    """Resolve the day part of a phrase to an inclusive date range plus a label."""
    # "day after tomorrow" is checked before "tomorrow" so it is not shadowed.
    if re.search(r"\bday after tomorrow\b", text):
        d = today + timedelta(days=2)
        return d, d, _day_label(d)
    if re.search(r"\b(today|tonight|now|asap)\b", text):
        return today, today, "today"
    if re.search(r"\btomorrow\b", text):
        d = today + timedelta(days=1)
        return d, d, "tomorrow"

    m = re.search(r"\bin (\d{1,2}) days?\b", text)
    if m:
        d = today + timedelta(days=int(m.group(1)))
        return d, d, _day_label(d)

    if re.search(r"\bnext week\b", text):
        start = today + timedelta(days=7 - today.weekday())
        return start, start + timedelta(days=6), "next week"
    if re.search(r"\bthis week\b", text):
        return today, today + timedelta(days=6 - today.weekday()), "this week"

    for parser in (_parse_named_date, _parse_numeric_date):
        d = parser(text, today)
        if d is not None:
            return d, d, _day_label(d)

    # Weekday names last, so "next week" is not mistaken for a weekday phrase.
    # Longest name first: a bare "tue" must not match inside "tuesday".
    for name in sorted(_WEEKDAYS, key=len, reverse=True):
        if not re.search(rf"\b{name}\b", text):
            continue
        following = bool(re.search(rf"\bnext {name}\b", text))
        d = _next_weekday(today, _WEEKDAYS[name], following_week=following)
        return d, d, _day_label(d)

    return None


def _apply_meridiem(hour: int, minute: int, meridiem: str | None, text: str) -> time | None:
    if not (0 <= hour <= 23 and 0 <= minute <= 59):
        return None
    if meridiem == "pm" and hour < 12:
        hour += 12
    elif meridiem == "am" and hour == 12:
        hour = 0
    elif meridiem is None and hour <= 12:
        # No am/pm given. A part-of-day word settles it; otherwise fall back to
        # the shape of a working day -- "at 6" is 6pm, "at 9" is 9am. This is
        # wrong occasionally, which is why the agent confirms the time aloud.
        if hour < 12 and re.search(r"\b(evening|night|afternoon)\b", text):
            hour += 12
        elif re.search(r"\bmorning\b", text):
            pass
        elif 1 <= hour <= 7:
            hour += 12
    return time(hour, minute) if hour <= 23 else None


def _parse_clock(text: str) -> time | None:
    """Pull an explicit clock time out of the phrase, if there is one."""
    m = re.search(r"\b(\d{1,2})[:.](\d{2}) ?(am|pm)?\b", text)
    if m:
        return _apply_meridiem(int(m.group(1)), int(m.group(2)), m.group(3), text)

    m = re.search(r"\b(half|quarter) past (\d{1,2})\b", text)
    if m:
        return _apply_meridiem(int(m.group(2)), 30 if m.group(1) == "half" else 15, None, text)

    m = re.search(r"\b(\d{1,2}) ?(am|pm)\b", text)
    if m:
        return _apply_meridiem(int(m.group(1)), 0, m.group(2), text)

    m = re.search(r"\b(\d{1,2}) ?o'?clock\b", text)
    if m:
        return _apply_meridiem(int(m.group(1)), 0, None, text)

    # Spoken "six thirty", which normalises to "6 30". The preposition is
    # required so a date like "30 09" cannot be read as a time.
    m = re.search(r"\b(?:at|around|by) (\d{1,2}) (\d{2})\b", text)
    if m:
        return _apply_meridiem(int(m.group(1)), int(m.group(2)), None, text)

    # A bare hour, again only after a preposition -- "can you do around 6".
    # Requiring it keeps us from reading the "3" in "in 3 days" as a time.
    m = re.search(r"\b(?:at|around|by) (\d{1,2})\b", text)
    if m:
        return _apply_meridiem(int(m.group(1)), 0, None, text)

    return None


def _format_clock(t: time) -> str:
    return t.strftime("%I:%M %p").lstrip("0").lower()


def _shift_hours(t: time, hours: int) -> time:
    """Clamp rather than wrap: a window must not spill onto the next day."""
    total = (t.hour + hours) * 60 + t.minute
    total = max(0, min(total, 23 * 60 + 59))
    return time(total // 60, total % 60)


# How far either side of a requested clock time we look for alternatives.
NEAR_MISS_HOURS = 2


def parse_when(phrase: str, *, now_local: datetime, tz: ZoneInfo) -> DateWindow | None:
    """Parse a spoken date/time phrase into a window of business-local time.

    Returns ``None`` when the phrase names no day and no time we recognise, so
    that ``check_availability`` can ask the caller to be more specific instead
    of inventing a date.
    """
    if now_local.tzinfo is None:
        raise ValueError("now_local must be timezone-aware")

    text = _normalise(phrase)
    if not text:
        return None

    day = _parse_day(text, now_local.date())
    clock = _parse_clock(text)

    part = next((name for name in PART_OF_DAY if re.search(rf"\b{name}\b", text)), None)
    if part is None and re.search(r"\btonight\b", text):
        part = "evening"

    if day is None:
        # A bare time with no day ("at 6:30") means the next time that clock
        # time comes around: today if it has not passed, otherwise tomorrow.
        if clock is None:
            return None
        candidate = datetime.combine(now_local.date(), clock, tzinfo=tz)
        if candidate <= now_local:
            candidate += timedelta(days=1)
        start_date = end_date = candidate.date()
        label = "today" if candidate.date() == now_local.date() else "tomorrow"
    else:
        start_date, end_date, label = day

    lo, hi = PART_OF_DAY.get(part, (DAY_START, DAY_END)) if part else (DAY_START, DAY_END)

    if clock is not None:
        preferred = datetime.combine(start_date, clock, tzinfo=tz)
        # A named time gets a narrow window around it, so the alternatives we
        # offer are near misses ("6:15 or 6:45") rather than the whole day.
        # A part-of-day word, if one was also said, bounds it further.
        near_lo = _shift_hours(clock, -NEAR_MISS_HOURS)
        near_hi = _shift_hours(clock, NEAR_MISS_HOURS)
        if part:
            near_lo, near_hi = max(near_lo, lo), min(near_hi, hi)
            # Keep the requested time inside its own window even if it sits
            # outside the part-of-day range ("7:30 in the afternoon").
            near_lo, near_hi = min(near_lo, clock), max(near_hi, _shift_hours(clock, 1))
        window_start = datetime.combine(start_date, near_lo, tzinfo=tz)
        window_end = datetime.combine(end_date, near_hi, tzinfo=tz)
        label = f"{label} at {_format_clock(clock)}"
    else:
        preferred = None
        window_start = datetime.combine(start_date, lo, tzinfo=tz)
        window_end = datetime.combine(end_date, hi, tzinfo=tz)
        if part:
            label = f"{label} {part}"

    # Never search the past. Someone asking about "today" at 7pm should be
    # offered tonight's remaining slots, not this morning's.
    window_start = max(window_start, now_local)
    if window_end <= window_start:
        return None

    return DateWindow(
        start=window_start, end=window_end, label=label, preferred_start=preferred
    )
