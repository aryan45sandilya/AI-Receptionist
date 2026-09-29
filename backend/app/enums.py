"""Closed sets of values, shared between Python and Postgres.

These are native Postgres enum types rather than text-plus-CHECK. The sets are
genuinely closed (a booking is not going to grow a sixth status by accident),
native enums keep the dashboard's ORDER BY meaningful, and since Postgres 12
``ALTER TYPE ... ADD VALUE`` runs inside a transaction, so adding a value later
is still an ordinary migration.
"""

from __future__ import annotations

import enum


class BookingStatus(enum.StrEnum):
    BOOKED = "booked"
    CANCELLED = "cancelled"
    COMPLETED = "completed"
    NO_SHOW = "no_show"


class BookingSource(enum.StrEnum):
    VOICE = "voice"
    WHATSAPP = "whatsapp"
    DASHBOARD = "dashboard"


class CallChannel(enum.StrEnum):
    WEB_DEMO = "web_demo"
    TWILIO_VOICE = "twilio_voice"
    WHATSAPP = "whatsapp"
    TERMINAL = "terminal"  # the Phase 1 text agent


class CallStatus(enum.StrEnum):
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    HANDED_OFF = "handed_off"


class TurnRole(enum.StrEnum):
    CALLER = "caller"
    AGENT = "agent"
