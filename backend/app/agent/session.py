"""Per-conversation state.

The important thing here is ``offered``: the registry of slots that
``check_availability`` has actually returned during this conversation.

``book_appointment`` does not accept a date or a time. It accepts a ``slot_id``,
and that id has to already be in this registry. So the hard rule -- the model
must never invent a time -- is enforced by the *shape of the tool*, not by asking
the model nicely in a prompt. A fabricated time is not a thing the schema can
express; the worst the model can do is pass an id that was never issued, which
comes straight back as an error telling it to check availability first.

In Phase 4 this state moves to Redis or Postgres so a WhatsApp thread can span
process restarts. The interface stays the same.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from zoneinfo import ZoneInfo

from ..domain.timeslots import Slot


@dataclass(frozen=True)
class OfferedSlot:
    """A slot that was genuinely offered, with everything needed to book it."""

    slot_id: str
    start: datetime
    end: datetime
    service_id: uuid.UUID
    service_name: str
    # None means "no particular staff member", which is what the caller hears as
    # "with whoever is free".
    staff_id: uuid.UUID | None
    staff_name: str | None
    label: str


@dataclass
class HandoffRequest:
    reason: str
    summary: str


@dataclass
class AgentSession:
    """Everything one conversation needs to remember."""

    business_id: uuid.UUID
    business_name: str
    tz: ZoneInfo
    caller_phone: str
    call_id: uuid.UUID | None = None

    # Slot ids issued so far, oldest first. Kept for the whole conversation so a
    # caller can circle back ("actually, the first one you said") and the id
    # still resolves.
    offered: dict[str, OfferedSlot] = field(default_factory=dict)
    _slot_counter: int = 0

    # Consecutive turns where the agent could not make progress. Two of these
    # is one of the documented handoff triggers.
    failed_attempts: int = 0
    handoff: HandoffRequest | None = None

    def offer(self, slots: list[Slot], *, service_id: uuid.UUID, service_name: str,
              staff_lookup: dict[uuid.UUID, str]) -> list[OfferedSlot]:
        """Register slots and hand back their ids.

        ``staff_lookup`` maps a resource id to a staff name. A resource id that
        is not in it is the business itself, i.e. "anyone", so ``staff_id``
        stays None -- which is exactly the value that makes the database's
        generated ``resource_id`` column fall back to the business.
        """
        issued: list[OfferedSlot] = []
        for slot in slots:
            self._slot_counter += 1
            staff_name = staff_lookup.get(slot.resource_id)
            offered = OfferedSlot(
                slot_id=f"s{self._slot_counter}",
                start=slot.start,
                end=slot.end,
                service_id=service_id,
                service_name=service_name,
                staff_id=slot.resource_id if staff_name else None,
                staff_name=staff_name,
                label=slot.label(self.tz),
            )
            self.offered[offered.slot_id] = offered
            issued.append(offered)
        return issued

    def resolve(self, slot_id: str) -> OfferedSlot | None:
        """Look up an id. Returns None for anything never issued."""
        return self.offered.get(slot_id.strip().lower())

    def record_failure(self) -> int:
        self.failed_attempts += 1
        return self.failed_attempts

    def record_success(self) -> None:
        self.failed_attempts = 0
