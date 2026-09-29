"""The five tools, their schemas, and the dispatcher.

Design notes worth knowing before reading the code:

*   Every tool is declared ``strict``. With ``additionalProperties: false`` and a
    complete ``required`` list, arguments are guaranteed to validate against the
    schema before they reach us, so nothing below has to defend against a
    missing key or a wrong type. Optional parameters are expressed as a nullable
    type rather than by omission, which is how optionality works under strict
    schemas.

*   ``book_appointment`` takes a ``slot_id``, never a time. See
    ``agent/session.py`` for why.

*   Tool results are plain prose, not JSON. The model is the only consumer, and
    prose lets a result carry both the facts and what to do next ("that slot
    went, call check_availability again") in the same breath. Every result is
    written so that the most obvious next action is also the correct one.

*   Results never contain a time the caller could act on unless it came from the
    database this turn.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta

from sqlalchemy.ext.asyncio import AsyncSession

from ..config import Settings
from ..domain import availability, booking as booking_domain
from ..domain.datetime_parse import parse_when
from ..domain.timeslots import spread
from ..enums import BookingSource
from .session import AgentSession, HandoffRequest

# Reasons a call may legitimately leave the agent. Kept as an enum in the schema
# so the dashboard can group handoffs without parsing free text.
HANDOFF_REASONS = [
    "caller_asked_for_human",
    "caller_frustrated",
    "outside_booking_scope",
    "agent_cannot_proceed",
]


@dataclass
class ToolContext:
    """Everything a tool needs, passed explicitly rather than read from globals.

    ``now`` is passed in for the same reason the parser takes it: a conversation
    should be replayable in a test at a fixed point in time.
    """

    db: AsyncSession
    session: AgentSession
    settings: Settings
    now: datetime
    source: BookingSource
    # Slow-changing facts about the business (services, staff, opening hours),
    # read once per conversation. See domain/availability.py for why.
    snapshot: availability.Snapshot | None = None

    async def get_snapshot(self) -> availability.Snapshot:
        """The call's snapshot, loading it on first use.

        A voice call loads it up front, while the greeting is playing, so the
        caller never waits for it. Anything that does not (a test, the terminal
        demo) still works: the first tool call pays for it instead.
        """
        if self.snapshot is None:
            self.snapshot = await availability.load_snapshot(self.db, self.session.business_id)
        return self.snapshot


def _tool_definitions() -> list[dict]:
    """The tools, in a provider-neutral shape: name, description, parameters.

    Kept separate from the wire format so that changing provider is a change to
    ``tool_schemas`` below and nothing else. The order is fixed rather than
    derived from a set, so the prompt a model sees is identical run to run --
    which keeps behaviour reproducible and lets any provider with prefix
    caching actually hit it.
    """
    return [
        {
            "name": "check_availability",
            "description": (
                "Find real, bookable appointment slots. This is the ONLY source of "
                "appointment times. Call it before offering any time to the caller, "
                "and offer only the slots it returns, using the exact wording it "
                "gives. Never state or imply a time this tool has not returned."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "service": {
                        "type": "string",
                        "description": "Service the caller asked for, in their words.",
                    },
                    "when": {
                        "type": "string",
                        "description": (
                            "When they want it, in their words: 'next Tuesday evening', "
                            "'tomorrow at 6:30', 'day after tomorrow'. Pass the phrase "
                            "through as spoken; do not convert it to a date yourself."
                        ),
                    },
                    "staff_name": {
                        "type": ["string", "null"],
                        "description": "Staff member requested by name, else null.",
                    },
                },
                "required": ["service", "when", "staff_name"],
                "additionalProperties": False,
            },
        },
        {
            "name": "book_appointment",
            "description": (
                "Confirm a booking. Takes the slot_id of a slot that check_availability "
                "returned in this conversation -- it does not take a date or a time. "
                "Only call this once the caller has agreed to a specific slot and you "
                "have their name."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "slot_id": {
                        "type": "string",
                        "description": "Slot id from check_availability, e.g. 's2'.",
                    },
                    "customer_name": {
                        "type": "string",
                        "description": "Caller's name, as they gave it.",
                    },
                    "customer_phone": {
                        "type": ["string", "null"],
                        "description": (
                            "Only if the caller asked to use a different number from "
                            "the one they are calling from. Otherwise null."
                        ),
                    },
                },
                "required": ["slot_id", "customer_name", "customer_phone"],
                "additionalProperties": False,
            },
        },
        {
            "name": "cancel_appointment",
            "description": (
                "Cancel an existing booking. Call with booking_reference null first to "
                "see what this caller has booked; if there is more than one, ask which "
                "and call again with the reference."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "booking_reference": {
                        "type": ["string", "null"],
                        "description": "Six-character reference, or null to list.",
                    },
                    "customer_phone": {
                        "type": ["string", "null"],
                        "description": "Different number from the caller's, else null.",
                    },
                },
                "required": ["booking_reference", "customer_phone"],
                "additionalProperties": False,
            },
        },
        {
            "name": "reschedule_appointment",
            "description": (
                "Move an existing booking to a new slot. Needs a new_slot_id from "
                "check_availability. As with cancelling, pass booking_reference null "
                "first if you do not know which booking is meant."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "new_slot_id": {
                        "type": "string",
                        "description": "Slot id from check_availability.",
                    },
                    "booking_reference": {
                        "type": ["string", "null"],
                        "description": "Six-character reference, or null to list.",
                    },
                    "customer_phone": {
                        "type": ["string", "null"],
                        "description": "Different number from the caller's, else null.",
                    },
                },
                "required": ["new_slot_id", "booking_reference", "customer_phone"],
                "additionalProperties": False,
            },
        },
        {
            "name": "handoff_to_human",
            "description": (
                "Hand the call to a member of staff. Call this when the caller asks "
                "for a person, when they are clearly frustrated after two failed "
                "attempts, or when they want something outside booking, changing and "
                "cancelling appointments -- medical advice, prices you do not know, "
                "complaints. Do not guess at an answer instead of calling this."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "reason": {"type": "string", "enum": HANDOFF_REASONS},
                    "summary": {
                        "type": "string",
                        "description": (
                            "One or two sentences for the person picking up: what the "
                            "caller wants and what has happened so far."
                        ),
                    },
                },
                "required": ["reason", "summary"],
                "additionalProperties": False,
            },
        },
    ]


def tool_schemas() -> list[dict]:
    """The definitions above in OpenAI's ``tools`` wire format, which is what
    Groq's endpoint speaks. Swapping provider means changing this function."""
    return [{"type": "function", "function": definition} for definition in _tool_definitions()]


# ---------------------------------------------------------------------------
# Argument validation
#
# A JSON schema in a tool definition is a hint to the model, not a guarantee:
# nothing on the wire enforces it. An open-weights model omits a required field,
# writes the string "null", or invents a parameter often enough that all three
# have to be handled here.
#
# Handled means answering the model in a way it can act on. Raising would end a
# call over a mistake the model would have corrected on the next turn.
# ---------------------------------------------------------------------------

# The many ways a model says "nothing" when it means null.
_NULLISH = {"", "null", "none", "nil", "n/a", "na", "undefined", "unknown"}


def _allowed_types(spec: dict) -> set[str]:
    declared = spec.get("type", "string")
    return set(declared) if isinstance(declared, list) else {declared}


def _clean(value: object) -> object:
    if isinstance(value, str):
        stripped = value.strip()
        return None if stripped.lower() in _NULLISH else stripped
    return value


def validate_arguments(name: str, args: dict) -> tuple[dict | None, str | None]:
    """Check and normalise one tool call's arguments.

    Returns ``(cleaned, None)`` on success or ``(None, message)`` on failure,
    where the message is written for the model rather than for a log.
    """
    definition = next((d for d in _tool_definitions() if d["name"] == name), None)
    if definition is None:
        # Reachable only if the schema list and the handler table drift apart.
        return None, f"ERROR: no tool named {name!r}."

    schema = definition["parameters"]
    properties: dict[str, dict] = schema["properties"]

    # Drop anything invented; keep only declared parameters.
    cleaned = {key: _clean(value) for key, value in args.items() if key in properties}

    missing = [
        key
        for key in schema["required"]
        if cleaned.get(key) is None and "null" not in _allowed_types(properties[key])
    ]
    if missing:
        wanted = ", ".join(f"{key} ({properties[key].get('description', '')})" for key in missing)
        return None, (
            f"ERROR: {name} was called without {', '.join(missing)}. "
            f"It needs: {wanted}. Ask the caller for what is missing, then call it again."
        )

    # Optional parameters the model left out become an explicit None, so the
    # handlers can assume every declared key is present.
    for key in properties:
        cleaned.setdefault(key, None)
    return cleaned, None


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

async def execute_tool(name: str, args: dict, ctx: ToolContext) -> str:
    cleaned, error = validate_arguments(name, args)
    if error is not None:
        if not error.startswith("ERROR: no tool named"):
            # A malformed call is a turn that made no progress, which counts
            # toward the handoff policy. An unknown tool name is our bug, not
            # the conversation's, so it does not.
            ctx.session.record_failure()
        return error

    # Built here rather than at module level because the handlers are defined
    # further down the file.
    handlers = {
        "check_availability": _check_availability,
        "book_appointment": _book_appointment,
        "cancel_appointment": _cancel_appointment,
        "reschedule_appointment": _reschedule_appointment,
        "handoff_to_human": _handoff_to_human,
    }
    return await handlers[name](cleaned, ctx)


def _phone(args: dict, ctx: ToolContext) -> str:
    """The caller's own number unless they explicitly gave another.

    Defaulting to the number we already have avoids a whole class of failure
    where a digit is lost in transcription and the confirmation goes nowhere.
    """
    given = (args.get("customer_phone") or "").strip()
    return given or ctx.session.caller_phone


# ---------------------------------------------------------------------------


async def _check_availability(args: dict, ctx: ToolContext) -> str:
    session = ctx.session
    snapshot = await ctx.get_snapshot()
    tz = snapshot.business.tz
    now_local = ctx.now.astimezone(tz)

    services = snapshot.find_services(args["service"])
    if not services:
        session.record_failure()
        names = ", ".join(s.name for s in snapshot.services) or "none configured"
        return (
            f"No service matching {args['service']!r}. This business offers: {names}. "
            "Ask the caller which of these they want; do not invent a service."
        )
    if len(services) > 1:
        names = ", ".join(s.name for s in services)
        return (
            f"{args['service']!r} matches more than one service: {names}. "
            "Ask the caller which one they mean, then call check_availability again."
        )
    service = services[0]

    staff_id = None
    staff_name = None
    if args.get("staff_name"):
        matches = snapshot.find_staff(args["staff_name"])
        if not matches:
            session.record_failure()
            return (
                f"Nobody here matches {args['staff_name']!r}. Ask the caller to repeat "
                "the name, or offer the next available person instead."
            )
        if len(matches) > 1:
            return (
                f"{args['staff_name']!r} matches several people: "
                f"{', '.join(m.name for m in matches)}. Ask which one."
            )
        staff_id, staff_name = matches[0].id, matches[0].name

    window = parse_when(args["when"], now_local=now_local, tz=tz)
    if window is None:
        session.record_failure()
        return (
            f"Could not read {args['when']!r} as a date or time. Ask the caller for a "
            "specific day -- 'this Thursday', 'the 14th', 'tomorrow afternoon'. "
            "Do not guess a date."
        )

    slots = await availability.find_free_slots(
        ctx.db,
        snapshot=snapshot,
        duration_min=service.duration_min,
        window=window,
        now=ctx.now,
        staff_id=staff_id,
        granularity_min=ctx.settings.slot_granularity_min,
        min_lead_min=ctx.settings.min_lead_min,
    )
    if staff_id is None:
        slots = availability.collapse_by_time(slots)
    slots = spread(
        slots,
        min_gap=timedelta(minutes=service.duration_min),
        limit=ctx.settings.max_slots_offered,
    )

    if not slots:
        session.record_failure()
        who = f" with {staff_name}" if staff_name else ""
        return (
            f"Nothing free for {service.name}{who} in {window.label}. "
            "Ask the caller for a different day or a wider time of day, then call "
            "check_availability again. Do not offer any time this tool has not returned."
        )

    issued = session.offer(
        slots,
        service_id=service.id,
        service_name=service.name,
        staff_lookup=snapshot.staff_names(),
    )
    session.record_success()

    lines = [
        f"{service.name}, {service.duration_min} minutes. Free slots in {window.label}:"
    ]
    for slot in issued:
        with_whom = f" with {slot.staff_name}" if slot.staff_name else ""
        lines.append(f"  [{slot.slot_id}] {slot.label}{with_whom}")
    lines.append(
        "Offer at most the first two, in one short sentence, using this exact wording "
        "for each time. To book, call book_appointment with the slot_id in brackets."
    )
    return "\n".join(lines)


async def _book_appointment(args: dict, ctx: ToolContext) -> str:
    session = ctx.session
    slot = session.resolve(args["slot_id"])
    if slot is None:
        session.record_failure()
        # The structural guard. An id that was never issued cannot be booked,
        # which is what stops a hallucinated time from ever reaching the table.
        known = ", ".join(session.offered) or "none yet"
        return (
            f"No slot {args['slot_id']!r} has been offered in this conversation "
            f"(ids issued so far: {known}). Call check_availability and offer only "
            "the slots it returns."
        )

    name = args["customer_name"].strip()
    if not session.name_was_spoken(name):
        # Nothing has been booked yet, so this costs a question, not a mistake.
        session.record_failure()
        return (
            f"The caller has not told you a name like {name!r}. Ask \"May I have your "
            "name?\", wait for their answer, and book under exactly what they say. "
            "Never use a placeholder."
        )

    try:
        created = await booking_domain.create_booking(
            ctx.db,
            business_id=session.business_id,
            service_id=slot.service_id,
            staff_id=slot.staff_id,
            customer_name=name,
            customer_phone_e164=_phone(args, ctx),
            slot_start=slot.start,
            slot_end=slot.end,
            source=ctx.source,
            call_id=session.call_id,
        )
    except booking_domain.SlotTakenError:
        session.record_failure()
        # The exclusion constraint rejected it: someone else got the slot during
        # this conversation. Nothing to retry -- the slot is genuinely gone.
        return (
            f"{slot.label} was taken by someone else while you were talking. "
            "Apologise briefly, call check_availability again, and offer the new times."
        )
    except booking_domain.BookingError as exc:
        session.record_failure()
        return f"Booking failed: {exc}. Apologise and offer to have someone call back."

    await ctx.db.commit()
    session.record_success()
    with_whom = f" with {slot.staff_name}" if slot.staff_name else ""
    return (
        f"Booked. {slot.service_name}{with_whom}, {slot.label}. "
        f"Reference {created.reference}. Read the day, the time and the reference back "
        "to the caller, then ask if they need anything else."
    )


async def _resolve_booking(args: dict, ctx: ToolContext, verb: str):
    """Shared lookup for cancel and reschedule.

    Returns either a booking or a string to hand back to the model, because
    "there are two, ask which" is a perfectly good outcome that is not an error.
    """
    session = ctx.session
    reference = (args.get("booking_reference") or "").strip()

    if reference:
        try:
            found = await booking_domain.get_booking_by_reference(
                ctx.db, business_id=session.business_id, reference=reference
            )
        except booking_domain.BookingNotFoundError:
            session.record_failure()
            return (
                f"No booking with reference {reference!r}. Ask the caller to read it "
                "again, or call this tool with booking_reference null to list what "
                "their number has booked."
            )
        return found

    active = await booking_domain.find_active_bookings(
        ctx.db,
        business_id=session.business_id,
        customer_phone_e164=_phone(args, ctx),
        after=ctx.now,
    )
    if not active:
        session.record_failure()
        return (
            "No upcoming bookings found for that number. Ask the caller which number "
            "the appointment was booked under, or offer to hand over to a colleague."
        )
    if len(active) > 1:
        lines = [f"That number has {len(active)} upcoming bookings:"]
        for row in active:
            local = row.slot_start.astimezone(session.tz)
            lines.append(
                f"  {row.reference}: {row.service.name}, "
                f"{local.strftime('%A')} {local.day} {local.strftime('%B')} at "
                f"{local.strftime('%I:%M %p').lstrip('0').lower()}"
            )
        lines.append(f"Ask which one to {verb}, then call again with its reference.")
        return "\n".join(lines)
    return active[0]


async def _cancel_appointment(args: dict, ctx: ToolContext) -> str:
    found = await _resolve_booking(args, ctx, "cancel")
    if isinstance(found, str):
        return found

    local = found.slot_start.astimezone(ctx.session.tz)
    await booking_domain.cancel_booking(ctx.db, found, now=ctx.now)
    await ctx.db.commit()
    ctx.session.record_success()
    return (
        f"Cancelled {found.reference}: {found.service.name} on "
        f"{local.strftime('%A')} {local.day} {local.strftime('%B')}. "
        "Tell the caller it is cancelled and ask if they want to rebook."
    )


async def _reschedule_appointment(args: dict, ctx: ToolContext) -> str:
    session = ctx.session
    slot = session.resolve(args["new_slot_id"])
    if slot is None:
        session.record_failure()
        known = ", ".join(session.offered) or "none yet"
        return (
            f"No slot {args['new_slot_id']!r} has been offered in this conversation "
            f"(ids issued so far: {known}). Call check_availability first."
        )

    found = await _resolve_booking(args, ctx, "move")
    if isinstance(found, str):
        return found

    was = found.slot_start.astimezone(session.tz)
    try:
        await booking_domain.reschedule_booking(
            ctx.db, found, slot_start=slot.start, slot_end=slot.end, staff_id=slot.staff_id
        )
    except booking_domain.SlotTakenError:
        session.record_failure()
        return (
            f"{slot.label} was taken while you were talking. The original booking is "
            "unchanged. Call check_availability again and offer the new times."
        )
    except booking_domain.BookingError as exc:
        session.record_failure()
        return f"Could not move the booking: {exc}."

    await ctx.db.commit()
    session.record_success()
    return (
        f"Moved {found.reference} from {was.strftime('%A')} {was.day} "
        f"{was.strftime('%B')} to {slot.label}. Confirm the new day and time to the "
        "caller, and tell them the reference is unchanged."
    )


async def _handoff_to_human(args: dict, ctx: ToolContext) -> str:
    ctx.session.handoff = HandoffRequest(reason=args["reason"], summary=args["summary"])
    target = (await ctx.get_snapshot()).business.handoff_phone_e164
    where = f" to {target}" if target else ""
    return (
        f"Handoff recorded ({args['reason']}). Tell the caller you are putting them "
        f"through{where} and stop trying to handle the request yourself."
    )
