"""Phase 1: the agent as a terminal conversation.

Same agent, same tools and same database writes that the phone call will use in
Phase 3 -- only the transport is different. Proving the booking logic in text
first means that when audio is added, a failure is a voice-pipeline failure and
nothing else.

    python -m backend.app.cli.chat
    python -m backend.app.cli.chat --phone +919876543210

Type 'quit' to hang up.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import uuid
from datetime import UTC, datetime
from zoneinfo import ZoneInfo

from openai import AsyncOpenAI
from sqlalchemy import select

from ..agent.loop import TurnResult, build_runner
from ..agent.session import AgentSession
from ..agent.tools import ToolContext
from ..config import get_settings
from ..runtime import run_async
from ..db import dispose_engine, get_session_factory
from ..domain import calls as call_log
from ..enums import BookingSource, CallChannel, CallStatus, TurnRole
from ..models import Business

DEFAULT_CALLER = "+919876543210"


def _dim(text: str) -> str:
    return f"\x1b[2m{text}\x1b[0m"


async def _load_business(session, phone: str | None) -> Business:
    stmt = select(Business).order_by(Business.created_at)
    if phone:
        stmt = stmt.where(Business.phone_e164 == phone)
    business = (await session.execute(stmt)).scalars().first()
    if business is None:
        raise SystemExit(
            "No business found. Run:  python -m scripts.seed_demo"
            + (f"\n(looked for phone {phone})" if phone else "")
        )
    return business


async def converse(caller_phone: str, business_phone: str | None, show_latency: bool) -> None:
    settings = get_settings()
    if not settings.llm_api_key:
        raise SystemExit(
            "LLM_API_KEY is not set. Get a free key at https://console.groq.com/keys "
            "and put it in .env."
        )

    client = AsyncOpenAI(api_key=settings.llm_api_key, base_url=settings.llm_base_url)

    async with get_session_factory()() as db:
        business = await _load_business(db, business_phone)
        tz = ZoneInfo(business.timezone)

        call = await call_log.start_call(
            db,
            business_id=business.id,
            channel=CallChannel.TERMINAL,
            from_phone_e164=caller_phone,
        )
        # Committed immediately so bookings made during the call can reference it.
        await db.commit()

        agent_session = AgentSession(
            business_id=business.id,
            business_name=business.name,
            tz=tz,
            caller_phone=caller_phone,
            call_id=call.id,
        )
        ctx = ToolContext(
            db=db,
            session=agent_session,
            settings=settings,
            now=datetime.now(UTC),
            source=BookingSource.VOICE,
        )
        runner = await build_runner(client=client, settings=settings, ctx=ctx)

        print(f"\n  {business.name} -- terminal demo")
        print(_dim(f"  call {call.id}  caller {caller_phone}  timezone {business.timezone}"))
        print(_dim(f"  local time now: {datetime.now(tz).strftime('%A %d %B %Y, %I:%M %p')}"))
        print(_dim("  type 'quit' to hang up\n"))

        # The greeting is a fixed string, not a model call: it is the same every
        # time and paying for a round trip to produce it would add latency to the
        # one moment the caller is definitely waiting.
        print(f"Agent: {business.greeting}")
        await call_log.log_turn(
            db,
            call_id=call.id,
            role=TurnRole.AGENT,
            text=business.greeting,
            turn_started_at=datetime.now(UTC),
        )
        await db.commit()

        status = CallStatus.COMPLETED
        while True:
            try:
                user_text = (await asyncio.to_thread(input, "\nYou:   ")).strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if not user_text:
                continue
            if user_text.lower() in {"quit", "exit", "bye"}:
                break

            now = datetime.now(UTC)
            await call_log.log_turn(
                db, call_id=call.id, role=TurnRole.CALLER, text=user_text,
                turn_started_at=now,
            )

            print("Agent: ", end="", flush=True)
            result = await runner.run_turn(
                user_text, now=now, on_text=lambda chunk: print(chunk, end="", flush=True)
            )
            print()

            await _log_agent_turn(db, call.id, result)
            await db.commit()

            if show_latency:
                print(_dim(f"       {_latency_line(result)}"))

            if result.handoff:
                reason = (
                    agent_session.handoff.reason if agent_session.handoff else result.stop_reason
                )
                print(_dim(f"\n  [handoff: {reason}] call would transfer to a human here"))
                status = CallStatus.HANDED_OFF
                break

        await call_log.end_call(
            db,
            call,
            now=datetime.now(UTC),
            status=status,
            handoff_reason=(
                agent_session.handoff.summary if agent_session.handoff else None
            ),
        )
        await db.commit()
        print(_dim(f"\n  call {call.id} ended: {status.value}"))
        print(_dim("  turns and latency are in call_turns; bookings are in bookings\n"))


async def _log_agent_turn(db, call_id: uuid.UUID, result: TurnResult) -> None:
    m = result.metrics
    await call_log.log_turn(
        db,
        call_id=call_id,
        role=TurnRole.AGENT,
        text=result.text,
        turn_started_at=m.turn_started_at,
        stt_final_ms=m.stt_final_ms,
        llm_first_token_ms=m.llm_first_token_ms,
        llm_done_ms=m.llm_done_ms,
        tts_first_audio_ms=m.tts_first_audio_ms,
        total_ms=m.total_ms,
        tool_calls=result.tool_calls,
        model=m.model,
        input_tokens=m.input_tokens,
        output_tokens=m.output_tokens,
        cache_read_tokens=m.cache_read_tokens,
    )


def _latency_line(result: TurnResult) -> str:
    m = result.metrics
    parts = [
        f"first token {m.llm_first_token_ms}ms" if m.llm_first_token_ms is not None else "first token n/a",
        f"llm done {m.llm_done_ms}ms",
        f"total {m.total_ms}ms",
        f"in {m.input_tokens} out {m.output_tokens} cached {m.cache_read_tokens}",
    ]
    if result.tool_calls:
        names = ", ".join(
            f"{c['name']} {c['duration_ms']}ms" for c in result.tool_calls
        )
        parts.append(f"tools: {names}")
    return " | ".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser(description="Terminal receptionist demo")
    parser.add_argument(
        "--phone", default=DEFAULT_CALLER, help="the number the caller is calling from"
    )
    parser.add_argument(
        "--business", default=None, help="business phone number, if more than one is seeded"
    )
    parser.add_argument("--no-latency", action="store_true", help="hide the latency line")
    args = parser.parse_args()

    try:
        run_async(_run(args.phone, args.business, show_latency=not args.no_latency))
    except KeyboardInterrupt:
        sys.exit(0)


async def _run(caller: str, business: str | None, *, show_latency: bool) -> None:
    try:
        await converse(caller, business, show_latency)
    finally:
        await dispose_engine()


if __name__ == "__main__":
    main()
