"""Run a scripted conversation end to end, without a human at the keyboard.

A smoke test, not a unit test: it calls the real model, writes a real booking,
and prints per-turn latency. That makes it the first thing to run after changing
provider, model or prompt, and the seed of the Phase 6 eval harness -- which is
this loop over forty scripted conversations with grading on top.

    python -m scripts.smoke_chat
    python -m scripts.smoke_chat --scenario handoff
"""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
from zoneinfo import ZoneInfo

from openai import AsyncOpenAI
from sqlalchemy import select

from backend.app.agent.loop import build_runner
from backend.app.agent.session import AgentSession
from backend.app.agent.tools import ToolContext
from backend.app.config import get_settings
from backend.app.db import dispose_engine, get_session_factory
from backend.app.domain import calls as call_log
from backend.app.enums import BookingSource, CallChannel, CallStatus, TurnRole
from backend.app.models import Booking, Business
from backend.app.runtime import run_async

SCENARIOS: dict[str, list[str]] = {
    # The happy path, with a vague date the parser has to resolve.
    "booking": [
        "hi, i need a dental cleaning next tuesday evening",
        "yeah the first one works",
        "Asha Menon",
    ],
    # Asks for something outside booking, then for a person.
    "handoff": [
        "hi, my tooth has been hurting for three days, what should i do",
        "can i just talk to someone please",
    ],
    # A date nobody can resolve: the agent must ask, not guess.
    "vague": [
        "i want to book a cleaning",
        "sometime whenever you have space",
    ],
}


async def run(scenario: str) -> None:
    settings = get_settings()
    if not settings.llm_api_key:
        raise SystemExit("LLM_API_KEY is not set. See .env")

    client = AsyncOpenAI(api_key=settings.llm_api_key, base_url=settings.llm_base_url)

    async with get_session_factory()() as db:
        business = (
            await db.execute(select(Business).order_by(Business.created_at))
        ).scalars().first()
        if business is None:
            raise SystemExit("No business seeded. Run: python -m scripts.seed_demo")

        tz = ZoneInfo(business.timezone)
        call = await call_log.start_call(
            db, business_id=business.id, channel=CallChannel.TERMINAL,
            from_phone_e164="+919876543210",
        )
        await db.commit()

        agent_session = AgentSession(
            business_id=business.id, business_name=business.name, tz=tz,
            caller_phone="+919876543210", call_id=call.id,
        )
        ctx = ToolContext(
            db=db, session=agent_session, settings=settings,
            now=datetime.now(UTC), source=BookingSource.VOICE,
        )
        runner = await build_runner(client=client, settings=settings, ctx=ctx)

        print(f"\n  scenario: {scenario}   model: {settings.llm_model}")
        print(f"  local time: {datetime.now(tz).strftime('%A %d %B %Y, %I:%M %p')}\n")
        print(f"Agent: {business.greeting}")

        for line in SCENARIOS[scenario]:
            print(f"\nYou:   {line}")
            now = datetime.now(UTC)
            await call_log.log_turn(
                db, call_id=call.id, role=TurnRole.CALLER, text=line, turn_started_at=now
            )
            result = await runner.run_turn(line, now=now)
            print(f"Agent: {result.text or '(no text)'}")

            m = result.metrics
            print(
                f"       [first token {m.llm_first_token_ms}ms | total {m.total_ms}ms | "
                f"in {m.input_tokens} out {m.output_tokens} | stop {result.stop_reason}]"
            )
            for tc in result.tool_calls:
                flag = "ERROR " if tc["is_error"] else ""
                first = tc["output"].splitlines()[0] if tc["output"] else ""
                print(f"       -> {flag}{tc['name']}({tc['input']}) {tc['duration_ms']}ms")
                print(f"          {first}")

            await call_log.log_turn(
                db, call_id=call.id, role=TurnRole.AGENT, text=result.text,
                turn_started_at=m.turn_started_at,
                llm_first_token_ms=m.llm_first_token_ms, llm_done_ms=m.llm_done_ms,
                total_ms=m.total_ms, tool_calls=result.tool_calls, model=m.model,
                input_tokens=m.input_tokens, output_tokens=m.output_tokens,
                cache_read_tokens=m.cache_read_tokens,
            )
            await db.commit()
            if result.handoff:
                reason = agent_session.handoff.reason if agent_session.handoff else result.stop_reason
                print(f"\n  [handoff: {reason}]")
                break

        await call_log.end_call(
            db, call, now=datetime.now(UTC),
            status=CallStatus.HANDED_OFF if agent_session.handoff else CallStatus.COMPLETED,
        )
        await db.commit()

        made = (
            await db.execute(select(Booking).where(Booking.call_id == call.id))
        ).unique().scalars().all()
        print(f"\n  bookings created by this call: {len(made)}")
        for row in made:
            local = row.slot_start.astimezone(tz)
            print(
                f"    {row.reference}  {row.customer_name}  "
                f"{local.strftime('%A %d %B %I:%M %p')}  ({row.status.value})"
            )

    await dispose_engine()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", default="booking", choices=sorted(SCENARIOS))
    run_async(run(parser.parse_args().scenario))


if __name__ == "__main__":
    main()
