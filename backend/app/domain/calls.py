"""Call and turn records, including the latency log.

Every transport (terminal now; browser, phone and WhatsApp later) writes through
these three functions, so the Phase 6 latency report reads one table with one
shape regardless of where the conversation happened.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from ..enums import CallChannel, CallStatus, TurnRole
from ..models import Call, CallTurn


async def start_call(
    session: AsyncSession,
    *,
    business_id: uuid.UUID,
    channel: CallChannel,
    from_phone_e164: str | None = None,
    twilio_call_sid: str | None = None,
) -> Call:
    call = Call(
        business_id=business_id,
        channel=channel,
        from_phone_e164=from_phone_e164,
        twilio_call_sid=twilio_call_sid,
        status=CallStatus.IN_PROGRESS,
    )
    session.add(call)
    await session.flush()
    return call


async def next_turn_index(session: AsyncSession, call_id: uuid.UUID) -> int:
    result = await session.execute(
        select(func.coalesce(func.max(CallTurn.turn_index), -1)).where(
            CallTurn.call_id == call_id
        )
    )
    return int(result.scalar_one()) + 1


async def log_turn(
    session: AsyncSession,
    *,
    call_id: uuid.UUID,
    role: TurnRole,
    text: str | None,
    turn_started_at: datetime,
    stt_final_ms: int | None = None,
    llm_first_token_ms: int | None = None,
    llm_done_ms: int | None = None,
    tts_first_audio_ms: int | None = None,
    total_ms: int | None = None,
    filler_ms: int | None = None,
    barge_in: bool = False,
    tool_calls: list[dict[str, Any]] | None = None,
    model: str | None = None,
    input_tokens: int | None = None,
    output_tokens: int | None = None,
    cache_read_tokens: int | None = None,
) -> CallTurn:
    """Append one turn.

    Called even when a turn failed halfway: a row with llm_first_token_ms set and
    tts_first_audio_ms NULL is how a TTS failure is recorded, and dropping the
    row would erase the only evidence.
    """
    turn = CallTurn(
        call_id=call_id,
        turn_index=await next_turn_index(session, call_id),
        role=role,
        text=text,
        turn_started_at=turn_started_at,
        stt_final_ms=stt_final_ms,
        llm_first_token_ms=llm_first_token_ms,
        llm_done_ms=llm_done_ms,
        tts_first_audio_ms=tts_first_audio_ms,
        total_ms=total_ms,
        filler_ms=filler_ms,
        barge_in=barge_in,
        # A JSONB column, so the list is wrapped in an object -- top-level arrays
        # are awkward to query with the -> operators.
        tool_calls={"calls": tool_calls} if tool_calls else None,
        model=model,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cache_read_tokens=cache_read_tokens,
    )
    session.add(turn)
    await session.flush()
    return turn


async def end_call(
    session: AsyncSession,
    call: Call,
    *,
    now: datetime,
    status: CallStatus,
    handoff_reason: str | None = None,
    transcript_summary: str | None = None,
) -> Call:
    call.status = status
    call.ended_at = now
    call.handoff_reason = handoff_reason
    call.transcript_summary = transcript_summary
    started = call.started_at
    if started is not None:
        call.duration_ms = int((now - started).total_seconds() * 1000)
    await session.flush()
    return call
