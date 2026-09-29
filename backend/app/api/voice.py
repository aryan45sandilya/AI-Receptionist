"""The voice WebSocket: microphone in, speech out.

One socket carries a whole call. The browser streams 16 kHz mono PCM up; the
server finds utterances in it, transcribes them, runs the same agent the
terminal demo uses, and streams the spoken reply back as PCM. Control messages
travel as JSON text frames, audio as binary frames.

Where the time goes, which is what this phase exists to measure::

    caller stops speaking
      |  ~800 ms   endpointing: the detector waits to be sure they are done
      v
    stt_final_ms          transcript back from the speech service
      v
    llm_first_token_ms    first word of the reply exists
      v
    tts_first_audio_ms    first audio frame leaves the server
    total_ms              == tts_first_audio_ms: what the caller actually waited

All of them are measured from the moment the caller *stopped talking*, not from
the moment the server noticed -- the endpointing wait is real waiting, and a
number that left it out would flatter us. Every stage lands in ``call_turns``.

Structure worth knowing before reading the code:

*   The socket is read continuously by ``_pump``, and each turn runs as its own
    task. Reading never pauses for the model, so the connection stays live and
    Phase 3 can hear the caller start talking over the agent.

*   Until then the call is half duplex: mic audio is discarded while a turn is
    running. Without echo cancellation the agent would otherwise hear its own
    voice and answer itself.

*   Database work that the caller is not waiting for happens while something
    else holds their attention: the business's services and hours are read while
    the greeting plays, and each turn is written to the database after its reply
    has been sent, while the last of it is still being spoken.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any
from zoneinfo import ZoneInfo

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from openai import AsyncOpenAI
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ..agent.loop import AgentRunner, TurnResult, build_runner
from ..agent.session import AgentSession
from ..agent.tools import ToolContext
from ..config import Settings, get_settings
from ..db import get_session_factory
from ..domain import calls as call_log
from ..enums import BookingSource, CallChannel, CallStatus, TurnRole
from ..models import Business, Call
from ..voice.audio import Pcm
from ..voice.pipeline import TurnSpeaker, loop_time
from ..voice.speech import STT_SAMPLE_RATE, GroqOrpheus, GroqWhisper, SpeechToText, TextToSpeech
from ..voice.vad import Event, VoiceActivityDetector

log = logging.getLogger(__name__)
router = APIRouter()

# The browser demo has no caller ID.
WEB_DEMO_CALLER = "+919876543210"

# A phone call must never go silent, so an empty reply is replaced with
# something the caller can act on rather than spoken as nothing.
EMPTY_REPLY_FALLBACK = "Sorry, I didn't catch that. Could you say it again?"
HANDOFF_FALLBACK = "Let me put you through to a colleague."


def build_clients(settings: Settings) -> tuple[AsyncOpenAI, SpeechToText, TextToSpeech]:
    """The three service clients. A function so tests can substitute fakes."""
    client = AsyncOpenAI(api_key=settings.llm_api_key, base_url=settings.llm_base_url)
    stt = GroqWhisper(client, model=settings.stt_model, language=settings.stt_language)
    tts = GroqOrpheus(client, model=settings.tts_model, voice=settings.tts_voice)
    return client, stt, tts


def _since(anchor: float) -> int:
    return int((loop_time() - anchor) * 1000)


def _shift(value: int | None, offset_ms: int) -> int | None:
    return None if value is None else value + offset_ms


def _is_failure(result: TurnResult) -> bool:
    """The turn ended on our fallback text rather than something the model said."""
    reason = result.stop_reason or ""
    return reason.startswith("error") or reason == "max_iterations"


@dataclass
class CallState:
    """What one call holds: its database session, its agent and its identity.

    Plain values, read once at the start of the call. The greeting is kept as a
    string for the same reason: an ORM attribute read after a rollback has to go
    back to the database, and from async code that fails in a confusing place.
    """

    db: AsyncSession
    call_id: uuid.UUID
    greeting: str
    agent_session: AgentSession
    ctx: ToolContext
    client: AsyncOpenAI
    settings: Settings
    runner: AgentRunner | None = None

    async def ensure_runner(self) -> AgentRunner:
        """The agent, built on first use from the call's snapshot.

        Called while the greeting plays so the caller never waits for it, and
        again at the start of every turn, where it is free unless that first
        attempt failed -- in which case the turn retries it.
        """
        if self.runner is None:
            self.runner = await build_runner(
                client=self.client, settings=self.settings, ctx=self.ctx
            )
        return self.runner


class VoiceCall:
    """One browser call: its socket, its agent and its database rows."""

    def __init__(self, websocket: WebSocket, *, business_id: uuid.UUID | None = None) -> None:
        self.ws = websocket
        # Which business is answering. None picks the oldest one, which is what
        # the single-business browser demo wants; a telephone call will pass the
        # business that owns the number that was dialled.
        self.business_id = business_id
        self.settings = get_settings()
        self.detector = VoiceActivityDetector()
        self._turn: asyncio.Task[None] | None = None
        self._speaker: TurnSpeaker | None = None
        # Two tasks send on one socket (the pump's events, the turn's audio).
        # Serialised so frames can never interleave, whatever the server does.
        self._send_lock = asyncio.Lock()
        self._background: set[asyncio.Task[None]] = set()

    # ------------------------------------------------------------------ socket

    async def send_event(self, kind: str, **fields: Any) -> None:
        async with self._send_lock:
            await self.ws.send_text(json.dumps({"type": kind, **fields}))

    async def send_audio(self, data: bytes) -> None:
        async with self._send_lock:
            await self.ws.send_bytes(data)

    async def _safe_event(self, kind: str, **fields: Any) -> None:
        """An event that may not be deliverable: the caller may already be gone."""
        with contextlib.suppress(Exception):
            await self.send_event(kind, **fields)

    def _fire(self, kind: str, **fields: Any) -> None:
        """Send an event from synchronous code (the model's text callback).

        Held in a set so the task is not garbage collected mid-flight, and
        errors are swallowed because a failed UI update must not fail a turn.
        """
        task = asyncio.ensure_future(self._safe_event(kind, **fields))
        self._background.add(task)
        task.add_done_callback(self._background.discard)

    @property
    def turn_running(self) -> bool:
        return self._turn is not None and not self._turn.done()

    # -------------------------------------------------------------------- call

    async def run(self) -> None:
        if not self.settings.llm_api_key:
            await self.send_event("fatal", message="LLM_API_KEY is not set. See .env")
            return

        client, stt, tts = build_clients(self.settings)

        async with get_session_factory()() as db:
            query = select(Business).order_by(Business.created_at)
            if self.business_id is not None:
                query = query.where(Business.id == self.business_id)
            business = (await db.execute(query)).scalars().first()
            if business is None:
                await self.send_event(
                    "fatal", message="No business seeded. Run: python -m scripts.seed_demo"
                )
                return

            # Everything needed from the row is read now, into plain values.
            business_id, greeting = business.id, business.greeting
            tz = ZoneInfo(business.timezone)
            business_name = business.name

            call = await call_log.start_call(
                db,
                business_id=business_id,
                channel=CallChannel.WEB_DEMO,
                from_phone_e164=WEB_DEMO_CALLER,
            )
            await db.commit()
            call_id = call.id

            agent_session = AgentSession(
                business_id=business_id,
                business_name=business_name,
                tz=tz,
                caller_phone=WEB_DEMO_CALLER,
                call_id=call_id,
            )
            state = CallState(
                db=db,
                call_id=call_id,
                greeting=greeting,
                agent_session=agent_session,
                ctx=ToolContext(
                    db=db,
                    session=agent_session,
                    settings=self.settings,
                    now=datetime.now(UTC),
                    source=BookingSource.VOICE,
                ),
                client=client,
                settings=self.settings,
            )

            await self.send_event(
                "ready",
                call_id=str(call_id),
                business=business_name,
                greeting=greeting,
                sample_rate=STT_SAMPLE_RATE,
                local_time=datetime.now(tz).strftime("%A %d %B %Y, %I:%M %p"),
            )

            # The greeting runs as a turn like any other, so the pump is already
            # reading -- and discarding -- mic audio while it plays. Awaiting it
            # here instead would leave frames queueing in the socket to be
            # replayed at the detector afterwards.
            self._turn = asyncio.create_task(self._greet(state, tts))

            try:
                await self._pump(state, stt, tts)
            except WebSocketDisconnect:
                pass
            except RuntimeError as exc:
                # Starlette raises this for a receive() after the socket closed,
                # which is what happens when we close it ourselves on handoff.
                if "disconnect" not in str(exc).lower():
                    raise
            finally:
                await self._stop_turn()
                with contextlib.suppress(Exception):
                    await db.rollback()

        await self._finish_call(call_id, agent_session)

    async def _finish_call(self, call_id: uuid.UUID, agent_session: AgentSession) -> None:
        """Close the call's record on a fresh session.

        A new session rather than the call's own: the turn may have been
        cancelled halfway through a flush, and a session in that state cannot be
        trusted with one more write.
        """
        async with get_session_factory()() as db:
            call = await db.get(Call, call_id)
            if call is None:
                return
            await call_log.end_call(
                db,
                call,
                now=datetime.now(UTC),
                status=CallStatus.HANDED_OFF if agent_session.handoff else CallStatus.COMPLETED,
                handoff_reason=agent_session.handoff.summary if agent_session.handoff else None,
            )
            await db.commit()

    async def _stop_turn(self) -> None:
        if self._speaker is not None:
            await self._speaker.cancel()
        if self._turn is not None and not self._turn.done():
            self._turn.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await self._turn

    # -------------------------------------------------------------------- pump

    async def _pump(self, state: CallState, stt: SpeechToText, tts: TextToSpeech) -> None:
        """Read the socket until the call ends, starting a turn per utterance."""
        while True:
            message = await self.ws.receive()
            if message.get("type") == "websocket.disconnect":
                return

            text = message.get("text")
            if text is not None:
                with contextlib.suppress(json.JSONDecodeError):
                    if json.loads(text).get("type") == "hangup":
                        return
                continue

            chunk = message.get("bytes")
            if not chunk:
                continue

            if self.turn_running:
                # Half duplex, for now: the agent is thinking or talking, so this
                # is at best silence and at worst the agent's own voice coming
                # back through the caller's speakers. Phase 3 replaces this with
                # barge-in detection.
                continue

            for event, audio in self.detector.feed(chunk):
                if event is Event.SPEECH_STARTED:
                    await self.send_event("speech_started")
                elif event is Event.SPEECH_ENDED:
                    if audio is None:
                        # Long enough to trip the detector, too short to be a word.
                        await self.send_event("speech_ended", used=False)
                        continue
                    await self.send_event("speech_ended", used=True)
                    self._turn = asyncio.create_task(
                        self._handle_utterance(
                            state, stt, tts, audio, self.detector.endpoint_silence_ms
                        )
                    )
                    break

    # -------------------------------------------------------------------- turns

    async def _greet(self, state: CallState, tts: TextToSpeech) -> None:
        speaking: asyncio.Future[None] | None = None
        try:
            speaker = self._new_speaker(tts, anchor=loop_time())
            speaker.start()
            speaker.say(state.greeting)
            speaking = asyncio.ensure_future(speaker.finish())

            # The database work happens while the greeting plays, not before it.
            # The speaker uses no database, so nothing here contends with it.
            setup_error: Exception | None = None
            try:
                await state.ensure_runner()
            except Exception as exc:  # noqa: BLE001 - finish the greeting first, then report
                setup_error = exc
                with contextlib.suppress(Exception):
                    await state.db.rollback()

            await speaking

            # Logged after the audio is out, off the path the caller is waiting on.
            await call_log.log_turn(
                state.db,
                call_id=state.call_id,
                role=TurnRole.AGENT,
                text=speaker.spoken_text or state.greeting,
                turn_started_at=datetime.now(UTC),
                tts_first_audio_ms=speaker.first_audio_ms,
                total_ms=speaker.first_audio_ms,
            )
            await state.db.commit()
            await speaker.drain_playback()
            if setup_error is not None:
                raise setup_error
        except asyncio.CancelledError:
            if speaking is not None:
                speaking.cancel()
            raise
        except Exception as exc:  # noqa: BLE001
            log.exception("greeting failed")
            with contextlib.suppress(Exception):
                await state.db.rollback()
            await self._safe_event("turn_error", message=f"{type(exc).__name__}: {exc}"[:300])
        finally:
            self._speaker = None
            self.detector.reset()
        # Reached only if the turn was not cancelled: the call is still live and
        # the gate has reopened, so say so. A client (the page, or a test) needs
        # an explicit signal -- the server alone knows when it stopped discarding.
        await self._safe_event("listening")

    def _new_speaker(self, tts: TextToSpeech, *, anchor: float) -> TurnSpeaker:
        speaker = TurnSpeaker(
            tts=tts, send_event=self.send_event, send_audio=self.send_audio, anchor=anchor
        )
        self._speaker = speaker
        return speaker

    async def _handle_utterance(
        self,
        state: CallState,
        stt: SpeechToText,
        tts: TextToSpeech,
        audio: bytes,
        endpoint_silence_ms: int,
    ) -> None:
        try:
            await self._run_turn(state, stt, tts, audio, endpoint_silence_ms)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - one bad turn must not end the call
            log.exception("turn failed")
            with contextlib.suppress(Exception):
                await state.db.rollback()
            await self._safe_event("turn_error", message=f"{type(exc).__name__}: {exc}"[:300])
        finally:
            self._speaker = None
            self.detector.reset()
        await self._safe_event("listening")

    async def _run_turn(
        self,
        state: CallState,
        stt: SpeechToText,
        tts: TextToSpeech,
        audio: bytes,
        endpoint_silence_ms: int,
    ) -> None:
        db, agent_session = state.db, state.agent_session

        # The anchor is when the caller stopped talking. The detector needed
        # ``endpoint_silence_ms`` of quiet to be sure, and they waited through
        # all of it, so the clock starts that far in the past.
        anchor = loop_time() - endpoint_silence_ms / 1000
        started_at = datetime.now(UTC) - timedelta(milliseconds=endpoint_silence_ms)

        transcript = await stt.transcribe(Pcm(data=audio, sample_rate=STT_SAMPLE_RATE))
        stt_final_ms = _since(anchor)

        if not transcript:
            # A breath, a cough, a door. Say nothing rather than answer it.
            await self.send_event("transcript", text="", discarded=True)
            return
        await self.send_event("transcript", text=transcript, discarded=False)

        runner = await state.ensure_runner()
        speaker = self._new_speaker(tts, anchor=anchor)
        speaker.start()

        def on_text(chunk: str) -> None:
            speaker.on_text(chunk)
            self._fire("agent_delta", text=chunk)

        # run_turn measures its own stages from the moment it is called. The
        # offset re-bases them onto the shared origin.
        llm_offset_ms = _since(anchor)
        try:
            result = await runner.run_turn(transcript, now=datetime.now(UTC), on_text=on_text)

            if _is_failure(result):
                speaker.say(result.text)
                self._fire("agent_delta", text=" " + result.text)
            elif not result.text.strip():
                speaker.say(HANDOFF_FALLBACK if result.handoff else EMPTY_REPLY_FALLBACK)
            await speaker.finish()
        except asyncio.CancelledError:
            await speaker.cancel()
            raise

        m = result.metrics
        m.stt_final_ms = stt_final_ms
        m.llm_first_token_ms = _shift(m.llm_first_token_ms, llm_offset_ms)
        m.llm_done_ms = _shift(m.llm_done_ms, llm_offset_ms)
        m.tts_first_audio_ms = speaker.first_audio_ms
        # None, not a stand-in, when nothing was ever heard: a turn whose voice
        # failed did not have a 900 ms response time, it had none.
        m.total_ms = speaker.first_audio_ms

        reply = speaker.spoken_text or result.text
        await self.send_event(
            "turn_done",
            stt_final_ms=m.stt_final_ms,
            llm_first_token_ms=m.llm_first_token_ms,
            llm_done_ms=m.llm_done_ms,
            tts_first_audio_ms=m.tts_first_audio_ms,
            total_ms=m.total_ms,
            endpoint_ms=endpoint_silence_ms,
            input_tokens=m.input_tokens,
            output_tokens=m.output_tokens,
            tools=[t["name"] for t in result.tool_calls],
            stop_reason=result.stop_reason,
            reply=reply,
        )

        # Written after the reply is out, while its tail is still playing.
        await call_log.log_turn(
            db,
            call_id=state.call_id,
            role=TurnRole.CALLER,
            text=transcript,
            turn_started_at=started_at,
            stt_final_ms=stt_final_ms,
        )
        await call_log.log_turn(
            db,
            call_id=state.call_id,
            role=TurnRole.AGENT,
            text=reply,
            turn_started_at=started_at,
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
        await db.commit()

        if result.handoff or agent_session.handoff is not None:
            reason = agent_session.handoff.reason if agent_session.handoff else result.stop_reason
            await self.send_event("handoff", reason=reason)
            await speaker.drain_playback()
            # Closing is what ends the call: the pump sees the disconnect.
            with contextlib.suppress(Exception):
                await self.ws.close(code=1000)
            return

        # The gate stays shut until the listener has heard the last word.
        await speaker.drain_playback()


@router.websocket("/ws/voice")
async def voice_socket(websocket: WebSocket) -> None:
    await websocket.accept()
    call = VoiceCall(websocket)
    try:
        await call.run()
    except WebSocketDisconnect:
        pass
    except Exception as exc:  # noqa: BLE001 - tell the page rather than leave it hanging
        log.exception("voice call failed")
        await call._safe_event("fatal", message=f"{type(exc).__name__}: {exc}"[:400])
    finally:
        with contextlib.suppress(Exception):
            await websocket.close()
