"""One call: who is speaking, when to answer, and when to stop talking.

This is the heart of the voice agent, and it knows nothing about wires. A
``CallTransport`` hands it the caller's audio and carries the agent's back; the
browser demo and a Twilio phone call differ only in which transport they plug in.

Where the time goes, which is what this exists to measure::

    caller stops speaking
      |  ~400 ms   the transcription service waits to be sure they are done,
      |            and a network round trip passes before it can say so
      v
    stt_final_ms          the finished transcript is back
      v
    llm_first_token_ms    first word of the reply exists
      v
    tts_first_audio_ms    first audio of the *answer* leaves the server
    total_ms              == tts_first_audio_ms: what the caller waited for
    filler_ms             when the line stopped being silent, if a stopgap
                          ("one moment") was said first while a tool ran

All of them are measured from the moment the caller *stopped talking*, not from
the moment the server noticed -- the wait for the end of the utterance is real
waiting, and a number that left it out would flatter us. The stopgap is timed on
its own so that it cannot make the answer look quicker than it was. Every stage
of every turn lands in ``call_turns``, interrupted turns included.

How the caller's audio is handled
---------------------------------

A voice activity detector watches every frame. It decides what reaches the
transcription service -- only speech does, never silence, room tone or the
agent's own voice -- and it is the clock for barge-in. What it has heard is in
one of four phases::

    idle     nobody is speaking
    open     the caller is speaking and the agent is not: audio streams to the
             transcriber, and the utterance ends when the *service* decides they
             have paused (usually first) or the detector hears enough silence
    held     the caller started speaking while the agent was talking. Not yet an
             interruption: a cough or a "mm-hm" should not cut the agent off, so
             the audio is kept, not sent, until it has lasted ``barge_in_ms``
    ignored  as ``held``, with barge-in switched off (the agent talks through it)

``held`` becomes ``open`` -- and the agent stops mid-word -- once the caller has
kept talking for long enough. Everything after that is an ordinary utterance.

What an interruption does
-------------------------

Speech stops at the next frame (the speaker is muted synchronously), the far end
is told to throw away audio it is still holding, and the agent's turn is
cancelled. The model's tool calls are not: a booking that has begun finishes
(see ``agent/loop.py``). The turn is then logged as interrupted, with the words
the caller actually heard rather than the ones the model had planned.

If the interrupted turn had done nothing the caller could have noticed -- no tool
run, no word heard -- it is erased and the caller's words are carried into the
next utterance. That is what makes a pause in the middle of a sentence cost
nothing: the agent had started to answer half a sentence, the caller carried on,
and the model sees one complete request.

The cut-in can land at three points, and the model's history is repaired
differently at each (``AgentRunner.settle_interruption``): before the model has
been asked; while it is still writing; or -- the usual case, because a reply is
written in under a second and takes several to say -- after it has finished and
is only being spoken, when the whole planned reply is already in history and has
to be cut back to what was heard.

Other structure worth knowing before reading the code:

*   The transport is read continuously by ``_pump`` and each turn runs as its own
    task, so reading never pauses for the model and the agent can be interrupted.

*   Database work the caller is not waiting for happens while something else
    holds their attention: the business's services and hours are read while the
    greeting plays, and each turn is written after its reply has been sent. The
    call has one database session and it is used by one task at a time.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import uuid
from collections import deque
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any
from zoneinfo import ZoneInfo

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
from .pipeline import TurnSpeaker, loop_time
from .services import ConfigurationError, build_clients
from .transport import CallTransport, Control
from .vad import FRAME_MS, Event, VadConfig, VoiceActivityDetector

log = logging.getLogger(__name__)

# The browser demo has no caller ID.
WEB_DEMO_CALLER = "+919876543210"

# A phone call must never go silent, so an empty reply is replaced with
# something the caller can act on rather than spoken as nothing.
EMPTY_REPLY_FALLBACK = "Sorry, I didn't catch that. Could you say it again?"
HANDOFF_FALLBACK = "Let me put you through to a colleague."

# Said while a tool runs, if the model has said nothing. Several, rotated, so a
# long call does not sound like a recording; each short enough to finish before
# the answer is ready. Not for a handoff: the model says goodbye in its own words.
FILLERS = ("One moment.", "Let me check that.", "Sure, one moment.")
NO_FILLER_FOR = frozenset({"handoff_to_human"})

# How much recent audio is kept so an utterance can start with the sound that
# came before the detector was sure it was speech: the detector's own pre-roll
# (160 ms) and the 100 ms it takes to decide, with a chunk of margin.
RING_MS = 400

# Words carried over from a turn that was cut short before anything happened are
# only worth joining to what comes next if it comes soon.
CARRY_TTL_S = 15.0


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
    """One call: its transport, its agent and its database rows."""

    def __init__(
        self,
        transport: CallTransport,
        *,
        business_id: uuid.UUID | None = None,
        dialed_number: str | None = None,
        channel: CallChannel = CallChannel.WEB_DEMO,
        caller: str = WEB_DEMO_CALLER,
        call_sid: str | None = None,
        settings: Settings | None = None,
    ) -> None:
        self.transport = transport
        # Which business is answering: the one named, else the one that owns the
        # number that was dialled, else the oldest -- which is what the
        # single-business browser demo wants.
        self.business_id = business_id
        self.dialed_number = dialed_number
        self.channel = channel
        self.caller = caller
        self.call_sid = call_sid
        self.settings = settings or get_settings()
        self.detector = VoiceActivityDetector(sample_rate=transport.input_rate)

        self._state: CallState | None = None
        self._transcriber: Any = None
        self._tts: Any = None

        # What can be in flight, as stages of one exchange: the caller's last
        # utterance being turned into text, and the agent's turn (model, tools,
        # speech). A second utterance can end while the first is still being
        # transcribed, hence a set.
        self._finishers: set[asyncio.Task[None]] = set()
        self._turn: asyncio.Task[None] | None = None
        self._warmup: asyncio.Task[None] | None = None
        self._speaker: TurnSpeaker | None = None
        self._background: set[asyncio.Task[Any]] = set()

        self._phase = "idle"
        self._held: list[bytes] = []
        self._ring: deque[bytes] = deque()
        self._ring_bytes = 0
        self._ring_limit = transport.input_rate * 2 * RING_MS // 1000

        self._barged_in = False
        self._carry: tuple[str, float] | None = None
        self._fillers_said = 0

    # ------------------------------------------------------------------ events

    async def send_event(self, kind: str, **fields: Any) -> None:
        await self.transport.send_event(kind, **fields)

    async def _safe_event(self, kind: str, **fields: Any) -> None:
        """An event that may not be deliverable: the caller may already be gone."""
        with contextlib.suppress(Exception):
            await self.send_event(kind, **fields)

    def _spawn(self, coro: Any) -> asyncio.Task[Any]:
        """Run something in the background, kept alive and with errors swallowed:
        a failed UI update or a socket that would not open must not fail a turn."""
        async def guarded() -> None:
            with contextlib.suppress(Exception):
                await coro

        task = asyncio.ensure_future(guarded())
        self._background.add(task)
        task.add_done_callback(self._background.discard)
        return task

    def _fire(self, kind: str, **fields: Any) -> None:
        """Send an event from synchronous code (the model's text callback)."""
        self._spawn(self.send_event(kind, **fields))

    async def report_fatal(self, message: str) -> None:
        await self._safe_event("fatal", message=message[:400])

    @property
    def turn_running(self) -> bool:
        return self._turn is not None and not self._turn.done()

    def _agent_active(self) -> bool:
        """Is the agent doing something the caller could interrupt?

        Thinking, or speaking, or still being heard. Not merely "a turn is
        running": for a second or two after the agent has finished speaking its turn
        is still writing its record to the database, and a caller who starts
        talking then is replying to the agent, not interrupting it.
        """
        if not self.turn_running:
            return False
        speaker = self._speaker
        return speaker is None or speaker.audible

    # -------------------------------------------------------------------- call

    async def run(self) -> None:
        settings = self.settings
        if not settings.llm_api_key:
            await self.report_fatal("LLM_API_KEY is not set. See .env")
            return

        try:
            client, transcriber, tts = build_clients(
                settings,
                input_rate=self.transport.input_rate,
                output_rate=self.transport.output_rate,
            )
        except ConfigurationError as exc:
            await self.report_fatal(str(exc))
            return
        self._transcriber, self._tts = transcriber, tts
        transcriber.on_interim = self._on_interim
        transcriber.on_endpoint = self._on_service_endpoint
        self.detector = self._make_detector(transcriber)

        # The voice's socket takes over a second to open. Start it now, behind
        # the database work, so the greeting does not pay for it.
        self._spawn(tts.start())

        try:
            async with get_session_factory()() as db:
                business = await self._find_business(db)
                if business is None:
                    await self.report_fatal("No business seeded. Run: python -m scripts.seed_demo")
                    return

                # Everything needed from the row is read now, into plain values.
                business_id, greeting = business.id, business.greeting
                tz = ZoneInfo(business.timezone)
                business_name = business.name

                call = await call_log.start_call(
                    db,
                    business_id=business_id,
                    channel=self.channel,
                    from_phone_e164=self.caller,
                    twilio_call_sid=self.call_sid,
                )
                await db.commit()
                call_id = call.id

                agent_session = AgentSession(
                    business_id=business_id,
                    business_name=business_name,
                    tz=tz,
                    caller_phone=self.caller,
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
                        settings=settings,
                        now=datetime.now(UTC),
                        source=BookingSource.VOICE,
                    ),
                    client=client,
                    settings=settings,
                )
                self._state = state

                await self.send_event(
                    "ready",
                    call_id=str(call_id),
                    business=business_name,
                    greeting=greeting,
                    sample_rate=self.transport.input_rate,
                    local_time=datetime.now(tz).strftime("%A %d %B %Y, %I:%M %p"),
                )

                # The greeting runs as a turn like any other, so the caller can
                # talk over it, and the pump is already reading while it plays.
                self._warmup = asyncio.create_task(self._warm_up(state))
                self._start_turn(state, self._greet(state))

                try:
                    await self._pump()
                finally:
                    await self._shutdown()
                    with contextlib.suppress(Exception):
                        await db.rollback()

            await self._finish_call(call_id, agent_session)
        finally:
            await self._close_services()

    def _make_detector(self, transcriber: Any) -> VoiceActivityDetector:
        """A detector whose silence timer suits what is transcribing.

        With a batch service it is the clock that ends every utterance. With one
        that ends utterances itself it is only a backstop, so it waits long enough
        that the service -- which needs less silence, but takes a round trip to
        say so -- is normally first.
        """
        settings = self.settings
        silence_ms = (
            settings.vad_backstop_ms
            if getattr(transcriber, "endpoints_itself", False)
            else settings.vad_end_silence_ms
        )
        return VoiceActivityDetector(
            config=VadConfig(end_frames=max(1, silence_ms // FRAME_MS)),
            sample_rate=self.transport.input_rate,
        )

    async def _find_business(self, db: AsyncSession) -> Business | None:
        query = select(Business).order_by(Business.created_at)
        if self.business_id is not None:
            return (await db.execute(query.where(Business.id == self.business_id))).scalars().first()
        if self.dialed_number:
            owner = (
                await db.execute(query.where(Business.phone_e164 == self.dialed_number))
            ).scalars().first()
            if owner is not None:
                return owner
            log.warning("no business owns %s; answering as the first one", self.dialed_number)
        return (await db.execute(query)).scalars().first()

    async def _warm_up(self, state: CallState) -> None:
        """Everything the first turn needs that the caller should not wait for.

        Reads the business's services, staff and hours (the only database work
        before the first turn) and, now that the names are known, opens the
        transcription socket with them as words to favour: "Dr Rao" is not in
        a general vocabulary. The socket takes about a second to open, which the
        greeting hides.
        """
        terms: list[str] = []
        try:
            await state.ensure_runner()
            snapshot = await state.ctx.get_snapshot()
            terms = [
                snapshot.business.name,
                *(service.name for service in snapshot.services),
                *snapshot.staff_names().values(),
            ]
        except Exception:  # noqa: BLE001 - the first turn retries, and reports
            log.exception("warm-up failed")
            with contextlib.suppress(Exception):
                await state.db.rollback()
        await self._transcriber.start(keyterms=terms)

    async def _wait_warm_up(self) -> None:
        if self._warmup is not None:
            # Shielded: a barge-in cancels the turn that is waiting, never the
            # database read it is waiting for.
            await asyncio.shield(self._warmup)

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

    async def _shutdown(self) -> None:
        """The call is over: stop whatever is still going, in a safe order."""
        for task in (*self._finishers, self._warmup):
            if task is not None and not task.done():
                task.cancel()
        if self._speaker is not None:
            await self._speaker.cancel()
        if self._turn is not None and not self._turn.done():
            self._turn.cancel()
        pending = [t for t in (*self._finishers, self._turn, self._warmup) if t is not None]
        await asyncio.gather(*pending, return_exceptions=True)
        for task in list(self._background):
            task.cancel()

    async def _close_services(self) -> None:
        for service in (self._transcriber, self._tts):
            if service is not None:
                with contextlib.suppress(Exception):
                    await service.close()

    # -------------------------------------------------------------------- pump

    async def _pump(self) -> None:
        """Read the transport until the call ends."""
        while True:
            incoming = await self.transport.receive()
            if incoming is None:
                return
            if isinstance(incoming, Control):
                if incoming.kind == "hangup":
                    return
                if incoming.kind == "config":
                    self._apply_config(incoming.data)
                continue
            await self._on_audio(incoming)

    def _apply_config(self, data: dict[str, Any]) -> None:
        """Settings a caller may change for their own call.

        Only barge-in, and only as a boolean: the page's "allow me to interrupt"
        box. Someone using speakers hears the agent's voice come back through
        their microphone and needs a way to stop it interrupting itself that
        does not involve editing a file and restarting the server. Nothing else is
        read from the message, so it cannot be used to change anything that costs
        money or touches data.
        """
        allowed = data.get("barge_in")
        if isinstance(allowed, bool):
            self.settings = self.settings.model_copy(update={"barge_in": allowed})

    def _remember(self, chunk: bytes) -> None:
        self._ring.append(chunk)
        self._ring_bytes += len(chunk)
        while self._ring_bytes > self._ring_limit and len(self._ring) > 1:
            self._ring_bytes -= len(self._ring.popleft())

    async def _on_audio(self, chunk: bytes) -> None:
        self._remember(chunk)
        # Judged by where things stood before this chunk. The chunk that *starts*
        # an utterance is already in the pre-roll that opens it, and must not be
        # sent or kept a second time.
        before = self._phase

        for event, audio in self.detector.feed(chunk):
            if event is Event.SPEECH_STARTED:
                await self._speech_started()
            elif event is Event.SPEECH_ENDED:
                self._speech_ended(audio)

        if before == "open" and self._phase == "open":
            # A continuing utterance, including the chunk that held the silence
            # the service needs in order to hear that it has ended.
            await self._transcriber.send(chunk)
        elif before == "held" and self._phase == "held":
            self._held.append(chunk)

        if self._phase == "held":
            if not self._agent_active():
                # The agent finished while this was being held: nothing is left to
                # interrupt, so it is simply the start of the caller's turn.
                held, self._held = b"".join(self._held), []
                await self._open_utterance(held)
            elif self.detector.voiced_ms >= self.settings.barge_in_ms:
                await self._barge_in()

    async def _speech_started(self) -> None:
        snapshot = b"".join(self._ring)
        if not self._agent_active():
            await self._open_utterance(snapshot)
        elif self.settings.barge_in:
            self._phase = "held"
            self._held = [snapshot]
        else:
            # Half duplex: the agent talks through whatever this is. It is most
            # likely its own voice coming back through the caller's speakers.
            self._phase = "ignored"

    def _speech_ended(self, audio: bytes | None) -> None:
        phase = self._phase
        if phase == "open":
            if audio is None:
                # Long enough to trip the detector, too short to be a word.
                self._phase = "idle"
                self._transcriber.abandon()
                self._fire("speech_ended", used=False)
            else:
                self._close_utterance(audio, self.detector.endpoint_silence_ms)
        elif phase in {"held", "ignored"}:
            # It never lasted long enough to be an interruption.
            self._phase = "idle"
            self._held = []

    # --------------------------------------------------------------- utterances

    async def _open_utterance(self, initial: bytes) -> None:
        """The caller has started speaking and nothing is in the way."""
        self._phase = "open"
        self._held = []
        self._transcriber.begin()
        await self._transcriber.send(initial)
        await self._safe_event("speech_started")

    def _on_interim(self, text: str) -> None:
        if self._phase == "open":
            self._fire("interim", text=text)

    def _on_service_endpoint(self) -> None:
        """The transcription service has decided the caller finished.

        Usually before the detector would have: the service waits for a shorter
        pause. Closing here is what makes streaming transcription faster, and the
        detector is reset so the same utterance is not ended a second time by its
        own, slower, timer.
        """
        if self._phase != "open":
            return
        silence_ms = self.detector.quiet_ms
        self.detector.reset()
        self._close_utterance(None, silence_ms)

    def _close_utterance(self, audio: bytes | None, silence_ms: int) -> None:
        if self._phase != "open":
            return
        self._phase = "idle"

        # The caller stopped talking ``silence_ms`` ago. They waited through all
        # of it, so the clock starts that far in the past.
        anchor = loop_time() - silence_ms / 1000
        started_at = datetime.now(UTC) - timedelta(milliseconds=silence_ms)
        self._fire("speech_ended", used=True)
        task = asyncio.ensure_future(self._finish_utterance(audio, anchor, started_at, silence_ms))
        self._finishers.add(task)
        task.add_done_callback(self._finishers.discard)

    async def _finish_utterance(
        self, audio: bytes | None, anchor: float, started_at: datetime, endpoint_ms: int
    ) -> None:
        state = self._state
        assert state is not None
        try:
            text = await self._transcriber.finish(audio)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - one bad utterance must not end the call
            log.exception("transcription failed")
            await self._safe_event("turn_error", message=f"{type(exc).__name__}: {exc}"[:300])
            text = ""
        stt_final_ms = _since(anchor)

        # An interrupted turn may still be unwinding (a tool finishing its write,
        # a record being written). What it leaves behind -- words to carry over --
        # is only there once it has finished, so wait for it before looking.
        await self._settle()
        text = self._with_carry(text)
        if not text:
            # A breath, a cough, a door. Say nothing rather than answer it.
            await self.send_event("transcript", text="", discarded=True)
            if not self.turn_running and self._phase != "open":
                await self._safe_event("listening")
            return

        if self._phase == "open":
            # The caller started again while this was being transcribed: they had
            # not finished. Keep the words for the rest of the sentence.
            self._carry = (text, loop_time())
            return

        await self.send_event("transcript", text=text, discarded=False)
        self._start_turn(
            state,
            self._run_turn(
                state, text, anchor=anchor, started_at=started_at,
                stt_final_ms=stt_final_ms, endpoint_ms=endpoint_ms,
            ),
        )

    def _with_carry(self, text: str) -> str:
        """Join words carried over from a turn that was cut short, if recent."""
        carried, self._carry = self._carry, None
        if carried is None:
            return text
        words, when = carried
        if loop_time() - when > CARRY_TTL_S:
            return text
        return f"{words} {text}".strip() if text else words

    # --------------------------------------------------------------- barge-in

    async def _barge_in(self) -> None:
        """The caller has talked over the agent for long enough to count."""
        # In this order, and the first two without waiting for anything: the
        # voice stops, then the far end is told to drop what it is holding. The
        # caller began talking some time ago; every millisecond of the agent's
        # voice after this point is the thing being fixed.
        speaker = self._speaker
        if speaker is not None:
            speaker.mute()
        await self.transport.clear_audio()

        self._barged_in = True
        if self._turn is not None and not self._turn.done():
            self._turn.cancel()

        held, self._held = b"".join(self._held), []
        await self._open_utterance(held)

    async def _settle(self) -> None:
        """Wait for the agent's turn to finish unwinding, if it is."""
        turn = self._turn
        if turn is not None and turn is not asyncio.current_task() and not turn.done():
            await asyncio.gather(turn, return_exceptions=True)

    # ------------------------------------------------------------------ turns

    def _start_turn(self, state: CallState, work: Any) -> None:
        self._barged_in = False
        self._turn = asyncio.create_task(self._guard(state, work))

    async def _guard(self, state: CallState, work: Any) -> None:
        """Run a turn; one that fails must not end the call."""
        try:
            await work
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001
            log.exception("turn failed")
            with contextlib.suppress(Exception):
                await state.db.rollback()
            await self._safe_event("turn_error", message=f"{type(exc).__name__}: {exc}"[:300])
        finally:
            self._speaker = None
        await self._turn_finished()

    async def _turn_finished(self) -> None:
        """Reached when a turn ran to its end -- not when it was cancelled."""
        if self._phase == "ignored":
            # Half duplex: what the detector heard was the agent's own voice.
            self.detector.reset()
            self._phase = "idle"
        if self._phase != "open":
            # (If the caller is already speaking again they have the floor; telling
            # the page it is listening would be telling it about the wrong moment.)
            await self._safe_event("listening")
        if self._phase == "held":
            held, self._held = b"".join(self._held), []
            await self._open_utterance(held)

    def _new_speaker(self, *, anchor: float) -> TurnSpeaker:
        speaker = TurnSpeaker(
            tts=self._tts,
            send_event=self.send_event,
            send_audio=self.transport.send_audio,
            anchor=anchor,
            output_rate=self.transport.output_rate,
        )
        self._speaker = speaker
        return speaker

    async def _greet(self, state: CallState) -> None:
        speaker = self._new_speaker(anchor=loop_time())
        started_at = datetime.now(UTC)
        logged = False
        try:
            speaker.start()
            speaker.say(state.greeting)
            await speaker.finish()

            # The speaker uses no database; this is where the warm-up's finishes.
            await self._wait_warm_up()
            await call_log.log_turn(
                state.db,
                call_id=state.call_id,
                role=TurnRole.AGENT,
                text=speaker.spoken_text or state.greeting,
                turn_started_at=started_at,
                tts_first_audio_ms=speaker.first_audio_ms,
                total_ms=speaker.first_audio_ms,
            )
            await state.db.commit()
            logged = True
            await speaker.drain_playback()
        except asyncio.CancelledError:
            await speaker.cancel()
            if self._barged_in and not logged:
                await self._log_interrupted_greeting(state, speaker, started_at)
            raise

    async def _log_interrupted_greeting(
        self, state: CallState, speaker: TurnSpeaker, started_at: datetime
    ) -> None:
        try:
            await self._wait_warm_up()
            await call_log.log_turn(
                state.db,
                call_id=state.call_id,
                role=TurnRole.AGENT,
                text=speaker.heard_text() or None,
                turn_started_at=started_at,
                tts_first_audio_ms=speaker.first_audio_ms,
                total_ms=speaker.first_audio_ms,
                barge_in=True,
            )
            await state.db.commit()
        except Exception:  # noqa: BLE001
            log.exception("could not log the interrupted greeting")
            with contextlib.suppress(Exception):
                await state.db.rollback()

    async def _run_turn(
        self,
        state: CallState,
        transcript: str,
        *,
        anchor: float,
        started_at: datetime,
        stt_final_ms: int,
        endpoint_ms: int,
    ) -> None:
        agent_session = state.agent_session

        await self._wait_warm_up()
        runner = await state.ensure_runner()
        runner.interruption = None
        speaker = self._new_speaker(anchor=anchor)
        speaker.start()

        def on_text(chunk: str) -> None:
            speaker.on_text(chunk)
            self._fire("agent_delta", text=chunk)

        def on_tool_call(name: str) -> None:
            """The model has asked for a tool and will write nothing until it has
            the result. Fill the silence, unless it already has."""
            if not self.settings.tool_filler or name in NO_FILLER_FOR:
                return
            phrase = FILLERS[self._fillers_said % len(FILLERS)]
            if speaker.say_filler(phrase):
                self._fillers_said += 1
                self._fire("agent_delta", text=phrase + " ")

        # run_turn measures its own stages from the moment it is called. The
        # offset re-bases them onto the shared origin.
        llm_offset_ms = _since(anchor)
        # Where an interruption would find this turn, which decides how it is
        # settled: not yet asked of the model; the model still working; the
        # model finished and the reply being spoken; reported and logged.
        asked = announced = logged = False
        result: TurnResult | None = None
        try:
            asked = True
            result = await runner.run_turn(
                transcript, now=datetime.now(UTC), on_text=on_text, on_tool_call=on_tool_call
            )

            if _is_failure(result):
                speaker.say(result.text)
                self._fire("agent_delta", text=" " + result.text)
            elif not result.text.strip():
                speaker.say(HANDOFF_FALLBACK if result.handoff else EMPTY_REPLY_FALLBACK)
            # The long wait, and where a caller most often cuts in: the reply was
            # written in well under a second and takes several to say.
            await speaker.finish()

            m = result.metrics
            self._rebase(m, stt_final_ms, llm_offset_ms, speaker)
            reply = speaker.spoken_text or result.text
            announced = True
            await self._announce(
                m, result.tool_calls, result.stop_reason, reply, endpoint_ms, interrupted=False
            )

            # Written after the reply is out, while its tail is still playing. From
            # here nothing waits on the caller: an interruption now finds a reply
            # that has been spoken, and the write is allowed to finish.
            write = asyncio.ensure_future(
                self._log_exchange(
                    state, transcript, reply, started_at,
                    metrics=m, tool_calls=result.tool_calls, barge_in=False,
                )
            )
            try:
                await asyncio.shield(write)
            except asyncio.CancelledError:
                with contextlib.suppress(Exception):
                    await write
                logged = True
                raise
            logged = True

            if result.handoff or agent_session.handoff is not None:
                reason = agent_session.handoff.reason if agent_session.handoff else result.stop_reason
                await self._hand_off(reason, speaker)
                return

            # Whatever waits on the end of the agent's speech waits for the
            # listener to have heard the last word.
            await speaker.drain_playback()
        except asyncio.CancelledError:
            await speaker.cancel()
            if self._barged_in and not logged:
                await self._record_interruption(
                    state, runner, speaker, transcript,
                    result=result, asked=asked, announced=announced,
                    started_at=started_at, stt_final_ms=stt_final_ms,
                    llm_offset_ms=llm_offset_ms, endpoint_ms=endpoint_ms,
                )
            raise

    @staticmethod
    def _rebase(m: Any, stt_final_ms: int, llm_offset_ms: int, speaker: TurnSpeaker) -> None:
        """Put a turn's stage times on the call's clock, once.

        The model layer times itself from when it was asked; the call times
        everything from when the caller stopped talking. The answer's audio time
        is None, not a stand-in, when it was never heard: a turn whose voice failed
        did not have a 900 ms response time, it had none. A stopgap said while a
        tool ran is timed apart from it, so it cannot make the answer look quicker.
        """
        if m.rebased:
            return
        m.rebased = True
        m.stt_final_ms = stt_final_ms
        m.llm_first_token_ms = _shift(m.llm_first_token_ms, llm_offset_ms)
        m.llm_done_ms = _shift(m.llm_done_ms, llm_offset_ms)
        m.tts_first_audio_ms = speaker.answer_audio_ms
        m.total_ms = speaker.answer_audio_ms
        m.filler_ms = speaker.filler_audio_ms

    async def _announce(
        self,
        m: Any,
        tool_calls: list[dict[str, Any]],
        stop_reason: str | None,
        reply: str,
        endpoint_ms: int,
        *,
        interrupted: bool,
    ) -> None:
        await self._safe_event(
            "turn_done",
            stt_final_ms=m.stt_final_ms,
            llm_first_token_ms=m.llm_first_token_ms,
            llm_done_ms=m.llm_done_ms,
            tts_first_audio_ms=m.tts_first_audio_ms,
            total_ms=m.total_ms,
            filler_ms=m.filler_ms,
            endpoint_ms=endpoint_ms,
            input_tokens=m.input_tokens,
            output_tokens=m.output_tokens,
            tools=[t["name"] for t in tool_calls],
            stop_reason=stop_reason,
            reply=reply,
            interrupted=interrupted,
        )

    async def _record_interruption(
        self,
        state: CallState,
        runner: AgentRunner,
        speaker: TurnSpeaker,
        transcript: str,
        *,
        result: TurnResult | None,
        asked: bool,
        announced: bool,
        started_at: datetime,
        stt_final_ms: int,
        llm_offset_ms: int,
        endpoint_ms: int,
    ) -> None:
        """A turn the caller cut short: settle the history, and log what happened.

        The turn may be in any of three states, and the model's history differs in
        each (see ``AgentRunner.settle_interruption``): never asked of the model,
        so there is nothing to undo; cancelled while the model worked, so there
        is a record of the partial turn; or finished, so ``result`` says what it
        did and its whole planned reply is in history, to be cut back to what
        was heard.
        """
        interruption = runner.interruption if result is None else None
        if not asked or (result is None and interruption is None):
            # Cut off before the model was even asked: the words are still the
            # caller's to finish.
            self._carry = (transcript, loop_time())
            return

        heard = speaker.heard_text()
        # The model is told only what it wrote that got out, not the stopgap.
        if runner.settle_interruption(speaker.heard_text(fillers=False)):
            # Nothing the caller could have noticed had happened. Their words go
            # back to them, to be joined to what they say next.
            self._carry = (transcript, loop_time())
            return

        turn = result if result is not None else interruption
        m = turn.metrics
        self._rebase(m, stt_final_ms, llm_offset_ms, speaker)
        if not announced:
            await self._announce(
                m, turn.tool_calls, "interrupted", heard, endpoint_ms, interrupted=True
            )
        try:
            await self._log_exchange(
                state, transcript, heard, started_at,
                metrics=m, tool_calls=turn.tool_calls, barge_in=True,
            )
        except Exception:  # noqa: BLE001 - the call goes on; the record is what suffers
            log.exception("could not log the interrupted turn")
            with contextlib.suppress(Exception):
                await state.db.rollback()
            return

        if state.agent_session.handoff is not None:
            # The handoff had already been decided when the caller cut in.
            await self._hand_off(state.agent_session.handoff.reason, None)

    async def _log_exchange(
        self,
        state: CallState,
        transcript: str,
        reply: str,
        started_at: datetime,
        *,
        metrics: Any,
        tool_calls: list[dict[str, Any]],
        barge_in: bool,
    ) -> None:
        db = state.db
        await call_log.log_turn(
            db,
            call_id=state.call_id,
            role=TurnRole.CALLER,
            text=transcript,
            turn_started_at=started_at,
            stt_final_ms=metrics.stt_final_ms,
        )
        await call_log.log_turn(
            db,
            call_id=state.call_id,
            role=TurnRole.AGENT,
            text=reply,
            turn_started_at=started_at,
            stt_final_ms=metrics.stt_final_ms,
            llm_first_token_ms=metrics.llm_first_token_ms,
            llm_done_ms=metrics.llm_done_ms,
            tts_first_audio_ms=metrics.tts_first_audio_ms,
            total_ms=metrics.total_ms,
            filler_ms=metrics.filler_ms,
            barge_in=barge_in,
            tool_calls=tool_calls,
            model=metrics.model,
            input_tokens=metrics.input_tokens,
            output_tokens=metrics.output_tokens,
            cache_read_tokens=metrics.cache_read_tokens,
        )
        await db.commit()

    async def _hand_off(self, reason: str | None, speaker: TurnSpeaker | None) -> None:
        """The agent has given the call up: say so, let the last words finish,
        then put the caller through to a person if the line can, and otherwise end
        it. Either way the pump sees the far end go away, which ends the call."""
        await self.send_event("handoff", reason=reason)
        if speaker is not None:
            await speaker.drain_playback()
        number = await self._handoff_number()
        if number and await self.transport.transfer(number):
            return
        await self.transport.close()

    async def _handoff_number(self) -> str | None:
        """The business's own number for a person, if it has one."""
        state = self._state
        if state is None:
            return None
        try:
            snapshot = await state.ctx.get_snapshot()
        except Exception:  # noqa: BLE001 - no number is a reason to hang up, not to fail
            return None
        return snapshot.business.handoff_phone_e164
