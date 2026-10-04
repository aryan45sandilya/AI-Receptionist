"""Doubles and helpers for tests that run a whole call.

Real: the call controller, the voice activity detector, the turn timing, the
agent loop, the tools, the per-call snapshot and the Postgres database.
Faked: the transcriber, the voice and the language model, each replaced by a
scripted double -- which keeps the run free and deterministic while still
exercising every line between "audio arrives" and "rows are written".
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import json
import uuid

import httpx2 as httpx
from fastapi import WebSocketDisconnect
from sqlalchemy import select

from backend.app.enums import CallChannel
from backend.app.models import Call, CallTurn
from backend.app.telephony.twilio import TwilioRest
from backend.app.voice import call as call_module
from backend.app.voice.audio import Pcm
from backend.app.voice.call import VoiceCall
from backend.app.voice.codec import ulaw_encode
from backend.app.voice.speech import BatchTranscriber, ClipSynthesizer
from backend.app.voice.transport import BrowserTransport
from backend.app.voice.twilio_transport import TwilioTransport
from backend.app.voice.vad import frame_rms
from tests.audio_helpers import silence, tone
from tests.fakes import finish, text, tool, usage

CHUNK_BYTES = 1280  # 40 ms of 16 kHz PCM16, the browser's chunk size

CHECK_ARGS = '{"service": "Dental cleaning", "when": "next tuesday evening", "staff_name": null}'
OFFER = [
    [tool(0, id="c1", name="check_availability", args=CHECK_ARGS), finish("tool_calls")],
    [
        text("We have Tuesday at five o'clock. "),
        text("Which would suit you best?"),
        finish("stop"),
        usage(1234, 56),
    ],
]


HANDOFF = [
    [
        tool(
            0,
            id="h1",
            name="handoff_to_human",
            args='{"reason": "caller_asked_for_human", "summary": "Wants to speak to a person."}',
        ),
        finish("tool_calls"),
    ],
    [text("Of course, I am putting you through now."), finish("stop")],
]


# ---------------------------------------------------------------------------
# Doubles
# ---------------------------------------------------------------------------

class FakeSocket:
    """Stands in for the WebSocket, and records everything the server sends."""

    def __init__(self) -> None:
        self._incoming: asyncio.Queue = asyncio.Queue()
        self.events: list[dict] = []
        self.audio: list[bytes] = []
        # Everything sent, in the order it was sent: "audio", or "event:<kind>".
        # Events and audio are kept apart above; this is for asking what came
        # between two events.
        self.timeline: list[str] = []
        self.closed = False

    # -- what the server calls
    async def receive(self) -> dict:
        return await self._incoming.get()

    async def send_text(self, data: str) -> None:
        event = json.loads(data)
        self.events.append(event)
        self.timeline.append(f"event:{event['type']}")

    async def send_bytes(self, data: bytes) -> None:
        self.audio.append(data)
        self.timeline.append("audio")

    async def close(self, code: int = 1000) -> None:
        self.closed = True
        self._incoming.put_nowait({"type": "websocket.disconnect"})

    # -- what the test does
    def feed(self, pcm: bytes) -> None:
        for offset in range(0, len(pcm), CHUNK_BYTES):
            self._incoming.put_nowait(
                {"type": "websocket.receive", "bytes": pcm[offset : offset + CHUNK_BYTES]}
            )

    def hang_up(self) -> None:
        self._incoming.put_nowait(
            {"type": "websocket.receive", "text": json.dumps({"type": "hangup"})}
        )

    def kinds(self) -> list[str]:
        return [event["type"] for event in self.events]

    def of(self, kind: str) -> list[dict]:
        return [event for event in self.events if event["type"] == kind]

    async def wait_for(self, kind: str, count: int = 1, timeout: float = 120.0) -> None:
        async def poll() -> None:
            while len(self.of(kind)) < count:
                await asyncio.sleep(0.02)

        try:
            await asyncio.wait_for(poll(), timeout)
        except TimeoutError:
            raise AssertionError(
                f"timed out waiting for {count}x {kind!r}; saw {self.kinds()}"
            ) from None

    async def wait_until(self, predicate, what: str, timeout: float = 30.0) -> None:
        async def poll() -> None:
            while not predicate():
                await asyncio.sleep(0.02)

        try:
            await asyncio.wait_for(poll(), timeout)
        except TimeoutError:
            raise AssertionError(f"timed out waiting for {what}; saw {self.kinds()}") from None


class FakeSTT:
    """A batch transcription service that answers with scripted text."""

    def __init__(self, *transcripts: str) -> None:
        self._transcripts = list(transcripts)
        self.calls = 0

    async def transcribe(self, pcm: Pcm) -> str:
        self.calls += 1
        return self._transcripts.pop(0) if self._transcripts else ""


class FakeStreamingSTT:
    """A transcriber that, like Deepgram, decides for itself that the caller has
    finished: it listens to the audio it is sent and calls the end of the
    utterance after ``endpointing_ms`` of quiet that follows some speech.
    """

    def __init__(self, *transcripts: str, endpointing_ms: int = 500) -> None:
        self.on_interim = None
        self.on_endpoint = None
        self.endpoints_itself = True
        self._transcripts = list(transcripts)
        self._endpointing_ms = endpointing_ms
        self.keyterms: list[str] = []
        self.begun = 0
        self.finished = 0
        self.abandoned = 0
        self.finish_audio: list[bytes | None] = []
        self.sent = bytearray()
        self._quiet_ms = 0
        self._heard_speech = False
        self._interim_sent = False
        self._ended = False

    async def start(self, *, keyterms=()) -> None:
        self.keyterms = list(keyterms)

    def begin(self) -> None:
        self.begun += 1
        self._quiet_ms = 0
        self._heard_speech = False
        self._interim_sent = False
        self._ended = False

    async def send(self, pcm: bytes) -> None:
        self.sent.extend(pcm)
        chunk_ms = len(pcm) // 32  # 16 kHz, 16-bit
        if frame_rms(pcm) > 0.02:
            self._heard_speech = True
            self._quiet_ms = 0
            if not self._interim_sent and self.on_interim:
                self._interim_sent = True
                self.on_interim("I need a")
        else:
            self._quiet_ms += chunk_ms
        if (
            self._heard_speech
            and not self._ended
            and self._quiet_ms >= self._endpointing_ms
            and self.on_endpoint
        ):
            self._ended = True
            self.on_endpoint()

    async def finish(self, utterance=None) -> str:
        self.finished += 1
        self.finish_audio.append(utterance)
        return self._transcripts.pop(0) if self._transcripts else ""

    def abandon(self) -> None:
        self.abandoned += 1

    async def close(self) -> None: ...


class FakeTTS(ClipSynthesizer):
    """A voice whose every sentence is ``seconds`` of audio (100 ms by default)."""

    def __init__(self, *, fail: bool = False, delay: float = 0.0, seconds: float = 0.1) -> None:
        self.fail = fail
        self.delay = delay
        self.seconds = seconds
        self.spoken: list[str] = []

    async def synthesize(self, text: str) -> Pcm:
        self.spoken.append(text)
        if self.delay:
            await asyncio.sleep(self.delay)
        if self.fail:
            raise RuntimeError("the model requires terms acceptance")
        return Pcm(data=b"\x01\x00" * int(24_000 * self.seconds), sample_rate=24_000)


@contextlib.asynccontextmanager
async def running_call(
    monkeypatch, seeded, *, client, stt, tts, hang_up: bool = True, **settings
):
    """Run a call against the seeded business; yield its socket.

    ``settings`` overrides configuration for this call (``barge_in=False``, say).
    A scripted batch service is wrapped in the batch transcriber here, so tests
    can script transcripts as plain strings; a streaming double is used as is.
    """
    transcriber = stt if hasattr(stt, "begin") else BatchTranscriber(stt)
    monkeypatch.setattr(call_module, "build_clients", lambda s, **kw: (client, transcriber, tts))
    # Off unless a test asks: a spoken stopgap on every tool turn would change what
    # every other test sees in the voice's output.
    settings.setdefault("tool_filler", False)
    socket = FakeSocket()
    call = VoiceCall(BrowserTransport(socket), business_id=seeded.business_id)
    call.settings = call.settings.model_copy(
        update={"llm_api_key": "test-key", "stt_provider": "groq", "tts_provider": "groq", **settings}
    )
    task = asyncio.create_task(call.run())
    try:
        yield socket
    finally:
        if hang_up:
            socket.hang_up()
        try:
            await asyncio.wait_for(task, 90)
        except TimeoutError:
            task.cancel()
            raise


async def speak_turn(socket: FakeSocket, *, turn: int) -> None:
    """Say something and wait for the server to be listening again.

    ``turn`` counts listening events: 1 is the greeting finishing, so the first
    spoken turn ends at 2.
    """
    await socket.wait_for("listening", count=turn)
    socket.feed(tone(700) + silence(1200))
    await socket.wait_for("listening", count=turn + 1)


async def stored_turns(sessionmaker_for, socket: FakeSocket) -> tuple[Call, list[CallTurn]]:
    call_id = uuid.UUID(socket.of("ready")[0]["call_id"])
    async with sessionmaker_for() as db:
        call = await db.get(Call, call_id)
        turns = (
            await db.execute(
                select(CallTurn).where(CallTurn.call_id == call_id).order_by(CallTurn.turn_index)
            )
        ).scalars().all()
    return call, list(turns)


# ---------------------------------------------------------------------------
# The telephone
# ---------------------------------------------------------------------------

AUTH_TOKEN = "s3cret-auth-token"


class FakeTwilioSocket:
    """Plays Twilio's part of the WebSocket."""

    def __init__(self) -> None:
        self._incoming: asyncio.Queue = asyncio.Queue()
        self.sent: list[dict] = []
        self.closed = False

    async def receive_text(self) -> str:
        item = await self._incoming.get()
        if item is None:
            raise WebSocketDisconnect(1000)
        return item

    async def send_text(self, data: str) -> None:
        self.sent.append(json.loads(data))

    async def close(self, code: int = 1000) -> None:
        self.closed = True
        self._incoming.put_nowait(None)

    # -- the test's side
    def say(self, message: dict | str) -> None:
        self._incoming.put_nowait(message if isinstance(message, str) else json.dumps(message))

    def media(self, pcm: bytes, *, track: str = "inbound") -> None:
        """Send audio the way Twilio does: mu-law, 20 ms at a time, base64."""
        mulaw = ulaw_encode(pcm)
        for offset in range(0, len(mulaw), 160):
            payload = base64.b64encode(mulaw[offset : offset + 160]).decode()
            self.say({"event": "media", "streamSid": "MZ1",
                      "media": {"track": track, "chunk": "1", "timestamp": "0", "payload": payload}})


def rest_with(handler) -> tuple[TwilioRest, list[httpx.Request]]:
    seen: list[httpx.Request] = []

    def record(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return handler(request)

    client = httpx.AsyncClient(transport=httpx.MockTransport(record))
    return TwilioRest("AC123", AUTH_TOKEN, client=client), seen


class ObservedTwilio(TwilioTransport):
    """A Twilio transport that also records the events a screen would have shown.

    On the wire a telephone ignores them; here they are how a test knows the call
    is listening again, or that a turn has finished.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.events: list[dict] = []

    async def send_event(self, kind: str, **fields) -> None:
        self.events.append({"type": kind, **fields})

    def of(self, kind: str) -> list[dict]:
        return [e for e in self.events if e["type"] == kind]

    def kinds(self) -> list[str]:
        return [e["type"] for e in self.events]

    async def wait_for(self, kind: str, count: int = 1, timeout: float = 120.0) -> None:
        async def poll() -> None:
            while len(self.of(kind)) < count:
                await asyncio.sleep(0.02)

        try:
            await asyncio.wait_for(poll(), timeout)
        except TimeoutError:
            raise AssertionError(
                f"timed out waiting for {count}x {kind!r}; saw {self.kinds()}"
            ) from None


@contextlib.asynccontextmanager
async def running_phone_call(
    monkeypatch, seeded, *, client, stt, tts,
    rest: TwilioRest | None = None, hang_up: bool = True,
    caller: str = "+919811112222", call_sid: str = "CA-test-1", **settings,
):
    """Run a phone call against the seeded business; yield its transport.

    The business is found the way a real call finds it: by the number dialled.
    The transport's ``socket`` is the stand-in for Twilio's end.
    """
    transcriber = stt if hasattr(stt, "begin") else BatchTranscriber(stt, sample_rate=8_000)
    monkeypatch.setattr(call_module, "build_clients", lambda s, **kw: (client, transcriber, tts))
    settings.setdefault("tool_filler", False)  # see running_call
    socket = FakeTwilioSocket()
    transport = ObservedTwilio(socket, stream_sid="MZ1", call_sid=call_sid, rest=rest)
    transport.socket = socket
    call = VoiceCall(
        transport,
        dialed_number=seeded.business.phone_e164,
        channel=CallChannel.TWILIO_VOICE,
        caller=caller,
        call_sid=call_sid,
    )
    call.settings = call.settings.model_copy(
        update={"llm_api_key": "test-key", "stt_provider": "groq", "tts_provider": "groq", **settings}
    )
    task = asyncio.create_task(call.run())
    try:
        yield transport
    finally:
        if hang_up:
            socket.say({"event": "stop", "stop": {"callSid": call_sid}})
        try:
            await asyncio.wait_for(task, 90)
        except TimeoutError:
            task.cancel()
            raise
