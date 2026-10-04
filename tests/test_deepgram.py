"""The Deepgram clients, tested against a fake service that speaks the same protocol.

Real WebSockets on localhost, no network and no key: each test starts a small
server that plays the part of Deepgram, scripted for the behaviour under test.
That keeps the checks on what the *client* does -- which messages it sends, how
it reads the events, what it does when the service goes quiet or goes away --
while the live service is exercised separately by hand.

The scripted behaviours are the ones measured against the real service, and the
details that matter are noted where they are reproduced: ``Cleared`` arrives
*after* the stale audio it refers to, ``from_finalize`` results answer a
``Finalize``, and a socket with nothing sent is closed.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
from urllib.parse import parse_qs, urlsplit

import pytest
from websockets.asyncio.server import serve

from backend.app.voice.audio import Pcm
from backend.app.voice.deepgram import DeepgramTTS, DeepgramTranscriber

SECRET = "dg-test-secret-key"


@contextlib.asynccontextmanager
async def fake_service(handler):
    server = await serve(handler, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    try:
        yield f"ws://127.0.0.1:{port}"
    finally:
        server.close()
        await server.wait_closed()


@pytest.fixture
def unreachable(monkeypatch):
    """The service cannot be reached. Patched rather than pointed at a closed
    port: Windows takes about two seconds to report a refused connection."""
    async def refuse(*args, **kwargs):
        raise OSError("connection refused")

    monkeypatch.setattr("backend.app.voice.deepgram.connect", refuse)


async def eventually(predicate, timeout: float = 3.0) -> None:
    async def poll() -> None:
        while not predicate():
            await asyncio.sleep(0.01)

    await asyncio.wait_for(poll(), timeout)


# ===========================================================================
# Transcription
# ===========================================================================


def result(text: str, *, final: bool = False, speech_final: bool = False, from_finalize: bool = False) -> str:
    return json.dumps(
        {
            "type": "Results",
            "is_final": final,
            "speech_final": speech_final,
            "from_finalize": from_finalize,
            "start": 0.0,
            "duration": 1.0,
            "channel": {"alternatives": [{"transcript": text, "confidence": 0.99}]},
        }
    )


class FakeListen:
    """A scripted transcription service."""

    def __init__(self, *, on_audio=None, on_text=None, close_after_text: bool = False) -> None:
        self.audio: list[bytes] = []
        self.texts: list[str] = []
        self.connections = 0
        self.paths: list[str] = []
        self.headers: list[dict] = []
        self._on_audio = on_audio
        self._on_text = on_text
        self._close_after_text = close_after_text

    async def handler(self, ws) -> None:
        self.connections += 1
        self.paths.append(ws.request.path)
        self.headers.append({k.lower(): v for k, v in ws.request.headers.items()})
        async for message in ws:
            if isinstance(message, bytes):
                self.audio.append(message)
                if self._on_audio:
                    await self._on_audio(ws, len(self.audio))
            else:
                kind = json.loads(message)["type"]
                self.texts.append(kind)
                if self._on_text:
                    await self._on_text(ws, kind)
                if self._close_after_text and kind == "Finalize":
                    await ws.close()


def make_transcriber(url: str, **overrides) -> DeepgramTranscriber:
    options = {"base_url": url, "keepalive_s": 30.0, "finalize_timeout_s": 1.0}
    options.update(overrides)
    return DeepgramTranscriber(SECRET, **options)


# --------------------------------------------------------------------------
# The request
# --------------------------------------------------------------------------

async def test_the_key_travels_in_a_header_and_never_in_the_url():
    service = FakeListen()
    async with fake_service(service.handler) as url:
        stt = make_transcriber(url)
        await stt.start()
        await stt.close()

    assert service.headers[0]["authorization"] == f"Token {SECRET}"
    assert SECRET not in service.paths[0]


async def test_the_url_names_the_model_the_audio_format_and_the_pause_to_wait_for():
    service = FakeListen()
    async with fake_service(service.handler) as url:
        stt = make_transcriber(url, model="nova-3", language="en-IN", sample_rate=8000, endpointing_ms=450)
        await stt.start()
        await stt.close()

    parts = urlsplit(service.paths[0])
    query = parse_qs(parts.query)
    assert parts.path == "/v1/listen"
    assert query["model"] == ["nova-3"]
    assert query["language"] == ["en-IN"]
    assert query["encoding"] == ["linear16"]
    assert query["sample_rate"] == ["8000"]
    assert query["channels"] == ["1"]
    assert query["endpointing"] == ["450"]
    assert query["interim_results"] == ["true"]
    assert query["utterance_end_ms"] == ["1000"]


async def test_domain_words_are_passed_as_keyterms_so_names_are_recognised():
    service = FakeListen()
    async with fake_service(service.handler) as url:
        stt = make_transcriber(url)
        await stt.start(keyterms=["Dental cleaning", "Dr Rao", "  "])
        await stt.close()

    assert parse_qs(urlsplit(service.paths[0]).query)["keyterm"] == ["Dental cleaning", "Dr Rao"]


# --------------------------------------------------------------------------
# Sending audio
# --------------------------------------------------------------------------

async def test_audio_reaches_the_service_in_order():
    service = FakeListen()
    async with fake_service(service.handler) as url:
        stt = make_transcriber(url)
        await stt.start()
        stt.begin()
        for chunk in (b"one", b"two", b"three"):
            await stt.send(chunk)
        await eventually(lambda: len(service.audio) == 3)
        await stt.close()

    assert service.audio == [b"one", b"two", b"three"]


async def test_audio_spoken_before_the_socket_is_open_is_held_and_sent_first():
    """The caller talked over the greeting, and the 1.2 s handshake was still in
    progress. Their first words must not be lost, and must stay in order."""
    service = FakeListen()
    async with fake_service(service.handler) as url:
        stt = make_transcriber(url)
        stt.begin()
        await stt.send(b"first")
        await stt.send(b"second")
        await eventually(lambda: stt.connected)
        await stt.send(b"third")
        await eventually(lambda: len(service.audio) == 3)
        await stt.close()

    assert service.audio == [b"first", b"second", b"third"]


# --------------------------------------------------------------------------
# Reading results
# --------------------------------------------------------------------------

async def test_the_finished_utterance_is_the_final_segments_joined():
    async def script(ws, n):
        if n == 1:
            await ws.send(result("hello"))
            await ws.send(result("hello there", final=True))
            await ws.send(result("how are you", final=True, speech_final=True))

    service = FakeListen(on_audio=script)
    async with fake_service(service.handler) as url:
        stt = make_transcriber(url)
        ended = asyncio.Event()
        stt.on_endpoint = ended.set
        await stt.start()
        stt.begin()
        await stt.send(b"audio")
        await asyncio.wait_for(ended.wait(), 2)
        text = await stt.finish()
        await stt.close()

    assert text == "hello there how are you"


async def test_when_the_service_has_already_called_the_end_nothing_is_flushed():
    """speech_final means the transcript is complete. Sending a Finalize anyway
    would add a round trip to every turn for nothing."""
    async def script(ws, n):
        await ws.send(result("book me a cleaning", final=True, speech_final=True))

    service = FakeListen(on_audio=script)
    async with fake_service(service.handler) as url:
        stt = make_transcriber(url)
        ended = asyncio.Event()
        stt.on_endpoint = ended.set
        await stt.start()
        stt.begin()
        await stt.send(b"audio")
        await asyncio.wait_for(ended.wait(), 2)
        assert await stt.finish() == "book me a cleaning"
        await stt.close()

    assert "Finalize" not in service.texts


async def test_interim_text_is_reported_with_the_finished_segments_in_front():
    async def script(ws, n):
        await ws.send(result("i need", final=False))
        await ws.send(result("i need a cleaning", final=True))
        await ws.send(result("next tuesday", final=False))

    service = FakeListen(on_audio=script)
    seen: list[str] = []
    async with fake_service(service.handler) as url:
        stt = make_transcriber(url)
        stt.on_interim = seen.append
        await stt.start()
        stt.begin()
        await stt.send(b"audio")
        await eventually(lambda: len(seen) == 3)
        await stt.close()

    assert seen == ["i need", "i need a cleaning", "i need a cleaning next tuesday"]


async def test_finish_flushes_what_the_service_has_not_yet_reported():
    """The detector decided the caller had finished before the service did."""
    async def on_audio(ws, n):
        await ws.send(result("i need a cleaning", final=True))

    async def on_text(ws, kind):
        if kind == "Finalize":
            await ws.send(result("next tuesday", final=True, from_finalize=True))

    service = FakeListen(on_audio=on_audio, on_text=on_text)
    async with fake_service(service.handler) as url:
        stt = make_transcriber(url)
        await stt.start()
        stt.begin()
        await stt.send(b"audio")
        await eventually(lambda: len(service.audio) == 1)
        await asyncio.sleep(0.05)
        text = await stt.finish()
        await stt.close()

    assert service.texts.count("Finalize") == 1
    assert text == "i need a cleaning next tuesday"


async def test_a_service_that_never_answers_a_flush_cannot_hang_the_turn():
    async def on_audio(ws, n):
        await ws.send(result("book a cleaning", final=True))

    service = FakeListen(on_audio=on_audio)  # ignores Finalize
    async with fake_service(service.handler) as url:
        stt = make_transcriber(url, finalize_timeout_s=0.2)
        await stt.start()
        stt.begin()
        await stt.send(b"audio")
        await eventually(lambda: len(service.audio) == 1)
        await asyncio.sleep(0.05)
        started = asyncio.get_running_loop().time()
        text = await stt.finish()
        elapsed = asyncio.get_running_loop().time() - started
        await stt.close()

    assert text == "book a cleaning"
    assert elapsed < 0.6


async def test_the_utterance_end_backstop_closes_an_utterance_that_never_endpointed():
    async def on_audio(ws, n):
        await ws.send(result("hello there", final=True))
        await ws.send(json.dumps({"type": "UtteranceEnd", "last_word_end": 1.2}))

    service = FakeListen(on_audio=on_audio)
    async with fake_service(service.handler) as url:
        stt = make_transcriber(url)
        ended = asyncio.Event()
        stt.on_endpoint = ended.set
        await stt.start()
        stt.begin()
        await stt.send(b"audio")
        await asyncio.wait_for(ended.wait(), 2)
        await stt.close()


async def test_an_empty_speech_final_is_not_an_endpoint():
    """Silence that the service finalises is not something to answer."""
    async def on_audio(ws, n):
        await ws.send(result("", final=True, speech_final=True))

    service = FakeListen(on_audio=on_audio)
    fired: list[bool] = []
    async with fake_service(service.handler) as url:
        stt = make_transcriber(url)
        stt.on_endpoint = lambda: fired.append(True)
        await stt.start()
        stt.begin()
        await stt.send(b"audio")
        await asyncio.sleep(0.2)
        assert await stt.finish() == ""
        await stt.close()

    assert fired == []


async def test_the_endpoint_fires_once_however_many_results_follow_it():
    async def on_audio(ws, n):
        await ws.send(result("one", final=True, speech_final=True))
        await ws.send(result("two", final=True, speech_final=True))
        await ws.send(json.dumps({"type": "UtteranceEnd", "last_word_end": 1.0}))

    service = FakeListen(on_audio=on_audio)
    fired: list[bool] = []
    async with fake_service(service.handler) as url:
        stt = make_transcriber(url)
        stt.on_endpoint = lambda: fired.append(True)
        await stt.start()
        stt.begin()
        await stt.send(b"audio")
        await asyncio.sleep(0.2)
        await stt.close()

    assert fired == [True]


async def test_results_arriving_outside_an_utterance_are_ignored():
    """Late words from the last utterance must not leak into the next one."""
    async def on_audio(ws, n):
        await ws.send(result("stale words", final=True, speech_final=True))

    service = FakeListen(on_audio=on_audio)
    fired: list[bool] = []
    async with fake_service(service.handler) as url:
        stt = make_transcriber(url)
        stt.on_endpoint = lambda: fired.append(True)
        await stt.start()
        stt.begin()
        await stt.send(b"audio")
        await eventually(lambda: len(fired) == 1)
        assert await stt.finish() == "stale words"

        # The service says something more, after the utterance was collected.
        stt.begin()
        assert await stt.finish() == ""
        await stt.close()


async def test_a_new_utterance_does_not_inherit_the_last_ones_words():
    async def on_audio(ws, n):
        await ws.send(result(f"utterance {n}", final=True, speech_final=True))

    service = FakeListen(on_audio=on_audio)
    async with fake_service(service.handler) as url:
        stt = make_transcriber(url)
        ended = asyncio.Event()
        stt.on_endpoint = ended.set
        await stt.start()

        stt.begin()
        await stt.send(b"a")
        await asyncio.wait_for(ended.wait(), 2)
        first = await stt.finish()

        ended.clear()
        stt.begin()
        await stt.send(b"b")
        await asyncio.wait_for(ended.wait(), 2)
        second = await stt.finish()
        await stt.close()

    assert (first, second) == ("utterance 1", "utterance 2")


async def test_abandoning_an_utterance_discards_what_was_heard_so_far():
    async def on_audio(ws, n):
        await ws.send(result("something the caller took back", final=True))

    service = FakeListen(on_audio=on_audio)
    async with fake_service(service.handler) as url:
        stt = make_transcriber(url)
        await stt.start()
        stt.begin()
        await stt.send(b"audio")
        await eventually(lambda: len(service.audio) == 1)
        await asyncio.sleep(0.05)
        stt.abandon()
        assert await stt.finish() == ""
        await stt.close()


async def test_speech_that_starts_while_the_last_utterance_is_flushing_is_held_back():
    """The service cannot tell two utterances apart on the wire. If the caller
    starts again while the last one is still being flushed, the new words must
    not be mistaken for the old ones, and the old ones must not be lost."""
    async def on_audio(ws, n):
        if n == 1:
            await ws.send(result("first utterance", final=True))
        elif n == 2:  # the first chunk of the second utterance
            await ws.send(result("second utterance", final=True, speech_final=True))

    async def on_text(ws, kind):
        if kind == "Finalize":
            await asyncio.sleep(0.3)
            await ws.send(result("", final=True, from_finalize=True))

    service = FakeListen(on_audio=on_audio, on_text=on_text)
    async with fake_service(service.handler) as url:
        stt = make_transcriber(url)
        ended = asyncio.Event()
        stt.on_endpoint = ended.set
        await stt.start()
        stt.begin()
        await stt.send(b"a")
        await eventually(lambda: len(service.audio) == 1)
        await asyncio.sleep(0.05)

        finishing = asyncio.create_task(stt.finish())
        await asyncio.sleep(0.1)  # the flush is in flight
        stt.begin()  # the caller starts again
        await stt.send(b"b")
        await stt.send(b"c")

        first = await finishing
        await asyncio.wait_for(ended.wait(), 2)
        second = await stt.finish()
        await stt.close()

    assert (first, second) == ("first utterance", "second utterance")
    assert service.audio == [b"a", b"b", b"c"]


# --------------------------------------------------------------------------
# Keeping the socket alive, and losing it
# --------------------------------------------------------------------------

async def test_a_keepalive_is_sent_while_the_call_is_quiet():
    """Measured: a socket with nothing sent is closed after about twelve
    seconds, and the call is quiet for longer than that between turns."""
    service = FakeListen()
    async with fake_service(service.handler) as url:
        stt = make_transcriber(url, keepalive_s=0.05)
        await stt.start()
        await asyncio.sleep(0.4)
        await stt.close()

    assert service.texts.count("KeepAlive") >= 3


async def test_no_keepalive_is_needed_while_audio_is_flowing():
    service = FakeListen()
    async with fake_service(service.handler) as url:
        stt = make_transcriber(url, keepalive_s=0.2)
        await stt.start()
        stt.begin()
        for _ in range(20):
            await stt.send(b"x")
            await asyncio.sleep(0.02)
        await stt.close()

    assert service.texts.count("KeepAlive") == 0


async def test_a_service_that_cannot_be_reached_falls_back_to_the_recording(unreachable):
    """The caller must not be left unanswered because a socket would not open."""
    class Recorder:
        def __init__(self) -> None:
            self.heard: Pcm | None = None

        async def transcribe(self, pcm: Pcm) -> str:
            self.heard = pcm
            return "transcribed from the recording"

    fallback = Recorder()
    stt = make_transcriber("ws://127.0.0.1:1", sample_rate=8000, fallback=fallback)
    await stt.start()
    assert stt.connected is False
    assert stt.last_error

    stt.begin()
    await stt.send(b"live audio")
    text = await stt.finish(utterance=b"the whole utterance")
    await stt.close()

    assert text == "transcribed from the recording"
    assert fallback.heard == Pcm(data=b"the whole utterance", sample_rate=8000)


async def test_without_a_fallback_an_unreachable_service_gives_an_empty_transcript(unreachable):
    stt = make_transcriber("ws://127.0.0.1:1")
    await stt.start()
    stt.begin()
    await stt.send(b"live audio")
    assert await stt.finish(utterance=b"audio") == ""
    await stt.close()


async def test_a_service_that_is_down_is_tried_once_per_utterance_not_once_per_chunk(monkeypatch):
    """Audio arrives every 20-40 ms. A client that reconnected on each chunk would
    hammer a service that is already struggling."""
    attempts = 0

    async def refuse(*args, **kwargs):
        nonlocal attempts
        attempts += 1
        raise OSError("connection refused")

    monkeypatch.setattr("backend.app.voice.deepgram.connect", refuse)
    stt = make_transcriber("ws://127.0.0.1:1")
    stt.begin()
    for _ in range(200):  # five seconds of 25 ms chunks
        await stt.send(b"x" * 800)
        await asyncio.sleep(0)
    await stt.finish()
    assert attempts == 1

    stt.begin()  # the next utterance gets its own attempt
    await stt.send(b"x")
    await stt.finish()
    assert attempts == 2
    await stt.close()


async def test_a_healthy_service_that_hears_silence_does_not_trigger_the_fallback():
    class Boom:
        async def transcribe(self, pcm: Pcm) -> str:
            raise AssertionError("the service answered; there is nothing to fall back from")

    async def on_text(ws, kind):
        if kind == "Finalize":
            await ws.send(result("", final=True, from_finalize=True))

    service = FakeListen(on_text=on_text)
    async with fake_service(service.handler) as url:
        stt = make_transcriber(url, fallback=Boom())
        await stt.start()
        stt.begin()
        await stt.send(b"audio")
        assert await stt.finish(utterance=b"audio") == ""
        await stt.close()


async def test_a_socket_the_service_closed_is_reopened_for_the_next_utterance():
    async def on_audio(ws, n):
        await ws.send(result(f"turn {n}", final=True, speech_final=True))

    service = FakeListen(on_audio=on_audio, on_text=None)
    async with fake_service(service.handler) as url:
        stt = make_transcriber(url)
        ended = asyncio.Event()
        stt.on_endpoint = ended.set
        await stt.start()

        stt.begin()
        await stt.send(b"a")
        await asyncio.wait_for(ended.wait(), 2)
        assert await stt.finish() == "turn 1"

        # The service drops the connection while the call is quiet.
        await stt._ws.close()  # noqa: SLF001 - simulating the far end going away
        await eventually(lambda: not stt.connected)

        ended.clear()
        stt.begin()
        await stt.send(b"b")
        await asyncio.wait_for(ended.wait(), 3)
        assert await stt.finish() == "turn 2"
        await stt.close()

    assert service.connections == 2


async def test_closing_is_clean_and_idempotent():
    service = FakeListen()
    async with fake_service(service.handler) as url:
        stt = make_transcriber(url)
        await stt.start()
        await stt.close()
        await stt.close()
        # Nothing may reconnect once the call is over.
        stt.begin()
        await stt.send(b"late")
        await asyncio.sleep(0.1)
    assert service.connections == 1


# ===========================================================================
# Speech
# ===========================================================================

CLIP = b"\x01\x00" * 960  # 40 ms at 24 kHz
STALE = b"\x7f\x7f" * 960


class FakeSpeak:
    """A scripted speech service.

    Mirrors what was measured against the real one: a request is Speak then
    Flush; audio streams back and ends with Flushed; a Clear while audio is on
    its way is answered with whatever was already made (the stale audio) and
    *then* Cleared.
    """

    def __init__(
        self,
        *,
        chunks: int = 6,
        gap: float = 0.0,
        stale: int = 3,
        acknowledge_clear: bool = True,
        close_on_flush: int | None = None,
    ) -> None:
        self.chunks = chunks
        self.gap = gap
        self.stale = stale
        self.acknowledge_clear = acknowledge_clear
        self.close_on_flush = close_on_flush
        self.messages: list[dict] = []
        self.close_received = False
        self.connections = 0
        self.paths: list[str] = []
        self.headers: list[dict] = []
        self._flushes = 0

    async def handler(self, ws) -> None:
        self.connections += 1
        self.paths.append(ws.request.path)
        self.headers.append({k.lower(): v for k, v in ws.request.headers.items()})
        sender: asyncio.Task | None = None
        try:
            async for message in ws:
                data = json.loads(message)
                if data["type"] == "Close":
                    self.close_received = True
                    continue
                self.messages.append(data)
                if data["type"] == "Flush":
                    self._flushes += 1
                    if self.close_on_flush is not None and self._flushes <= self.close_on_flush:
                        await ws.close()
                        return
                    sender = asyncio.create_task(self._say(ws))
                elif data["type"] == "Clear":
                    if sender is not None:
                        sender.cancel()
                    if self.acknowledge_clear:
                        for _ in range(self.stale):
                            await ws.send(STALE)
                        await ws.send(json.dumps({"type": "Cleared", "sequence_id": 0}))
        finally:
            if sender is not None:
                sender.cancel()

    async def _say(self, ws) -> None:
        for _ in range(self.chunks):
            await ws.send(CLIP)
            if self.gap:
                await asyncio.sleep(self.gap)
        await ws.send(json.dumps({"type": "Flushed", "sequence_id": 0}))

    def kinds(self) -> list[str]:
        return [m["type"] for m in self.messages]


def make_tts(url: str, **overrides) -> DeepgramTTS:
    options = {"base_url": url, "clear_timeout_s": 0.3}
    options.update(overrides)
    return DeepgramTTS(SECRET, **options)


async def collect(tts: DeepgramTTS, text: str) -> list[Pcm]:
    return [chunk async for chunk in tts.stream(text)]


# --------------------------------------------------------------------------
# The request
# --------------------------------------------------------------------------

async def test_speech_authenticates_with_a_header_and_names_the_voice_and_format():
    service = FakeSpeak()
    async with fake_service(service.handler) as url:
        tts = make_tts(url, model="aura-2-cordelia-en", sample_rate=8000)
        await tts.start()
        await tts.close()

    assert service.headers[0]["authorization"] == f"Token {SECRET}"
    parts = urlsplit(service.paths[0])
    query = parse_qs(parts.query)
    assert parts.path == "/v1/speak"
    assert query["model"] == ["aura-2-cordelia-en"]
    assert query["encoding"] == ["linear16"]
    assert query["sample_rate"] == ["8000"]
    assert SECRET not in service.paths[0]


async def test_a_request_is_a_speak_then_a_flush():
    service = FakeSpeak()
    async with fake_service(service.handler) as url:
        tts = make_tts(url)
        await collect(tts, "Hello there.")
        await tts.close()

    assert service.messages == [
        {"type": "Speak", "text": "Hello there."},
        {"type": "Flush"},
    ]


async def test_audio_is_yielded_chunk_by_chunk_as_it_arrives_then_stops_at_flushed():
    service = FakeSpeak(chunks=6)
    async with fake_service(service.handler) as url:
        tts = make_tts(url, sample_rate=24_000)
        chunks = await collect(tts, "Hello there.")
        await tts.close()

    assert len(chunks) == 6
    assert all(c == Pcm(data=CLIP, sample_rate=24_000) for c in chunks)


async def test_the_first_chunk_is_available_before_the_request_has_finished():
    """The point of streaming: playback can start while the rest is still being
    made."""
    service = FakeSpeak(chunks=10, gap=0.05)
    async with fake_service(service.handler) as url:
        tts = make_tts(url)
        loop = asyncio.get_running_loop()
        started = loop.time()
        first_at = None
        count = 0
        async for _ in tts.stream("A longer sentence."):
            count += 1
            if first_at is None:
                first_at = loop.time() - started
        total = loop.time() - started
        await tts.close()

    assert count == 10
    assert first_at < total / 3


async def test_requests_run_one_after_another_on_a_single_socket():
    service = FakeSpeak(chunks=3)
    async with fake_service(service.handler) as url:
        tts = make_tts(url)
        a = await collect(tts, "First.")
        b = await collect(tts, "Second.")
        await tts.close()

    assert len(a) == len(b) == 3
    assert service.connections == 1
    assert service.kinds() == ["Speak", "Flush", "Speak", "Flush"]


async def test_two_requests_started_together_do_not_interleave_their_audio():
    service = FakeSpeak(chunks=5, gap=0.01)
    async with fake_service(service.handler) as url:
        tts = make_tts(url)
        a, b = await asyncio.gather(collect(tts, "First."), collect(tts, "Second."))
        await tts.close()

    assert len(a) == 5 and len(b) == 5
    assert service.kinds() == ["Speak", "Flush", "Speak", "Flush"]


async def test_the_socket_is_opened_ahead_of_the_first_request():
    service = FakeSpeak()
    async with fake_service(service.handler) as url:
        tts = make_tts(url)
        assert tts.connected is False
        await tts.start()
        assert tts.connected is True
        assert service.connections == 1
        await collect(tts, "Hi.")
        assert service.connections == 1
        await tts.close()


# --------------------------------------------------------------------------
# Barge-in: stopping mid-sentence
# --------------------------------------------------------------------------

async def test_stopping_early_sends_a_clear():
    service = FakeSpeak(chunks=50, gap=0.02)
    async with fake_service(service.handler) as url:
        tts = make_tts(url)
        stream = tts.stream("A very long sentence that the caller interrupts.")
        async for _ in stream:
            break
        await stream.aclose()
        await eventually(lambda: "Clear" in service.kinds())
        await tts.close()

    assert service.kinds() == ["Speak", "Flush", "Clear"]


async def test_audio_made_before_the_clear_took_effect_never_reaches_the_next_request():
    """Measured: the service has already made a second or two of audio when the
    Clear arrives, and all of it comes down the socket before ``Cleared``. If
    the client kept it, the caller would hear the end of the interrupted
    sentence at the start of the next one."""
    service = FakeSpeak(chunks=50, gap=0.02, stale=4)
    async with fake_service(service.handler) as url:
        tts = make_tts(url)
        stream = tts.stream("A very long sentence that the caller interrupts.")
        async for _ in stream:
            break
        await stream.aclose()

        service.chunks, service.gap = 4, 0.0
        chunks = await collect(tts, "Next.")
        await tts.close()

    assert len(chunks) == 4
    assert all(c.data == CLIP for c in chunks), "stale audio leaked into the next request"


async def test_cancelling_the_consuming_task_clears_the_request_too():
    """This is what barge-in does: it cancels the task that is playing the reply."""
    service = FakeSpeak(chunks=50, gap=0.02, stale=4)
    async with fake_service(service.handler) as url:
        tts = make_tts(url)
        got: list[Pcm] = []

        async def play() -> None:
            async for chunk in tts.stream("A very long sentence that the caller interrupts."):
                got.append(chunk)

        task = asyncio.create_task(play())
        await eventually(lambda: len(got) >= 2)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        await eventually(lambda: "Clear" in service.kinds())

        service.chunks, service.gap = 3, 0.0
        chunks = await collect(tts, "Next.")
        await tts.close()

    assert len(chunks) == 3 and all(c.data == CLIP for c in chunks)


async def test_the_next_request_waits_for_the_clear_to_be_acknowledged():
    service = FakeSpeak(chunks=50, gap=0.02, stale=2)
    async with fake_service(service.handler) as url:
        tts = make_tts(url)
        stream = tts.stream("Long.")
        async for _ in stream:
            break
        await stream.aclose()
        service.chunks, service.gap = 2, 0.0
        await collect(tts, "Next.")
        await tts.close()

    kinds = service.kinds()
    # Clear is on the wire before the next Speak, never after it.
    assert kinds.index("Clear") < len(kinds) - 2
    assert kinds[-2:] == ["Speak", "Flush"]


async def test_an_unacknowledged_clear_means_a_fresh_socket_not_a_guess():
    service = FakeSpeak(chunks=50, gap=0.02, acknowledge_clear=False)
    async with fake_service(service.handler) as url:
        tts = make_tts(url, clear_timeout_s=0.15)
        stream = tts.stream("Long.")
        async for _ in stream:
            break
        await stream.aclose()

        service.chunks, service.gap = 3, 0.0
        chunks = await collect(tts, "Next.")
        await tts.close()

    assert len(chunks) == 3
    assert service.connections == 2


async def test_cancel_when_nothing_is_being_said_does_nothing():
    service = FakeSpeak()
    async with fake_service(service.handler) as url:
        tts = make_tts(url)
        await tts.cancel()
        await collect(tts, "Hello.")
        await tts.cancel()
        await tts.close()

    assert "Clear" not in service.kinds()


async def test_cancel_stops_the_request_in_flight():
    service = FakeSpeak(chunks=50, gap=0.02)
    async with fake_service(service.handler) as url:
        tts = make_tts(url)
        got: list[Pcm] = []

        async def play() -> None:
            with contextlib.suppress(Exception):
                async for chunk in tts.stream("Long."):
                    got.append(chunk)

        task = asyncio.create_task(play())
        await eventually(lambda: len(got) >= 2)
        await tts.cancel()
        await eventually(lambda: "Clear" in service.kinds())
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        await tts.close()


# --------------------------------------------------------------------------
# Losing the socket
# --------------------------------------------------------------------------

async def test_a_socket_the_service_closed_during_a_quiet_spell_is_replaced():
    service = FakeSpeak(chunks=3, close_on_flush=None)
    async with fake_service(service.handler) as url:
        tts = make_tts(url)
        await collect(tts, "First.")
        await tts._ws.close()  # noqa: SLF001 - the far end drops the connection while quiet
        await eventually(lambda: not tts.connected)
        chunks = await collect(tts, "Second.")
        await tts.close()

    assert len(chunks) == 3
    assert service.connections == 2


async def test_a_request_that_finds_the_socket_closed_is_retried_once_before_any_audio():
    """The service closes the first connection the moment it sees a Flush. The
    caller must hear the sentence anyway -- once -- not silence."""
    service = FakeSpeak(chunks=3, close_on_flush=1)
    async with fake_service(service.handler) as url:
        tts = make_tts(url)
        chunks = await collect(tts, "Hello.")
        await tts.close()

    assert len(chunks) == 3
    assert service.connections == 2


async def test_a_second_failure_is_raised_not_retried_forever():
    service = FakeSpeak(close_on_flush=99)
    async with fake_service(service.handler) as url:
        tts = make_tts(url)
        with pytest.raises((ConnectionError, Exception)):
            await collect(tts, "Hello.")
        await tts.close()

    assert service.connections == 2


async def test_a_service_that_cannot_be_reached_raises_at_the_first_sentence(unreachable):
    tts = make_tts("ws://127.0.0.1:1")
    await tts.start()  # swallowed: opening early is an optimisation
    assert tts.connected is False
    assert tts.last_error
    with pytest.raises(Exception):  # noqa: B017 - any connection error will do
        await collect(tts, "Hello.")
    await tts.close()


async def test_nothing_reconnects_after_close():
    service = FakeSpeak()
    async with fake_service(service.handler) as url:
        tts = make_tts(url)
        await tts.start()
        await tts.close()
        with pytest.raises(ConnectionError):
            await collect(tts, "Too late.")
    assert service.connections == 1
