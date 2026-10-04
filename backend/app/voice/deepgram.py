"""Deepgram over WebSockets: transcription that listens while the caller talks, and
speech that starts playing before the sentence has been finished.

Written against the wire protocol rather than their SDK. The protocol is small
-- audio in, JSON events out -- and what matters for a phone call lives in the
details the SDK would hide: when exactly an utterance is over, what happens to
audio already in flight when the caller interrupts, how long a quiet socket
survives. Those are the things measured against the live service while this was
written, and they are recorded where they decide the code.

Both clients hold one socket for the whole call. Opening one costs about 1.2 s
from India (TLS plus the WebSocket handshake), which is far too long to pay in
the middle of a conversation, so it is opened while the greeting plays.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
from collections.abc import AsyncIterator, Callable, Sequence
from urllib.parse import urlencode

from websockets.asyncio.client import ClientConnection, connect
from websockets.exceptions import ConnectionClosed

from .audio import Pcm
from .speech import SpeechToText

log = logging.getLogger(__name__)

DEFAULT_BASE_URL = "wss://api.deepgram.com"

# Deepgram closes a transcription socket that has heard nothing for ten seconds.
# Between utterances nothing is sent (only speech goes upstream), so a KeepAlive
# goes out well inside that.
KEEPALIVE_S = 4.0

# How long to wait for the service to flush what it has when the caller stops.
# The round trip is ~350 ms from India, so this is generous: it exists so a dead
# socket costs one slow turn, not a hung call.
FINALIZE_TIMEOUT_S = 2.5

CONNECT_TIMEOUT_S = 8.0

# Audio that arrives before the socket is ready (the caller spoke during the
# greeting) is held, not dropped. Bounded, so a socket that never opens cannot
# make the process grow without limit: ten seconds of 16 kHz 16-bit mono.
BACKLOG_LIMIT_BYTES = 10 * 16_000 * 2

# Speech: how long to wait for the confirmation that a Clear took effect.
CLEAR_TIMEOUT_S = 1.5


def _auth(api_key: str) -> dict[str, str]:
    return {"Authorization": f"Token {api_key}"}


# ---------------------------------------------------------------------------
# Transcription
# ---------------------------------------------------------------------------


class DeepgramTranscriber:
    """Streaming transcription of the caller, an utterance at a time.

    Only speech goes upstream. The detector that opens and closes utterances sits
    in front of this, so silence, room tone and the agent's own voice coming back
    through the caller's speakers are never transcribed -- the same job it does
    for the batch service, where the cost of failing is a hallucinated "Thank
    you." and here is the agent answering itself.

    What streaming buys is the end of the utterance. Deepgram listens for a pause
    (``endpointing_ms``) and reports the finished transcript the moment it has
    decided, which is sooner than the detector's own silence timer, and the words
    were transcribed as they were spoken rather than after. Measured from India:
    the transcript is ready about 0.6 s after the caller stops, against 1.15 s
    for end-of-silence plus a batch request.

    If the socket is unavailable when an utterance ends, the caller is not left
    unanswered: given a ``fallback`` and the detector's copy of the audio, the
    utterance is transcribed as a recording instead.
    """

    def __init__(
        self,
        api_key: str,
        *,
        model: str = "nova-3",
        language: str = "en",
        sample_rate: int = 16_000,
        endpointing_ms: int = 500,
        utterance_end_ms: int = 1000,
        base_url: str = DEFAULT_BASE_URL,
        keepalive_s: float = KEEPALIVE_S,
        finalize_timeout_s: float = FINALIZE_TIMEOUT_S,
        fallback: SpeechToText | None = None,
    ) -> None:
        self._api_key = api_key
        self._model = model
        self._language = language
        self._rate = sample_rate
        self._endpointing_ms = endpointing_ms
        self._utterance_end_ms = utterance_end_ms
        self._base_url = base_url.rstrip("/")
        self._keepalive_s = keepalive_s
        self._finalize_timeout_s = finalize_timeout_s
        self._fallback = fallback

        self.on_interim: Callable[[str], None] | None = None
        self.on_endpoint: Callable[[], None] | None = None
        self.endpoints_itself = True

        self._keyterms: tuple[str, ...] = ()
        self._ws: ClientConnection | None = None
        self._connecting: asyncio.Task[None] | None = None
        self._reader: asyncio.Task[None] | None = None
        self._keeper: asyncio.Task[None] | None = None
        self._backlog: list[bytes] = []
        self._backlog_bytes = 0
        self._closed = False
        self._last_sent = 0.0
        self.last_error: str | None = None

        # The utterance in progress.
        self._collecting = False
        self._finals: list[str] = []
        self._endpoint_seen = False
        self._flushed = asyncio.Event()
        self._unavailable = False

        # The caller started speaking again while the last utterance was still
        # being flushed. The service cannot tell the two apart on the wire, so the
        # new one is held back -- its audio kept in order -- until the old one
        # has been collected, rather than being allowed to blur into it.
        self._finishing = False
        self._begin_deferred = False
        self._deferred: list[bytes] = []

    # ------------------------------------------------------------- connection

    @property
    def connected(self) -> bool:
        return self._ws is not None

    def _url(self) -> str:
        params: list[tuple[str, str | int]] = [
            ("model", self._model),
            ("language", self._language),
            ("encoding", "linear16"),
            ("sample_rate", self._rate),
            ("channels", 1),
            ("interim_results", "true"),
            ("endpointing", self._endpointing_ms),
            # A backstop for an utterance the endpointing never closes (noise
            # that keeps the detector open). Needs interim results.
            ("utterance_end_ms", self._utterance_end_ms),
            ("smart_format", "true"),
        ]
        params.extend(("keyterm", term) for term in self._keyterms)
        return f"{self._base_url}/v1/listen?{urlencode(params)}"

    async def start(self, *, keyterms: Sequence[str] = ()) -> None:
        """Open the socket. Safe to call again; reconnects if it has dropped."""
        self._keyterms = tuple(term for term in keyterms if term.strip())
        self._kick_connect()
        if self._connecting is not None:
            await asyncio.shield(self._connecting)

    def _kick_connect(self) -> None:
        if self._closed or self._ws is not None:
            return
        if self._connecting is None or self._connecting.done():
            self._connecting = asyncio.create_task(self._connect(), name="deepgram-stt-connect")

    async def _connect(self) -> None:
        try:
            ws = await connect(
                self._url(),
                additional_headers=_auth(self._api_key),
                open_timeout=CONNECT_TIMEOUT_S,
                max_size=None,
            )
        except Exception as exc:  # noqa: BLE001 - reported, then retried at the next utterance
            self.last_error = f"{type(exc).__name__}: {exc}"[:300]
            log.warning("deepgram transcription socket failed to open: %s", self.last_error)
            return

        # Everything that piled up while the socket was opening goes first, in
        # order, and the socket is published only once the backlog is empty. No
        # await sits between that check and the assignment, so audio arriving
        # now either lands in the backlog (and is sent here) or goes straight to
        # the socket, never past it.
        try:
            while self._backlog:
                chunk = self._backlog.pop(0)
                self._backlog_bytes -= len(chunk)
                await ws.send(chunk)
        except ConnectionClosed:
            return
        self._ws = ws
        self.last_error = None
        self._last_sent = asyncio.get_running_loop().time()
        self._reader = asyncio.create_task(self._read(ws), name="deepgram-stt-read")
        if self._keeper is None or self._keeper.done():
            self._keeper = asyncio.create_task(self._keep_alive(), name="deepgram-stt-keepalive")

    async def _keep_alive(self) -> None:
        loop = asyncio.get_running_loop()
        while not self._closed:
            await asyncio.sleep(self._keepalive_s)
            ws = self._ws
            if ws is None:
                continue
            if loop.time() - self._last_sent >= self._keepalive_s * 0.9:
                with contextlib.suppress(ConnectionClosed):
                    await ws.send(json.dumps({"type": "KeepAlive"}))
                    self._last_sent = loop.time()

    # ---------------------------------------------------------------- reading

    async def _read(self, ws: ClientConnection) -> None:
        try:
            async for message in ws:
                if isinstance(message, bytes):
                    continue
                try:
                    event = json.loads(message)
                except json.JSONDecodeError:
                    continue
                self._handle(event)
        except ConnectionClosed:
            pass
        except Exception:  # noqa: BLE001 - a reader that dies silently would deafen the call
            log.exception("deepgram transcription reader failed")
        finally:
            if self._ws is ws:
                self._ws = None
            # Whoever is waiting for a flush will not get one from this socket.
            self._unavailable = True
            self._flushed.set()

    def _handle(self, event: dict) -> None:
        kind = event.get("type")

        if kind == "Results":
            if not self._collecting:
                return
            text = (event["channel"]["alternatives"][0].get("transcript") or "").strip()
            if event.get("from_finalize"):
                if text:
                    self._finals.append(text)
                self._flushed.set()
                return
            if event.get("is_final"):
                if text:
                    self._finals.append(text)
                self._interim(" ".join(self._finals))
                # The service has decided the caller finished. Only worth
                # reporting if there is something to answer.
                if event.get("speech_final") and self._finals:
                    self._endpoint()
            elif text:
                self._interim(" ".join([*self._finals, text]))

        elif kind == "UtteranceEnd":
            # The backstop: a long silence after words, with no speech_final.
            if self._collecting and self._finals:
                self._endpoint()

        elif kind in {"Error", "Warning"}:
            log.warning("deepgram %s: %s", kind.lower(), event)

    def _interim(self, text: str) -> None:
        if self.on_interim is not None and text:
            try:
                self.on_interim(text)
            except Exception:  # noqa: BLE001 - a caption must not break transcription
                log.exception("interim callback failed")

    def _endpoint(self) -> None:
        if self._endpoint_seen:
            return
        self._endpoint_seen = True
        if self.on_endpoint is not None:
            try:
                self.on_endpoint()
            except Exception:  # noqa: BLE001
                log.exception("endpoint callback failed")

    # -------------------------------------------------------------- utterance

    def begin(self) -> None:
        if self._finishing:
            self._begin_deferred = True
            self._deferred = []
            return
        self._start_utterance()

    def _start_utterance(self) -> None:
        self._collecting = True
        self._finals = []
        self._endpoint_seen = False
        self._unavailable = False
        self._flushed.clear()
        # An utterance is the moment to find out the socket has gone, and to
        # start opening another while the caller is still speaking.
        if self._ws is None:
            self._kick_connect()

    async def send(self, pcm: bytes) -> None:
        if self._begin_deferred:
            self._deferred.append(pcm)
            return
        ws = self._ws
        if ws is None:
            # Held for when the socket opens. No attempt to open one is made here:
            # audio arrives every 20-40 ms, and a service that is down would be
            # hit with a connection attempt per chunk. One is made at the start
            # of each utterance (``begin``) and when an open socket drops, below.
            self._unavailable = True
            if self._backlog_bytes + len(pcm) <= BACKLOG_LIMIT_BYTES:
                self._backlog.append(pcm)
                self._backlog_bytes += len(pcm)
            return
        try:
            await ws.send(pcm)
            self._last_sent = asyncio.get_running_loop().time()
        except ConnectionClosed:
            self._ws = None
            self._unavailable = True
            self._kick_connect()

    async def finish(self, utterance: bytes | None = None) -> str:
        if not self._collecting:
            return ""
        self._finishing = True
        try:
            answered = await self._flush()
            text = " ".join(self._finals).strip()
            had_trouble = self._unavailable or not answered
            self._collecting = False
            self._finals = []
            self._endpoint_seen = False

            if not text and had_trouble and utterance and self._fallback is not None:
                log.warning("transcribing as a recording: the streaming socket was unavailable")
                return await self._fallback.transcribe(Pcm(data=utterance, sample_rate=self._rate))
            return text
        finally:
            self._finishing = False
            if self._begin_deferred:
                # Whatever the caller said in the meantime starts now.
                self._begin_deferred = False
                held, self._deferred = self._deferred, []
                self._start_utterance()
                for chunk in held:
                    await self.send(chunk)

    async def _flush(self) -> bool:
        """Ask the service for whatever it has not yet reported; True if it did.

        Skipped when the service has already called the end of the utterance: its
        final transcript is in hand and there is nothing left to wait for.
        """
        if self._endpoint_seen:
            return True
        if self._ws is None:
            # Still opening (the caller spoke straight away): give it a moment.
            if self._connecting is not None and not self._connecting.done():
                with contextlib.suppress(Exception):
                    await asyncio.wait_for(asyncio.shield(self._connecting), self._finalize_timeout_s)
            if self._ws is None:
                return False
        ws = self._ws
        self._flushed.clear()
        try:
            await ws.send(json.dumps({"type": "Finalize"}))
            await asyncio.wait_for(self._flushed.wait(), self._finalize_timeout_s)
        except (ConnectionClosed, TimeoutError):
            return False
        return not self._unavailable

    def abandon(self) -> None:
        self._collecting = False
        self._finals = []
        self._endpoint_seen = False
        self._begin_deferred = False
        self._deferred = []

    async def close(self) -> None:
        self._closed = True
        for task in (self._keeper, self._connecting):
            if task is not None:
                task.cancel()
        ws, self._ws = self._ws, None
        if ws is not None:
            with contextlib.suppress(Exception):
                await ws.send(json.dumps({"type": "CloseStream"}))
            with contextlib.suppress(Exception):
                await ws.close()
        if self._reader is not None:
            self._reader.cancel()
        await asyncio.gather(
            *(t for t in (self._keeper, self._connecting, self._reader) if t is not None),
            return_exceptions=True,
        )


# ---------------------------------------------------------------------------
# Speech
# ---------------------------------------------------------------------------

_END = object()  # the service has sent everything for this request
_BROKEN = object()  # the socket closed under the request


class DeepgramTTS:
    """Streaming speech synthesis over one socket per call.

    A request is ``Speak`` (the text) and ``Flush`` (say it now), and the answer is
    a stream of small audio chunks followed by ``Flushed``. Measured from India
    on a warm socket, the first chunk arrives 300-600 ms after the request and
    the rest at about twice real time -- so playback starts after one chunk,
    not after the whole sentence, and never underruns.

    One request is in flight at a time. The pipeline already asks for sentence N+1
    only once sentence N has been taken, and the audio is generated faster than
    it plays, so the next sentence is waiting before the last one ends.

    **Barge-in.** When the caller interrupts, a ``Clear`` tells the service to
    drop what it has not yet made -- but it has typically made a second or two of
    audio already, and that arrives after the Clear. Measured: all of it arrives
    *before* the ``Cleared`` acknowledgement and none after. So the rule is
    simple and does not need the socket thrown away: after a Clear, discard every
    audio frame until ``Cleared``; whatever follows belongs to the next request.
    """

    def __init__(
        self,
        api_key: str,
        *,
        model: str = "aura-2-cordelia-en",
        sample_rate: int = 24_000,
        base_url: str = DEFAULT_BASE_URL,
        clear_timeout_s: float = CLEAR_TIMEOUT_S,
    ) -> None:
        self._api_key = api_key
        self._model = model
        self.sample_rate = sample_rate
        self._base_url = base_url.rstrip("/")
        self._clear_timeout_s = clear_timeout_s

        self._ws: ClientConnection | None = None
        self._reader: asyncio.Task[None] | None = None
        self._connect_lock = asyncio.Lock()
        self._request_lock = asyncio.Lock()
        self._inbox: asyncio.Queue[object] = asyncio.Queue()
        self._active = False
        self._discarding = False
        self._cleared = asyncio.Event()
        self._cleared.set()
        self._closed = False
        self.last_error: str | None = None

    # ------------------------------------------------------------- connection

    @property
    def connected(self) -> bool:
        return self._ws is not None

    def _url(self) -> str:
        query = urlencode(
            {"model": self._model, "encoding": "linear16", "sample_rate": self.sample_rate}
        )
        return f"{self._base_url}/v1/speak?{query}"

    async def start(self) -> None:
        try:
            await self._ensure()
        except Exception as exc:  # noqa: BLE001 - the first sentence retries, and reports
            self.last_error = f"{type(exc).__name__}: {exc}"[:300]
            log.warning("deepgram speech socket failed to open: %s", self.last_error)

    async def _ensure(self) -> ClientConnection:
        async with self._connect_lock:
            if self._ws is not None:
                return self._ws
            if self._closed:
                raise ConnectionError("the speech connection was closed")
            ws = await connect(
                self._url(),
                additional_headers=_auth(self._api_key),
                open_timeout=CONNECT_TIMEOUT_S,
                max_size=None,
            )
            self._ws = ws
            self._discarding = False
            self._cleared.set()
            self.last_error = None
            self._reader = asyncio.create_task(self._read(ws), name="deepgram-tts-read")
            return ws

    # ---------------------------------------------------------------- reading

    async def _read(self, ws: ClientConnection) -> None:
        try:
            async for message in ws:
                if self._ws is not ws:
                    # A socket that has been dropped keeps delivering for a
                    # moment. Nothing from it belongs to any request any more.
                    continue
                if isinstance(message, bytes):
                    if not self._discarding:
                        self._inbox.put_nowait(message)
                    continue
                try:
                    event = json.loads(message)
                except json.JSONDecodeError:
                    continue
                kind = event.get("type")
                if kind == "Flushed":
                    if not self._discarding:
                        self._inbox.put_nowait(_END)
                elif kind == "Cleared":
                    self._discarding = False
                    self._cleared.set()
                elif kind in {"Error", "Warning"}:
                    log.warning("deepgram speech %s: %s", kind.lower(), event)
        except ConnectionClosed:
            pass
        except Exception:  # noqa: BLE001
            log.exception("deepgram speech reader failed")
        finally:
            # Only the socket currently in use may fail a request. One that was
            # dropped on purpose (to reconnect) must not fail the retry.
            if self._ws is ws:
                self._ws = None
                self._cleared.set()
                self._inbox.put_nowait(_BROKEN)

    # --------------------------------------------------------------- requests

    async def stream(self, text: str) -> AsyncIterator[Pcm]:
        """Speak ``text``, yielding audio as it is made.

        If the consumer stops early -- a barge-in cancels it, or it simply
        stops iterating -- the request is abandoned cleanly: a Clear goes out and
        the stale audio is discarded, whichever way the generator is closed.

        A socket the service closed while the call was quiet (nothing is sent
        between turns) is found out here, and the request is retried once on a
        fresh one -- but only if no audio has been produced yet. After that a
        retry would repeat words the caller has already heard.
        """
        async with self._request_lock:
            attempt = 0
            while True:
                attempt += 1
                ws = await self._ready_for_request()
                inbox: asyncio.Queue[object] = asyncio.Queue()
                self._inbox = inbox
                self._active = True
                finished = False
                yielded = False
                try:
                    await ws.send(json.dumps({"type": "Speak", "text": text}))
                    await ws.send(json.dumps({"type": "Flush"}))
                    while True:
                        item = await inbox.get()
                        if item is _END:
                            finished = True
                            return
                        if item is _BROKEN:
                            raise ConnectionError("the speech connection closed mid-sentence")
                        yielded = True
                        yield Pcm(data=item, sample_rate=self.sample_rate)  # type: ignore[arg-type]
                except (ConnectionClosed, ConnectionError):
                    if yielded or attempt >= 2:
                        raise
                    log.warning("deepgram speech: connection was closed; retrying on a new one")
                    await self._drop()
                    continue
                finally:
                    self._active = False
                    if not finished:
                        await self._abandon(ws)

    async def _ready_for_request(self) -> ClientConnection:
        # A Clear from the previous request whose acknowledgement has not come
        # back yet: audio from the old request may still be on its way, and a
        # new request must not start collecting it. Normally already settled.
        if not self._cleared.is_set():
            try:
                await asyncio.wait_for(self._cleared.wait(), self._clear_timeout_s)
            except TimeoutError:
                # The acknowledgement never came. Rather than guess where the old
                # audio ends, start over on a fresh socket.
                log.warning("deepgram speech: no Cleared acknowledgement; reconnecting")
                await self._drop()
        return await self._ensure()

    async def _abandon(self, ws: ClientConnection) -> None:
        """Tell the service to stop, and ignore audio until it confirms."""
        if self._ws is not ws:
            return
        self._discarding = True
        self._cleared.clear()
        with contextlib.suppress(Exception):
            await ws.send(json.dumps({"type": "Clear"}))

    async def cancel(self) -> None:
        """Barge-in: stop what is being said. Idempotent, and a no-op between
        requests -- the generator's own cleanup normally does this first."""
        ws = self._ws
        if self._active and ws is not None:
            await self._abandon(ws)

    async def _drop(self) -> None:
        ws, self._ws = self._ws, None
        if ws is not None:
            with contextlib.suppress(Exception):
                await ws.close()
        self._discarding = False
        self._cleared.set()

    async def close(self) -> None:
        self._closed = True
        ws, self._ws = self._ws, None
        if ws is not None:
            with contextlib.suppress(Exception):
                await ws.send(json.dumps({"type": "Close"}))
            with contextlib.suppress(Exception):
                await ws.close()
        if self._reader is not None:
            self._reader.cancel()
            await asyncio.gather(self._reader, return_exceptions=True)
