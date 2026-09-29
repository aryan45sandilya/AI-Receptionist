"""Turning a streaming reply into paced audio: the speaking half of a voice turn.

The model writes text a few tokens at a time; the synthesis service turns one
sentence into one audio clip; the browser plays PCM as it arrives. This module
is the seam between them, and it does three things that decide how the call
feels:

1.  **Overlap.** The first sentence is synthesised the moment it is complete,
    while the model is still writing the second. Time to first audio is then the
    time to *one sentence*, not the time to the whole reply.

2.  **Synthesis and playback are separate workers.** Sentence two is being
    synthesised while sentence one is still playing, so the gap between them is
    zero rather than one synthesis round trip.

3.  **Pacing.** Audio is released at real-time speed with a short lead, not
    dumped onto the socket as fast as it can be made. That is not for the
    network's sake: it keeps the server's idea of "what has been played"
    within a quarter second of the listener's, which is what makes stopping on
    barge-in (Phase 3) land where the caller expects, and what tells the half
    duplex gate when the agent has actually finished talking.

Deliberately knows nothing about WebSockets, Groq or the database: it is handed
callbacks. Its timing is all ``asyncio`` loop time and ``asyncio.sleep``, so a
test can run it under a virtual clock and assert exact numbers.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import Any

from .audio import Pcm
from .speech import ORPHEUS_MAX_CHARS, SentenceStream, TextToSpeech, split_sentences

SendEvent = Callable[..., Awaitable[None]]
SendAudio = Callable[[bytes], Awaitable[None]]

# How far ahead of the listener the server is allowed to run. Enough to absorb
# scheduling jitter and a slow frame without an audible gap; small enough that
# stopping now leaves at most this much stale audio to flush.
DEFAULT_LEAD_S = 0.25
DEFAULT_CHUNK_MS = 40

# Time for the browser's own output buffer to drain after the last byte arrives.
OUTPUT_LATENCY_MARGIN_S = 0.15


def loop_time() -> float:
    """Monotonic seconds on the running event loop.

    One clock for the whole pipeline. It is also the clock a virtual-time test
    loop replaces, which is why nothing here calls ``time.perf_counter``.
    """
    return asyncio.get_running_loop().time()


class TurnSpeaker:
    """Speak one agent reply, sentence by sentence, as it is written.

    Usage, from the code running the model::

        speaker = TurnSpeaker(tts=tts, send_event=..., send_audio=..., anchor=t0)
        speaker.start()
        result = await runner.run_turn(text, on_text=speaker.on_text)
        await speaker.finish()
        await speaker.drain_playback()
    """

    def __init__(
        self,
        *,
        tts: TextToSpeech,
        send_event: SendEvent,
        send_audio: SendAudio,
        anchor: float | None = None,
        lead_s: float = DEFAULT_LEAD_S,
        chunk_ms: int = DEFAULT_CHUNK_MS,
        max_chars: int = ORPHEUS_MAX_CHARS,
        eager_first: bool = True,
    ) -> None:
        self._tts = tts
        self._send_event = send_event
        self._send_audio = send_audio
        self._lead_s = lead_s
        self._chunk_ms = chunk_ms
        self._sentences = SentenceStream(max_chars=max_chars, eager_first=eager_first)
        self._max_chars = max_chars

        # ``anchor`` is when the caller stopped talking, on loop time. first
        # audio is measured from there, because that is when they started
        # waiting -- not from when this object happened to be built.
        self._anchor = anchor if anchor is not None else loop_time()

        self._text_q: asyncio.Queue[str | None] = asyncio.Queue()
        self._clip_q: asyncio.Queue[Pcm | None] = asyncio.Queue()
        self._tasks: list[asyncio.Task[None]] = []
        self._audio_started = False
        self._finished = False

        self.first_audio_ms: int | None = None
        self.error: str | None = None
        # Loop time at which everything sent so far will have finished playing.
        self.play_until: float = 0.0
        # Every sentence handed to synthesis, in order: what the caller was told.
        self.spoken: list[str] = []

    # ------------------------------------------------------------------ input

    def start(self) -> None:
        """Start the two workers. Call before any text arrives."""
        if self._tasks:
            return
        self.play_until = loop_time()
        self._tasks = [
            asyncio.create_task(self._synth_worker(), name="tts-synth"),
            asyncio.create_task(self._play_worker(), name="tts-play"),
        ]

    def on_text(self, chunk: str) -> None:
        """Model output as it streams. Synchronous, so it can be the LLM callback."""
        if self._finished:
            return
        self._enqueue(self._sentences.feed(chunk))

    def say(self, text: str) -> None:
        """Speak a fixed message after whatever has been streamed so far.

        For replies the model did not write: an error apology, or the fallback
        when it produced no words at all. Anything still buffered is released
        first, so the order the caller hears is the order things happened.
        """
        if self._finished:
            return
        self._enqueue(self._sentences.flush())
        self._enqueue(split_sentences(text, max_chars=self._max_chars))

    def _enqueue(self, sentences: list[str]) -> None:
        for sentence in sentences:
            self.spoken.append(sentence)
            self._text_q.put_nowait(sentence)

    @property
    def spoken_text(self) -> str:
        return " ".join(self.spoken)

    # -------------------------------------------------------------- lifecycle

    async def finish(self) -> None:
        """No more text is coming: say what is left, wait until it is all sent."""
        if self._finished:
            return
        self._enqueue(self._sentences.flush())
        self._finished = True
        self._text_q.put_nowait(None)
        if self._tasks:
            await asyncio.gather(*self._tasks)
        if self._audio_started:
            await self._send_event("audio_end")

    async def drain_playback(self) -> None:
        """Wait until the listener has, in all likelihood, heard everything.

        ``finish`` returns when the last byte is *sent*; the browser is still
        playing for up to a quarter second after that, plus whatever its own
        output buffer holds. The half-duplex gate must stay closed through that
        tail or the agent hears its own last words.
        """
        remaining = self.play_until + OUTPUT_LATENCY_MARGIN_S - loop_time()
        if remaining > 0:
            await asyncio.sleep(remaining)

    async def cancel(self) -> None:
        """Stop immediately: the caller has started talking (Phase 3 barge-in).

        Audio already handed to the socket cannot be recalled. At most
        ``lead_s`` of it exists, and the browser is told to discard its queue.
        """
        self._finished = True
        for task in self._tasks:
            task.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()
        self.play_until = loop_time()

    # ---------------------------------------------------------------- workers

    async def _synth_worker(self) -> None:
        failed = False
        while True:
            sentence = await self._text_q.get()
            if sentence is None:
                break
            if failed:
                # Every later sentence would fail the same way (an unaccepted
                # licence, a bad key), so stop spending requests on them -- but
                # keep draining, so nothing upstream ever blocks on this queue.
                continue
            try:
                clip = await self._tts.synthesize(sentence)
            except Exception as exc:  # noqa: BLE001 - a silent voice must not end the call
                failed = True
                self.error = f"{type(exc).__name__}: {exc}"[:300]
                await self._send_event("tts_error", message=self.error)
                continue
            await self._clip_q.put(clip)
        await self._clip_q.put(None)

    async def _play_worker(self) -> None:
        while (clip := await self._clip_q.get()) is not None:
            await self._stream(clip)

    async def _stream(self, clip: Pcm) -> None:
        """Send one clip in small frames, never far ahead of real time."""
        bytes_per_second = clip.sample_rate * 2 * clip.channels
        # Even, or a frame could split a 16-bit sample in two.
        step = (bytes_per_second * self._chunk_ms // 1000) & ~1
        step = max(step, 2)

        for offset in range(0, len(clip.data), step):
            piece = clip.data[offset : offset + step]

            ahead = self.play_until - loop_time()
            if ahead > self._lead_s:
                await asyncio.sleep(ahead - self._lead_s)

            if not self._audio_started:
                self._audio_started = True
                self.first_audio_ms = int((loop_time() - self._anchor) * 1000)
                await self._send_event(
                    "audio_start",
                    sample_rate=clip.sample_rate,
                    first_audio_ms=self.first_audio_ms,
                )

            await self._send_audio(piece)
            self.play_until = max(self.play_until, loop_time()) + len(piece) / bytes_per_second


def describe(speaker: TurnSpeaker) -> dict[str, Any]:
    """A snapshot for logs and tests."""
    return {
        "first_audio_ms": speaker.first_audio_ms,
        "error": speaker.error,
        "sentences": len(speaker.spoken),
    }
