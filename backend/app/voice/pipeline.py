"""Turning a streaming reply into paced audio: the speaking half of a voice turn.

The model writes text a few tokens at a time; the synthesis service turns a
sentence into audio; the caller's device plays PCM as it arrives. This module is
the seam between them, and it does four things that decide how the call feels:

1.  **Overlap.** The first sentence is synthesised the moment it is complete,
    while the model is still writing the second. Time to first audio is then the
    time to *one sentence*, not the time to the whole reply. With a streaming
    service it is shorter still: the first chunk of the sentence, not all of it.

2.  **Synthesis and playback are separate workers.** Sentence two is being
    synthesised while sentence one is still playing, so the gap between them is
    zero rather than one synthesis round trip.

3.  **Pacing.** Audio is released at real-time speed with a short lead, not
    dumped onto the socket as fast as it can be made. That is not for the
    network's sake: it keeps the server's idea of "what has been played" within
    a quarter second of the listener's, which is what makes stopping on barge-in
    land where the caller expects, and lets the server say truthfully what the
    caller heard.

4.  **Stopping.** When the caller talks over the agent, ``mute`` ends the audio
    at once and ``cancel`` unwinds the workers. At most the lead's worth of audio
    is already on the wire -- the transport tells the listener's device to throw
    it away -- and ``heard_text`` says which words got out, so the model's idea
    of the conversation matches the caller's.

Deliberately knows nothing about WebSockets, Deepgram or the database: it is
handed callbacks. Its timing is all ``asyncio`` loop time and ``asyncio.sleep``,
so a test can run it under a virtual clock and assert exact numbers.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from contextlib import aclosing
from typing import Any

from .audio import Pcm, resample
from .codec import decimate
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


def match_rate(clip: Pcm, rate: int | None) -> Pcm:
    """Bring a clip to the rate the listener's line carries, if it has to be one.

    ``None`` means the listener takes any rate (a browser resamples for its own
    speakers). A telephone line is 8 kHz, so a 24 kHz voice is averaged down --
    by the whole ratio where there is one, since that filters as it goes --
    and anything else falls back to plain interpolation.
    """
    if rate is None or clip.sample_rate == rate:
        return clip
    if clip.sample_rate > rate and clip.sample_rate % rate == 0 and clip.channels == 1:
        return decimate(clip, rate)
    return resample(clip, rate)


class TurnSpeaker:
    """Speak one agent reply, sentence by sentence, as it is written.

    Usage, from the code running the model::

        speaker = TurnSpeaker(tts=tts, send_event=..., send_audio=..., anchor=t0)
        speaker.start()
        result = await runner.run_turn(text, on_text=speaker.on_text)
        await speaker.finish()
        await speaker.drain_playback()

    and, if the caller starts talking::

        speaker.mute()              # synchronous: not one more frame is sent
        await speaker.cancel()
        heard = speaker.heard_text()
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
        output_rate: int | None = None,
    ) -> None:
        self._tts = tts
        self._send_event = send_event
        self._send_audio = send_audio
        self._lead_s = lead_s
        self._chunk_ms = chunk_ms
        self._sentences = SentenceStream(max_chars=max_chars, eager_first=eager_first)
        self._max_chars = max_chars
        self._output_rate = output_rate

        # ``anchor`` is when the caller stopped talking, on loop time. first
        # audio is measured from there, because that is when they started
        # waiting -- not from when this object happened to be built.
        self._anchor = anchor if anchor is not None else loop_time()

        self._text_q: asyncio.Queue[tuple[int, str] | None] = asyncio.Queue()
        # Audio chunks, each tagged with the sentence it belongs to; None ends it.
        self._clip_q: asyncio.Queue[tuple[int, Pcm] | None] = asyncio.Queue()
        self._tasks: list[asyncio.Task[None]] = []
        self._audio_started = False
        self._finished = False
        self._muted = False
        # Index of the last sentence any audio of which was sent to the listener.
        self._last_voiced = -1

        # First sound of any kind, then split by what it was: a stopgap said while a
        # tool ran, and the answer itself. The caller's line stops being dead at
        # ``first_audio_ms``; what they were waiting *for* arrives at
        # ``answer_audio_ms``. They are the same unless there was a filler.
        self.first_audio_ms: int | None = None
        self.filler_audio_ms: int | None = None
        self.answer_audio_ms: int | None = None
        self._fillers: set[int] = set()
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

    def say_filler(self, text: str) -> bool:
        """Say a short stopgap, if nothing has been said yet and the reply is open.

        For the silence while a tool runs: the model has asked for one and will
        write nothing until it has the result, and a phone line that goes quiet for
        three seconds feels dead. Refused (False) if the model has already
        said something -- it is not silent, and a filler on top would be noise.
        Counted separately in the timings so it cannot flatter the answer's.
        """
        if self._finished or self.spoken or self._sentences.has_pending:
            return False
        self._fillers.add(len(self.spoken))
        self._enqueue([text])
        return True

    def _enqueue(self, sentences: list[str]) -> None:
        for sentence in sentences:
            self._text_q.put_nowait((len(self.spoken), sentence))
            self.spoken.append(sentence)

    @property
    def spoken_text(self) -> str:
        return " ".join(self.spoken)

    def heard_text(self, *, fillers: bool = True) -> str:
        """The words the caller actually got to hear.

        Everything handed to synthesis is in ``spoken``, but a reply that was cut
        off never played its later sentences, and the one it was cut in the
        middle of played only part. A sentence counts as heard once any of its
        audio left; the last of them is marked as cut short if the reply did not
        run to its end.

        ``fillers=False`` leaves out the stopgaps ("one moment"): they are said
        by the voice layer, not written by the model, and telling the model it
        said them would only confuse it. The record of what the caller heard
        keeps them; what goes back into the model's history does not.
        """
        if self._last_voiced < 0:
            return ""
        heard = [
            sentence
            for index, sentence in enumerate(self.spoken[: self._last_voiced + 1])
            if fillers or index not in self._fillers
        ]
        text = " ".join(heard)
        cut_short = self._last_voiced < len(self.spoken) - 1 or not self._finished_cleanly
        return f"{text} ..." if cut_short and text else text

    @property
    def _finished_cleanly(self) -> bool:
        return not self._muted

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
        if self._audio_started and not self._muted:
            await self._send_event("audio_end")

    async def drain_playback(self) -> None:
        """Wait until the listener has, in all likelihood, heard everything.

        ``finish`` returns when the last byte is *sent*; the browser is still
        playing for up to a quarter second after that, plus whatever its own
        output buffer holds. Whatever waits on the end of the agent's speech has
        to wait through that tail too.
        """
        remaining = self.play_until + OUTPUT_LATENCY_MARGIN_S - loop_time()
        if remaining > 0:
            await asyncio.sleep(remaining)

    @property
    def audible(self) -> bool:
        """Is there still something to interrupt: a reply being written or
        synthesised, or one whose last audio the listener may still be hearing?

        False once the reply is over and its tail has played out. A turn goes on
        for a while after that -- its record is being written -- but the caller
        is no longer interrupting anyone by speaking.
        """
        if not self._finished:
            return True
        return loop_time() < self.play_until + OUTPUT_LATENCY_MARGIN_S

    def mute(self) -> None:
        """Send nothing more, starting now. Synchronous on purpose.

        ``cancel`` has to wait for the workers to unwind, and the synthesis
        service to be told to stop; the caller has already begun talking and every
        millisecond of the agent's voice after that is the thing being fixed. So
        the sending stops first, here, and the clean-up follows.
        """
        self._muted = True
        self._finished = True

    @property
    def muted(self) -> bool:
        return self._muted

    async def cancel(self) -> None:
        """Stop immediately: the caller has started talking.

        Audio already handed to the socket cannot be recalled. At most
        ``lead_s`` of it exists, and the transport tells the listener's device to
        discard its queue.
        """
        self.mute()
        for task in self._tasks:
            task.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()
        self.play_until = loop_time()
        await self._tts.cancel()

    # ---------------------------------------------------------------- workers

    async def _synth_worker(self) -> None:
        failed = False
        while True:
            item = await self._text_q.get()
            if item is None:
                break
            index, sentence = item
            if failed:
                # Every later sentence would fail the same way (an unaccepted
                # licence, a bad key), so stop spending requests on them -- but
                # keep draining, so nothing upstream ever blocks on this queue.
                continue
            try:
                # aclosing: if this task is cancelled mid-sentence the stream is
                # closed *now*, which is what tells the service to stop, rather
                # than whenever the generator happens to be collected.
                async with aclosing(self._tts.stream(sentence)) as chunks:
                    async for chunk in chunks:
                        await self._clip_q.put((index, chunk))
            except Exception as exc:  # noqa: BLE001 - a silent voice must not end the call
                failed = True
                self.error = f"{type(exc).__name__}: {exc}"[:300]
                await self._send_event("tts_error", message=self.error)
        await self._clip_q.put(None)

    async def _play_worker(self) -> None:
        while (item := await self._clip_q.get()) is not None:
            index, clip = item
            await self._stream(index, match_rate(clip, self._output_rate))

    async def _stream(self, index: int, clip: Pcm) -> None:
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
            if self._muted:
                return

            if not self._audio_started:
                self._audio_started = True
                self.first_audio_ms = int((loop_time() - self._anchor) * 1000)
                await self._send_event(
                    "audio_start",
                    sample_rate=clip.sample_rate,
                    first_audio_ms=self.first_audio_ms,
                )
                if self._muted:
                    return

            await self._send_audio(piece)
            self._last_voiced = max(self._last_voiced, index)
            self.play_until = max(self.play_until, loop_time()) + len(piece) / bytes_per_second

            heard_at = int((loop_time() - self._anchor) * 1000)
            if index in self._fillers:
                if self.filler_audio_ms is None:
                    self.filler_audio_ms = heard_at
            elif self.answer_audio_ms is None:
                self.answer_audio_ms = heard_at


def describe(speaker: TurnSpeaker) -> dict[str, Any]:
    """A snapshot for logs and tests."""
    return {
        "first_audio_ms": speaker.first_audio_ms,
        "error": speaker.error,
        "sentences": len(speaker.spoken),
    }
