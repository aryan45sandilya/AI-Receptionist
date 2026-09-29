"""Tests for the speaking pipeline: overlap, pacing, failure and cancellation.

Everything runs on a virtual clock (see virtual_time.py), so the numbers asserted
here are exact. "The second sentence began synthesising while the first was still
playing" is a statement about time, and it is checked as one.

No network and no sockets: the TTS service and the WebSocket are fakes that
record what they were asked to do and when.
"""

from __future__ import annotations

import asyncio

import pytest

from backend.app.voice.audio import Pcm
from backend.app.voice.pipeline import OUTPUT_LATENCY_MARGIN_S, TurnSpeaker, loop_time
from tests.virtual_time import run_virtual

RATE = 24_000
BYTES_PER_SECOND = RATE * 2


class FakeTTS:
    """Synthesis that takes ``delay`` seconds and yields ``seconds`` of audio."""

    def __init__(self, *, delay: float = 0.3, seconds: float = 2.0, fail: bool = False):
        self.delay = delay
        self.seconds = seconds
        self.fail = fail
        self.calls: list[tuple[float, str]] = []

    async def synthesize(self, text: str) -> Pcm:
        self.calls.append((loop_time(), text))
        await asyncio.sleep(self.delay)
        if self.fail:
            raise RuntimeError("model requires terms acceptance")
        samples = int(RATE * self.seconds)
        return Pcm(data=b"\x01\x00" * samples, sample_rate=RATE)


class Sink:
    """Stands in for the WebSocket: records events and audio with timestamps."""

    def __init__(self) -> None:
        self.events: list[tuple[float, str, dict]] = []
        self.audio: list[tuple[float, int]] = []

    async def send_event(self, kind: str, **fields) -> None:
        self.events.append((loop_time(), kind, fields))

    async def send_audio(self, data: bytes) -> None:
        self.audio.append((loop_time(), len(data)))

    def kinds(self) -> list[str]:
        return [kind for _, kind, _ in self.events]

    @property
    def total_audio_seconds(self) -> float:
        return sum(n for _, n in self.audio) / BYTES_PER_SECOND


def make(tts: FakeTTS, sink: Sink, *, anchor: float | None = 0.0, **kw) -> TurnSpeaker:
    speaker = TurnSpeaker(
        tts=tts, send_event=sink.send_event, send_audio=sink.send_audio, anchor=anchor, **kw
    )
    speaker.start()
    return speaker


TWO = "Sure, let me check that for you. I have five o'clock or six thirty available. "


# --------------------------------------------------------------------------
# Latency measurement
# --------------------------------------------------------------------------

def test_first_audio_is_measured_from_the_anchor_not_from_construction():
    async def scenario():
        tts, sink = FakeTTS(delay=0.3), Sink()
        # The caller stopped talking 800 ms before this object existed -- the
        # endpointing wait. That time is theirs, and must be counted.
        anchor = loop_time() - 0.8
        speaker = make(tts, sink, anchor=anchor)
        speaker.on_text(TWO)
        await speaker.finish()
        return speaker

    speaker = run_virtual(scenario())
    assert speaker.first_audio_ms == 1100  # 800 ms already waited + 300 ms to synthesise


def test_first_audio_does_not_wait_for_the_rest_of_the_reply():
    """The reason for streaming at all: audio starts after one sentence, not
    after the whole reply."""
    async def scenario():
        tts, sink = FakeTTS(delay=0.3), Sink()
        speaker = make(tts, sink, anchor=loop_time())
        speaker.on_text("Sure, let me check that for you. ")  # sentence one is complete
        await asyncio.sleep(2.0)  # ...the model is still writing for two more seconds
        speaker.on_text("I have five o'clock or six thirty available. ")
        await speaker.finish()
        return speaker

    speaker = run_virtual(scenario())
    assert speaker.first_audio_ms == 300


# --------------------------------------------------------------------------
# Overlap and gaplessness
# --------------------------------------------------------------------------

def test_next_sentence_is_synthesised_while_the_current_one_plays():
    async def scenario():
        tts, sink = FakeTTS(delay=0.3, seconds=2.0), Sink()
        speaker = make(tts, sink)
        speaker.on_text(TWO)
        await speaker.finish()
        return tts

    tts = run_virtual(scenario())
    first, second = (when for when, _ in tts.calls)
    # Sentence one is 2 s of audio, but sentence two was requested at 0.3 s --
    # the moment sentence one finished synthesising -- not at 2.3 s.
    assert first == 0.0
    assert second == pytest.approx(0.3)


def test_there_is_no_gap_between_sentences():
    """Total playing time is the sum of the clips: sentence two was ready before
    sentence one ended, so nothing waited on a synthesis round trip."""
    async def scenario():
        tts, sink = FakeTTS(delay=0.3, seconds=2.0), Sink()
        speaker = make(tts, sink)
        speaker.on_text(TWO)
        await speaker.finish()
        return speaker, sink

    speaker, sink = run_virtual(scenario())
    first_audio_at = 0.3
    assert speaker.play_until == pytest.approx(first_audio_at + 4.0, abs=0.05)


def test_sentences_are_spoken_in_the_order_they_were_written():
    async def scenario():
        tts, sink = FakeTTS(), Sink()
        speaker = make(tts, sink)
        speaker.on_text("First sentence is right here. Second one follows it. ")
        speaker.on_text("Third comes last of all.")
        await speaker.finish()
        return tts, speaker

    tts, speaker = run_virtual(scenario())
    assert [text for _, text in tts.calls] == [
        "First sentence is right here.",
        "Second one follows it.",
        "Third comes last of all.",
    ]
    assert speaker.spoken_text == "First sentence is right here. Second one follows it. Third comes last of all."


# --------------------------------------------------------------------------
# Pacing
# --------------------------------------------------------------------------

def test_audio_is_never_sent_far_ahead_of_real_time():
    """Bounded lead: the server's idea of what has played stays close to the
    listener's, which is what lets barge-in stop where the caller expects."""
    lead, chunk = 0.25, 0.04

    async def scenario():
        tts, sink = FakeTTS(delay=0.1, seconds=3.0), Sink()
        speaker = make(tts, sink, lead_s=lead)
        speaker.on_text("A single reasonably long sentence to say out loud here.")
        await speaker.finish()
        return sink

    sink = run_virtual(scenario())
    start = sink.audio[0][0]
    sent = 0.0
    for at, size in sink.audio:
        sent += size / BYTES_PER_SECOND
        # Audio handed over, minus audio the listener has had time to play.
        assert sent - (at - start) <= lead + chunk + 1e-6


def test_playback_takes_as_long_as_the_audio_lasts():
    async def scenario():
        tts, sink = FakeTTS(delay=0.1, seconds=3.0), Sink()
        speaker = make(tts, sink, lead_s=0.25)
        speaker.on_text("A single reasonably long sentence to say out loud here.")
        await speaker.finish()
        return sink

    sink = run_virtual(scenario())
    started, last = sink.audio[0][0], sink.audio[-1][0]
    # The last frame is released one lead before the audio ends.
    assert last - started == pytest.approx(3.0 - 0.25 - 0.04, abs=0.06)


def test_frames_are_whole_samples_at_awkward_sample_rates():
    """An odd byte count would cut a 16-bit sample in half and click."""
    class Odd:
        async def synthesize(self, text: str) -> Pcm:
            return Pcm(data=b"\x01\x00" * 12345, sample_rate=12345)

    async def scenario():
        sink = Sink()
        speaker = TurnSpeaker(
            tts=Odd(), send_event=sink.send_event, send_audio=sink.send_audio, anchor=0.0
        )
        speaker.start()
        speaker.on_text("One sentence, long enough to be said alone.")
        await speaker.finish()
        return sink

    sink = run_virtual(scenario())
    assert sink.audio and all(size % 2 == 0 for _, size in sink.audio)
    assert sum(size for _, size in sink.audio) == 12345 * 2


def test_drain_waits_for_the_tail_of_playback():
    """finish() returns when the last byte is sent; the listener is still hearing
    up to a lead's worth, and the gate must stay shut through it."""
    async def scenario():
        tts, sink = FakeTTS(delay=0.1, seconds=2.0), Sink()
        speaker = make(tts, sink)
        speaker.on_text("A single reasonably long sentence to say out loud here.")
        await speaker.finish()
        sent_at = loop_time()
        await speaker.drain_playback()
        return sent_at, loop_time(), speaker

    sent_at, drained_at, speaker = run_virtual(scenario())
    assert drained_at > sent_at
    assert drained_at == pytest.approx(speaker.play_until + OUTPUT_LATENCY_MARGIN_S)


def test_drain_returns_immediately_when_nothing_was_said():
    async def scenario():
        speaker = make(FakeTTS(), Sink())
        await speaker.finish()
        before = loop_time()
        await speaker.drain_playback()
        return loop_time() - before

    assert run_virtual(scenario()) <= OUTPUT_LATENCY_MARGIN_S


# --------------------------------------------------------------------------
# Events
# --------------------------------------------------------------------------

def test_events_bracket_the_audio_exactly_once():
    async def scenario():
        tts, sink = FakeTTS(), Sink()
        speaker = make(tts, sink)
        speaker.on_text(TWO)
        await speaker.finish()
        return sink

    kinds = run_virtual(scenario()).kinds()
    assert kinds.count("audio_start") == 1
    assert kinds.count("audio_end") == 1
    assert kinds.index("audio_start") < kinds.index("audio_end")


def test_audio_start_announces_the_sample_rate():
    async def scenario():
        tts, sink = FakeTTS(), Sink()
        speaker = make(tts, sink)
        speaker.on_text(TWO)
        await speaker.finish()
        return sink

    start = next(f for _, k, f in run_virtual(scenario()).events if k == "audio_start")
    assert start["sample_rate"] == RATE


def test_silence_produces_no_events_at_all():
    async def scenario():
        sink = Sink()
        speaker = make(FakeTTS(), sink)
        await speaker.finish()
        return sink, speaker

    sink, speaker = run_virtual(scenario())
    assert sink.events == [] and sink.audio == []
    assert speaker.first_audio_ms is None


# --------------------------------------------------------------------------
# Fixed messages
# --------------------------------------------------------------------------

def test_a_fixed_message_follows_whatever_was_streamed():
    async def scenario():
        tts, sink = FakeTTS(), Sink()
        speaker = make(tts, sink)
        speaker.on_text("Let me check that for you. One moment")  # unfinished
        speaker.say("Sorry, I lost that for a moment.")
        await speaker.finish()
        return tts

    tts = run_virtual(scenario())
    # The unfinished fragment is released as it stood, then the fixed message
    # follows as its own utterance: the caller hears things in the order they
    # happened, and unrelated text is never fused into one sentence.
    assert [text for _, text in tts.calls] == [
        "Let me check that for you.",
        "One moment",
        "Sorry, I lost that for a moment.",
    ]


def test_text_after_finish_is_ignored():
    async def scenario():
        tts, sink = FakeTTS(), Sink()
        speaker = make(tts, sink)
        speaker.on_text("A complete sentence, said before the end. ")
        await speaker.finish()
        speaker.on_text("Too late to be spoken at all. ")
        speaker.say("Neither will this one be.")
        return speaker

    assert len(run_virtual(scenario()).spoken) == 1


# --------------------------------------------------------------------------
# Failure
# --------------------------------------------------------------------------

def test_a_failing_voice_is_reported_once_and_never_hangs_the_turn():
    """The most likely real failure: the model's terms were never accepted, so
    every request fails identically. One report, no retry storm, no deadlock."""
    async def scenario():
        tts, sink = FakeTTS(fail=True), Sink()
        speaker = make(tts, sink)
        speaker.on_text("First sentence here is long enough. Second sentence is also long. ")
        speaker.on_text("Third sentence closes the reply out.")
        await speaker.finish()
        return tts, sink, speaker

    tts, sink, speaker = run_virtual(scenario())
    assert sink.kinds() == ["tts_error"]
    assert "terms acceptance" in sink.events[0][2]["message"]
    assert len(tts.calls) == 1, "should stop asking a service that just said no"
    assert speaker.first_audio_ms is None
    assert speaker.error is not None
    assert sink.audio == []


# --------------------------------------------------------------------------
# Cancellation (the mechanism Phase 3's barge-in uses)
# --------------------------------------------------------------------------

def test_cancel_stops_audio_at_once():
    async def scenario():
        tts, sink = FakeTTS(delay=0.1, seconds=5.0), Sink()
        speaker = make(tts, sink)
        speaker.on_text("A long reply that would take several seconds to say aloud. ")
        speaker.on_text("And then a second sentence that follows straight after it.")
        await asyncio.sleep(1.0)
        await speaker.cancel()
        sent_at_cancel = len(sink.audio)
        await asyncio.sleep(10.0)
        return sent_at_cancel, len(sink.audio), sink

    at_cancel, later, sink = run_virtual(scenario())
    assert at_cancel > 0, "audio should have been flowing before the cancel"
    assert later == at_cancel, "nothing may be sent after cancel"
    assert "audio_end" not in sink.kinds()


def test_cancel_before_anything_is_said_is_harmless():
    async def scenario():
        speaker = make(FakeTTS(), Sink())
        await speaker.cancel()
        await speaker.cancel()  # and again

    run_virtual(scenario())


def test_cancel_resets_the_playback_clock_so_the_gate_can_reopen():
    async def scenario():
        tts, sink = FakeTTS(delay=0.1, seconds=5.0), Sink()
        speaker = make(tts, sink)
        speaker.on_text("A long reply that would take several seconds to say aloud.")
        await asyncio.sleep(1.0)
        await speaker.cancel()
        before = loop_time()
        await speaker.drain_playback()
        return loop_time() - before

    # Without the reset, drain would wait out the audio that was never played.
    assert run_virtual(scenario()) <= OUTPUT_LATENCY_MARGIN_S + 1e-9
