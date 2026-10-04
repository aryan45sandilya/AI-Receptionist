"""Tests for voice activity detection.

Audio is synthesised, so these need no microphone, no network and no hardware --
the same reason the booking logic lives in ``domain/``. A tone stands in for
speech: the detector keys on energy, and a tone at speaking level is
indistinguishable from speech as far as it is concerned.
"""

from __future__ import annotations

import array
import math

import pytest

from tests.audio_helpers import hiss, silence, tone
from backend.app.voice.vad import (
    FRAME_BYTES,
    FRAME_MS,
    SAMPLE_RATE,
    Event,
    VadConfig,
    VoiceActivityDetector,
    frame_rms,
)


def events_of(detector: VoiceActivityDetector, pcm: bytes) -> list[Event]:
    return [event for event, _ in detector.feed(pcm)]


# --------------------------------------------------------------------------
# Energy measurement
# --------------------------------------------------------------------------

def test_silence_has_no_energy():
    assert frame_rms(silence(FRAME_MS)) == 0.0


def test_a_tone_has_energy_near_its_amplitude():
    # RMS of a sine is its amplitude over root two.
    assert frame_rms(tone(FRAME_MS, amplitude=0.5)) == pytest.approx(0.5 / math.sqrt(2), abs=0.02)


def test_empty_and_partial_frames_do_not_crash():
    assert frame_rms(b"") == 0.0
    assert frame_rms(b"\x01") == 0.0  # a single byte is not a whole sample


# --------------------------------------------------------------------------
# Detecting the start and end of speech
# --------------------------------------------------------------------------

def test_silence_alone_produces_nothing():
    """The reason this module exists: the transcription model answers silence
    with " Thank you.", so silence must never reach it."""
    detector = VoiceActivityDetector()
    assert events_of(detector, silence(2000)) == []


def test_room_tone_alone_produces_nothing():
    detector = VoiceActivityDetector()
    assert events_of(detector, hiss(3000)) == []


def test_speech_then_silence_yields_one_utterance():
    detector = VoiceActivityDetector()
    results = detector.feed(silence(200) + tone(700) + silence(1200))
    kinds = [event for event, _ in results]
    assert kinds == [Event.SPEECH_STARTED, Event.SPEECH_ENDED]
    audio = results[-1][1]
    assert audio is not None and len(audio) > 0


def test_utterance_is_not_closed_by_a_short_pause():
    """A pause between words must not cut the caller off mid-sentence."""
    detector = VoiceActivityDetector()
    kinds = events_of(detector, tone(400) + silence(300) + tone(400) + silence(1200))
    assert kinds == [Event.SPEECH_STARTED, Event.SPEECH_ENDED]


def test_two_utterances_separated_by_a_long_pause():
    detector = VoiceActivityDetector()
    kinds = events_of(
        detector, tone(400) + silence(1200) + tone(400) + silence(1200)
    )
    assert kinds == [
        Event.SPEECH_STARTED,
        Event.SPEECH_ENDED,
        Event.SPEECH_STARTED,
        Event.SPEECH_ENDED,
    ]


def test_a_brief_click_does_not_open_an_utterance():
    """One loud frame is a door, not a word."""
    detector = VoiceActivityDetector()
    assert events_of(detector, silence(200) + tone(40) + silence(1500)) == []


def test_a_very_short_burst_is_reported_but_not_transcribed():
    """Long enough to trip the start threshold, too short to be a word: the turn
    closes, but there is no audio worth sending."""
    config = VadConfig(min_utterance_ms=400)
    detector = VoiceActivityDetector(config=config)
    results = detector.feed(tone(150) + silence(1200))
    assert [e for e, _ in results] == [Event.SPEECH_STARTED, Event.SPEECH_ENDED]
    assert results[-1][1] is None


# --------------------------------------------------------------------------
# Audio handed back
# --------------------------------------------------------------------------

def test_utterance_includes_preroll_so_the_first_consonant_survives():
    """Speech is only recognised after start_frames, so without pre-roll the
    utterance begins already clipped."""
    detector = VoiceActivityDetector()
    speech_ms = 600
    results = detector.feed(silence(400) + tone(speech_ms) + silence(1200))
    audio = results[-1][1]
    bytes_per_ms = SAMPLE_RATE * 2 // 1000
    # At least the spoken part, since pre-roll only adds to it.
    assert len(audio) >= speech_ms * bytes_per_ms * 0.9


def test_trailing_silence_is_trimmed():
    """Transcribing the pause that ended the turn costs latency and buys nothing."""
    detector = VoiceActivityDetector()
    results = detector.feed(tone(500) + silence(2000))
    audio = results[-1][1]
    bytes_per_ms = SAMPLE_RATE * 2 // 1000
    # Well short of speech plus the full two seconds of silence.
    assert len(audio) < (500 + 900) * bytes_per_ms


def test_a_stuck_microphone_cannot_buffer_forever():
    config = VadConfig(max_utterance_ms=1000)
    detector = VoiceActivityDetector(config=config)
    results = detector.feed(tone(3000))
    assert Event.SPEECH_ENDED in [e for e, _ in results]


# --------------------------------------------------------------------------
# Chunking, state, and adapting to the room
# --------------------------------------------------------------------------

def test_audio_split_across_arbitrary_chunks_behaves_identically():
    """Network delivery sizes must not change what the detector hears."""
    pcm = silence(200) + tone(700) + silence(1200)

    whole = VoiceActivityDetector()
    expected = events_of(whole, pcm)

    piecemeal = VoiceActivityDetector()
    got: list[Event] = []
    step = 111  # deliberately not a frame multiple
    for i in range(0, len(pcm), step):
        got.extend(events_of(piecemeal, pcm[i : i + step]))
    assert got == expected


def test_partial_frames_are_held_until_complete():
    detector = VoiceActivityDetector()
    assert events_of(detector, tone(FRAME_MS)[: FRAME_BYTES // 2]) == []


def test_in_speech_reports_the_current_state():
    detector = VoiceActivityDetector()
    assert detector.in_speech is False
    detector.feed(tone(300))
    assert detector.in_speech is True
    detector.feed(silence(1200))
    assert detector.in_speech is False


def test_flush_closes_an_utterance_left_open():
    """The caller hung up mid-sentence; what they said should not be lost."""
    detector = VoiceActivityDetector()
    detector.feed(tone(600))
    assert detector.in_speech is True
    audio = detector.flush()
    assert audio is not None and len(audio) > 0
    assert detector.in_speech is False


def test_flush_with_nothing_open_returns_none():
    assert VoiceActivityDetector().flush() is None


def test_threshold_rises_in_a_noisy_room():
    """Room tone that would otherwise trip the detector pushes the bar up."""
    quiet = VoiceActivityDetector()
    quiet.feed(silence(1000))

    noisy = VoiceActivityDetector()
    noisy.feed(hiss(3000, amplitude=0.02))

    assert noisy.threshold() > quiet.threshold()


def test_a_fixed_threshold_can_be_forced():
    detector = VoiceActivityDetector(config=VadConfig(adapt_noise_floor=False))
    detector.feed(hiss(2000, amplitude=0.02))
    assert detector.threshold() == pytest.approx(VadConfig().speech_threshold)


def test_speech_is_still_detected_over_a_noisy_background():
    detector = VoiceActivityDetector()
    detector.feed(hiss(2000))  # learn the room first
    kinds = events_of(detector, tone(700, amplitude=0.3) + silence(1200))
    assert kinds == [Event.SPEECH_STARTED, Event.SPEECH_ENDED]


def test_reset_clears_a_part_heard_utterance():
    detector = VoiceActivityDetector()
    detector.feed(tone(400))
    detector.reset()
    assert detector.in_speech is False
    assert events_of(detector, silence(1200)) == []


# --------------------------------------------------------------------------
# Telephone audio, and the readings barge-in is built on
# --------------------------------------------------------------------------

PHONE = 8_000


def test_a_telephone_rate_detector_finds_an_utterance_in_eight_kilohertz_audio():
    """A call arrives at 8 kHz, so a frame is half as many bytes. The detector
    keys on energy, so only the frame size needed to change."""
    detector = VoiceActivityDetector(sample_rate=PHONE)
    pcm = silence(200, PHONE) + tone(700, rate=PHONE) + silence(1200, PHONE)
    kinds = [event for event, _ in detector.feed(pcm)]
    assert kinds == [Event.SPEECH_STARTED, Event.SPEECH_ENDED]


def test_a_telephone_rate_detector_still_ignores_silence_and_room_tone():
    detector = VoiceActivityDetector(sample_rate=PHONE)
    assert detector.feed(silence(2000, PHONE) + hiss(2000, rate=PHONE)) == []


def test_a_telephone_rate_utterance_is_the_right_length():
    detector = VoiceActivityDetector(sample_rate=PHONE)
    results = detector.feed(tone(600, rate=PHONE) + silence(1200, PHONE))
    audio = results[-1][1]
    bytes_per_ms = PHONE * 2 // 1000
    assert audio is not None
    assert 600 * bytes_per_ms * 0.9 <= len(audio) < (600 + 900) * bytes_per_ms


def test_speech_ms_counts_up_while_the_caller_keeps_talking():
    detector = VoiceActivityDetector()
    assert detector.speech_ms == 0
    detector.feed(tone(400))
    assert detector.in_speech
    # From the frame that first sounded like speech, including the 100 ms that
    # it took to be sure.
    assert 380 <= detector.speech_ms <= 420
    detector.feed(tone(200))
    assert 580 <= detector.speech_ms <= 620


def test_speech_ms_is_zero_before_speech_and_after_it_ends():
    detector = VoiceActivityDetector()
    detector.feed(silence(500))
    assert detector.speech_ms == 0
    detector.feed(tone(500) + silence(1200))
    assert detector.in_speech is False
    assert detector.speech_ms == 0


def test_a_short_burst_never_reaches_a_barge_in_length():
    """The reason barge-in measures duration: a cough or a click can trip the
    detector, but it is over long before it counts as the caller talking."""
    detector = VoiceActivityDetector()
    detector.feed(tone(160))
    assert detector.in_speech
    assert detector.speech_ms < 300


def test_quiet_ms_counts_the_silence_since_the_last_loud_frame():
    detector = VoiceActivityDetector()
    detector.feed(tone(500))
    assert detector.quiet_ms == 0
    detector.feed(silence(300))
    assert 280 <= detector.quiet_ms <= 320


def test_quiet_ms_starts_again_when_the_caller_resumes():
    detector = VoiceActivityDetector()
    detector.feed(tone(400) + silence(400) + tone(100))
    assert detector.quiet_ms == 0


def test_quiet_ms_is_zero_when_no_utterance_is_open():
    detector = VoiceActivityDetector()
    detector.feed(silence(1000))
    assert detector.quiet_ms == 0


def test_reset_clears_the_speech_clock():
    detector = VoiceActivityDetector()
    detector.feed(tone(400))
    detector.reset()
    assert detector.speech_ms == 0 and detector.quiet_ms == 0


def test_voiced_ms_counts_talking_not_the_silence_after_it():
    """A cough is 120 ms of sound and then nothing. Elapsed time since it began
    keeps growing through the silence; voiced time does not, which is what makes
    the difference between a cough and an interruption."""
    detector = VoiceActivityDetector()
    detector.feed(tone(120))
    assert detector.in_speech
    voiced = detector.voiced_ms
    assert 100 <= voiced <= 140

    detector.feed(silence(600))
    assert detector.in_speech, "still open: 800 ms of quiet has not passed"
    assert detector.speech_ms > 600, "elapsed time keeps counting"
    assert detector.voiced_ms == voiced, "talking time does not"


def test_voiced_ms_keeps_growing_while_the_caller_talks_with_short_pauses():
    detector = VoiceActivityDetector()
    detector.feed(tone(300) + silence(200) + tone(300))
    assert 760 <= detector.voiced_ms <= 820


def test_voiced_ms_is_zero_outside_an_utterance():
    detector = VoiceActivityDetector()
    assert detector.voiced_ms == 0
    detector.feed(tone(500) + silence(1200))
    assert detector.voiced_ms == 0
