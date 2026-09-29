"""Tests for audio containers, silence rejection and sentence splitting.

No network: these cover the parts that decide what gets sent to a speech
service and what comes back, which is where the bugs that are hard to see
actually live. A wrong sample rate does not raise -- it just makes the agent
sound wrong.
"""

from __future__ import annotations

import struct

import pytest

from backend.app.voice.audio import Pcm, from_wav, to_mono, to_wav, wav_header
from backend.app.voice.speech import (
    ORPHEUS_MAX_CHARS,
    SentenceStream,
    looks_like_silence,
    split_sentences,
)


def pcm(ms: int = 100, rate: int = 16_000, channels: int = 1) -> Pcm:
    frames = rate * ms // 1000
    return Pcm(data=b"\x01\x02" * frames * channels, sample_rate=rate, channels=channels)


# ---------------------------------------------------------------------------
# WAV round trip
# ---------------------------------------------------------------------------

def test_wav_round_trip_preserves_audio_and_rate():
    original = pcm(250, rate=24_000)
    restored = from_wav(to_wav(original))
    assert restored.data == original.data
    assert restored.sample_rate == 24_000
    assert restored.channels == 1


def test_header_is_the_canonical_44_bytes():
    assert len(wav_header(sample_rate=16_000, channels=1, data_bytes=0)) == 44


def test_rate_is_read_from_the_file_not_assumed():
    """A voice's output rate is a property of the voice. Assuming 16 kHz for a
    24 kHz clip plays it slow and deep; the reverse makes it a chipmunk."""
    assert from_wav(to_wav(pcm(rate=44_100))).sample_rate == 44_100


def test_extra_chunks_before_the_audio_are_skipped():
    """Services insert LIST or fact chunks. A reader that slices at byte 44
    would return metadata and call it sound."""
    body = pcm(50)
    extra = b"LIST" + struct.pack("<I", 10) + b"INFOhello "
    head = wav_header(sample_rate=body.sample_rate, channels=1, data_bytes=len(body.data))
    # Rebuild with the extra chunk between 'fmt ' and 'data'.
    doctored = head[:36] + extra + head[36:] + body.data
    doctored = b"RIFF" + struct.pack("<I", len(doctored) - 8) + doctored[8:]
    assert from_wav(doctored).data == body.data


def test_an_odd_sized_chunk_is_padded():
    body = pcm(20)
    extra = b"fact" + struct.pack("<I", 3) + b"abc" + b"\x00"  # 3 bytes + pad
    head = wav_header(sample_rate=body.sample_rate, channels=1, data_bytes=len(body.data))
    doctored = head[:36] + extra + head[36:] + body.data
    doctored = b"RIFF" + struct.pack("<I", len(doctored) - 8) + doctored[8:]
    assert from_wav(doctored).data == body.data


@pytest.mark.parametrize("payload", [b"", b"NOTRIFF", b"RIFF____NOTWAVE"])
def test_rubbish_is_rejected(payload):
    with pytest.raises(ValueError):
        from_wav(payload)


def test_missing_data_chunk_is_rejected():
    with pytest.raises(ValueError):
        from_wav(wav_header(sample_rate=16_000, channels=1, data_bytes=0)[:36])


def test_duration_is_reported_in_milliseconds():
    assert pcm(500).duration_ms == pytest.approx(500, abs=2)


def test_stereo_is_mixed_to_mono():
    stereo = Pcm(
        data=struct.pack("<hhhh", 100, 200, 300, 400), sample_rate=16_000, channels=2
    )
    mono = to_mono(stereo)
    assert mono.channels == 1
    assert struct.unpack("<hh", mono.data) == (150, 350)


def test_mono_passes_through_untouched():
    original = pcm()
    assert to_mono(original) is original


# ---------------------------------------------------------------------------
# Silence and hallucination rejection
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "text",
    ["", "   ", "Thank you.", "thank you", "THANKS FOR WATCHING!", "you", ".", "Bye.",
     "[BLANK_AUDIO]", "[Music]"],
)
def test_known_hallucinations_are_treated_as_silence(text):
    """The transcription model was trained on subtitled video, so silence comes
    back as subtitle filler. Measured: half a second of digital zero returns
    " Thank you." -- which is plausible enough that the agent would answer it."""
    assert looks_like_silence(text) is True


@pytest.mark.parametrize(
    "text",
    [
        "I need a cleaning next Tuesday",
        "yes",
        "no thanks, thank you though",
        "Thank you, can I book for Friday?",
        "six thirty",
    ],
)
def test_real_speech_is_not_discarded(text):
    assert looks_like_silence(text) is False


# ---------------------------------------------------------------------------
# Sentence splitting for synthesis
# ---------------------------------------------------------------------------

def test_a_reply_splits_into_sentences():
    parts = split_sentences(
        "I have five o'clock or six thirty. Which would you prefer?"
    )
    assert parts == ["I have five o'clock or six thirty.", "Which would you prefer?"]


def test_a_single_sentence_stays_whole():
    assert split_sentences("Sunrise Dental, how can I help?") == [
        "Sunrise Dental, how can I help?"
    ]


def test_empty_text_produces_nothing():
    assert split_sentences("") == []
    assert split_sentences("   \n  ") == []


def test_whitespace_is_normalised():
    """Newlines and runs of spaces collapse; the split still happens.

    Both halves are kept comfortably above the merge threshold, since a short
    fragment would legitimately be joined to its neighbour instead.
    """
    messy = "Your appointment is confirmed.\n\n   Is there anything else I can help with?"
    assert split_sentences(messy) == [
        "Your appointment is confirmed.",
        "Is there anything else I can help with?",
    ]


def test_a_short_fragment_is_merged_rather_than_spoken_alone():
    """"Booked." on its own synthesises as a clipped bark."""
    parts = split_sentences("Booked. Your reference is GK48HX.")
    assert len(parts) == 1
    assert parts[0].startswith("Booked.")


def test_a_very_long_sentence_is_capped():
    """One runaway reply must not stall playback while it synthesises."""
    long = "We have " + ", ".join(f"slot number {i}" for i in range(80)) + "."
    parts = split_sentences(long, max_chars=100)
    assert len(parts) > 1
    assert all(len(p) <= 100 for p in parts)


def test_splitting_preserves_every_word():
    text = "First sentence here. Second one follows. And a third to finish."
    assert " ".join(split_sentences(text)).split() == text.split()


def test_decimal_numbers_do_not_split_the_sentence():
    parts = split_sentences("The cost is 1500.50 rupees for the cleaning today.")
    assert len(parts) == 1


# ---------------------------------------------------------------------------
# Incremental splitting: what lets speech start before the reply is finished
# ---------------------------------------------------------------------------

REPLY = (
    "Sure, let me check that for you. I have Tuesday the sixth at five o'clock, "
    "or six thirty. Which would you prefer? Dr. Rao is free at both."
)


def stream_in_chunks(text: str, size: int) -> list[str]:
    stream = SentenceStream()
    out: list[str] = []
    for i in range(0, len(text), size):
        out.extend(stream.feed(text[i : i + size]))
    out.extend(stream.flush())
    return out


@pytest.mark.parametrize("size", [1, 2, 3, 7, 13, 50, 1000])
def test_streaming_gives_the_same_sentences_however_the_text_is_chunked(size):
    """Tokens arrive in whatever pieces the model emits. Where the boundaries
    fall must never change what is spoken."""
    assert stream_in_chunks(REPLY, size) == split_sentences(REPLY)


def test_the_first_sentence_is_released_before_the_reply_is_finished():
    """The whole point: playback can start while the model is still writing."""
    stream = SentenceStream()
    released: list[str] = []
    for word in REPLY.split(" "):
        released.extend(stream.feed(word + " "))
        if released:
            break
    assert released == ["Sure, let me check that for you."]
    # ...and it came out long before the last word.
    assert word != REPLY.split(" ")[-1]


def test_a_sentence_is_not_released_until_whitespace_confirms_it_ended():
    """"1500." might still become "1500.50", so the full stop alone is not proof."""
    stream = SentenceStream()
    assert stream.feed("The fee is 1500.") == []
    assert stream.feed("50 rupees today. Anything else?") == ["The fee is 1500.50 rupees today."]
    assert stream.flush() == ["Anything else?"]


def test_the_final_sentence_needs_flush():
    stream = SentenceStream()
    assert stream.feed("Your booking is confirmed for Tuesday.") == []
    assert stream.flush() == ["Your booking is confirmed for Tuesday."]


def test_flush_releases_a_held_short_fragment_rather_than_losing_it():
    stream = SentenceStream()
    assert stream.feed("Booked. ") == []  # too short to say alone
    assert stream.flush() == ["Booked."]


def test_a_stream_with_no_punctuation_is_cut_rather_than_waited_on():
    """A model that never writes a full stop must not leave the caller in
    silence: past one clip's worth, the text is cut on a space."""
    stream = SentenceStream(max_chars=60)
    words = "we have several openings on tuesday and wednesday and thursday " * 3
    out = stream.feed(words)
    assert out, "expected an early cut instead of waiting for punctuation"
    assert all(len(piece) <= 60 for piece in out)


def test_every_piece_respects_the_api_length_limit():
    """Groq rejects a request over 200 characters outright, so this is a hard
    guarantee, not a preference."""
    long_reply = "We have " + ", ".join(f"slot number {i} at some time" for i in range(60)) + "."
    pieces = split_sentences(long_reply)
    assert len(pieces) > 1
    assert all(len(piece) <= ORPHEUS_MAX_CHARS for piece in pieces)


def test_an_unbroken_run_longer_than_the_limit_is_still_cut():
    pieces = split_sentences("x" * 450)
    assert all(len(piece) <= ORPHEUS_MAX_CHARS for piece in pieces)
    assert "".join(pieces) == "x" * 450


def test_streamed_text_keeps_every_word_in_order():
    words = REPLY.split()
    assert " ".join(stream_in_chunks(REPLY, 5)).split() == words


def test_a_stream_can_be_reused_after_flush():
    stream = SentenceStream()
    stream.feed("First reply is here. ")
    stream.flush()
    assert stream.feed("Second reply is here.") == []
    assert stream.flush() == ["Second reply is here."]


def test_limits_that_cannot_fit_a_merged_fragment_are_rejected():
    with pytest.raises(ValueError):
        SentenceStream(max_chars=20, min_chars=16)


# ---------------------------------------------------------------------------
# Eager first chunk: audio starts sooner, at the cost of one extra clip
# ---------------------------------------------------------------------------

LONG_FIRST = (
    "We have three openings on Tuesday the sixth in the evening: "
    "5:30 pm, 5:45 pm, or 6:00 pm with Dr Rao. Which would you like?"
)


def test_the_first_chunk_may_end_at_a_colon_or_comma_once_long_enough():
    """One sentence, but the caller need not wait for the whole of it."""
    eager = split_sentences(LONG_FIRST, eager_first=True)
    assert eager[0] == "We have three openings on Tuesday the sixth in the evening:"
    assert len(eager[0]) < len(split_sentences(LONG_FIRST)[0])


def test_without_the_option_nothing_changes():
    assert split_sentences(LONG_FIRST) == [
        "We have three openings on Tuesday the sixth in the evening: "
        "5:30 pm, 5:45 pm, or 6:00 pm with Dr Rao.",
        "Which would you like?",
    ]


def test_a_short_lead_in_is_not_cut_off_on_its_own():
    """"Sure," alone is a clipped fragment. The cut waits until there is enough
    to stand as a phrase, and here that is the full stop."""
    assert split_sentences("Sure, let me check that for you. One moment please.", eager_first=True) == [
        "Sure, let me check that for you.",
        "One moment please.",
    ]


def test_only_the_first_chunk_is_cut_early():
    """Once audio is playing there is time to synthesise the rest properly, and
    whole sentences sound better than clauses."""
    text = (
        "We have three openings on Tuesday the sixth in the evening: five thirty, "
        "or a quarter to six, or six o'clock with Dr Rao, all of which are free. "
        "Which of those would you prefer, or would another day be better for you?"
    )
    pieces = split_sentences(text, eager_first=True)
    assert pieces[0].endswith(":")
    # The second sentence is left whole even though it contains commas.
    assert "Which of those would you prefer, or would another day be better for you?" in pieces


def test_a_time_of_day_is_never_split_at_its_colon():
    """"5:30" has a colon but no space after it, so it is not a boundary."""
    assert split_sentences("Your appointment is at 5:30 pm on Tuesday the sixth.", eager_first=True) == [
        "Your appointment is at 5:30 pm on Tuesday the sixth."
    ]


@pytest.mark.parametrize("size", [1, 2, 5, 13, 1000])
def test_eager_streaming_is_independent_of_chunking(size):
    def stream(text: str) -> list[str]:
        s = SentenceStream(eager_first=True)
        out: list[str] = []
        for i in range(0, len(text), size):
            out.extend(s.feed(text[i : i + size]))
        return [*out, *s.flush()]

    assert stream(LONG_FIRST) == split_sentences(LONG_FIRST, eager_first=True)


def test_the_eager_option_keeps_every_word_in_order():
    pieces = split_sentences(LONG_FIRST, eager_first=True)
    assert " ".join(pieces).split() == LONG_FIRST.split()
