"""Tests for the telephone audio format.

Mu-law is lossy by design, so a round trip is checked against the error the
format promises rather than for equality; the fixed points (silence, the loudest
values) are checked exactly, because those are the ones an off-by-one in the
tables would corrupt.
"""

from __future__ import annotations

import array
import math

import pytest

from backend.app.voice.audio import Pcm
from backend.app.voice.codec import decimate, ulaw_decode, ulaw_encode


def pcm_of(*samples: int) -> bytes:
    return array.array("h", samples).tobytes()


def samples_of(pcm: bytes) -> list[int]:
    out = array.array("h")
    out.frombytes(pcm)
    return list(out)


# --------------------------------------------------------------------------
# Fixed points of G.711
# --------------------------------------------------------------------------

def test_silence_encodes_to_the_idle_byte():
    # 0xFF is what an idle telephone line carries.
    assert ulaw_encode(pcm_of(0)) == b"\xff"


def test_the_idle_byte_decodes_to_silence():
    assert samples_of(ulaw_decode(b"\xff")) == [0]


def test_the_extremes_map_to_the_extreme_codes():
    assert ulaw_encode(pcm_of(32767)) == b"\x80"
    assert ulaw_encode(pcm_of(-32768)) == b"\x00"


def test_decode_of_the_extreme_codes_is_the_loudest_value_the_format_has():
    assert samples_of(ulaw_decode(b"\x80")) == [32124]
    assert samples_of(ulaw_decode(b"\x00")) == [-32124]


def test_the_format_is_symmetric_around_zero():
    for sample in (1, 7, 100, 1000, 8000, 20000, 32000):
        positive = samples_of(ulaw_decode(ulaw_encode(pcm_of(sample))))[0]
        negative = samples_of(ulaw_decode(ulaw_encode(pcm_of(-sample))))[0]
        assert positive == -negative


# --------------------------------------------------------------------------
# Round trip
# --------------------------------------------------------------------------

def test_a_round_trip_stays_within_the_formats_own_error():
    """Companding: fine steps when quiet, coarse when loud.

    The bound is the format's own: a value is rounded to the nearest step of its
    segment, and a segment's step is 2**(exponent + 3), where the exponent is read
    off the position of the top set bit of the biased magnitude. So the worst
    error is half that step, derived here from the specification rather than from
    the implementation under test."""
    for sample in range(-32768, 32768, 7):
        back = samples_of(ulaw_decode(ulaw_encode(pcm_of(sample))))[0]
        top_bit = int(math.log2(abs(sample) + 132))
        assert abs(back - sample) <= 2 ** (top_bit - 5), (sample, back)


def test_a_speech_level_tone_survives_with_good_signal_to_noise():
    n = 8000
    tone = [int(0.3 * 32767 * math.sin(2 * math.pi * 440 * i / 8000)) for i in range(n)]
    back = samples_of(ulaw_decode(ulaw_encode(pcm_of(*tone))))
    signal = sum(s * s for s in tone)
    noise = sum((a - b) ** 2 for a, b in zip(tone, back, strict=True))
    snr_db = 10 * math.log10(signal / noise)
    # G.711 is specified for roughly 38 dB at speech levels.
    assert snr_db > 30


def test_encoding_halves_the_length_and_decoding_doubles_it():
    pcm = pcm_of(*range(160))
    encoded = ulaw_encode(pcm)
    assert len(encoded) == 160
    assert len(ulaw_decode(encoded)) == 320


def test_a_trailing_odd_byte_is_dropped_not_misread():
    assert ulaw_encode(pcm_of(0, 0) + b"\x01") == b"\xff\xff"


def test_empty_input_is_empty_output():
    assert ulaw_encode(b"") == b""
    assert ulaw_decode(b"") == b""


def test_every_possible_byte_decodes_and_re_encodes_to_itself():
    """The codes are the format's own alphabet, so they must round-trip exactly.
    The one exception is negative zero (0x7F), which encodes as positive zero."""
    for code in range(256):
        if code == 0x7F:
            continue
        sample = samples_of(ulaw_decode(bytes([code])))[0]
        assert ulaw_encode(pcm_of(sample)) == bytes([code]), code


# --------------------------------------------------------------------------
# Bringing a synthesiser's rate down to the telephone's
# --------------------------------------------------------------------------

def test_decimate_averages_each_group_of_samples():
    pcm = Pcm(data=pcm_of(0, 300, 600, 900, 1200, 1500), sample_rate=24_000)
    out = decimate(pcm, 8_000)
    assert out.sample_rate == 8_000
    assert samples_of(out.data) == [300, 1200]


def test_decimate_by_two_for_sixteen_kilohertz():
    out = decimate(Pcm(data=pcm_of(10, 20, 30, 50), sample_rate=16_000), 8_000)
    assert samples_of(out.data) == [15, 40]


def test_decimating_to_the_same_rate_changes_nothing():
    pcm = Pcm(data=pcm_of(1, 2, 3), sample_rate=8_000)
    assert decimate(pcm, 8_000) is pcm


def test_a_partial_group_at_the_end_is_dropped():
    out = decimate(Pcm(data=pcm_of(3, 3, 3, 9, 9), sample_rate=24_000), 8_000)
    assert samples_of(out.data) == [3]


def test_decimate_keeps_a_tone_below_the_new_nyquist_and_damps_one_above():
    """The reason for averaging rather than dropping samples: a 6 kHz component
    cannot exist on an 8 kHz line, and plain dropping would fold it down to
    2 kHz at full strength. Averaging three samples passes it at a third."""
    def level(hz: int) -> float:
        n = 2400
        tone = [int(20000 * math.sin(2 * math.pi * hz * i / 24_000)) for i in range(n)]
        out = samples_of(decimate(Pcm(data=pcm_of(*tone), sample_rate=24_000), 8_000).data)
        return math.sqrt(sum(s * s for s in out) / len(out))

    assert level(1_000) > 12_000                  # speech band: passes
    assert level(6_000) < level(1_000) * 0.4      # above the line's limit: damped


def test_decimate_refuses_a_fractional_ratio():
    with pytest.raises(ValueError, match="whole multiple"):
        decimate(Pcm(data=pcm_of(0, 0, 0), sample_rate=22_050), 8_000)


def test_decimate_refuses_stereo():
    with pytest.raises(ValueError, match="mono"):
        decimate(Pcm(data=pcm_of(0, 0), sample_rate=16_000, channels=2), 8_000)
