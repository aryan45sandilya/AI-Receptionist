"""G.711 mu-law: the audio format of the telephone network.

A phone call is 8 kHz, one byte per sample, companded rather than linear: the
quiet parts of speech get fine steps and the loud parts coarse ones, which is
how eight bits carry speech that would need twelve or thirteen as plain PCM.
Twilio's media streams send and expect exactly this, base64 encoded, 20 ms at a
time (160 bytes).

Everything inside this project moves 16-bit PCM, so mu-law exists only at the
edge: decoded on the way in, encoded on the way out. Written out by hand because
Python removed ``audioop`` in 3.13, and a pair of lookup tables is all it takes.
"""

from __future__ import annotations

import array
import struct
from functools import cache

from .audio import Pcm

# Standard G.711 constants. The bias lifts every magnitude into the range where
# the segment (exponent) can be read off the position of the top set bit; the
# clip is the largest magnitude that still fits once the bias is added.
_BIAS = 0x84
_CLIP = 32_635

TELEPHONE_RATE = 8_000


@cache
def _decode_table() -> tuple[bytes, ...]:
    """Each mu-law byte -> its 16-bit sample, already packed little-endian."""
    table = []
    for byte in range(256):
        # Bits are stored inverted, so line noise on an idle (all-ones) line
        # decodes to silence rather than to a loud tone.
        inverted = ~byte & 0xFF
        sign = inverted & 0x80
        exponent = (inverted >> 4) & 0x07
        mantissa = inverted & 0x0F
        magnitude = (((mantissa << 3) + _BIAS) << exponent) - _BIAS
        table.append(struct.pack("<h", -magnitude if sign else magnitude))
    return tuple(table)


def _encode_sample(sample: int) -> int:
    sign = 0
    if sample < 0:
        sample = -sample
        sign = 0x80
    sample = min(sample, _CLIP) + _BIAS

    exponent = 7
    mask = 0x4000
    while exponent > 0 and not sample & mask:
        exponent -= 1
        mask >>= 1
    mantissa = (sample >> (exponent + 3)) & 0x0F
    return ~(sign | (exponent << 4) | mantissa) & 0xFF


@cache
def _encode_table() -> bytes:
    """Every possible 16-bit sample -> its mu-law byte, indexed by the sample's
    two's-complement bit pattern. Built once; encoding is then one lookup each."""
    table = bytearray(65_536)
    for pattern in range(65_536):
        sample = pattern - 65_536 if pattern >= 32_768 else pattern
        table[pattern] = _encode_sample(sample)
    return bytes(table)


def ulaw_decode(data: bytes) -> bytes:
    """Mu-law bytes -> little-endian 16-bit PCM, twice as long."""
    table = _decode_table()
    return b"".join(table[byte] for byte in data)


def ulaw_encode(pcm: bytes) -> bytes:
    """Little-endian 16-bit PCM -> mu-law bytes, half as long."""
    table = _encode_table()
    samples = array.array("h")
    # An odd trailing byte is half a sample; dropping it is correct.
    samples.frombytes(pcm[: len(pcm) - (len(pcm) % 2)])
    return bytes(table[sample & 0xFFFF] for sample in samples)


def decimate(pcm: Pcm, target_rate: int) -> Pcm:
    """Bring audio down to ``target_rate`` by averaging, when the ratio is whole.

    Voice from a 24 kHz synthesiser has energy above 4 kHz, and a phone line
    cannot carry it. Dropping samples without filtering folds that energy back
    down into the speech band as a hiss, so each output sample is the mean of the
    ``factor`` input samples it replaces -- a crude low-pass, but enough to keep
    the sibilants from turning into noise. Mono 16-bit only.

    Raises for a ratio that is not a whole number: the callers that need a
    fractional one should say so by using ``audio.resample`` instead, not get
    this quietly.
    """
    if pcm.channels != 1:
        raise ValueError("decimate expects mono audio")
    if pcm.sample_rate == target_rate:
        return pcm
    if pcm.sample_rate % target_rate:
        raise ValueError(f"{pcm.sample_rate} Hz is not a whole multiple of {target_rate} Hz")

    factor = pcm.sample_rate // target_rate
    samples = array.array("h")
    samples.frombytes(pcm.data[: len(pcm.data) - (len(pcm.data) % 2)])
    usable = len(samples) - (len(samples) % factor)
    out = array.array(
        "h",
        (sum(samples[i : i + factor]) // factor for i in range(0, usable, factor)),
    )
    return Pcm(data=out.tobytes(), sample_rate=target_rate, channels=1)
