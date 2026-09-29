"""WAV container handling.

Speech services exchange audio as WAV files, but everything inside this project
moves raw 16-bit PCM: the detector measures it, the browser queues it, and
Phase 3 will hand it to a telephony stream. So WAV headers get built on the way
out and parsed on the way in, and nothing else has to know the format exists.

Small enough to do by hand. Python's ``wave`` module works on file objects and
would mean wrapping every buffer in BytesIO to read four integers.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass

# Minimal canonical WAV: 'RIFF' + size + 'WAVE' + 'fmt ' chunk + 'data' chunk.
_HEADER_BYTES = 44
_PCM_FORMAT = 1


@dataclass(frozen=True)
class Pcm:
    """Raw little-endian 16-bit PCM, with the rate needed to play it."""

    data: bytes
    sample_rate: int
    channels: int = 1

    @property
    def duration_ms(self) -> int:
        frames = len(self.data) // (2 * self.channels)
        return int(frames * 1000 / self.sample_rate) if self.sample_rate else 0


def wav_header(*, sample_rate: int, channels: int, data_bytes: int) -> bytes:
    byte_rate = sample_rate * channels * 2
    return (
        b"RIFF"
        + struct.pack("<I", 36 + data_bytes)
        + b"WAVEfmt "
        + struct.pack(
            "<IHHIIHH", 16, _PCM_FORMAT, channels, sample_rate, byte_rate, channels * 2, 16
        )
        + b"data"
        + struct.pack("<I", data_bytes)
    )


def to_wav(pcm: Pcm) -> bytes:
    """Wrap PCM in a WAV container, for a service that wants a file."""
    return (
        wav_header(
            sample_rate=pcm.sample_rate, channels=pcm.channels, data_bytes=len(pcm.data)
        )
        + pcm.data
    )


def from_wav(payload: bytes) -> Pcm:
    """Pull the PCM and sample rate out of a WAV file.

    Chunks are walked rather than assumed to sit at fixed offsets: services add
    'LIST' or 'fact' chunks between the header and the audio, and a reader that
    slices at byte 44 would hand back metadata as if it were sound.
    """
    if len(payload) < 12 or payload[:4] != b"RIFF" or payload[8:12] != b"WAVE":
        raise ValueError("not a RIFF/WAVE payload")

    sample_rate = 0
    channels = 1
    offset = 12
    while offset + 8 <= len(payload):
        chunk_id = payload[offset : offset + 4]
        (chunk_size,) = struct.unpack("<I", payload[offset + 4 : offset + 8])
        body = offset + 8

        if chunk_id == b"fmt " and chunk_size >= 16:
            audio_format, channels, sample_rate = struct.unpack(
                "<HHI", payload[body : body + 8]
            )
            bits = struct.unpack("<H", payload[body + 14 : body + 16])[0]
            if audio_format != _PCM_FORMAT or bits != 16:
                raise ValueError(f"expected 16-bit PCM, got format={audio_format} bits={bits}")
        elif chunk_id == b"data":
            data = payload[body : body + chunk_size]
            if not sample_rate:
                raise ValueError("data chunk appeared before fmt")
            return Pcm(data=data, sample_rate=sample_rate, channels=channels)

        # Chunks are word-aligned: an odd size is followed by a pad byte.
        offset = body + chunk_size + (chunk_size % 2)

    raise ValueError("no data chunk in WAV payload")


def to_mono(pcm: Pcm) -> Pcm:
    """Average channels down to one. Speech models expect mono."""
    if pcm.channels == 1:
        return pcm
    samples = memoryview(pcm.data).cast("h")
    step = pcm.channels
    mixed = bytearray()
    for i in range(0, len(samples) - step + 1, step):
        mixed += struct.pack("<h", sum(samples[i : i + step]) // step)
    return Pcm(data=bytes(mixed), sample_rate=pcm.sample_rate, channels=1)


def resample(pcm: Pcm, target_rate: int) -> Pcm:
    """Linear-interpolation resampling of mono 16-bit PCM.

    Adequate for speech going into a recogniser, and dependency-free. Not a
    mastering-grade converter: there is no anti-alias filter, so downsampling
    audio with real content above the new Nyquist will alias. For a 44.1 or
    48 kHz microphone recording brought down to 16 kHz that is inaudible to the
    transcription model.
    """
    if pcm.channels != 1:
        pcm = to_mono(pcm)
    if pcm.sample_rate == target_rate or not pcm.data:
        return Pcm(data=pcm.data, sample_rate=target_rate, channels=1)

    samples = memoryview(pcm.data[: len(pcm.data) - (len(pcm.data) % 2)]).cast("h")
    n_in = len(samples)
    n_out = max(1, int(n_in * target_rate / pcm.sample_rate))
    step = pcm.sample_rate / target_rate
    out = bytearray()
    for k in range(n_out):
        position = k * step
        i = int(position)
        if i + 1 >= n_in:
            value = samples[n_in - 1]
        else:
            frac = position - i
            value = int(samples[i] + (samples[i + 1] - samples[i]) * frac)
        out += struct.pack("<h", value)
    return Pcm(data=bytes(out), sample_rate=target_rate, channels=1)
