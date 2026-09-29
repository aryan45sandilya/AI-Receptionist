"""Synthetic audio for tests: a tone stands in for speech, zeros for silence.

The voice detector keys on energy, and a tone at speaking level is
indistinguishable from speech as far as it is concerned, so tests need no
recordings and no microphone.
"""

from __future__ import annotations

import array
import math

from backend.app.voice.vad import SAMPLE_RATE


def tone(ms: int, amplitude: float = 0.3, hz: int = 220) -> bytes:
    """A sine wave: stands in for speech."""
    n = SAMPLE_RATE * ms // 1000
    samples = array.array(
        "h",
        (int(amplitude * 32767 * math.sin(2 * math.pi * hz * i / SAMPLE_RATE)) for i in range(n)),
    )
    return samples.tobytes()


def silence(ms: int) -> bytes:
    return b"\x00\x00" * (SAMPLE_RATE * ms // 1000)


def hiss(ms: int, amplitude: float = 0.004) -> bytes:
    """Low-level room tone: present, but far too quiet to be speech."""
    n = SAMPLE_RATE * ms // 1000
    # Deterministic pseudo-noise, so a failure is always reproducible.
    samples = array.array("h", (int(amplitude * 32767 * math.sin(i * 12.9898)) for i in range(n)))
    return samples.tobytes()
