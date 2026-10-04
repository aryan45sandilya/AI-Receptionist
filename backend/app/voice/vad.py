"""Voice activity detection: deciding when the caller has started and stopped.

This is the clock that drives a voice turn. Speech onset is what stops the
agent talking over the caller (barge-in, Phase 3); the silence that follows is
what says "this utterance is finished, transcribe it now".

Energy-based with hysteresis, rather than a neural detector. webrtcvad needs a C
extension with no wheel for this Python, and silero-vad drags in torch to decide
whether someone is talking. Neither is worth it: the detector only has to
separate speech from near-silence on a single close-mic stream, and being pure
Python means the whole thing is testable with synthesised audio and no hardware.

The reason this is not optional: the transcription model hallucinates on silence.
Sent half a second of digital zero, it returns " Thank you." -- so an agent that
transcribes everything would answer remarks the caller never made. Speech has to
be detected before any audio is sent.
"""

from __future__ import annotations

import array
import math
from dataclasses import dataclass, field
from enum import Enum

# 16 kHz mono 16-bit PCM is the default: what the browser sends, and small enough
# to stream comfortably over a WebSocket. A telephone call arrives at 8 kHz, and
# the detector is told so with ``sample_rate``; nothing in it depends on the rate
# except how many bytes make a frame.
SAMPLE_RATE = 16_000
SAMPLE_WIDTH = 2

# 20 ms frames. Long enough that one glottal pulse does not swing the energy
# reading, short enough to catch the start of speech without a hearable delay.
FRAME_MS = 20
FRAME_SAMPLES = SAMPLE_RATE * FRAME_MS // 1000
FRAME_BYTES = FRAME_SAMPLES * SAMPLE_WIDTH


class Event(Enum):
    """What the detector noticed in the audio it was just given."""

    NOTHING = "nothing"
    SPEECH_STARTED = "speech_started"
    SPEECH_ENDED = "speech_ended"


@dataclass
class VadConfig:
    # Root-mean-square amplitude, 0..1, above which a frame counts as loud.
    # 0.02 is roughly -34 dBFS: comfortably above room tone on a laptop mic and
    # well below speech. Calibrated against the noise floor at runtime.
    speech_threshold: float = 0.02

    # Hysteresis. Speech must be sustained to start, and silence sustained to
    # end, so a cough does not open a turn and a pause mid-sentence does not
    # close one. Asymmetric on purpose: starting late clips the caller's first
    # word, ending early cuts them off -- the second is far more annoying, so
    # the end threshold is the more patient of the two.
    start_frames: int = 5    # 100 ms of speech opens an utterance
    end_frames: int = 40     # 800 ms of silence closes it

    # Refuse to transcribe something too short to be a word. Guards against a
    # door slam becoming " Thank you."
    min_utterance_ms: int = 250

    # Hard stop, so a stuck-open mic cannot buffer forever.
    max_utterance_ms: int = 30_000

    # Audio kept from before speech was detected. Without it the utterance
    # starts 100 ms late and loses the initial consonant -- "six" becomes "ix".
    preroll_frames: int = 8  # 160 ms

    # Adapt to the room. The floor tracks the quietest recent frames, and the
    # working threshold sits above it, so a noisy cafe and a quiet room both
    # work without anyone tuning a constant.
    adapt_noise_floor: bool = True
    noise_floor_margin: float = 3.5


def frame_rms(frame: bytes) -> float:
    """Root-mean-square amplitude of one PCM frame, normalised to 0..1."""
    if not frame:
        return 0.0
    samples = array.array("h")
    # An odd trailing byte is not a whole sample; dropping it is correct.
    samples.frombytes(frame[: len(frame) - (len(frame) % SAMPLE_WIDTH)])
    if not samples:
        return 0.0
    total = math.fsum(float(s) * float(s) for s in samples)
    return math.sqrt(total / len(samples)) / 32768.0


@dataclass
class VoiceActivityDetector:
    """Feed it PCM, it tells you when an utterance starts and ends.

    Stateful and single-conversation: one instance per call.
    """

    config: VadConfig = field(default_factory=VadConfig)
    sample_rate: int = SAMPLE_RATE

    _buffer: bytearray = field(default_factory=bytearray, init=False)
    _speech: bytearray = field(default_factory=bytearray, init=False)
    _preroll: list[bytes] = field(default_factory=list, init=False)
    _loud_run: int = field(default=0, init=False)
    _quiet_run: int = field(default=0, init=False)
    _speech_frames: int = field(default=0, init=False)
    _in_speech: bool = field(default=False, init=False)
    _noise_floor: float | None = field(default=None, init=False)
    _endpoint_silence_ms: int = field(default=0, init=False)
    _frame_bytes: int = field(default=FRAME_BYTES, init=False)

    def __post_init__(self) -> None:
        self._frame_bytes = self.sample_rate * FRAME_MS // 1000 * SAMPLE_WIDTH

    @property
    def speech_ms(self) -> int:
        """How long the current utterance has been going, counted from the frame
        that first sounded like speech. Zero between utterances.

        What barge-in measures: the caller has not interrupted until they have
        kept talking for a while, and "a while" is this."""
        return self._speech_frames * FRAME_MS if self._in_speech else 0

    @property
    def voiced_ms(self) -> int:
        """How long the caller has actually been *talking* in this utterance: from
        the first frame that sounded like speech to the last one that did.

        Not ``speech_ms``, which keeps counting through the silence that follows
        until the utterance is closed. A cough is 120 ms of sound and then 800 ms
        of nothing; ``speech_ms`` would read 920 and call it an interruption. This
        reads 120, and stays there."""
        if not self._in_speech:
            return 0
        return max(0, self._speech_frames - self._quiet_run) * FRAME_MS

    @property
    def quiet_ms(self) -> int:
        """Silence since the last loud frame, while an utterance is open.

        The moment the caller actually stopped talking is this far in the past.
        A transcription service that decides the utterance is over before this
        detector does (it listens for a shorter pause) still needs the answer, to
        anchor the latency clock where the caller's waiting began."""
        return self._quiet_run * FRAME_MS if self._in_speech else 0

    @property
    def endpoint_silence_ms(self) -> int:
        """Silence the last utterance waited through before it was closed.

        The caller stopped talking this long before the detector could say so,
        and they experience that wait as part of the reply latency. The turn
        timer anchors on the moment they actually stopped, so it has to know.
        Zero when the utterance was closed by the length cap rather than by
        silence.
        """
        return self._endpoint_silence_ms

    @property
    def in_speech(self) -> bool:
        return self._in_speech

    @property
    def noise_floor(self) -> float:
        return self._noise_floor or 0.0

    def threshold(self) -> float:
        """The level a frame must exceed right now to count as speech."""
        floor = self._noise_floor
        if not self.config.adapt_noise_floor or floor is None:
            return self.config.speech_threshold
        # Whichever is stricter: the configured floor, or well above the room.
        return max(self.config.speech_threshold, floor * self.config.noise_floor_margin)

    def _observe_noise(self, rms: float, loud: bool) -> None:
        """Track the quiet end of the signal, so the threshold follows the room.

        Falls fast and rises slowly: a genuinely quieter room should be believed
        almost at once, but a pause between words must not drag the floor up
        into the middle of speech.

        Two guards keep the floor from climbing into speech and deafening the
        detector -- which it will otherwise do, permanently, on a call that
        opens with the caller already talking:

        *   frames already judged loud are ignored. Speech before the start
            threshold has been met is still speech, and must not teach the
            detector that the room is that loud;
        *   the floor is capped at ``speech_threshold``. Anything above that
            level is speech by definition, so it cannot be room tone, whatever
            it looks like.
        """
        if not self.config.adapt_noise_floor or self._in_speech or loud:
            return
        observed = min(rms, self.config.speech_threshold)
        if self._noise_floor is None:
            self._noise_floor = observed
        elif observed < self._noise_floor:
            self._noise_floor = 0.7 * self._noise_floor + 0.3 * observed
        else:
            self._noise_floor = 0.995 * self._noise_floor + 0.005 * observed

    def feed(self, pcm: bytes) -> list[tuple[Event, bytes | None]]:
        """Consume PCM and return whatever it completed.

        Audio arrives in whatever sizes the network delivers, so it is buffered
        and consumed in whole frames. A SPEECH_ENDED event carries the finished
        utterance; every other event carries None.
        """
        self._buffer.extend(pcm)
        events: list[tuple[Event, bytes | None]] = []

        while len(self._buffer) >= self._frame_bytes:
            frame = bytes(self._buffer[: self._frame_bytes])
            del self._buffer[: self._frame_bytes]
            event = self._consume_frame(frame)
            if event is not None:
                events.append(event)
        return events

    def _consume_frame(self, frame: bytes) -> tuple[Event, bytes | None] | None:
        rms = frame_rms(frame)
        # Judged against the current threshold first, so the noise tracker can
        # be told whether this frame was speech and skip it if so.
        loud = rms > self.threshold()
        self._observe_noise(rms, loud)

        if not self._in_speech:
            # Keep a rolling pre-roll so the opening consonant survives.
            self._preroll.append(frame)
            if len(self._preroll) > self.config.preroll_frames:
                self._preroll.pop(0)

            self._loud_run = self._loud_run + 1 if loud else 0
            if self._loud_run >= self.config.start_frames:
                self._in_speech = True
                self._quiet_run = 0
                # The frames that tripped the threshold count towards the length.
                self._speech_frames = self._loud_run
                self._speech = bytearray(b"".join(self._preroll))
                self._preroll.clear()
                return (Event.SPEECH_STARTED, None)
            return None

        self._speech.extend(frame)
        self._speech_frames += 1
        self._quiet_run = 0 if loud else self._quiet_run + 1

        if len(self._speech) >= self._max_bytes():
            return self._close_utterance()
        if self._quiet_run >= self.config.end_frames:
            return self._close_utterance()
        return None

    def _max_bytes(self) -> int:
        return self.config.max_utterance_ms * self.sample_rate * SAMPLE_WIDTH // 1000

    def _min_bytes(self) -> int:
        return self.config.min_utterance_ms * self.sample_rate * SAMPLE_WIDTH // 1000

    def _close_utterance(self) -> tuple[Event, bytes | None]:
        # Drop the trailing silence that ended the utterance; there is no point
        # paying to transcribe it, and it adds latency.
        tail = self._quiet_run * self._frame_bytes
        self._endpoint_silence_ms = self._quiet_run * FRAME_MS
        audio = bytes(self._speech[: len(self._speech) - tail] if tail else self._speech)

        self._in_speech = False
        self._speech = bytearray()
        self._speech_frames = 0
        self._loud_run = 0
        self._quiet_run = 0

        if len(audio) < self._min_bytes():
            # Too short to be speech. Report the end so playback can resume,
            # but hand back nothing to transcribe.
            return (Event.SPEECH_ENDED, None)
        return (Event.SPEECH_ENDED, audio)

    def flush(self) -> bytes | None:
        """Close any utterance still open, for when the call ends mid-sentence."""
        if not self._in_speech:
            return None
        _, audio = self._close_utterance()
        return audio

    def reset(self) -> None:
        """Forget everything except the learned noise floor, which still holds."""
        self._buffer.clear()
        self._speech = bytearray()
        self._preroll.clear()
        self._loud_run = 0
        self._quiet_run = 0
        self._speech_frames = 0
        self._in_speech = False
