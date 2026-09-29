"""Speech to text, and text to speech.

Both sit behind small protocols so the rest of the pipeline never learns which
service is on the other end. That is not decoration: the two implementations
here are batch services, and a streaming one (Deepgram for transcription,
Cartesia for speech) has a different shape -- swapping one in should touch this
file only.

The batch/streaming distinction is the main latency story in this phase:

*   Transcription only starts once the caller stops talking, because a batch
    service needs the whole utterance. A streaming service would have most of
    the transcript ready by then. Costs roughly the length of the utterance.

*   Speech is synthesised a sentence at a time rather than all at once, so the
    caller hears the first sentence while the second is still being made. With
    a batch service this is the difference between speaking after ~200 ms and
    speaking after the whole reply is finished.
"""

from __future__ import annotations

import io
import re
from typing import Protocol

from openai import AsyncOpenAI

from .audio import Pcm, from_wav, to_mono, to_wav

# What the detector produces and the transcription service expects.
STT_SAMPLE_RATE = 16_000


class SpeechToText(Protocol):
    async def transcribe(self, pcm: Pcm) -> str: ...


class TextToSpeech(Protocol):
    async def synthesize(self, text: str) -> Pcm: ...


# ---------------------------------------------------------------------------
# Transcription
# ---------------------------------------------------------------------------

# Things the model emits for audio containing no speech. It was trained on
# subtitled video, so silence comes back as the phrases that pad subtitle
# tracks. The detector should stop most of this ever being sent, but a half
# second of breath still gets through, and "Thank you." is a plausible enough
# thing for a caller to say that the agent would answer it.
_HALLUCINATIONS = {
    "thank you.", "thanks for watching!", "thank you for watching!",
    "you", "bye.", ".", "so", "[blank_audio]", "[music]", "[silence]",
    "please subscribe", "subtitles by the amara.org community",
}


def looks_like_silence(text: str) -> bool:
    cleaned = text.strip().lower().strip(" .!?,-")
    if not cleaned:
        return True
    return cleaned in {h.strip(" .!?,-") for h in _HALLUCINATIONS}


class GroqWhisper:
    """Batch transcription.

    The service takes a file, so the utterance is wrapped in a WAV container on
    the way out. ``language`` is pinned rather than detected: this is an
    India-facing product where callers mix English and Tamil mid-sentence, and
    left to auto-detect the model will sometimes decide a whole utterance is
    Tamil and translate it, silently changing what the caller said.
    """

    def __init__(
        self,
        client: AsyncOpenAI,
        *,
        model: str = "whisper-large-v3-turbo",
        language: str | None = "en",
        prompt: str | None = None,
    ) -> None:
        self._client = client
        self._model = model
        self._language = language
        # A short prompt biases spelling toward words this business actually
        # uses -- names and services that generic English would mangle.
        self._prompt = prompt

    async def transcribe(self, pcm: Pcm) -> str:
        audio = to_mono(pcm)
        payload = io.BytesIO(to_wav(audio))
        # The service picks its parser from the filename extension.
        payload.name = "utterance.wav"

        extra = {}
        if self._language:
            extra["language"] = self._language
        if self._prompt:
            extra["prompt"] = self._prompt

        result = await self._client.audio.transcriptions.create(
            model=self._model, file=payload, **extra
        )
        text = (result.text or "").strip()
        return "" if looks_like_silence(text) else text


# ---------------------------------------------------------------------------
# Speech synthesis
# ---------------------------------------------------------------------------

# Groq rejects a synthesis request longer than this. It is a hard API limit, not
# a tuning knob: a longer sentence must be cut before it is sent, or the caller
# hears nothing at all for that part of the reply.
ORPHEUS_MAX_CHARS = 200

# Split after ., ? or ! followed by whitespace. Deliberately simple: the cost of
# a wrong split is one slightly short audio clip, and a full sentence tokeniser
# would be a dependency to fix a cosmetic problem.
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")

# For the first chunk of a reply only, a comma, colon or semicolon is also a place
# to cut. See ``SentenceStream`` for why.
_CLAUSE_END = re.compile(r"(?<=[.!?,;:])\s+")

# Below this, a "sentence" is an abbreviation or a stray word ("Booked.",
# "Dr."); synthesised alone it comes out as a clipped fragment, so it is held
# and joined to the sentence that follows.
_MIN_SENTENCE_CHARS = 16


class SentenceStream:
    """Cut streamed text into speakable sentences as it arrives.

    This is what lets the caller hear the first sentence while the model is
    still writing the second. The synthesis service returns whole clips, so the
    unit of overlap is the sentence: emit one the moment it is complete, rather
    than waiting for the reply to finish.

    A sentence is only known to be complete once whitespace follows its final
    punctuation -- "1500." might yet become "1500.50" -- so each sentence is
    released roughly one token after it ends. The last one is released by
    ``flush``.

    Fed a reply in any chunking, character by character or all at once, it
    yields the same sentences; ``split_sentences`` is defined in terms of it so
    the batch and streaming paths cannot drift apart.
    """

    def __init__(
        self,
        *,
        max_chars: int = ORPHEUS_MAX_CHARS,
        min_chars: int = _MIN_SENTENCE_CHARS,
        eager_first: bool = False,
        first_min_chars: int = 24,
    ) -> None:
        if max_chars < 2 * min_chars:
            # A held fragment plus its neighbour must be able to fit in one clip.
            raise ValueError("max_chars must be at least twice min_chars")
        self._max = max_chars
        self._min = min_chars
        self._eager_first = eager_first
        self._first_min = first_min_chars
        self._released = False  # has any piece been handed out yet?
        self._pending = ""  # text after the last confirmed sentence boundary
        self._held = ""  # a finished but too-short sentence, awaiting a neighbour

    def feed(self, chunk: str) -> list[str]:
        """Add streamed text; return any sentences it completed."""
        self._pending += chunk
        ready: list[str] = []

        while (boundary := self._next_boundary()) is not None:
            sentence = self._pending[: boundary.start()]
            self._pending = self._pending[boundary.end() :]
            ready.extend(self._accept(sentence))
            self._released = self._released or bool(ready)

        # No punctuation in sight but the text is already too long for one clip:
        # cut on a space rather than let the caller wait for a full stop that
        # may not come.
        while len(self._pending) > self._max:
            cut = self._pending.rfind(" ", 0, self._max)
            cut = cut if cut > 0 else self._max
            ready.extend(self._accept(self._pending[:cut]))
            self._pending = self._pending[cut:].lstrip()
        return ready

    def flush(self) -> list[str]:
        """The stream has ended: release everything still held."""
        ready = self._accept(self._pending, final=True)
        self._pending = ""
        if self._held:
            ready.append(self._held)
            self._held = ""
        self._released = self._released or bool(ready)
        return ready

    def _next_boundary(self) -> re.Match[str] | None:
        """Where the next piece ends, if it has yet.

        Ordinarily a full stop, question mark or exclamation mark. With
        ``eager_first``, the *first* piece of a reply may also end at a comma or
        colon, once it is long enough to stand alone.

        Why: speech synthesis time grows with text length (measured: about
        0.75 s for 60 characters, 1.3 s for 100), and until the first clip
        exists the caller hears nothing. "We have three openings on Tuesday the
        sixth in the evening: 5:30, 5:45 or 6:00 with Dr Rao." is one sentence
        and would wait for the whole of it; cut at the colon, the caller hears
        the first half while the second is still being made. Only the first
        piece is cut early -- once audio is playing there is time to make the
        rest properly, and whole sentences sound better than clauses.
        """
        if not (self._eager_first and not self._released):
            return _SENTENCE_END.search(self._pending)
        for match in _CLAUSE_END.finditer(self._pending):
            head = self._pending[: match.start()].strip()
            if head[-1:] in ".!?" or len(head) >= self._first_min:
                return match
        return None

    def _accept(self, raw: str, *, final: bool = False) -> list[str]:
        sentence = " ".join(raw.split())
        if not sentence:
            return []
        if self._held:
            sentence = f"{self._held} {sentence}"
            self._held = ""
        if len(sentence) < self._min and not final:
            self._held = sentence
            return []
        return self._cap(sentence)

    def _cap(self, sentence: str) -> list[str]:
        """Split anything longer than one clip on whitespace."""
        pieces: list[str] = []
        while len(sentence) > self._max:
            cut = sentence.rfind(" ", 0, self._max)
            cut = cut if cut > 0 else self._max
            pieces.append(sentence[:cut].strip())
            sentence = sentence[cut:].strip()
        if sentence:
            pieces.append(sentence)
        return pieces


def split_sentences(
    text: str, *, max_chars: int = ORPHEUS_MAX_CHARS, eager_first: bool = False
) -> list[str]:
    """Break a finished reply into synthesis-sized pieces."""
    stream = SentenceStream(max_chars=max_chars, eager_first=eager_first)
    return [*stream.feed(text), *stream.flush()]


class GroqOrpheus:
    """Batch speech synthesis.

    Returns a whole clip per request, so the caller feeds it one sentence at a
    time (see ``SentenceStream``) and plays each as it arrives. The output rate
    is read from the returned WAV rather than assumed -- it is a property of the
    voice, and guessing it wrong makes the agent sound like a chipmunk.
    """

    def __init__(
        self,
        client: AsyncOpenAI,
        *,
        model: str = "canopylabs/orpheus-v1-english",
        voice: str = "hannah",
    ) -> None:
        self._client = client
        self._model = model
        self._voice = voice

    async def synthesize(self, text: str) -> Pcm:
        if len(text) > ORPHEUS_MAX_CHARS:
            # Reaching here means a caller skipped the splitter. Failing loudly
            # beats sending a request the API will reject with a vaguer error.
            raise ValueError(f"text is {len(text)} characters; the limit is {ORPHEUS_MAX_CHARS}")
        response = await self._client.audio.speech.create(
            model=self._model,
            voice=self._voice,
            input=text,
            response_format="wav",
        )
        return from_wav(response.read())
