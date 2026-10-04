"""Choosing the speech services for a call.

One function, so the browser demo, the telephone route and the tests all build
their clients the same way -- and tests can replace it with fakes in one place.
"""

from __future__ import annotations

import logging

import httpx2 as httpx
from openai import AsyncOpenAI

from ..config import Settings
from .deepgram import DeepgramTranscriber, DeepgramTTS
from .speech import (
    STT_SAMPLE_RATE,
    BatchTranscriber,
    GroqOrpheus,
    GroqWhisper,
    TextToSpeech,
    Transcriber,
)

log = logging.getLogger(__name__)

# Aura voices are 24 kHz; the browser resamples to its own speakers.
DEFAULT_VOICE_RATE = 24_000


async def _note_rate_limit(response: httpx.Response) -> None:
    """Say so when the model provider turns a request away for going too fast.

    The OpenAI client retries a 429 on its own, waiting as long as the provider
    asks, and says nothing -- so on a free tier with a small tokens-per-minute
    allowance a reply can stall for seconds with no trace of why. This is that
    trace: one warning per refusal, with how long the provider asked it to wait.
    """
    if response.status_code == 429:
        log.warning(
            "model provider rate limit (429): retry-after=%ss, limit=%s tokens/min, remaining=%s",
            response.headers.get("retry-after", "?"),
            response.headers.get("x-ratelimit-limit-tokens", "?"),
            response.headers.get("x-ratelimit-remaining-tokens", "?"),
        )


def llm_client(settings: Settings) -> AsyncOpenAI:
    """The language model client: the OpenAI SDK pointed at the configured endpoint."""
    return AsyncOpenAI(
        api_key=settings.llm_api_key,
        base_url=settings.llm_base_url,
        http_client=httpx.AsyncClient(
            # The SDK's own defaults, restated because supplying a client replaces them:
            # generous overall (a stream can run a while), quick to give up connecting.
            timeout=httpx.Timeout(600.0, connect=5.0),
            event_hooks={"response": [_note_rate_limit]},
        ),
    )


class ConfigurationError(RuntimeError):
    """A setting that makes the call impossible, worded for whoever has to fix it."""


def build_clients(
    settings: Settings,
    *,
    input_rate: int = STT_SAMPLE_RATE,
    output_rate: int | None = None,
) -> tuple[AsyncOpenAI, Transcriber, TextToSpeech]:
    """The language model client, the transcriber and the voice for one call.

    ``input_rate`` is the sample rate of the caller's audio (16 kHz from a
    browser, 8 kHz from a telephone). ``output_rate`` is the rate the far end
    needs the agent's voice at, or None if it takes any: a service that can make
    speech at the right rate is asked for it, so nothing has to be converted.
    """
    llm = llm_client(settings)
    whisper = GroqWhisper(llm, model=settings.stt_model, language=settings.stt_language)

    if settings.stt_service == "deepgram":
        key = _deepgram_key(settings, "STT_PROVIDER")
        transcriber: Transcriber = DeepgramTranscriber(
            key,
            model=settings.deepgram_stt_model,
            language=settings.stt_language,
            sample_rate=input_rate,
            endpointing_ms=settings.deepgram_endpointing_ms,
            # If the streaming socket will not open, the utterance is still
            # transcribed -- as a recording, by the service that needs no socket.
            fallback=whisper,
        )
    else:
        transcriber = BatchTranscriber(whisper, sample_rate=input_rate)

    if settings.tts_service == "deepgram":
        key = _deepgram_key(settings, "TTS_PROVIDER")
        voice: TextToSpeech = DeepgramTTS(
            key,
            model=settings.deepgram_tts_model,
            sample_rate=output_rate or DEFAULT_VOICE_RATE,
        )
    else:
        voice = GroqOrpheus(llm, model=settings.tts_model, voice=settings.tts_voice)

    return llm, transcriber, voice


def _deepgram_key(settings: Settings, because: str) -> str:
    key = settings.deepgram_api_key.strip()
    if not key:
        raise ConfigurationError(
            f"{because}=deepgram needs DEEPGRAM_API_KEY. Add it to .env, "
            "or set the provider to groq."
        )
    return key
