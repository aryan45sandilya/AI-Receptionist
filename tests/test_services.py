"""Choosing the services for a call, and the model client's rate-limit trace."""

from __future__ import annotations

import logging

import httpx2 as httpx
import pytest

from backend.app.config import Settings
from backend.app.voice import services
from backend.app.voice.deepgram import DeepgramTranscriber, DeepgramTTS
from backend.app.voice.services import ConfigurationError, _note_rate_limit, build_clients
from backend.app.voice.speech import BatchTranscriber, GroqOrpheus


def settings(**overrides) -> Settings:
    """Settings that depend on nothing but what the test says -- not on the
    developer's own .env, which holds a real Deepgram key."""
    base = {
        "database_url": "postgresql+psycopg://u:p@localhost/db",
        "llm_api_key": "gsk-test",
        "deepgram_api_key": "",
    }
    return Settings(_env_file=None, **{**base, **overrides})


# ---------------------------------------------------------------------------
# Which services
# ---------------------------------------------------------------------------

def test_with_no_deepgram_key_the_free_groq_services_are_used():
    _, stt, tts = build_clients(settings())
    assert isinstance(stt, BatchTranscriber) and isinstance(tts, GroqOrpheus)


def test_a_deepgram_key_switches_both_to_deepgram_streaming():
    _, stt, tts = build_clients(settings(deepgram_api_key="dg-key"))
    assert isinstance(stt, DeepgramTranscriber) and isinstance(tts, DeepgramTTS)


def test_each_can_be_chosen_independently():
    _, stt, tts = build_clients(settings(deepgram_api_key="dg-key", tts_provider="groq"))
    assert isinstance(stt, DeepgramTranscriber) and isinstance(tts, GroqOrpheus)


def test_naming_deepgram_without_a_key_is_an_error_not_a_quiet_downgrade():
    with pytest.raises(ConfigurationError, match="DEEPGRAM_API_KEY"):
        build_clients(settings(stt_provider="deepgram"))
    with pytest.raises(ConfigurationError, match="DEEPGRAM_API_KEY"):
        build_clients(settings(tts_provider="deepgram"))


def test_the_transcriber_is_told_the_sample_rate_of_the_line():
    _, stt, _ = build_clients(settings(deepgram_api_key="dg-key"), input_rate=8_000)
    assert stt._rate == 8_000  # noqa: SLF001


def test_the_voice_is_made_at_the_rate_the_line_needs_or_the_browsers_default():
    _, _, phone = build_clients(settings(deepgram_api_key="dg-key"), output_rate=8_000)
    _, _, browser = build_clients(settings(deepgram_api_key="dg-key"))
    assert phone.sample_rate == 8_000
    assert browser.sample_rate == 24_000


def test_the_streaming_transcriber_can_fall_back_to_a_recording():
    _, stt, _ = build_clients(settings(deepgram_api_key="dg-key"))
    assert stt._fallback is not None  # noqa: SLF001


def test_the_services_follow_the_provider_settings():
    assert settings().stt_service == "groq"
    assert settings(deepgram_api_key="k").stt_service == "deepgram"
    assert settings(deepgram_api_key="k", stt_provider="groq").stt_service == "groq"
    assert settings(tts_provider="deepgram").tts_service == "deepgram"


# ---------------------------------------------------------------------------
# A rate limit leaves a trace
# ---------------------------------------------------------------------------

async def test_a_refused_request_is_logged_with_what_the_provider_said(caplog):
    response = httpx.Response(
        429,
        headers={"retry-after": "7", "x-ratelimit-limit-tokens": "8000", "x-ratelimit-remaining-tokens": "120"},
    )
    with caplog.at_level(logging.WARNING, logger=services.log.name):
        await _note_rate_limit(response)

    (record,) = caplog.records
    assert "429" in record.getMessage()
    assert "retry-after=7s" in record.getMessage()
    assert "8000" in record.getMessage() and "120" in record.getMessage()


async def test_a_successful_request_is_not_logged(caplog):
    with caplog.at_level(logging.WARNING, logger=services.log.name):
        await _note_rate_limit(httpx.Response(200))
    assert caplog.records == []


async def test_the_client_the_calls_use_has_the_trace_attached():
    client = services.llm_client(settings())
    hooks = client._client.event_hooks["response"]  # noqa: SLF001
    assert _note_rate_limit in hooks
