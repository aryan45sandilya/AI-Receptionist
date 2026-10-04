"""Settings, loaded from the environment (and .env in development).

Nothing here has a secret as its default. A missing DATABASE_URL should fail
loudly at startup rather than quietly fall back to something local, because the
failure mode of the quiet version is writing test bookings into production.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal
from urllib.parse import urlsplit

from pydantic_settings import BaseSettings, SettingsConfigDict

# Ports and hostnames that mean "transaction pooling" rather than a real session.
# Supabase's Supavisor uses 6543 for transaction mode and 5432 for session mode
# on the same host, so the port is the only thing that distinguishes them.
# Neon's PgBouncer endpoint is signalled by "-pooler" in the hostname instead.
_TRANSACTION_POOL_PORT = 6543
_NEON_POOLER_MARKER = "-pooler."


def uses_transaction_pooling(url: str) -> bool:
    """Whether this URL points at a transaction-pooled endpoint.

    It matters because psycopg3 prepares statements by default, and a
    transaction pooler hands the next statement to a different backend session
    where that prepared statement does not exist. The failure is intermittent
    and reads as "prepared statement _pg3_0 already exists", which is a
    miserable thing to debug, so we detect it instead.
    """
    try:
        parts = urlsplit(url)
        host = (parts.hostname or "").lower()
        port = parts.port
    except ValueError:
        # A malformed port. Let the driver produce the connection error, which
        # will be far more informative than a parse failure raised from config.
        return False
    if port == _TRANSACTION_POOL_PORT:
        return True
    return _NEON_POOLER_MARKER in host and host.endswith("neon.tech")


def _resolve_provider(choice: str, deepgram_key: str) -> Literal["deepgram", "groq"]:
    """Turn "auto" into a concrete service.

    Asking for Deepgram without a key is not quietly downgraded to Groq: someone
    who named a provider wants that provider, and finding out from a log line
    that they got the other one is how a comparison goes wrong. The call fails to
    start with a message saying which key is missing.
    """
    if choice == "auto":
        return "deepgram" if deepgram_key.strip() else "groq"
    return choice  # type: ignore[return-value]  # validated by the Literal


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",  # Phase 2+ keys can sit in .env before they are read
        case_sensitive=False,
    )

    # --- Database -----------------------------------------------------------
    # Use the direct (non-pooled) endpoint. psycopg3 prepares statements, which
    # pgBouncer's transaction pooling breaks, and the migration needs a real
    # session for CREATE EXTENSION.
    database_url: str

    # --- LLM (Groq, through its OpenAI-compatible endpoint) -----------------
    # Groq is chosen for latency: it is the fastest of the free tiers, and on a
    # phone call the time to first word is the entire experience. Because the
    # endpoint is OpenAI-compatible, moving to another provider (Gemini's compat
    # endpoint, OpenRouter, Together, or OpenAI itself) is a change to
    # llm_base_url and llm_model, not a change to any code.
    llm_api_key: str = ""
    llm_base_url: str = "https://api.groq.com/openai/v1"
    # gpt-oss-120b is the most capable chat model on Groq's free tier and is
    # built for agentic tool use. The Llama 3.x models are Enterprise-only and
    # 404 on a free key. openai/gpt-oss-20b and qwen/qwen3.8-27b are the
    # smaller free alternatives worth benchmarking against it in Phase 6.
    llm_model: str = "openai/gpt-oss-120b"
    llm_max_tokens: int = 1024
    # gpt-oss is a reasoning model: it spends hidden tokens thinking before the
    # first spoken word. Measured on Groq, "low" cut time to first word from
    # ~1500 ms to ~500 ms (reasoning tokens 64 -> 11), which is the difference
    # between a natural pause and dead air on a phone call. Empty string omits
    # the parameter, for models that reject it.
    llm_reasoning_effort: str = "low"
    # Low, but not zero. A receptionist should be predictable rather than
    # inventive; a little variation stops it sounding like a recording.
    llm_temperature: float = 0.3

    # --- Speech (also Groq: one key covers the whole pipeline) ---------------
    stt_model: str = "whisper-large-v3-turbo"
    # Pinned, not auto-detected. Callers mix English and Tamil mid-sentence, and
    # left to guess, the model sometimes decides a whole utterance is Tamil and
    # translates it -- silently changing what the caller said.
    stt_language: str = "en"
    # Orpheus needs its terms accepted once per Groq organisation:
    # https://console.groq.com/playground?model=canopylabs%2Forpheus-v1-english
    tts_model: str = "canopylabs/orpheus-v1-english"
    # Groq's Orpheus voices: autumn, diana, hannah, austin, daniel, troy.
    tts_voice: str = "hannah"

    # --- Which speech service ---------------------------------------------------
    # "auto" picks Deepgram when a key is set and Groq otherwise, so a fresh clone
    # with only a free Groq key still has a working voice. Name one explicitly to
    # force it -- useful for comparing the two on the same call.
    stt_provider: Literal["auto", "deepgram", "groq"] = "auto"
    tts_provider: Literal["auto", "deepgram", "groq"] = "auto"

    # Deepgram: streaming transcription and streaming speech over WebSockets.
    # Free signup credit, no card: https://console.deepgram.com
    deepgram_api_key: str = ""
    deepgram_stt_model: str = "nova-3"
    # How much silence Deepgram waits for before it calls the utterance finished.
    # Shorter answers sooner but cuts off a caller who pauses mid-sentence. The
    # time from the caller stopping to the transcript being ready is this plus the
    # network round trip (~300 ms from India), so every 100 ms here is 100 ms
    # off every reply. A split is cheap to repair -- see "a pause that was not the
    # end" in voice/call.py -- which is what makes a short value affordable.
    deepgram_endpointing_ms: int = 400
    # Aura-2 voice. Chosen from the catalogue for being described as warm,
    # approachable and polite; `python -m scripts.list_voices` lists the others.
    deepgram_tts_model: str = "aura-2-cordelia-en"

    # --- Hearing the end of an utterance --------------------------------------------
    # How much silence the detector waits for before it closes an utterance. This
    # is the only end-of-speech clock when the transcriber is a batch service.
    vad_end_silence_ms: int = 800
    # When the transcription service ends utterances itself, the detector's timer
    # is only a backstop for the service never answering (noise that keeps it
    # open, a dropped socket). It has to be well beyond the service's own wait or
    # it wins the race, and every turn then pays a flush round trip for nothing.
    vad_backstop_ms: int = 1500

    # --- Filling the silence --------------------------------------------------------
    # The model asks for a tool without writing a word, so while it runs (a few
    # seconds: a model round trip, a database read, a second round trip) the line
    # is dead. With this on, the agent says a short stopgap the moment it sees the
    # request. It is a few words of speech, and it is timed separately so it cannot
    # flatter the latency of the answer.
    tool_filler: bool = True

    # --- Barge-in -----------------------------------------------------------------
    # The caller can talk over the agent and the agent stops. Turn it off to get
    # the half-duplex behaviour of Phase 2 (the agent ignores the mic while it
    # speaks) -- the right setting if a laptop's speakers are feeding the agent's
    # own voice back into its microphone and it keeps interrupting itself.
    barge_in: bool = True
    # How long the caller must keep talking before it counts as an interruption.
    # Long enough that a cough, a "mm-hm" or a click of the speakers does not cut
    # the agent off; short enough that a real "wait, no" lands within a syllable
    # or two.
    barge_in_ms: int = 300

    # --- Telephony ----------------------------------------------------------------
    # The address Twilio reaches this server on, e.g. https://abc.trycloudflare.com
    # Used to check webhook signatures (they are computed over the exact URL
    # Twilio called) and to tell Twilio where to open the audio stream.
    public_base_url: str = ""
    twilio_account_sid: str = ""
    twilio_auth_token: str = ""
    twilio_phone_number: str = ""

    # --- Booking policy -----------------------------------------------------
    slot_granularity_min: int = 15
    min_lead_min: int = 30
    max_slots_offered: int = 3

    # --- Later phases (declared so .env validates) --------------------------
    cartesia_api_key: str = ""
    whatsapp_token: str = ""
    whatsapp_phone_number_id: str = ""
    redis_url: str = "redis://localhost:6379/0"

    @property
    def stt_service(self) -> Literal["deepgram", "groq"]:
        return _resolve_provider(self.stt_provider, self.deepgram_api_key)

    @property
    def tts_service(self) -> Literal["deepgram", "groq"]:
        return _resolve_provider(self.tts_provider, self.deepgram_api_key)

    @property
    def db_connect_args(self) -> dict[str, object]:
        """Driver arguments, shared by the app engine and Alembic.

        On a transaction-pooled endpoint, prepared statements are turned off.
        This costs a little planning time per query and buys a connection that
        actually works; a session-mode or direct endpoint keeps them on.
        """
        if uses_transaction_pooling(self.database_url):
            return {"prepare_threshold": None}
        return {}


@lru_cache
def get_settings() -> Settings:
    return Settings()  # type: ignore[call-arg]  # values come from the environment
