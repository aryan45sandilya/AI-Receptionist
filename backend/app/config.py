"""Settings, loaded from the environment (and .env in development).

Nothing here has a secret as its default. A missing DATABASE_URL should fail
loudly at startup rather than quietly fall back to something local, because the
failure mode of the quiet version is writing test bookings into production.
"""

from __future__ import annotations

from functools import lru_cache
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
    # Low, but not zero. A receptionist should be predictable rather than
    # inventive; a little variation stops it sounding like a recording.
    llm_temperature: float = 0.3

    # --- Booking policy -----------------------------------------------------
    slot_granularity_min: int = 15
    min_lead_min: int = 30
    max_slots_offered: int = 3

    # --- Phase 2+ (declared so .env validates; unused in Phase 1) -----------
    deepgram_api_key: str = ""
    cartesia_api_key: str = ""
    twilio_account_sid: str = ""
    twilio_auth_token: str = ""
    whatsapp_token: str = ""
    whatsapp_phone_number_id: str = ""
    redis_url: str = "redis://localhost:6379/0"

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
