"""Print the models the configured LLM key can actually use.

Provider model catalogues change, and free tiers carry a different set from the
one in the public docs -- a model listed there can still 404 on your key. That
failure surfaces as a generic NotFoundError mid-conversation, so this exists to
answer the question directly.

    python -m scripts.list_models
"""

from __future__ import annotations

from openai import AsyncOpenAI

from backend.app.config import get_settings
from backend.app.runtime import run_async

# Models that are not chat models, so not candidates for LLM_MODEL. Matched as
# substrings because providers version the names.
_NOT_CHAT = ("whisper", "prompt-guard", "orpheus", "safeguard", "embed", "tts")


async def main() -> None:
    settings = get_settings()
    if not settings.llm_api_key:
        raise SystemExit("LLM_API_KEY is not set. See .env")

    client = AsyncOpenAI(api_key=settings.llm_api_key, base_url=settings.llm_base_url)
    available = sorted(model.id for model in (await client.models.list()).data)

    chat = [m for m in available if not any(x in m.lower() for x in _NOT_CHAT)]
    other = [m for m in available if m not in chat]

    print(f"\n  {settings.llm_base_url}\n")
    print("  Chat models (candidates for LLM_MODEL):")
    for model in chat:
        marker = "  <- configured" if model == settings.llm_model else ""
        print(f"    {model}{marker}")

    if other:
        print("\n  Other (speech, moderation -- not chat):")
        for model in other:
            print(f"    {model}")

    print()
    if settings.llm_model in available:
        print(f"  LLM_MODEL={settings.llm_model} is available.")
    else:
        print(f"  LLM_MODEL={settings.llm_model} is NOT available on this key.")
        print("  Pick one of the chat models above and set it in .env.")
    print()


if __name__ == "__main__":
    run_async(main())
