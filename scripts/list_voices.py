"""List the Deepgram speech voices the configured key can use.

Aura voices differ in accent, register and tone. Pick one and set it as
DEEPGRAM_TTS_MODEL in .env; ``smoke_deepgram --save out.wav`` writes a sample
sentence in the configured voice so you can hear it before a call does.

    python -m scripts.list_voices              # English voices
    python -m scripts.list_voices --all        # every language
"""

from __future__ import annotations

import argparse

import httpx2 as httpx

from backend.app.config import get_settings
from backend.app.runtime import run_async, use_utf8_console


async def main(show_all: bool) -> None:
    settings = get_settings()
    key = settings.deepgram_api_key.strip()
    if not key:
        raise SystemExit("DEEPGRAM_API_KEY is not set. See .env")

    async with httpx.AsyncClient(timeout=20) as client:
        response = await client.get("https://api.deepgram.com/v1/models", headers={"Authorization": f"Token {key}"})
    if response.status_code != 200:
        raise SystemExit(f"Deepgram answered {response.status_code}: {response.text[:200]}")

    voices = sorted(response.json().get("tts", []), key=lambda v: v["canonical_name"])
    if not show_all:
        voices = [v for v in voices if "en" in (v.get("languages") or [])]

    print(f"\n  {'voice':<26}{'accent':<14}character")
    for voice in voices:
        meta = voice.get("metadata") or {}
        marker = "  <- configured" if voice["canonical_name"] == settings.deepgram_tts_model else ""
        print(f"  {voice['canonical_name']:<26}{meta.get('accent', ''):<14}{', '.join(meta.get('tags', []))}{marker}")
    print(f"\n  {len(voices)} voices. Set DEEPGRAM_TTS_MODEL in .env to change the one in use.\n")


if __name__ == "__main__":
    use_utf8_console()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--all", action="store_true", help="include voices for other languages")
    run_async(main(parser.parse_args().all))
