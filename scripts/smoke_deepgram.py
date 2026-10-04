"""Check the Deepgram key and settings against the live service.

Speaks a sentence with the configured voice, then feeds that audio back through
the transcriber -- so one run proves the key works for both services, that the
audio format round-trips, and shows the latencies from this machine:

    python -m scripts.smoke_deepgram            # browser audio: 16 kHz
    python -m scripts.smoke_deepgram --phone    # telephone audio: 8 kHz, what Twilio sends

Costs a fraction of a cent of the free credit. Prints times, never the key.
"""

from __future__ import annotations

import argparse
import asyncio
import time
from pathlib import Path

from backend.app.config import get_settings
from backend.app.runtime import run_async, use_utf8_console
from backend.app.voice.audio import Pcm, to_wav
from backend.app.voice.deepgram import DeepgramTranscriber, DeepgramTTS

SENTENCE = "Hi, I need a dental cleaning next Tuesday evening with Doctor Rao."


async def run(rate: int, save: Path | None, language: str | None) -> None:
    settings = get_settings()
    key = settings.deepgram_api_key.strip()
    if not key:
        raise SystemExit("DEEPGRAM_API_KEY is not set in .env")

    # --- speech -----------------------------------------------------------
    tts = DeepgramTTS(key, model=settings.deepgram_tts_model, sample_rate=rate)
    t = time.perf_counter()
    await tts.start()
    print(f"speech socket opened         {(time.perf_counter() - t) * 1000:6.0f} ms")
    if not tts.connected:
        raise SystemExit(f"could not open the speech socket: {tts.last_error}")

    chunks: list[Pcm] = []
    t = time.perf_counter()
    first = None
    async for chunk in tts.stream(SENTENCE):
        if first is None:
            first = (time.perf_counter() - t) * 1000
        chunks.append(chunk)
    total = (time.perf_counter() - t) * 1000
    audio = b"".join(c.data for c in chunks)
    seconds = len(audio) / 2 / rate
    print(f"first chunk of speech        {first:6.0f} ms")
    print(f"whole sentence made          {total:6.0f} ms   ({seconds:.1f} s of audio, {settings.deepgram_tts_model})")
    await tts.close()
    if save:
        save.write_bytes(to_wav(Pcm(data=audio, sample_rate=rate)))
        print(f"saved                        {save}")

    # --- transcription ----------------------------------------------------
    stt = DeepgramTranscriber(
        key,
        model=settings.deepgram_stt_model,
        language=language or settings.stt_language,
        sample_rate=rate,
        endpointing_ms=settings.deepgram_endpointing_ms,
    )
    interim: list[str] = []
    ended = asyncio.Event()
    stt.on_interim = interim.append
    stt.on_endpoint = ended.set

    t = time.perf_counter()
    await stt.start(keyterms=["Dental cleaning", "Dr Rao"])
    print(f"transcription socket opened  {(time.perf_counter() - t) * 1000:6.0f} ms")
    if not stt.connected:
        raise SystemExit(f"could not open the transcription socket: {stt.last_error}")

    chunk_bytes = rate * 2 * 20 // 1000  # 20 ms
    silence = b"\x00\x00" * (rate * settings.deepgram_endpointing_ms * 3 // 1000)
    stt.begin()
    started = time.perf_counter()
    for i, offset in enumerate(range(0, len(audio) + len(silence), chunk_bytes)):
        piece = (audio + silence)[offset : offset + chunk_bytes]
        await stt.send(piece)
        # Real-time pacing: the service's endpointing is a function of the audio
        # clock, and a client that blasted the file over would see none of it.
        await asyncio.sleep(max(0.0, started + (i + 1) * 0.02 - time.perf_counter()))
        if ended.is_set():
            break
    spoke_until = started + seconds
    heard_end = time.perf_counter()
    text = await stt.finish()
    print(f"words finished after speech  {(heard_end - spoke_until) * 1000:6.0f} ms   (endpointing {settings.deepgram_endpointing_ms} ms)")
    print(f"interim updates seen         {len(interim):6d}")
    print(f"said:        {SENTENCE}")
    print(f"transcribed: {text}")
    await stt.close()


def main() -> None:
    use_utf8_console()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--phone", action="store_true", help="8 kHz, the telephone rate")
    parser.add_argument("--save", type=Path, help="write the spoken sentence to this WAV file")
    parser.add_argument("--language", help="override STT_LANGUAGE, e.g. en-IN")
    args = parser.parse_args()
    run_async(run(8_000 if args.phone else 16_000, args.save, args.language))


if __name__ == "__main__":
    main()
