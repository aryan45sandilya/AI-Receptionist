"""Play Twilio against a local server, to test the phone path without a phone.

Does what Twilio does, in the order it does it:

  1. POST the call to /twilio/voice, signed with the auth token the way Twilio
     signs it, and read the TwiML that comes back;
  2. open the audio stream that TwiML names, and send ``connected`` and ``start``
     with the parameters the TwiML carried;
  3. stream the caller's audio as 8 kHz mu-law, 20 ms at a time, in real time;
  4. receive the agent's audio and ``clear`` messages, and send ``stop`` at the end.

So a run exercises the real signature check, the real stream token, the real
mu-law conversion both ways and the same call controller as a live call. What it
cannot show is Twilio itself: a real handshake, real network jitter, a real
handset. A call from a phone is checked by hand, and reported as such.

    TWILIO_AUTH_TOKEN=<any value> python -m backend.app.serve        # in one terminal
    TWILIO_AUTH_TOKEN=<same value> python -m scripts.twilio_sim scratch/speech/01_booking.wav

Add --talk-over to start each later file a second into the agent's reply.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import secrets
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from urllib.parse import urlsplit

import httpx2 as httpx
import websockets
from sqlalchemy import select

from backend.app.config import get_settings
from backend.app.db import dispose_engine, get_session_factory
from backend.app.models import Business, Call, CallTurn
from backend.app.runtime import run_async, use_utf8_console
from backend.app.telephony.twilio import compute_signature
from backend.app.voice.audio import from_wav, resample, to_mono
from backend.app.voice.codec import ulaw_encode

RATE = 8_000
FRAME_MS = 20
FRAME_BYTES = RATE * FRAME_MS // 1000  # mu-law is one byte per sample
TRAILING_SILENCE_MS = 1400


def load(path: Path) -> bytes:
    """A WAV file as the mu-law Twilio would send, with quiet after it so the
    server can tell the caller has finished."""
    pcm = resample(to_mono(from_wav(path.read_bytes())), RATE)
    return ulaw_encode(pcm.data + b"\x00\x00" * (RATE * TRAILING_SILENCE_MS // 1000))


async def stream_audio(ws, mulaw: bytes, stream_sid: str) -> None:
    started = time.perf_counter()
    for i, offset in enumerate(range(0, len(mulaw), FRAME_BYTES)):
        payload = base64.b64encode(mulaw[offset : offset + FRAME_BYTES]).decode()
        await ws.send(json.dumps({
            "event": "media", "streamSid": stream_sid,
            "media": {"track": "inbound", "chunk": str(i + 1), "timestamp": str(i * FRAME_MS), "payload": payload},
        }))
        await asyncio.sleep(max(0.0, started + (i + 1) * FRAME_MS / 1000 - time.perf_counter()))


async def run(host: str, files: list[Path], timeout: float, talk_over: float | None, dialed: str | None) -> None:
    settings = get_settings()
    token = settings.twilio_auth_token.strip()
    if not token:
        raise SystemExit("TWILIO_AUTH_TOKEN is not set (any value will do for the simulator, "
                         "as long as the server has the same one)")

    # The URL Twilio signs is the one it requested, which is the server's public
    # address when one is configured.
    base_url = (settings.public_base_url.strip().rstrip("/") or f"http://{host}")
    local_http = f"http://{host}"
    call_sid = "CA" + secrets.token_hex(16)

    if dialed is None:
        async with get_session_factory()() as db:
            business = (await db.execute(select(Business).order_by(Business.created_at))).scalars().first()
            dialed = business.phone_e164 if business else "+918000000000"
    form = {"CallSid": call_sid, "From": "+919876543210", "To": dialed, "Direction": "inbound", "CallStatus": "ringing"}

    t0 = time.perf_counter()

    def stamp() -> str:
        return f"{time.perf_counter() - t0:7.2f}s"

    async with httpx.AsyncClient(timeout=15) as http:
        # Twilio calls the public address; here that is the local server, so the
        # request goes there while the signature is over the configured URL.
        url = f"{base_url}/twilio/voice"
        sig = compute_signature(token, url, sorted(form.items()))
        response = await http.post(f"{local_http}/twilio/voice", data=form, headers={
            "X-Twilio-Signature": sig, "Host": urlsplit(base_url).netloc,
            "X-Forwarded-Proto": urlsplit(base_url).scheme,
        })
    if response.status_code != 200:
        raise SystemExit(f"the webhook answered {response.status_code}: {response.text[:200]}")
    stream = ET.fromstring(response.text).find("./Connect/Stream")
    if stream is None:
        raise SystemExit(f"the TwiML has no <Connect><Stream>: {response.text[:200]}")
    stream_url = stream.get("url", "")
    parameters = {p.get("name", ""): p.get("value", "") for p in stream.findall("Parameter")}
    print(f"  {stamp()}  webhook accepted; TwiML points at {stream_url}")
    print(f"  {stamp()}  stream parameters: {sorted(parameters)} (token {len(parameters.get('token', ''))} chars)")

    # The TwiML names the public address; connect to the local server instead.
    ws_url = f"ws://{host}/twilio/stream"
    stream_sid = "MZ" + secrets.token_hex(16)
    received_bytes = 0
    clears = 0
    first_media_at: float | None = None

    async with websockets.connect(ws_url, max_size=None) as ws:
        await ws.send(json.dumps({"event": "connected", "protocol": "Call", "version": "1.0.0"}))
        await ws.send(json.dumps({
            "event": "start", "sequenceNumber": "1", "streamSid": stream_sid,
            "start": {
                "accountSid": "AC" + "0" * 32, "streamSid": stream_sid, "callSid": call_sid,
                "tracks": ["inbound"],
                "mediaFormat": {"encoding": "audio/x-mulaw", "sampleRate": 8000, "channels": 1},
                "customParameters": parameters,
            },
        }))
        print(f"  {stamp()}  stream open")

        speaking = asyncio.Event()
        idle = asyncio.Event()
        last_media = 0.0

        async def reader() -> None:
            nonlocal received_bytes, clears, first_media_at, last_media
            try:
                async for raw in ws:
                    message = json.loads(raw)
                    event = message.get("event")
                    if event == "media":
                        now = time.perf_counter()
                        if first_media_at is None or now - last_media > 1.0:
                            print(f"  {stamp()}  agent audio begins")
                        first_media_at = first_media_at or now
                        last_media = now
                        received_bytes += len(base64.b64decode(message["media"]["payload"]))
                        speaking.set()
                        idle.clear()
                    elif event == "clear":
                        clears += 1
                        print(f"  {stamp()}  clear (the agent was interrupted)")
                        speaking.clear()
                    else:
                        print(f"  {stamp()}  {event}")
            except websockets.ConnectionClosed:
                pass

        async def watch_idle() -> None:
            """The agent is quiet when no audio has arrived for a while."""
            while True:
                await asyncio.sleep(0.2)
                if speaking.is_set() and time.perf_counter() - last_media > 1.2:
                    speaking.clear()
                    idle.set()

        tasks = [asyncio.create_task(reader()), asyncio.create_task(watch_idle())]
        try:
            for index, path in enumerate(files, start=1):
                if index == 1:
                    # The greeting plays first.
                    await asyncio.wait_for(speaking.wait(), timeout)
                    await asyncio.wait_for(idle.wait(), timeout)
                elif talk_over is not None:
                    await asyncio.wait_for(speaking.wait(), timeout)
                    await asyncio.sleep(talk_over)
                else:
                    await asyncio.wait_for(idle.wait(), timeout)
                print(f"\n  {stamp()}  >>> caller turn {index}: {path.name}"
                      + ("  (talking over the agent)" if talk_over is not None and index > 1 else ""))
                idle.clear()
                await stream_audio(ws, load(path), stream_sid)
                await asyncio.wait_for(speaking.wait(), timeout)
                await asyncio.wait_for(idle.wait(), timeout)
            await asyncio.sleep(0.5)
        finally:
            await ws.send(json.dumps({"event": "stop", "streamSid": stream_sid, "stop": {"callSid": call_sid}}))
            for task in tasks:
                task.cancel()

    print(f"\n  agent audio received: {received_bytes:,} bytes of mu-law = {received_bytes / RATE:.1f} s; "
          f"{clears} clear message(s)")
    await show_call(call_sid)


async def show_call(call_sid: str) -> None:
    await asyncio.sleep(1.5)  # the server finishes writing just after the stream ends
    async with get_session_factory()() as db:
        call = (await db.execute(select(Call).where(Call.twilio_call_sid == call_sid))).scalars().first()
        if call is None:
            print("\n  no stored call found for this sid")
            return
        turns = (await db.execute(
            select(CallTurn).where(CallTurn.call_id == call.id).order_by(CallTurn.turn_index)
        )).scalars().all()

    print(f"\n  stored call {call.id}")
    print(f"    status={call.status.value}  channel={call.channel.value}  from={call.from_phone_e164}  "
          f"sid={call.twilio_call_sid}  turns={len(turns)}")
    print(f"    {'#':>2} {'who':<7}{'stt':>6}{'llm1st':>8}{'llmdone':>8}{'tts1st':>8}{'total':>7}  text")
    for t in turns:
        def cell(v, w):
            return f"{v if v is not None else '-':>{w}}"
        print(
            f"    {t.turn_index:>2} {t.role.value:<7}"
            f"{cell(t.stt_final_ms, 6)}{cell(t.llm_first_token_ms, 8)}"
            f"{cell(t.llm_done_ms, 8)}{cell(t.tts_first_audio_ms, 8)}{cell(t.total_ms, 7)}"
            f"  {'[cut off] ' if t.barge_in else ''}{(t.text or '')[:60]}"
        )


def main() -> None:
    use_utf8_console()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("files", nargs="+", type=Path, help="WAV files, one per caller turn")
    parser.add_argument("--host", default="127.0.0.1:8000")
    parser.add_argument("--dialed", help="the number the caller dialled (default: the first business's)")
    parser.add_argument("--timeout", type=float, default=90.0, help="seconds to wait per turn")
    parser.add_argument("--talk-over", nargs="?", type=float, const=1.0, default=None, metavar="SECONDS",
                        help="start each file after the first this long into the agent's reply")
    args = parser.parse_args()

    missing = [f for f in args.files if not f.exists()]
    if missing:
        raise SystemExit(f"No such file: {missing[0]}")

    async def go() -> None:
        try:
            await run(args.host, args.files, args.timeout, args.talk_over, args.dialed)
        finally:
            await dispose_engine()

    run_async(go())


if __name__ == "__main__":
    main()
