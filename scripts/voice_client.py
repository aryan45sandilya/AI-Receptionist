"""Drive the voice WebSocket from WAV files, the way the browser does.

A headless caller: streams each file as microphone audio in real time, waits for
the agent to finish and go back to listening, then plays the next. Prints every
event with its timestamp and finishes by reading the call back out of the
database, so the latency columns can be checked against what actually happened.

    python -m scripts.voice_client scratch/speech/01_booking.wav scratch/speech/02_confirm.wav

To interrupt the agent instead of waiting for it to finish, add --talk-over: each
file after the first starts playing a second into the agent's reply.

    python -m scripts.voice_client --talk-over scratch/speech/01_booking.wav scratch/speech/02_confirm.wav

Audio is sent at real-time speed on purpose. The server's timing is measured from
when the caller stops talking, and a client that blasted the file over instantly
would make every latency figure meaningless.

Generate test speech without a microphone:

    powershell -ExecutionPolicy Bypass -File scripts/make_test_speech.ps1
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path

import websockets
from sqlalchemy import select

from backend.app.db import dispose_engine, get_session_factory
from backend.app.models import Call, CallTurn
from backend.app.runtime import run_async
from backend.app.voice.audio import from_wav, resample, to_mono
from backend.app.voice.vad import FRAME_MS, SAMPLE_RATE

# Matches the browser's capture chunk: 40 ms.
CHUNK_MS = 40
CHUNK_BYTES = SAMPLE_RATE * 2 * CHUNK_MS // 1000
# Enough quiet after each file for the server's detector to close the utterance.
TRAILING_SILENCE_MS = 1400


def load(path: Path) -> bytes:
    pcm = resample(to_mono(from_wav(path.read_bytes())), SAMPLE_RATE)
    return pcm.data + b"\x00\x00" * (SAMPLE_RATE * TRAILING_SILENCE_MS // 1000)


async def stream(ws, pcm: bytes) -> None:
    """Send audio at real-time speed, in the browser's chunk size."""
    started = time.perf_counter()
    for i, offset in enumerate(range(0, len(pcm), CHUNK_BYTES)):
        await ws.send(pcm[offset : offset + CHUNK_BYTES])
        # Pace against the wall clock rather than sleeping a fixed amount, so
        # send time and scheduling jitter do not accumulate into drift.
        due = started + (i + 1) * CHUNK_MS / 1000
        await asyncio.sleep(max(0.0, due - time.perf_counter()))


async def run(host: str, files: list[Path], timeout: float, talk_over: float | None = None) -> None:
    t0 = time.perf_counter()
    audio_bytes = 0
    call_id: str | None = None

    def stamp() -> str:
        return f"{(time.perf_counter() - t0):7.2f}s"

    async with websockets.connect(f"ws://{host}/ws/voice", max_size=None) as ws:
        listening = asyncio.Event()
        speaking = asyncio.Event()
        finished = asyncio.Event()
        turn_events: list[dict] = []

        async def reader() -> None:
            nonlocal audio_bytes, call_id
            async for message in ws:
                if isinstance(message, bytes):
                    audio_bytes += len(message)
                    continue
                event = json.loads(message)
                kind = event["type"]
                detail = {k: v for k, v in event.items() if k != "type"}
                # Deltas arrive a few words at a time; print them compactly.
                if kind == "agent_delta":
                    print(f"  {stamp()}  agent...  {detail['text']!r}")
                else:
                    print(f"  {stamp()}  {kind:<15}{json.dumps(detail, ensure_ascii=False)[:220]}")
                if kind == "ready":
                    call_id = event["call_id"]
                if kind == "turn_done":
                    turn_events.append(event)
                if kind == "audio_start":
                    speaking.set()
                if kind in {"clear", "audio_end"}:
                    speaking.clear()
                if kind == "listening":
                    listening.set()
                if kind in {"fatal", "turn_error"}:
                    listening.set()
                    finished.set()
                if kind == "handoff":
                    finished.set()

        reading = asyncio.create_task(reader())
        try:
            for index, path in enumerate(files, start=1):
                if finished.is_set():
                    break
                if talk_over is not None and index > 1:
                    # Cut in: wait until the agent is mid-reply, then talk.
                    await asyncio.wait_for(speaking.wait(), timeout)
                    await asyncio.sleep(talk_over)
                    print(f"\n  {stamp()}  >>> caller talks over the agent: {path.name}")
                else:
                    # The greeting plays first, and a turn is only heard once the
                    # server says it is listening again.
                    await asyncio.wait_for(listening.wait(), timeout)
                    print(f"\n  {stamp()}  >>> caller turn {index}: {path.name}")
                listening.clear()
                await stream(ws, load(path))
                if talk_over is not None and index < len(files):
                    # Do not wait for the reply: the next file is meant to cut in on it.
                    continue
                # Wait for this turn's reply to finish, or the call to end.
                await asyncio.wait_for(_first_of(listening, finished), timeout)
            await asyncio.sleep(0.5)
        finally:
            try:
                await ws.send(json.dumps({"type": "hangup"}))
            except websockets.ConnectionClosed:
                pass
            reading.cancel()

    print(f"\n  audio received from server: {audio_bytes:,} bytes "
          f"({audio_bytes / 2 / 24000:.1f}s at 24 kHz)")
    await show_call(call_id)


async def _first_of(*events: asyncio.Event) -> None:
    waiters = [asyncio.ensure_future(e.wait()) for e in events]
    try:
        await asyncio.wait(waiters, return_when=asyncio.FIRST_COMPLETED)
    finally:
        for w in waiters:
            w.cancel()


async def show_call(call_id: str | None) -> None:
    """Read the call back out of Postgres: what was stored is what counts."""
    if not call_id:
        return
    await asyncio.sleep(1.0)  # the server finishes writing just after hangup
    async with get_session_factory()() as db:
        call = await db.get(Call, call_id)
        turns = (
            await db.execute(
                select(CallTurn).where(CallTurn.call_id == call_id).order_by(CallTurn.turn_index)
            )
        ).scalars().all()

    print(f"\n  stored call {call_id}")
    print(f"    status={call.status.value}  channel={call.channel.value}  turns={len(turns)}")
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
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("files", nargs="+", type=Path, help="16 kHz mono WAV files, one per caller turn")
    parser.add_argument("--host", default="127.0.0.1:8000")
    parser.add_argument("--timeout", type=float, default=60.0, help="seconds to wait per turn")
    parser.add_argument(
        "--talk-over", nargs="?", type=float, const=1.0, default=None, metavar="SECONDS",
        help="start each file after the first this long into the agent's reply, "
             "instead of waiting for it to finish (default 1.0)",
    )
    args = parser.parse_args()

    missing = [f for f in args.files if not f.exists()]
    if missing:
        raise SystemExit(f"No such file: {missing[0]}\nCreate some with scripts/make_test_speech.ps1")

    async def go() -> None:
        try:
            await run(args.host, args.files, args.timeout, args.talk_over)
        finally:
            await dispose_engine()

    run_async(go())


if __name__ == "__main__":
    main()
