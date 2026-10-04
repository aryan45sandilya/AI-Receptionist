"""What a call is carried on.

A call is the same conversation whether it comes from a browser tab or a
telephone: the caller's audio arrives, the agent's audio goes back, and some
events travel alongside. What differs is the wire. The browser's WebSocket sends
raw 16 kHz PCM and JSON events; Twilio's sends JSON only, with the audio inside as
base64 mu-law at 8 kHz. Everything that decides how the conversation goes (when
the caller is speaking, when to answer, when to stop talking because they
interrupted) lives in ``call.py`` and sees only this interface.

Audio crossing it is always 16-bit PCM. A transport converts at its own edge.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
from dataclasses import dataclass, field
from typing import Any, Protocol

from fastapi import WebSocketDisconnect


@dataclass(frozen=True)
class Control:
    """Something from the far end that is not audio: a hang-up, a keypress."""

    kind: str
    data: dict[str, Any] = field(default_factory=dict)


class CallTransport(Protocol):
    # Sample rate of the PCM that ``receive`` returns.
    input_rate: int
    # Sample rate the far end needs audio sent at, or None if it takes any rate
    # (a browser resamples to suit its own speakers).
    output_rate: int | None

    async def receive(self) -> bytes | Control | None:
        """The next thing from the caller: PCM audio, a control message, or None
        once the call is over."""
        ...

    async def send_event(self, kind: str, **fields: Any) -> None:
        """A progress event, for a screen. A telephone has none and ignores it."""
        ...

    async def send_audio(self, pcm: bytes) -> None:
        """Agent audio, as PCM at ``output_rate`` (or at the rate last announced
        in an ``audio_start`` event, for a transport with no fixed one)."""
        ...

    async def clear_audio(self) -> None:
        """Throw away audio that has been sent but not yet played: the caller has
        started talking, and whatever the far end is still holding is stale."""
        ...

    async def transfer(self, number: str) -> bool:
        """Hand the call to a person at ``number``. False if this transport has no
        way to (a browser tab has no one to put through to), in which case the
        caller falls back to ending the call."""
        ...

    async def close(self) -> None: ...


class BrowserTransport:
    """The demo page's WebSocket: binary frames are audio, text frames are JSON.

    Takes anything with Starlette's WebSocket interface, so tests can drive it
    with a stand-in. Two tasks send on the one socket (the pump's events, a
    turn's audio); a lock keeps their frames from interleaving.
    """

    input_rate = 16_000
    output_rate: int | None = None

    def __init__(self, websocket: Any) -> None:
        self.ws = websocket
        self._send_lock = asyncio.Lock()

    async def receive(self) -> bytes | Control | None:
        try:
            message = await self.ws.receive()
        except WebSocketDisconnect:
            return None
        except RuntimeError as exc:
            # Starlette raises this for a receive() after the socket closed,
            # which is what happens when we close it ourselves on handoff.
            if "disconnect" in str(exc).lower():
                return None
            raise

        if message.get("type") == "websocket.disconnect":
            return None

        text = message.get("text")
        if text is not None:
            try:
                data = json.loads(text)
            except json.JSONDecodeError:
                return Control("ignored")
            return Control(str(data.get("type", "ignored")), data)

        chunk = message.get("bytes")
        return chunk if chunk else Control("ignored")

    async def send_event(self, kind: str, **fields: Any) -> None:
        async with self._send_lock:
            await self.ws.send_text(json.dumps({"type": kind, **fields}))

    async def send_audio(self, pcm: bytes) -> None:
        async with self._send_lock:
            await self.ws.send_bytes(pcm)

    async def clear_audio(self) -> None:
        # The page empties its playback queue on this.
        await self.send_event("clear")

    async def transfer(self, number: str) -> bool:
        return False

    async def close(self) -> None:
        with contextlib.suppress(Exception):
            await self.ws.close(code=1000)
