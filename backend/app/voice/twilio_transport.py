"""The Twilio side of a call: Media Streams over a WebSocket.

Twilio opens the socket after answering the call and speaks JSON only. Audio is
inside the JSON, base64 encoded, as 8 kHz mu-law -- 20 ms (160 bytes) at a time
inbound. The conversation is the same one the browser has; this class converts at
the edge so everything behind it sees 16-bit PCM and an ordinary ``CallTransport``.

Messages from Twilio::

    connected   the socket is open
    start       the stream began: its id, the call's id, and our own parameters
    media       20 ms of the caller's audio
    dtmf        a keypress
    mark        a marker we sent has been reached in the playback
    stop        the call ended

Messages to Twilio::

    media       audio to play, as mu-law. Buffered and played in the order sent.
    clear       drop everything buffered and not yet played -- barge-in
    mark        (unused) a marker to be told about when playback reaches it

The only thing here to be careful about is ``clear``: Twilio plays audio from its
own buffer, so what has been *sent* is not what the caller has *heard*. Pacing
audio to real time (see ``pipeline.py``) keeps that buffer to a quarter second,
and ``clear`` empties it the moment the caller interrupts.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import json
import logging
from dataclasses import dataclass, field
from typing import Any

from fastapi import WebSocketDisconnect

from ..telephony.twilio import TwilioRest, dial_twiml
from .codec import TELEPHONE_RATE, ulaw_decode, ulaw_encode
from .transport import Control

log = logging.getLogger(__name__)

# Twilio sends "connected" and "start" at once. A socket that says neither is not
# Twilio, or is Twilio on a call that has already gone.
HANDSHAKE_TIMEOUT_S = 10.0


@dataclass(frozen=True)
class StreamStart:
    """What the opening message says about the stream."""

    stream_sid: str
    call_sid: str
    parameters: dict[str, str] = field(default_factory=dict)


async def read_handshake(websocket: Any, *, timeout: float = HANDSHAKE_TIMEOUT_S) -> StreamStart | None:
    """Read Twilio's opening messages; None if they do not look like Twilio's."""
    try:
        async with asyncio.timeout(timeout):
            while True:
                message = json.loads(await websocket.receive_text())
                event = message.get("event")
                if event == "connected":
                    continue
                if event != "start":
                    return None
                start = message.get("start") or {}
                return StreamStart(
                    stream_sid=str(start.get("streamSid") or message.get("streamSid") or ""),
                    call_sid=str(start.get("callSid") or ""),
                    parameters={str(k): str(v) for k, v in (start.get("customParameters") or {}).items()},
                )
    except (TimeoutError, WebSocketDisconnect, ValueError, AttributeError):
        return None


class TwilioTransport:
    """A phone call, as the call controller sees it."""

    input_rate = TELEPHONE_RATE
    output_rate: int | None = TELEPHONE_RATE

    def __init__(
        self,
        websocket: Any,
        *,
        stream_sid: str,
        call_sid: str,
        rest: TwilioRest | None = None,
    ) -> None:
        self.ws = websocket
        self.stream_sid = stream_sid
        self.call_sid = call_sid
        self._rest = rest
        self._send_lock = asyncio.Lock()

    async def receive(self) -> bytes | Control | None:
        while True:
            try:
                message = json.loads(await self.ws.receive_text())
            except WebSocketDisconnect:
                return None
            except ValueError:
                continue  # not JSON; nothing to act on

            event = message.get("event")
            if event == "media":
                media = message.get("media") or {}
                # Only the caller's audio. (Bidirectional streams carry just the
                # inbound track, but a message for another is not ours to play.)
                if media.get("track", "inbound") != "inbound":
                    continue
                try:
                    payload = base64.b64decode(media.get("payload", ""), validate=True)
                except ValueError:
                    continue
                if payload:
                    return ulaw_decode(payload)
            elif event == "dtmf":
                return Control("dtmf", message.get("dtmf") or {})
            elif event == "mark":
                return Control("mark", message.get("mark") or {})
            elif event == "stop":
                return None

    async def send_event(self, kind: str, **fields: Any) -> None:
        # A telephone has no screen. Events are for the browser demo's page.
        return None

    async def send_audio(self, pcm: bytes) -> None:
        payload = base64.b64encode(ulaw_encode(pcm)).decode("ascii")
        await self._send({"event": "media", "streamSid": self.stream_sid, "media": {"payload": payload}})

    async def clear_audio(self) -> None:
        await self._send({"event": "clear", "streamSid": self.stream_sid})

    async def transfer(self, number: str) -> bool:
        """Move the live call onto a person's phone; False if that cannot be done.

        Done by replacing the call's instructions, not by closing the stream:
        Twilio ends the stream itself when the call is redirected.
        """
        if self._rest is None or not self.call_sid:
            return False
        try:
            await self._rest.redirect_call(self.call_sid, dial_twiml(number))
        except Exception:  # noqa: BLE001 - the caller is still on the line; fall back to hanging up
            log.exception("could not transfer the call")
            return False
        return True

    async def close(self) -> None:
        # Closing the stream is what ends a call that has no more instructions.
        with contextlib.suppress(Exception):
            await self.ws.close(code=1000)

    async def _send(self, message: dict[str, Any]) -> None:
        async with self._send_lock:
            await self.ws.send_text(json.dumps(message))
