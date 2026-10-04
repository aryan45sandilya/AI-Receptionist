"""The browser demo's voice WebSocket: the door into a call.

One socket carries a whole call. The browser streams 16 kHz mono PCM up and gets
the agent's speech back as PCM, with JSON events alongside (binary frames are
audio, text frames are control). The call itself -- turn-taking, interruption,
the model, the logging -- is ``voice/call.py`` and does not know it is talking to
a browser; this file only plugs the browser's transport into it.
"""

from __future__ import annotations

import contextlib
import logging

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from ..voice.call import VoiceCall
from ..voice.transport import BrowserTransport

log = logging.getLogger(__name__)
router = APIRouter()


@router.websocket("/ws/voice")
async def voice_socket(websocket: WebSocket) -> None:
    await websocket.accept()
    call = VoiceCall(BrowserTransport(websocket))
    try:
        await call.run()
    except WebSocketDisconnect:
        pass
    except Exception as exc:  # noqa: BLE001 - tell the page rather than leave it hanging
        log.exception("voice call failed")
        await call.report_fatal(f"{type(exc).__name__}: {exc}")
    finally:
        with contextlib.suppress(Exception):
            await websocket.close()
