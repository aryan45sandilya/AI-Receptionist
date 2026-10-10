"""Twilio's two doors: the webhook that answers a call, and the audio stream.

A call reaches the agent in two steps. Twilio first asks ``POST /twilio/voice``
what to do; the answer is TwiML telling it to open a WebSocket to
``/twilio/stream`` and hand its audio over. Everything after that is the same
conversation the browser demo has -- see ``voice/call.py``.

Both doors are closed by default. The webhook is public, so it answers only a
request that carries a valid Twilio signature -- or, for the unsigned requests a
trial account's calls turned out to send, one whose call Twilio's own API
confirms -- and the stream starts only for a caller holding the token the webhook
minted. Neither opens at all until ``TWILIO_AUTH_TOKEN`` is set.
"""

from __future__ import annotations

import contextlib
import logging
import re
import time
from collections import deque
from urllib.parse import parse_qsl

from fastapi import APIRouter, Request, Response, WebSocket

from ..config import Settings, get_settings
from ..enums import CallChannel
from ..telephony.twilio import (
    TwilioRest,
    call_matches,
    connect_twiml,
    say_twiml,
    valid_signature,
    valid_stream_token,
    mint_stream_token,
)
from ..voice.call import VoiceCall
from ..voice.twilio_transport import TwilioTransport, read_handshake

log = logging.getLogger(__name__)
router = APIRouter(prefix="/twilio")

# 1008: policy violation. What a WebSocket server says to a client it will not serve.
POLICY_VIOLATION = 1008

NOT_CONFIGURED = "This number is not set up yet. Goodbye."
SIGNATURE_HEADER = "X-Twilio-Signature"


def public_base(request: Request, settings: Settings) -> str:
    """The address Twilio reaches this server on.

    From ``PUBLIC_BASE_URL`` when set. Otherwise worked out from the request,
    trusting the forwarding headers a tunnel or a platform proxy adds -- which is
    convenient in development and exactly why production should set the variable:
    the signature check is only as sound as the URL it is computed over.
    """
    if settings.public_base_url.strip():
        return settings.public_base_url.strip().rstrip("/")
    scheme = request.headers.get("x-forwarded-proto", request.url.scheme).split(",")[0].strip()
    host = request.headers.get("x-forwarded-host") or request.headers.get("host") or request.url.netloc
    return f"{scheme}://{host.split(',')[0].strip()}"


def stream_url(base: str) -> str:
    """The WebSocket address for the same server: http becomes ws, https wss."""
    scheme = "wss" if base.startswith("https://") else "ws"
    return f"{scheme}://{base.split('://', 1)[1]}/twilio/stream"


def _parties(form: dict[str, str]) -> tuple[str, str]:
    """(the person calling, the number they reached).

    For an inbound call that is From and To. For one Twilio placed -- the test
    call the project makes to a phone -- Twilio's own number is From and the
    person is To, so the two swap: the caller is always the human.
    """
    if form.get("Direction", "").startswith("outbound"):
        return form.get("To", ""), form.get("From", "")
    return form.get("From", ""), form.get("To", "")


CALL_SID = re.compile(r"CA[0-9a-fA-F]{32}")
# Each unsigned request costs a call to Twilio's API, so a flood of forged ones must
# not become a flood of those: past this many a minute, they are refused unasked.
UNSIGNED_CHECKS_PER_MINUTE = 20
_unsigned_checks: deque[float] = deque()


def _within_unsigned_budget() -> bool:
    now = time.monotonic()
    while _unsigned_checks and now - _unsigned_checks[0] > 60:
        _unsigned_checks.popleft()
    if len(_unsigned_checks) >= UNSIGNED_CHECKS_PER_MINUTE:
        return False
    _unsigned_checks.append(now)
    return True


async def _authentic(
    request: Request, settings: Settings, token: str, url: str,
    params: list[tuple[str, str]], form: dict[str, str],
) -> bool:
    """Whether this request really is Twilio telling us about one of our calls.

    A request that carries a signature must carry a valid one: there is no
    falling back from a bad signature. A request that carries none -- which is
    what Twilio sent for a call on a trial account, checked against a real call
    -- is accepted only if Twilio's own API confirms the call it names.
    """
    header = request.headers.get(SIGNATURE_HEADER)
    if header:
        if valid_signature(token, url, params, header):
            return True
        log.warning("rejected a Twilio webhook with a bad signature for %s", url)
        return False

    account = settings.twilio_account_sid.strip()
    call_sid = form.get("CallSid", "")
    # Cheap checks first, so nothing that cannot be ours reaches Twilio's API.
    if not account or form.get("AccountSid") != account or not CALL_SID.fullmatch(call_sid):
        log.warning("rejected an unsigned Twilio webhook that does not name a call on this account (%s)", url)
        return False
    if not _within_unsigned_budget():
        log.warning("too many unsigned Twilio webhooks in a minute; refusing without asking Twilio")
        return False

    rest = TwilioRest(account, token)
    try:
        record = await rest.fetch_call(call_sid)
    except Exception:  # noqa: BLE001 - failing closed is the point; the log has why
        log.exception("could not ask Twilio about call %s; refusing the webhook", call_sid)
        return False
    finally:
        with contextlib.suppress(Exception):
            await rest.close()

    if record is None or not call_matches(record, form):
        log.warning("rejected an unsigned Twilio webhook: Twilio does not confirm call %s", call_sid)
        return False
    log.info("accepted an unsigned webhook for call %s after Twilio confirmed the call", call_sid)
    return True


@router.post("/voice")
async def answer(request: Request) -> Response:
    settings = get_settings()
    token = settings.twilio_auth_token.strip()
    if not token:
        log.warning("a Twilio webhook arrived but TWILIO_AUTH_TOKEN is not set")
        return Response(say_twiml(NOT_CONFIGURED), media_type="text/xml", status_code=503)

    body = (await request.body()).decode("utf-8", errors="replace")
    params = parse_qsl(body, keep_blank_values=True)
    base = public_base(request, settings)
    url = base + request.url.path + (f"?{request.url.query}" if request.url.query else "")

    form = dict(params)
    if not await _authentic(request, settings, token, url, params, form):
        return Response("Forbidden", status_code=403)

    caller, dialed = _parties(form)
    call_sid = form.get("CallSid", "")
    minted = mint_stream_token(token, call_sid=call_sid, caller=caller, dialed=dialed)

    twiml = connect_twiml(
        stream_url(base),
        {"token": minted, "caller": caller, "dialed": dialed},
    )
    return Response(twiml, media_type="text/xml")


@router.websocket("/stream")
async def stream(websocket: WebSocket) -> None:
    await websocket.accept()
    settings = get_settings()
    token = settings.twilio_auth_token.strip()

    start = await read_handshake(websocket)
    if start is None or not token:
        await websocket.close(code=POLICY_VIOLATION)
        return

    caller = start.parameters.get("caller", "")
    dialed = start.parameters.get("dialed", "")
    if not valid_stream_token(
        start.parameters.get("token"), token,
        call_sid=start.call_sid, caller=caller, dialed=dialed,
    ):
        log.warning("refused a stream with a missing or invalid token for call %s", start.call_sid)
        await websocket.close(code=POLICY_VIOLATION)
        return

    rest = None
    if settings.twilio_account_sid.strip():
        rest = TwilioRest(settings.twilio_account_sid.strip(), token)

    transport = TwilioTransport(
        websocket, stream_sid=start.stream_sid, call_sid=start.call_sid, rest=rest
    )
    call = VoiceCall(
        transport,
        dialed_number=dialed or None,
        channel=CallChannel.TWILIO_VOICE,
        caller=caller or "unknown",
        call_sid=start.call_sid,
    )
    try:
        await call.run()
    except Exception:  # noqa: BLE001 - nothing to tell a telephone; the log has it
        log.exception("phone call failed")
    finally:
        if rest is not None:
            with contextlib.suppress(Exception):
                await rest.close()
        with contextlib.suppress(Exception):
            await websocket.close()
