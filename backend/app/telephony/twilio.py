"""Twilio, the parts that are not audio.

Proving a request really came from Twilio, telling Twilio what to do with a call,
and asking it to change a call that is already live. All pure functions and one
small REST client, so every rule here is testable without a phone.

**Why a webhook has to be verified.** The URL Twilio calls is public, and what it
returns decides who is connected to the agent. Without a check, anyone who finds
the address can open streams, run up model and speech usage, and make the agent
answer as any business. Twilio signs every request with the account's auth token;
the signature is an HMAC over the exact URL and the form fields, so a request
that did not come from Twilio -- or was altered on the way -- fails.

**Why the audio stream is checked a different way.** The stream is a WebSocket
that Twilio opens *after* the webhook, to a URL the webhook handed it. Twilio
does sign that handshake too, but the exact URL it signs for a WebSocket has
quirks (their own documentation suggests adding a trailing slash if verification
fails), so the stream is authenticated by something this server controls: the
webhook mints a short-lived token bound to the call, passes it as a stream
parameter, and the stream refuses to start without it.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import time
from collections.abc import Mapping, Sequence
from typing import Any
from xml.sax.saxutils import quoteattr

import httpx2 as httpx

# How long a minted stream token is good for. Twilio opens the stream within a
# second or two of receiving the TwiML; this is generous without being useful to
# anyone who intercepts it later.
STREAM_TOKEN_TTL_S = 120


# ---------------------------------------------------------------------------
# Webhook signatures
# ---------------------------------------------------------------------------

def compute_signature(auth_token: str, url: str, params: Sequence[tuple[str, str]]) -> str:
    """The value Twilio puts in ``X-Twilio-Signature``.

    The URL as Twilio requested it, followed by each form field as name then
    value with nothing between, in order of field name; HMAC-SHA1 with the
    account's auth token as the key; base64.
    """
    payload = url + "".join(f"{name}{value}" for name, value in sorted(params))
    digest = hmac.new(auth_token.encode(), payload.encode(), hashlib.sha1).digest()
    return base64.b64encode(digest).decode()


def valid_signature(
    auth_token: str, url: str, params: Sequence[tuple[str, str]], header: str | None
) -> bool:
    if not auth_token or not header:
        return False
    expected = compute_signature(auth_token, url, params)
    # Constant time: a comparison that stops at the first wrong character lets
    # an attacker work out the signature one character at a time.
    return hmac.compare_digest(expected.encode(), header.encode())


# ---------------------------------------------------------------------------
# The stream token
# ---------------------------------------------------------------------------

def _token_mac(auth_token: str, call_sid: str, caller: str, dialed: str, expires: int) -> str:
    message = f"{call_sid}|{caller}|{dialed}|{expires}".encode()
    return hmac.new(auth_token.encode(), message, hashlib.sha256).hexdigest()


def mint_stream_token(
    auth_token: str,
    *,
    call_sid: str,
    caller: str,
    dialed: str,
    now: float | None = None,
    ttl_s: int = STREAM_TOKEN_TTL_S,
) -> str:
    """A token good for one call's stream, for a couple of minutes.

    Bound to the call and to both phone numbers: a token for one call cannot
    open a stream that claims to be another, or that claims a different caller.
    """
    expires = int((time.time() if now is None else now) + ttl_s)
    return f"{expires}.{_token_mac(auth_token, call_sid, caller, dialed, expires)}"


def valid_stream_token(
    token: str | None,
    auth_token: str,
    *,
    call_sid: str,
    caller: str,
    dialed: str,
    now: float | None = None,
) -> bool:
    if not token or not auth_token or "." not in token:
        return False
    expires_text, _, mac = token.partition(".")
    try:
        expires = int(expires_text)
    except ValueError:
        return False
    if (time.time() if now is None else now) > expires:
        return False
    expected = _token_mac(auth_token, call_sid, caller, dialed, expires)
    return hmac.compare_digest(expected.encode(), mac.encode())


# ---------------------------------------------------------------------------
# TwiML
# ---------------------------------------------------------------------------

_XML = '<?xml version="1.0" encoding="UTF-8"?>'


def connect_twiml(stream_url: str, parameters: Mapping[str, str]) -> str:
    """Answer the call and hand its audio to a WebSocket, both ways.

    ``<Connect><Stream>`` is the bidirectional form: Twilio sends the caller's
    audio and plays whatever audio comes back. (``<Start><Stream>`` only listens.)
    The parameters arrive in the stream's opening message.
    """
    params = "".join(
        f"<Parameter name={quoteattr(name)} value={quoteattr(value)}/>"
        for name, value in parameters.items()
    )
    return (
        f"{_XML}<Response><Connect><Stream url={quoteattr(stream_url)}>"
        f"{params}</Stream></Connect></Response>"
    )


def dial_twiml(number: str) -> str:
    """Put the caller through to a person."""
    return f"{_XML}<Response><Dial>{_escape(number)}</Dial></Response>"


def say_twiml(message: str) -> str:
    """Tell the caller something and hang up: for when the agent cannot answer."""
    return f"{_XML}<Response><Say>{_escape(message)}</Say><Hangup/></Response>"


def _escape(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


# ---------------------------------------------------------------------------
# REST
# ---------------------------------------------------------------------------

class TwilioRest:
    """The two calls to Twilio's API this project makes.

    Written against the HTTP API rather than their SDK: it is two requests, and
    the form each takes is the whole of what there is to know.
    """

    def __init__(
        self,
        account_sid: str,
        auth_token: str,
        *,
        base_url: str = "https://api.twilio.com",
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._sid = account_sid
        self._base = base_url.rstrip("/")
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(timeout=15)
        self._auth = (account_sid, auth_token)

    @property
    def _calls(self) -> str:
        return f"{self._base}/2010-04-01/Accounts/{self._sid}/Calls"

    async def create_call(self, *, to: str, from_: str, url: str) -> dict[str, Any]:
        """Ring ``to`` from ``from_``; when answered, Twilio asks ``url`` what to do.

        The same webhook an incoming call uses, so an outbound test call goes
        through exactly the path a real one does.
        """
        response = await self._client.post(
            f"{self._calls}.json",
            auth=self._auth,
            data={"To": to, "From": from_, "Url": url, "Method": "POST"},
        )
        _check(response)
        return response.json()

    async def redirect_call(self, call_sid: str, twiml: str) -> None:
        """Replace what a live call is doing with new TwiML.

        How a handoff reaches a person: the call is moved off the audio stream
        and onto a ``<Dial>``. Twilio closes the stream itself when it does.
        """
        response = await self._client.post(
            f"{self._calls}/{call_sid}.json", auth=self._auth, data={"Twiml": twiml}
        )
        _check(response)

    async def close(self) -> None:
        if self._owns_client:
            await self._client.aclose()


def _check(response: httpx.Response) -> None:
    if response.status_code >= 400:
        # Twilio's error bodies are JSON with a message; keep it, not just the code.
        try:
            detail = response.json().get("message") or response.text
        except ValueError:
            detail = response.text
        raise RuntimeError(f"Twilio returned {response.status_code}: {str(detail)[:300]}")
