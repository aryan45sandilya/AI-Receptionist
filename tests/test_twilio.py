"""The Twilio integration, without a phone.

Three layers, each tested on its own terms:

*   The pure rules -- signatures, the stream token, TwiML -- including a signature
    worked out by Twilio's own documentation, so the algorithm is checked against
    something other than this code's idea of it.
*   The REST calls, against a stand-in server that records what was sent.
*   The wire: the audio stream's messages in both directions, with a socket that
    plays Twilio's part.

What cannot be tested here is Twilio itself. A real call, and the handshake as
Twilio really sends it, is checked by hand and reported as such.
"""

from __future__ import annotations

import asyncio
import base64
import json
import time
import xml.etree.ElementTree as ET
from urllib.parse import parse_qsl

import httpx2 as httpx
import pytest
from fastapi import FastAPI, WebSocketDisconnect
from starlette.testclient import TestClient

from backend.app.api import twilio as twilio_api
from backend.app.config import Settings
from backend.app.telephony.twilio import (
    TwilioRest,
    compute_signature,
    connect_twiml,
    dial_twiml,
    mint_stream_token,
    say_twiml,
    valid_signature,
    valid_stream_token,
)
from backend.app.voice.codec import ulaw_decode, ulaw_encode
from backend.app.voice.transport import Control
from backend.app.voice.twilio_transport import TwilioTransport, read_handshake
from tests.audio_helpers import tone
from tests.voice_helpers import FakeTwilioSocket, rest_with

AUTH = "s3cret-auth-token"


# ===========================================================================
# Signatures
# ===========================================================================

# The worked example from Twilio's own security documentation. The expected value
# was not produced by this code, which is the point of using it.
DOC_URL = "https://example.com/myapp.php?foo=1&bar=2"
DOC_PARAMS = [
    ("CallSid", "CA1234567890ABCDE"),
    ("Caller", "+14158675310"),
    ("Digits", "1234"),
    ("From", "+14158675310"),
    ("To", "+18005551212"),
]
DOC_SIGNATURE = "L/OH5YylLD5NRKLltdqwSvS0BnU="


def test_the_signature_matches_the_worked_example_in_twilios_documentation():
    assert compute_signature("12345", DOC_URL, DOC_PARAMS) == DOC_SIGNATURE


def test_a_genuine_request_is_accepted():
    assert valid_signature("12345", DOC_URL, DOC_PARAMS, DOC_SIGNATURE)


def test_the_order_the_fields_arrive_in_does_not_matter():
    assert compute_signature("12345", DOC_URL, list(reversed(DOC_PARAMS))) == DOC_SIGNATURE


@pytest.mark.parametrize(
    "tamper",
    [
        pytest.param(lambda p: [(k, "+19999999999" if k == "From" else v) for k, v in p], id="a field changed"),
        pytest.param(lambda p: p[:-1], id="a field removed"),
        pytest.param(lambda p: [*p, ("Extra", "1")], id="a field added"),
    ],
)
def test_an_altered_request_is_rejected(tamper):
    assert not valid_signature("12345", DOC_URL, tamper(DOC_PARAMS), DOC_SIGNATURE)


def test_a_signature_for_another_url_is_rejected():
    """The signature covers the URL, so a request cannot be replayed at another."""
    assert not valid_signature("12345", "https://example.com/other", DOC_PARAMS, DOC_SIGNATURE)


def test_a_signature_made_with_another_token_is_rejected():
    assert not valid_signature("not-the-token", DOC_URL, DOC_PARAMS, DOC_SIGNATURE)


@pytest.mark.parametrize("header", [None, "", "garbage", DOC_SIGNATURE[:-2] + "xx"])
def test_a_missing_or_wrong_signature_is_rejected(header):
    assert not valid_signature("12345", DOC_URL, DOC_PARAMS, header)


def test_nothing_is_accepted_when_no_token_is_configured():
    """An unconfigured server must not treat an empty key as a valid one."""
    assert not valid_signature("", DOC_URL, DOC_PARAMS, compute_signature("", DOC_URL, DOC_PARAMS))


def test_repeated_field_names_are_ordered_by_value_so_the_result_is_stable():
    a = compute_signature("k", "https://x/y", [("Tag", "b"), ("Tag", "a")])
    b = compute_signature("k", "https://x/y", [("Tag", "a"), ("Tag", "b")])
    assert a == b


# ===========================================================================
# The stream token
# ===========================================================================

CALL = {"call_sid": "CA111", "caller": "+919876543210", "dialed": "+918000000000"}


def test_a_minted_token_opens_the_stream_it_was_minted_for():
    token = mint_stream_token(AUTH, **CALL)
    assert valid_stream_token(token, AUTH, **CALL)


@pytest.mark.parametrize("field,value", [
    ("call_sid", "CA222"),
    ("caller", "+910000000000"),
    ("dialed", "+917777777777"),
])
def test_a_token_is_bound_to_its_call_and_both_numbers(field, value):
    token = mint_stream_token(AUTH, **CALL)
    assert not valid_stream_token(token, AUTH, **{**CALL, field: value})


def test_a_token_expires():
    token = mint_stream_token(AUTH, **CALL, now=1_000_000, ttl_s=120)
    assert valid_stream_token(token, AUTH, **CALL, now=1_000_100)
    assert not valid_stream_token(token, AUTH, **CALL, now=1_000_121)


def test_a_token_made_with_another_key_is_rejected():
    assert not valid_stream_token(mint_stream_token("other", **CALL), AUTH, **CALL)


@pytest.mark.parametrize("token", [None, "", "no-dot", "abc.def", ".", "9999999999.", "x.y.z"])
def test_malformed_tokens_are_rejected(token):
    assert not valid_stream_token(token, AUTH, **CALL, now=time.time())


def test_a_tampered_token_is_rejected():
    token = mint_stream_token(AUTH, **CALL)
    expires, _, mac = token.partition(".")
    forged = f"{int(expires) + 100000}.{mac}"  # a longer life, same signature
    assert not valid_stream_token(forged, AUTH, **CALL)


# ===========================================================================
# TwiML
# ===========================================================================

def test_the_connect_twiml_opens_a_two_way_stream_with_its_parameters():
    twiml = connect_twiml("wss://example.test/twilio/stream", {"token": "t.k", "caller": "+91 98"})
    root = ET.fromstring(twiml)
    assert root.tag == "Response"
    stream = root.find("./Connect/Stream")
    assert stream is not None and stream.get("url") == "wss://example.test/twilio/stream"
    assert {p.get("name"): p.get("value") for p in stream.findall("Parameter")} == {
        "token": "t.k", "caller": "+91 98",
    }


def test_awkward_characters_in_values_survive_as_xml():
    twiml = connect_twiml("wss://x/s?a=1&b=2", {"name": 'Asha "A" <Menon> & co'})
    stream = ET.fromstring(twiml).find("./Connect/Stream")
    assert stream.get("url") == "wss://x/s?a=1&b=2"
    assert stream.find("Parameter").get("value") == 'Asha "A" <Menon> & co'


def test_the_dial_twiml_puts_the_caller_through():
    assert ET.fromstring(dial_twiml("+919900000000")).find("Dial").text == "+919900000000"


def test_the_say_twiml_speaks_and_hangs_up_and_escapes_its_text():
    root = ET.fromstring(say_twiml("Sorry <busy> & away"))
    assert root.find("Say").text == "Sorry <busy> & away"
    assert root.find("Hangup") is not None


# ===========================================================================
# REST
# ===========================================================================

async def test_a_test_call_is_placed_through_the_same_webhook_a_real_one_uses():
    rest, seen = rest_with(lambda r: httpx.Response(201, json={"sid": "CA999", "status": "queued"}))
    result = await rest.create_call(to="+919876543210", from_="+15005550006", url="https://x.test/twilio/voice")

    (request,) = seen
    assert request.method == "POST"
    assert str(request.url) == "https://api.twilio.com/2010-04-01/Accounts/AC123/Calls.json"
    assert request.headers["authorization"] == "Basic " + base64.b64encode(f"AC123:{AUTH}".encode()).decode()
    # Exactly these three: a trial account rejects any other parameter, Method included.
    assert dict(parse_qsl(request.content.decode())) == {
        "To": "+919876543210", "From": "+15005550006",
        "Url": "https://x.test/twilio/voice",
    }
    assert result["sid"] == "CA999"


async def test_a_live_call_is_redirected_by_replacing_its_twiml():
    rest, seen = rest_with(lambda r: httpx.Response(200, json={}))
    await rest.redirect_call("CA555", dial_twiml("+919900000000"))

    (request,) = seen
    assert str(request.url) == "https://api.twilio.com/2010-04-01/Accounts/AC123/Calls/CA555.json"
    assert "<Dial>+919900000000</Dial>" in dict(parse_qsl(request.content.decode()))["Twiml"]


async def test_an_error_from_twilio_is_raised_with_its_own_message():
    rest, _ = rest_with(lambda r: httpx.Response(400, json={"message": "The number is unverified"}))
    with pytest.raises(RuntimeError, match="400.*unverified"):
        await rest.create_call(to="+1", from_="+2", url="https://x")


# ===========================================================================
# The webhook and the stream door
# ===========================================================================

def settings(**overrides) -> Settings:
    base = {
        "database_url": "postgresql+psycopg://u:p@localhost/db",
        "twilio_auth_token": AUTH,
        "twilio_account_sid": "",
        "public_base_url": "https://example.test",
    }
    # Not from the developer's .env: once it holds real Twilio values, a test
    # that reads them would be talking to the wrong account.
    return Settings(_env_file=None, **{**base, **overrides})


@pytest.fixture
def client(monkeypatch):
    twilio_api._unsigned_checks.clear()  # the per-minute budget is module state

    def make(**overrides) -> TestClient:
        monkeypatch.setattr(twilio_api, "get_settings", lambda: settings(**overrides))
        app = FastAPI()
        app.include_router(twilio_api.router)
        return TestClient(app)

    return make


def signed(url: str, form: dict[str, str], token: str = AUTH) -> dict[str, str]:
    return {"X-Twilio-Signature": compute_signature(token, url, sorted(form.items()))}


INBOUND = {"CallSid": "CA777", "From": "+919876543210", "To": "+918000000000", "Direction": "inbound"}


def test_a_signed_webhook_is_answered_with_a_stream_for_this_call(client):
    http = client()
    response = http.post(
        "/twilio/voice", data=INBOUND,
        headers=signed("https://example.test/twilio/voice", INBOUND),
    )
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/xml")

    stream = ET.fromstring(response.text).find("./Connect/Stream")
    assert stream.get("url") == "wss://example.test/twilio/stream"
    params = {p.get("name"): p.get("value") for p in stream.findall("Parameter")}
    assert (params["caller"], params["dialed"]) == ("+919876543210", "+918000000000")
    # The token it minted opens exactly this call's stream, and nothing else.
    assert valid_stream_token(params["token"], AUTH, call_sid="CA777", caller=params["caller"], dialed=params["dialed"])
    assert not valid_stream_token(params["token"], AUTH, call_sid="CA888", caller=params["caller"], dialed=params["dialed"])


def test_a_webhook_with_no_signature_is_refused(client):
    assert client().post("/twilio/voice", data=INBOUND).status_code == 403


def test_a_webhook_signed_with_the_wrong_token_is_refused(client):
    http = client()
    response = http.post(
        "/twilio/voice", data=INBOUND,
        headers=signed("https://example.test/twilio/voice", INBOUND, token="someone-elses"),
    )
    assert response.status_code == 403


def test_a_signature_for_a_different_url_is_refused(client):
    """Replaying a captured request at this endpoint must fail."""
    http = client()
    response = http.post(
        "/twilio/voice", data=INBOUND,
        headers=signed("https://example.test/somewhere/else", INBOUND),
    )
    assert response.status_code == 403


def test_a_webhook_with_a_changed_field_is_refused(client):
    http = client()
    headers = signed("https://example.test/twilio/voice", INBOUND)
    forged = {**INBOUND, "From": "+910000000000"}
    assert http.post("/twilio/voice", data=forged, headers=headers).status_code == 403


# ---------------------------------------------------------------------------
# A webhook with no signature. A real call on a trial account sent exactly this:
# a genuine request -- its CallSid was a real call on the account, and Twilio's
# own record agreed with it -- but with no X-Twilio-Signature header at all. So
# the call is confirmed with Twilio's API instead, and a forgery still fails.
# ---------------------------------------------------------------------------

UNSIGNED_ACCOUNT = "AC" + "f" * 32
UNSIGNED_CALL_SID = "CA" + "0123456789abcdef" * 2
UNSIGNED = {
    "AccountSid": UNSIGNED_ACCOUNT, "ApiVersion": "2010-04-01", "CallSid": UNSIGNED_CALL_SID, "CallStatus": "in-progress",
    "Direction": "outbound-api", "From": "+15005550006", "To": "+919876543210",
}
RECORD = {"sid": UNSIGNED_CALL_SID, "status": "in-progress", "direction": "outbound-api",
          "from": "+15005550006", "to": "+919876543210"}


def twilio_says(monkeypatch, record=RECORD, status=200):
    """Stand in for Twilio's API; returns the requests the server makes to it."""
    def respond(request):
        if record is None:
            return httpx.Response(404, json={"message": "not found"})
        return httpx.Response(status, json=record)

    rest, seen = rest_with(respond)
    monkeypatch.setattr(twilio_api, "TwilioRest", lambda sid, token: rest)
    return seen


def post_unsigned(http, form=UNSIGNED):
    return http.post("/twilio/voice", data=form)


def test_an_unsigned_webhook_is_answered_when_twilio_confirms_the_call(client, monkeypatch):
    seen = twilio_says(monkeypatch)
    response = post_unsigned(client(twilio_account_sid=UNSIGNED_ACCOUNT))
    assert response.status_code == 200
    params = {p.get("name"): p.get("value") for p in ET.fromstring(response.text).findall(".//Parameter")}
    # Same rule as a signed one: for a call Twilio placed, the caller is the person rung.
    assert (params["caller"], params["dialed"]) == ("+919876543210", "+15005550006")
    assert [r.method for r in seen] == ["GET"]
    assert str(seen[0].url).endswith(f"/Calls/{UNSIGNED_CALL_SID}.json")


def test_an_unsigned_webhook_for_a_call_twilio_has_no_record_of_is_refused(client, monkeypatch):
    twilio_says(monkeypatch, record=None)
    assert post_unsigned(client(twilio_account_sid=UNSIGNED_ACCOUNT)).status_code == 403


def test_an_unsigned_webhook_that_disagrees_with_twilios_record_is_refused(client, monkeypatch):
    """Someone who knew a call's id still cannot change who it is between."""
    twilio_says(monkeypatch)
    forged = {**UNSIGNED, "To": "+910000000000"}
    assert post_unsigned(client(twilio_account_sid=UNSIGNED_ACCOUNT), forged).status_code == 403


def test_an_unsigned_webhook_about_a_finished_call_is_refused(client, monkeypatch):
    """A replay of an old request must not start a new conversation."""
    twilio_says(monkeypatch, record={**RECORD, "status": "completed"})
    assert post_unsigned(client(twilio_account_sid=UNSIGNED_ACCOUNT)).status_code == 403


def test_an_unsigned_webhook_is_refused_when_twilio_cannot_be_asked(client, monkeypatch):
    twilio_says(monkeypatch, status=500)
    assert post_unsigned(client(twilio_account_sid=UNSIGNED_ACCOUNT)).status_code == 403


@pytest.mark.parametrize(
    "form",
    [
        {**UNSIGNED, "AccountSid": "AC" + "0" * 32},   # someone else's account
        {**UNSIGNED, "CallSid": "CA777"},                # not shaped like a call id
        {k: v for k, v in UNSIGNED.items() if k != "CallSid"},
    ],
    ids=["another-account", "malformed-call-id", "no-call-id"],
)
def test_an_unsigned_webhook_that_cannot_be_ours_never_reaches_twilios_api(client, monkeypatch, form):
    seen = twilio_says(monkeypatch)
    assert post_unsigned(client(twilio_account_sid=UNSIGNED_ACCOUNT), form).status_code == 403
    assert seen == []


def test_a_bad_signature_is_never_rescued_by_twilios_record(client, monkeypatch):
    """Only the absence of a signature falls back. A wrong one is a forgery."""
    seen = twilio_says(monkeypatch)
    headers = signed("https://example.test/twilio/voice", UNSIGNED, token="someone-elses")
    response = client(twilio_account_sid=UNSIGNED_ACCOUNT).post("/twilio/voice", data=UNSIGNED, headers=headers)
    assert response.status_code == 403
    assert seen == []


def test_an_unsigned_webhook_is_refused_when_no_account_is_configured(client, monkeypatch):
    seen = twilio_says(monkeypatch)
    assert post_unsigned(client(twilio_account_sid="")).status_code == 403
    assert seen == []


def test_a_flood_of_unsigned_webhooks_does_not_become_a_flood_of_calls_to_twilio(client, monkeypatch):
    seen = twilio_says(monkeypatch, record=None)
    http = client(twilio_account_sid=UNSIGNED_ACCOUNT)
    for _ in range(twilio_api.UNSIGNED_CHECKS_PER_MINUTE + 5):
        assert post_unsigned(http).status_code == 403
    assert len(seen) == twilio_api.UNSIGNED_CHECKS_PER_MINUTE


def test_nothing_is_answered_until_the_auth_token_is_configured(client):
    """An unconfigured server must not be a way for anyone to start a call."""
    http = client(twilio_auth_token="")
    response = http.post("/twilio/voice", data=INBOUND, headers=signed("https://example.test/twilio/voice", INBOUND))
    assert response.status_code == 503
    assert "not set up" in ET.fromstring(response.text).find("Say").text


def test_for_a_call_twilio_placed_the_caller_is_the_person_it_rang(client):
    """Our test call: Twilio's number is From, the phone it rang is To."""
    form = {"CallSid": "CA1", "From": "+15005550006", "To": "+919876543210", "Direction": "outbound-api"}
    response = client().post(
        "/twilio/voice", data=form, headers=signed("https://example.test/twilio/voice", form)
    )
    params = {p.get("name"): p.get("value") for p in ET.fromstring(response.text).findall(".//Parameter")}
    assert (params["caller"], params["dialed"]) == ("+919876543210", "+15005550006")


def test_the_address_is_taken_from_the_tunnels_forwarding_headers_when_none_is_set(client):
    http = client(public_base_url="")
    url = "https://abc-def.trycloudflare.com/twilio/voice"
    response = http.post(
        "/twilio/voice", data=INBOUND,
        headers={
            **signed(url, INBOUND),
            "X-Forwarded-Proto": "https",
            "X-Forwarded-Host": "abc-def.trycloudflare.com",
        },
    )
    assert response.status_code == 200
    assert ET.fromstring(response.text).find("./Connect/Stream").get("url") == (
        "wss://abc-def.trycloudflare.com/twilio/stream"
    )


def test_a_plain_http_address_gets_a_plain_websocket(client):
    http = client(public_base_url="http://127.0.0.1:8000")
    response = http.post(
        "/twilio/voice", data=INBOUND, headers=signed("http://127.0.0.1:8000/twilio/voice", INBOUND)
    )
    assert ET.fromstring(response.text).find("./Connect/Stream").get("url") == "ws://127.0.0.1:8000/twilio/stream"


def start_message(**parameters) -> str:
    return json.dumps({
        "event": "start", "streamSid": "MZ1",
        "start": {"streamSid": "MZ1", "callSid": "CA777", "customParameters": parameters},
    })


def closes_with(http: TestClient, *messages: str) -> int:
    """Open the stream, send messages, and report the code the server closed with."""
    with http.websocket_connect("/twilio/stream") as ws:
        for message in messages:
            ws.send_text(message)
        with pytest.raises(WebSocketDisconnect) as closed:
            ws.receive_text()
        return closed.value.code


CONNECTED = json.dumps({"event": "connected", "protocol": "Call", "version": "1.0.0"})


def test_a_stream_without_a_token_is_refused(client):
    assert closes_with(client(), CONNECTED, start_message(caller="+91", dialed="+91")) == 1008


def test_a_stream_with_a_token_for_another_call_is_refused(client):
    token = mint_stream_token(AUTH, call_sid="CA-OTHER", caller="+91", dialed="+91")
    assert closes_with(client(), CONNECTED, start_message(token=token, caller="+91", dialed="+91")) == 1008


def test_a_stream_claiming_a_different_caller_than_the_token_was_minted_for_is_refused(client):
    token = mint_stream_token(AUTH, call_sid="CA777", caller="+91", dialed="+91")
    assert closes_with(client(), CONNECTED, start_message(token=token, caller="+92", dialed="+91")) == 1008


def test_a_stream_that_does_not_open_like_twilios_is_refused(client):
    assert closes_with(client(), json.dumps({"event": "media"})) == 1008
    assert closes_with(client(), "not even json") == 1008


def test_no_stream_is_served_until_the_auth_token_is_configured(client):
    token = mint_stream_token(AUTH, call_sid="CA777", caller="+91", dialed="+91")
    http = client(twilio_auth_token="")
    assert closes_with(http, CONNECTED, start_message(token=token, caller="+91", dialed="+91")) == 1008


# ===========================================================================
# The audio stream, as the call sees it
# ===========================================================================

def transport(socket: FakeTwilioSocket, rest: TwilioRest | None = None) -> TwilioTransport:
    return TwilioTransport(socket, stream_sid="MZ1", call_sid="CA777", rest=rest)


async def test_inbound_audio_arrives_as_pcm_at_eight_kilohertz():
    socket = FakeTwilioSocket()
    original = tone(20, rate=8000)  # 20 ms: one Twilio frame
    socket.media(original)

    got = await transport(socket).receive()
    assert isinstance(got, bytes) and len(got) == len(original)
    # Mu-law is lossy, so equal in level and shape rather than in every bit.
    assert ulaw_encode(got) == ulaw_encode(original)


async def test_audio_for_another_track_is_not_the_callers():
    socket = FakeTwilioSocket()
    socket.media(tone(20, rate=8000), track="outbound")
    socket.say({"event": "stop"})
    assert await transport(socket).receive() is None


@pytest.mark.parametrize("bad", ["not json at all", "{broken", json.dumps({"event": "media", "media": {"payload": "###"}}),
                                 json.dumps({"event": "media", "media": {"payload": ""}}),
                                 json.dumps({"event": "unheard-of"})])
async def test_a_message_that_is_not_usable_is_skipped_not_fatal(bad):
    socket = FakeTwilioSocket()
    socket.say(bad)
    socket.media(tone(20, rate=8000))
    assert isinstance(await transport(socket).receive(), bytes)


async def test_stop_ends_the_call():
    socket = FakeTwilioSocket()
    socket.say({"event": "stop", "stop": {"callSid": "CA777"}})
    assert await transport(socket).receive() is None


async def test_the_socket_closing_ends_the_call():
    socket = FakeTwilioSocket()
    await socket.close()
    assert await transport(socket).receive() is None


async def test_keypresses_and_marks_are_passed_up_as_control_messages():
    socket = FakeTwilioSocket()
    socket.say({"event": "dtmf", "dtmf": {"digit": "0", "track": "inbound_track"}})
    socket.say({"event": "mark", "mark": {"name": "end"}})
    t = transport(socket)
    assert await t.receive() == Control("dtmf", {"digit": "0", "track": "inbound_track"})
    assert await t.receive() == Control("mark", {"name": "end"})


async def test_outbound_audio_is_sent_as_base64_mulaw_for_this_stream():
    socket = FakeTwilioSocket()
    pcm = tone(40, rate=8000)
    await transport(socket).send_audio(pcm)

    (message,) = socket.sent
    assert message["event"] == "media" and message["streamSid"] == "MZ1"
    mulaw = base64.b64decode(message["media"]["payload"])
    assert len(mulaw) == len(pcm) // 2
    assert ulaw_decode(mulaw) == ulaw_decode(ulaw_encode(pcm))


async def test_clearing_tells_twilio_to_drop_what_it_has_buffered():
    socket = FakeTwilioSocket()
    await transport(socket).clear_audio()
    assert socket.sent == [{"event": "clear", "streamSid": "MZ1"}]


async def test_a_telephone_has_no_screen_so_events_send_nothing():
    socket = FakeTwilioSocket()
    await transport(socket).send_event("transcript", text="hello")
    assert socket.sent == []


async def test_a_transfer_redirects_the_live_call_to_a_dial():
    rest, seen = rest_with(lambda r: httpx.Response(200, json={}))
    socket = FakeTwilioSocket()
    assert await transport(socket, rest).transfer("+919900000000") is True
    assert str(seen[0].url).endswith("/Calls/CA777.json")
    assert socket.closed is False, "Twilio ends the stream itself when the call is redirected"


async def test_a_transfer_that_cannot_be_made_says_so_so_the_caller_can_hang_up():
    assert await transport(FakeTwilioSocket()).transfer("+919900000000") is False  # no REST client

    rest, _ = rest_with(lambda r: httpx.Response(400, json={"message": "unverified number"}))
    assert await transport(FakeTwilioSocket(), rest).transfer("+919900000000") is False


async def test_the_opening_messages_name_the_stream_the_call_and_our_parameters():
    socket = FakeTwilioSocket()
    socket.say(CONNECTED)
    socket.say(start_message(token="t", caller="+91"))
    start = await read_handshake(socket)
    assert (start.stream_sid, start.call_sid) == ("MZ1", "CA777")
    assert start.parameters == {"token": "t", "caller": "+91"}


async def test_a_handshake_that_never_arrives_gives_up():
    assert await read_handshake(FakeTwilioSocket(), timeout=0.05) is None


async def test_a_handshake_cut_short_gives_up():
    socket = FakeTwilioSocket()
    await socket.close()
    assert await read_handshake(socket, timeout=1) is None
