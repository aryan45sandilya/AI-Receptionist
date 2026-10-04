"""A phone call, end to end: Twilio's stream in, the real agent, Postgres behind it.

The same controller runs a phone call and a browser call; what these tests add is
the parts only a telephone has -- the business found by the number dialled, the
caller's own number on the booking, 8 kHz mu-law in and out, ``clear`` sent to
Twilio on an interruption, and a handoff that moves the live call to a person.

Twilio itself is a stand-in socket, so what is proved is this server's side of the
protocol. A call from a real phone is checked by hand.
"""

from __future__ import annotations

import asyncio
import base64
import json
from urllib.parse import parse_qsl

import httpx2 as httpx
import pytest
from sqlalchemy import select

from backend.app.enums import CallChannel, CallStatus
from backend.app.models import Booking
from backend.app.voice.codec import ulaw_decode
from tests.audio_helpers import silence, tone
from tests.fakes import FakeClient, finish, text, tool
from tests.voice_helpers import (
    HANDOFF,
    OFFER,
    FakeTwilioSocket,
    FakeSTT,
    FakeTTS,
    rest_with,
    running_phone_call,
    stored_turns,
)

pytestmark = pytest.mark.db

FIRST = "I need a dental cleaning next Tuesday evening"
PHONE = 8_000


def speech(ms: int = 700) -> bytes:
    return tone(ms, rate=PHONE) + silence(1200, PHONE)


def played(phone) -> bytes:
    """Everything the agent sent, decoded back to 16-bit PCM."""
    return b"".join(
        ulaw_decode(base64.b64decode(m["media"]["payload"]))
        for m in phone.socket.sent
        if m["event"] == "media"
    )


# ---------------------------------------------------------------------------
# Identity: which business, which caller, which call
# ---------------------------------------------------------------------------

async def test_a_phone_call_is_stored_with_its_channel_sid_and_caller(
    monkeypatch, sessionmaker_for, seeded
):
    async with running_phone_call(
        monkeypatch, seeded, client=FakeClient(*OFFER), stt=FakeSTT(FIRST), tts=FakeTTS()
    ) as phone:
        await phone.wait_for("listening")
        phone.socket.media(speech())
        await phone.wait_for("listening", count=2)

    call, turns = await stored_turns(sessionmaker_for, phone)
    assert call.channel is CallChannel.TWILIO_VOICE
    assert call.twilio_call_sid == "CA-test-1"
    assert call.from_phone_e164 == "+919811112222"
    # Found by the number that was dialled, not by being the first business.
    assert call.business_id == seeded.business_id
    assert call.status is CallStatus.COMPLETED
    assert turns[1].text == FIRST


async def test_every_stage_of_a_phone_turn_is_logged_just_as_for_the_browser(
    monkeypatch, sessionmaker_for, seeded
):
    async with running_phone_call(
        monkeypatch, seeded, client=FakeClient(*OFFER), stt=FakeSTT(FIRST), tts=FakeTTS()
    ) as phone:
        await phone.wait_for("listening")
        phone.socket.media(speech())
        await phone.wait_for("listening", count=2)

    _, turns = await stored_turns(sessionmaker_for, phone)
    reply = turns[-1]
    for column in ("stt_final_ms", "llm_first_token_ms", "llm_done_ms", "tts_first_audio_ms", "total_ms"):
        assert getattr(reply, column) is not None, f"{column} was not logged"
    # The 800 ms the detector waited for is part of what the caller waited for.
    assert reply.stt_final_ms >= 800


async def test_a_booking_made_over_the_phone_is_under_the_number_that_called(
    monkeypatch, sessionmaker_for, seeded
):
    book = [
        tool(0, id="b1", name="book_appointment",
             args='{"slot_id": "s1", "customer_name": "Asha Menon", "customer_phone": null}'),
        finish("tool_calls"),
    ]
    client = FakeClient(*OFFER, book, [text("You are booked. "), finish("stop")])
    stt = FakeSTT(FIRST, "Yes please, this is Asha Menon")
    async with running_phone_call(monkeypatch, seeded, client=client, stt=stt, tts=FakeTTS()) as phone:
        await phone.wait_for("listening")
        phone.socket.media(speech())
        await phone.wait_for("listening", count=2)
        phone.socket.media(speech())
        await phone.wait_for("listening", count=3)

    call, _ = await stored_turns(sessionmaker_for, phone)
    async with sessionmaker_for() as db:
        (booking,) = (
            await db.execute(select(Booking).where(Booking.call_id == call.id))
        ).unique().scalars().all()
    # The caller's own number, from the line, without asking them for it.
    assert booking.customer_phone_e164 == "+919811112222"
    assert booking.customer_name == "Asha Menon"


# ---------------------------------------------------------------------------
# The line
# ---------------------------------------------------------------------------

async def test_the_agents_voice_goes_out_as_eight_kilohertz_mulaw_in_twenty_ms_or_so_frames(
    monkeypatch, seeded
):
    async with running_phone_call(
        monkeypatch, seeded, client=FakeClient(), stt=FakeSTT(), tts=FakeTTS(seconds=1.0)
    ) as phone:
        await phone.wait_for("listening")

    media = [m for m in phone.socket.sent if m["event"] == "media"]
    assert media and all(m["streamSid"] == "MZ1" for m in media)
    # One second of greeting: 8000 samples at 8 kHz, however it was cut into frames.
    assert len(played(phone)) // 2 == pytest.approx(PHONE * 1.0, rel=0.02)


async def test_audio_for_the_wrong_track_does_not_reach_the_agent(monkeypatch, seeded):
    stt = FakeSTT("should never be asked for")
    async with running_phone_call(
        monkeypatch, seeded, client=FakeClient(), stt=stt, tts=FakeTTS()
    ) as phone:
        await phone.wait_for("listening")
        phone.socket.media(speech(), track="outbound")
        await asyncio.sleep(1.5)  # long enough for an utterance to have been found and heard

    assert stt.calls == 0


# ---------------------------------------------------------------------------
# Barge-in on the line
# ---------------------------------------------------------------------------

async def test_talking_over_the_agent_on_the_phone_clears_twilios_buffer(
    monkeypatch, sessionmaker_for, seeded
):
    """Twilio plays from its own buffer, so stopping our side is not enough: the
    caller would keep hearing what was already queued. ``clear`` empties it."""
    stt = FakeSTT(FIRST, "actually, Wednesday")
    client = FakeClient(*OFFER, [text("Wednesday works. "), finish("stop")])
    async with running_phone_call(
        monkeypatch, seeded, client=client, stt=stt, tts=FakeTTS(seconds=2.0)
    ) as phone:
        await phone.wait_for("listening")
        phone.socket.media(speech())
        await phone.wait_for("audio_start", count=2)  # the reply is playing
        phone.socket.media(tone(700, rate=PHONE))  # the caller cuts in
        await phone.wait_for("turn_done", count=1)
        phone.socket.media(silence(1200, PHONE))
        await phone.wait_for("listening", count=2)  # the greeting, then the answer to the interruption

    sent = phone.socket.sent
    clears = [i for i, m in enumerate(sent) if m["event"] == "clear"]
    assert len(clears) == 1 and sent[clears[0]] == {"event": "clear", "streamSid": "MZ1"}

    _, turns = await stored_turns(sessionmaker_for, phone)
    assert [t.barge_in for t in turns] == [False, False, True, False, False]


# ---------------------------------------------------------------------------
# Handing over to a person
# ---------------------------------------------------------------------------

async def test_a_handoff_moves_the_live_call_to_the_businesss_number(
    monkeypatch, sessionmaker_for, seeded
):
    rest, seen = rest_with(lambda request: httpx.Response(200, json={}))
    async with running_phone_call(
        monkeypatch, seeded,
        client=FakeClient(*HANDOFF), stt=FakeSTT("can I speak to a real person"), tts=FakeTTS(),
        rest=rest, hang_up=False,
    ) as phone:
        await phone.wait_for("listening")
        phone.socket.media(speech())
        await phone.wait_for("handoff")
        # Twilio ends the stream itself once the call is redirected.
        await asyncio.sleep(0.3)
        assert phone.socket.closed is False, "the server must not hang up on a call it is transferring"
        phone.socket.say({"event": "stop", "stop": {"callSid": "CA-test-1"}})

    (request,) = seen
    assert str(request.url).endswith("/Calls/CA-test-1.json")
    twiml = dict(parse_qsl(request.content.decode()))["Twiml"]
    assert "<Dial>+919900000000</Dial>" in twiml  # the seeded business's handoff number

    call, _ = await stored_turns(sessionmaker_for, phone)
    assert call.status is CallStatus.HANDED_OFF
    assert call.handoff_reason == "Wants to speak to a person."


async def test_a_handoff_that_cannot_be_transferred_hangs_up_rather_than_going_silent(
    monkeypatch, seeded
):
    rest, seen = rest_with(lambda request: httpx.Response(400, json={"message": "number not verified"}))
    async with running_phone_call(
        monkeypatch, seeded,
        client=FakeClient(*HANDOFF), stt=FakeSTT("can I speak to a real person"), tts=FakeTTS(),
        rest=rest, hang_up=False,
    ) as phone:
        await phone.wait_for("listening")
        phone.socket.media(speech())
        await phone.wait_for("handoff")
        await asyncio.sleep(0.3)

    assert len(seen) == 1, "the transfer was attempted"
    assert phone.socket.closed is True, "and when it failed, the call was ended"


async def test_a_handoff_with_no_twilio_credentials_hangs_up(monkeypatch, seeded):
    async with running_phone_call(
        monkeypatch, seeded,
        client=FakeClient(*HANDOFF), stt=FakeSTT("can I speak to a real person"), tts=FakeTTS(),
        rest=None, hang_up=False,
    ) as phone:
        await phone.wait_for("listening")
        phone.socket.media(speech())
        await phone.wait_for("handoff")
        await asyncio.sleep(0.3)

    assert phone.socket.closed is True


def test_the_stream_message_the_tests_use_is_what_twilio_documents():
    """Guard the double: if it drifts from the documented shape, every test above
    would be passing against something Twilio does not send."""
    socket = FakeTwilioSocket()
    socket.media(tone(20, rate=PHONE))
    message = json.loads(socket._incoming.get_nowait())  # noqa: SLF001
    assert message["event"] == "media"
    assert set(message["media"]) == {"track", "chunk", "timestamp", "payload"}
    assert message["media"]["track"] == "inbound"
    assert len(base64.b64decode(message["media"]["payload"])) == 160  # 20 ms of mu-law
