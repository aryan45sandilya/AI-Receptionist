"""Barge-in: the caller talks over the agent and the agent stops.

Run through the whole call -- the real controller, detector, agent loop, tools and
Postgres -- with only the services faked, because the behaviour that matters is
the coordination: the voice stops, the far end is told to drop its queue, the
turn is cancelled without abandoning a booking, history says what was heard, and
the database says the turn was cut off.

The checks that need the order of things (nothing was sent after the clear) read
the socket's timeline; the ones about what was stored read the rows back out of
Postgres.
"""

from __future__ import annotations

import asyncio
import json

import pytest
from sqlalchemy import select

from backend.app.agent import loop as loop_module
from backend.app.enums import TurnRole
from backend.app.models import Booking
from backend.app.voice import call as call_module
from tests.audio_helpers import silence, tone
from tests.fakes import FakeClient, HangingStream, finish, text, tool
from tests.voice_helpers import (
    CHECK_ARGS,
    OFFER,
    FakeSTT,
    FakeStreamingSTT,
    FakeTTS,
    running_call,
    stored_turns,
)

pytestmark = pytest.mark.db

FIRST = "I need a dental cleaning next Tuesday evening"


def audio_between(timeline: list[str], first: str, then: str) -> int:
    """How many audio frames were sent between two events."""
    start = timeline.index(f"event:{first}")
    end = timeline.index(f"event:{then}", start + 1)
    return timeline[start:end].count("audio")


# ---------------------------------------------------------------------------
# The interruption itself
# ---------------------------------------------------------------------------

async def test_talking_over_the_agent_stops_it_and_the_turn_is_logged_as_cut_off(
    monkeypatch, sessionmaker_for, seeded
):
    stt = FakeSTT(FIRST, "actually make it Wednesday")
    tts = FakeTTS(seconds=2.0)
    client = FakeClient(*OFFER, [text("Wednesday is fine. "), finish("stop")])

    async with running_call(monkeypatch, seeded, client=client, stt=stt, tts=tts) as ws:
        await ws.wait_for("listening")  # the greeting is over
        ws.feed(tone(700) + silence(1200))
        await ws.wait_for("audio_start", count=2)  # the reply is playing
        ws.feed(tone(700))  # the caller talks over it
        await ws.wait_for("clear")
        ws.feed(silence(1200))  # ...and finishes
        await ws.wait_for("turn_done", count=2)  # the cut-off reply, then the new one
        await ws.wait_for("listening", count=2)  # the greeting, then the answer: the cut-off reply sends none

    # The far end was told once, and none of the old reply followed the clear.
    assert len(ws.of("clear")) == 1
    assert audio_between(ws.timeline, "clear", "audio_start") == 0

    _, turns = await stored_turns(sessionmaker_for, ws)
    assert [(t.role, t.barge_in) for t in turns] == [
        (TurnRole.AGENT, False),   # the greeting
        (TurnRole.CALLER, False),  # the first request
        (TurnRole.AGENT, True),    # the reply, cut off
        (TurnRole.CALLER, False),  # the interruption
        (TurnRole.AGENT, False),   # the answer to it
    ]
    cut = turns[2]
    # What was stored is what the caller heard: the first sentence, marked as cut.
    assert cut.text == "We have Tuesday at five o'clock. ..."
    assert "Which would suit you best" not in cut.text
    # ...and the stages it got through were still measured.
    assert cut.stt_final_ms is not None and cut.llm_first_token_ms is not None
    assert cut.tts_first_audio_ms is not None
    assert cut.tool_calls["calls"][0]["name"] == "check_availability"
    assert turns[3].text == "actually make it Wednesday"


async def test_the_model_is_told_where_it_was_cut_off(monkeypatch, seeded):
    stt = FakeSTT(FIRST, "actually make it Wednesday")
    client = FakeClient(*OFFER, [text("Wednesday is fine. "), finish("stop")])
    async with running_call(
        monkeypatch, seeded, client=client, stt=stt, tts=FakeTTS(seconds=2.0)
    ) as ws:
        await ws.wait_for("listening")
        ws.feed(tone(700) + silence(1200))
        await ws.wait_for("audio_start", count=2)
        ws.feed(tone(700) + silence(1200))
        await ws.wait_for("listening", count=2)

    # The next request (after the two that made the first reply) carries the
    # history: the tool use, what was *heard* of the reply, then the interruption.
    history = client.requests[2]["messages"]
    assert history[-1] == {"role": "user", "content": "actually make it Wednesday"}
    assert history[-2] == {
        "role": "assistant",
        "content": "We have Tuesday at five o'clock. ... [interrupted]",
    }
    # Every tool call still has its answer, or the provider rejects the request.
    asked = [c["id"] for m in history if m.get("tool_calls") for c in m["tool_calls"]]
    answered = [m["tool_call_id"] for m in history if m["role"] == "tool"]
    assert asked == answered == ["c1"]


async def test_a_cough_does_not_interrupt_the_agent(monkeypatch, sessionmaker_for, seeded):
    """The reason the caller must keep talking for a while: a click, a cough or a
    "mm-hm" trips the detector but is over long before it counts."""
    stt = FakeSTT(FIRST)
    async with running_call(
        monkeypatch, seeded, client=FakeClient(*OFFER), stt=stt, tts=FakeTTS(seconds=1.5)
    ) as ws:
        await ws.wait_for("listening")
        ws.feed(tone(700) + silence(1200))
        await ws.wait_for("audio_start", count=2)
        ws.feed(tone(120) + silence(1000))  # a cough
        await ws.wait_for("listening", count=2)  # the reply plays out in full

    assert ws.of("clear") == []
    assert len(ws.of("audio_end")) == 2  # the greeting and the whole reply
    assert stt.calls == 1, "the cough must never have been transcribed"
    _, turns = await stored_turns(sessionmaker_for, ws)
    assert not any(t.barge_in for t in turns)


async def test_the_caller_can_talk_over_the_greeting(monkeypatch, sessionmaker_for, seeded):
    stt = FakeSTT("hello, I need a cleaning")
    client = FakeClient([text("Of course. "), finish("stop")])
    async with running_call(
        monkeypatch, seeded, client=client, stt=stt, tts=FakeTTS(seconds=3.0)
    ) as ws:
        await ws.wait_for("audio_start")  # the greeting has begun
        ws.feed(tone(700) + silence(1200))  # straight over it
        await ws.wait_for("clear")
        await ws.wait_for("turn_done")
        await ws.wait_for("listening")

    assert audio_between(ws.timeline, "clear", "audio_start") == 0
    _, turns = await stored_turns(sessionmaker_for, ws)
    assert turns[0].role is TurnRole.AGENT and turns[0].barge_in is True
    assert turns[0].text == "Test Clinic, how can I help? ..."
    assert turns[1].text == "hello, I need a cleaning"
    assert turns[2].text.startswith("Of course.")


async def test_nothing_more_is_sent_after_the_caller_cuts_in(monkeypatch, seeded):
    """The moment the caller has talked long enough, the voice is muted
    synchronously: no further frame may leave, however long the clean-up takes."""
    stt = FakeSTT(FIRST, "wait")
    client = FakeClient(*OFFER, [text("Yes? "), finish("stop")])
    async with running_call(
        monkeypatch, seeded, client=client, stt=stt, tts=FakeTTS(seconds=4.0)
    ) as ws:
        await ws.wait_for("listening")
        ws.feed(tone(700) + silence(1200))
        await ws.wait_for("audio_start", count=2)
        ws.feed(tone(700))
        await ws.wait_for("clear")
        frames_at_clear = len(ws.audio)
        await asyncio.sleep(1.0)  # a full second in which the old reply could have leaked
        frames_later = len(ws.audio)
        ws.feed(silence(1200))
        await ws.wait_for("listening", count=2)

    assert frames_later == frames_at_clear


async def test_the_page_can_switch_barge_in_off_for_its_own_call(monkeypatch, seeded):
    """Someone on speakers hears the agent's voice come back through their
    microphone. Untick the box and the agent talks through it."""
    stt = FakeSTT(FIRST, "this is never asked for")
    async with running_call(
        monkeypatch, seeded, client=FakeClient(*OFFER), stt=stt, tts=FakeTTS(seconds=1.5)
    ) as ws:
        ws._incoming.put_nowait(  # noqa: SLF001 - what the page sends when the box is unticked
            {"type": "websocket.receive", "text": json.dumps({"type": "config", "barge_in": False})}
        )
        await ws.wait_for("listening")
        ws.feed(tone(700) + silence(1200))
        await ws.wait_for("audio_start", count=2)
        ws.feed(tone(700) + silence(1200))  # would have interrupted
        await ws.wait_for("listening", count=2)

    assert ws.of("clear") == []
    assert stt.calls == 1


def test_only_a_real_boolean_changes_barge_in():
    from backend.app.config import Settings
    from backend.app.voice.call import VoiceCall

    class Door:
        input_rate, output_rate = 16_000, None

    call = VoiceCall(Door(), settings=Settings(
        _env_file=None, database_url="postgresql+psycopg://u:p@h/db", barge_in=True,
        llm_model="the-model", llm_api_key="the-key",
    ))
    call._apply_config({"barge_in": "no", "llm_model": "other", "llm_api_key": "stolen"})  # noqa: SLF001
    assert call.settings.barge_in is True
    assert (call.settings.llm_model, call.settings.llm_api_key) == ("the-model", "the-key")

    call._apply_config({"barge_in": False})  # noqa: SLF001
    assert call.settings.barge_in is False
    assert call.settings.llm_model == "the-model"


async def test_answering_the_agent_straight_after_it_stops_is_not_an_interruption(
    monkeypatch, sessionmaker_for, seeded
):
    """For a second or two after the agent stops talking its turn is still writing
    its record to the database. A caller who replies straight away is answering,
    not interrupting: no clear, no "you interrupted", and nothing deaf about the
    gap."""
    original = call_module.call_log.log_turn

    async def slow(*args, **kwargs):
        await asyncio.sleep(1.5)  # the database is far away
        return await original(*args, **kwargs)

    monkeypatch.setattr(call_module.call_log, "log_turn", slow)

    stt = FakeSTT(FIRST, "the first one please")
    client = FakeClient(*OFFER, [text("Booked. "), finish("stop")])
    async with running_call(
        monkeypatch, seeded, client=client, stt=stt, tts=FakeTTS(seconds=1.0)
    ) as ws:
        await ws.wait_for("listening")
        ws.feed(tone(700) + silence(1200))
        await ws.wait_for("audio_end", count=2)  # the reply has been sent in full
        await asyncio.sleep(1.0)  # past its tail; the record is still being written
        ws.feed(tone(700) + silence(1200))  # the caller answers at once
        await ws.wait_for("turn_done", count=2)
        await ws.wait_for("listening", count=3)

    assert ws.of("clear") == [], "answering is not interrupting"
    _, turns = await stored_turns(sessionmaker_for, ws)
    assert [(t.role, t.barge_in) for t in turns] == [
        (TurnRole.AGENT, False),
        (TurnRole.CALLER, False),
        (TurnRole.AGENT, False),
        (TurnRole.CALLER, False),
        (TurnRole.AGENT, False),
    ]
    assert turns[3].text == "the first one please"


# ---------------------------------------------------------------------------
# A pause that was not the end
# ---------------------------------------------------------------------------

async def test_a_sentence_split_by_a_pause_is_answered_once_as_a_whole(
    monkeypatch, sessionmaker_for, seeded
):
    """The agent began to answer half a sentence; the caller carried on. Nothing
    had been said or done yet, so the half-turn is erased and the model sees one
    complete request -- not two turns, the first of them wrong."""
    stt = FakeSTT("I need a dental cleaning", "next Tuesday evening")
    thinking = HangingStream()  # the model has been asked and has not answered
    client = FakeClient(thinking, [text("Tuesday evening, got it. "), finish("stop")])

    async with running_call(monkeypatch, seeded, client=client, stt=stt, tts=FakeTTS()) as ws:
        await ws.wait_for("listening")
        ws.feed(tone(700) + silence(1200))
        await asyncio.wait_for(thinking.reached_the_end.wait(), 30)
        ws.feed(tone(700))  # the caller carries on
        await ws.wait_for("clear")
        ws.feed(silence(1200))
        await ws.wait_for("turn_done")
        await ws.wait_for("listening", count=2)

    assert thinking.closed, "the abandoned request must be closed so the provider stops"
    sent = client.requests[1]["messages"]
    assert [m["content"] for m in sent if m["role"] == "user"] == [
        "I need a dental cleaning next Tuesday evening"
    ]

    _, turns = await stored_turns(sessionmaker_for, ws)
    assert [(t.role, t.barge_in) for t in turns] == [
        (TurnRole.AGENT, False),
        (TurnRole.CALLER, False),
        (TurnRole.AGENT, False),
    ]
    assert turns[1].text == "I need a dental cleaning next Tuesday evening"


# ---------------------------------------------------------------------------
# Never abandon a booking
# ---------------------------------------------------------------------------

async def test_interrupting_the_moment_a_booking_is_being_made_still_makes_it(
    monkeypatch, sessionmaker_for, seeded
):
    """The caller talks over the agent just as it writes their appointment. The
    speech and the model stop; the write does not. The booking exists, it is in
    the agent's history, and the call records that it happened."""
    started = asyncio.Event()
    real_execute = loop_module.execute_tool

    async def slow_booking(name, arguments, ctx):
        if name == "book_appointment":
            started.set()
            await asyncio.sleep(1.0)  # long enough for the interruption to land mid-write
        return await real_execute(name, arguments, ctx)

    monkeypatch.setattr(loop_module, "execute_tool", slow_booking)

    book = [
        tool(0, id="b1", name="book_appointment",
             args='{"slot_id": "s1", "customer_name": "Asha Menon", "customer_phone": null}'),
        finish("tool_calls"),
    ]
    stt = FakeSTT(FIRST, "Yes please, this is Asha Menon", "wait, never mind")
    client = FakeClient(*OFFER, book, [text("Understood, it stays booked. "), finish("stop")])

    async with running_call(monkeypatch, seeded, client=client, stt=stt, tts=FakeTTS()) as ws:
        await ws.wait_for("listening")
        ws.feed(tone(700) + silence(1200))
        await ws.wait_for("listening", count=2)  # slots offered
        ws.feed(tone(700) + silence(1200))
        await asyncio.wait_for(started.wait(), 30)  # the booking is being written
        ws.feed(tone(700))  # the caller cuts in
        await ws.wait_for("clear")
        ws.feed(silence(1200))
        await ws.wait_for("listening", count=3)

    call, turns = await stored_turns(sessionmaker_for, ws)
    async with sessionmaker_for() as db:
        bookings = (
            await db.execute(select(Booking).where(Booking.call_id == call.id))
        ).unique().scalars().all()

    assert len(bookings) == 1 and bookings[0].customer_name == "Asha Menon"

    # The interrupted turn is recorded, with the tool that completed.
    cut = [t for t in turns if t.barge_in]
    assert len(cut) == 1
    (booked,) = cut[0].tool_calls["calls"]
    assert booked["name"] == "book_appointment" and booked["is_error"] is False

    # And the model knows it happened, so it will not book twice.
    history = client.requests[-1]["messages"]
    results = [m["content"] for m in history if m["role"] == "tool"]
    assert any("booked" in r.lower() or "reference" in r.lower() for r in results)


# ---------------------------------------------------------------------------
# Streaming transcription ends utterances sooner
# ---------------------------------------------------------------------------

async def test_the_service_can_end_an_utterance_before_the_detector_would(
    monkeypatch, seeded
):
    """Deepgram waits for a shorter pause than the detector does. The detector
    needs 800 ms of silence; here the service decides after 500, and the turn
    starts then -- which is where the streaming latency saving comes from."""
    stt = FakeStreamingSTT(FIRST, endpointing_ms=500)
    async with running_call(
        monkeypatch, seeded, client=FakeClient(*OFFER), stt=stt, tts=FakeTTS()
    ) as ws:
        await ws.wait_for("listening")
        ws.feed(tone(700) + silence(560))  # 560 ms of quiet: under the detector's 800
        await ws.wait_for("listening", count=2)

    done = ws.of("turn_done")[0]
    # The wait the caller sat through was the service's, and the clock says so.
    assert 440 <= done["endpoint_ms"] <= 620
    assert done["stt_final_ms"] >= done["endpoint_ms"]
    assert stt.finished == 1
    # The service had it all already; the detector's copy of the audio is not needed.
    assert stt.finish_audio == [None]


async def test_with_a_service_that_ends_utterances_the_detector_is_only_a_backstop(
    monkeypatch, seeded
):
    """The detector's own 800 ms would win the race against a service that needs
    400 ms plus a network round trip, and every turn would then pay a flush for
    nothing. When the service ends utterances itself, the detector waits longer
    and acts only if the service never does."""
    stt = FakeStreamingSTT(FIRST, endpointing_ms=60_000)  # a service that never gets round to it
    async with running_call(
        monkeypatch, seeded, client=FakeClient(*OFFER), stt=stt, tts=FakeTTS()
    ) as ws:
        await ws.wait_for("listening")
        ws.feed(tone(700) + silence(1200))  # past a batch detector's 800 ms
        await asyncio.sleep(0.5)
        assert ws.of("speech_ended") == [], "the backstop is longer than that"

        ws.feed(silence(500))  # now past the 1500 ms backstop
        await ws.wait_for("transcript")

    # Closed by the detector, so it handed over its own copy of the audio.
    assert stt.finish_audio[0] is not None


async def test_the_detector_does_not_end_the_same_utterance_a_second_time(monkeypatch, seeded):
    stt = FakeStreamingSTT(FIRST, endpointing_ms=500)
    async with running_call(
        monkeypatch, seeded, client=FakeClient(*OFFER), stt=stt, tts=FakeTTS()
    ) as ws:
        await ws.wait_for("listening")
        ws.feed(tone(700) + silence(560))
        await ws.wait_for("listening", count=2)
        ws.feed(silence(1500))  # well past the detector's own 800 ms
        await asyncio.sleep(0.5)

    assert len(ws.of("speech_ended")) == 1
    assert stt.finished == 1
    assert len(ws.of("turn_done")) == 1, "one utterance, one turn"


async def test_words_are_captioned_as_they_are_heard(monkeypatch, seeded):
    stt = FakeStreamingSTT(FIRST, endpointing_ms=500)
    async with running_call(
        monkeypatch, seeded, client=FakeClient(*OFFER), stt=stt, tts=FakeTTS()
    ) as ws:
        await ws.wait_for("listening")
        ws.feed(tone(700) + silence(560))
        await ws.wait_for("listening", count=2)

    interim = ws.of("interim")
    assert interim and interim[0]["text"]
    kinds = ws.kinds()
    assert kinds.index("interim") < kinds.index("transcript")


async def test_the_businesss_own_words_are_given_to_the_transcriber(monkeypatch, seeded):
    """"Dr Rao" is not in a general vocabulary. The services and staff names are
    handed over so the service favours them."""
    stt = FakeStreamingSTT()
    async with running_call(
        monkeypatch, seeded, client=FakeClient(), stt=stt, tts=FakeTTS()
    ) as ws:
        await ws.wait_for("listening")

    assert "Dental cleaning" in stt.keyterms
    assert "Root canal" in stt.keyterms
    assert "Dr Rao" in stt.keyterms and "Dr Menon" in stt.keyterms


async def test_a_pause_longer_than_the_services_patience_does_not_split_the_request(
    monkeypatch, sessionmaker_for, seeded
):
    """The service called the end of the utterance at 500 ms of quiet, mid
    sentence. The agent started thinking; the caller carried on. Same outcome as
    with the detector: one request, answered once."""
    stt = FakeStreamingSTT("I need a dental cleaning", "next Tuesday evening", endpointing_ms=500)
    thinking = HangingStream()
    client = FakeClient(thinking, [text("Tuesday evening, got it. "), finish("stop")])

    async with running_call(monkeypatch, seeded, client=client, stt=stt, tts=FakeTTS()) as ws:
        await ws.wait_for("listening")
        ws.feed(tone(700) + silence(560))  # the service ends it here
        await asyncio.wait_for(thinking.reached_the_end.wait(), 30)
        ws.feed(tone(700) + silence(560))  # the rest of the sentence
        await ws.wait_for("turn_done")
        await ws.wait_for("listening", count=2)

    sent = client.requests[1]["messages"]
    assert [m["content"] for m in sent if m["role"] == "user"] == [
        "I need a dental cleaning next Tuesday evening"
    ]
    _, turns = await stored_turns(sessionmaker_for, ws)
    assert [t.role for t in turns] == [TurnRole.AGENT, TurnRole.CALLER, TurnRole.AGENT]
    assert stt.begun == 2, "the second half opened a fresh utterance"


async def test_the_start_of_an_interruption_reaches_the_transcriber(monkeypatch, seeded):
    """The caller has to talk for a while before it counts as an interruption.
    That audio was held back, not thrown away: the first words must still be
    transcribed along with the rest."""
    stt = FakeStreamingSTT(FIRST, "never mind", endpointing_ms=500)
    async with running_call(
        monkeypatch, seeded, client=FakeClient(*OFFER, [text("Okay. "), finish("stop")]),
        stt=stt, tts=FakeTTS(seconds=2.0),
    ) as ws:
        await ws.wait_for("listening")
        ws.feed(tone(700) + silence(560))
        await ws.wait_for("audio_start", count=2)
        before = len(stt.sent)
        ws.feed(tone(700) + silence(560))
        await ws.wait_for("listening", count=2)

    interruption_bytes = len(stt.sent) - before
    # All 700 ms of speech (32 bytes per millisecond), plus the pre-roll.
    assert interruption_bytes >= 700 * 32
