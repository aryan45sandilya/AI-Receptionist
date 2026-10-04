"""The whole voice endpoint, end to end, with only the paid services faked.

Real: the WebSocket handler, the voice activity detector, the turn timing, the
agent loop, the tools, the per-call snapshot and the Postgres database.
Faked: speech-to-text, text-to-speech and the language model, each replaced by a
scripted double. That keeps the run free and deterministic while still exercising
every line between "audio arrives" and "rows are written".

The tests are organised around the rule they exist to prove: *every stage of every
turn is logged*. Reading the rows back out of Postgres, rather than trusting the
events the server sent, is what makes that a claim about what was stored.
"""

from __future__ import annotations

import asyncio

import pytest
from sqlalchemy import select

from backend.app.enums import CallStatus, TurnRole
from tests.audio_helpers import hiss, silence, tone
from tests.fakes import FakeClient, finish, text, tool
from tests.voice_helpers import (
    CHECK_ARGS,
    HANDOFF,
    OFFER,
    FakeSTT,
    FakeTTS,
    running_call,
    speak_turn,
    stored_turns,
)

pytestmark = pytest.mark.db


# ---------------------------------------------------------------------------
# The rule: every stage of every turn is logged
# ---------------------------------------------------------------------------

LATENCY_COLUMNS = (
    "stt_final_ms",
    "llm_first_token_ms",
    "llm_done_ms",
    "tts_first_audio_ms",
    "total_ms",
)


async def test_every_stage_of_a_turn_is_written_to_the_database(
    monkeypatch, sessionmaker_for, seeded
):
    stt, tts = FakeSTT("I need a dental cleaning next Tuesday evening"), FakeTTS()
    async with running_call(monkeypatch, seeded, client=FakeClient(*OFFER), stt=stt, tts=tts) as ws:
        await speak_turn(ws, turn=1)

    call, turns = await stored_turns(sessionmaker_for, ws)
    agent_reply = turns[-1]

    assert agent_reply.role is TurnRole.AGENT
    for column in LATENCY_COLUMNS:
        assert getattr(agent_reply, column) is not None, f"{column} was not logged"

    # The stages happen in this order, and each is measured from the same origin.
    assert agent_reply.stt_final_ms < agent_reply.llm_first_token_ms
    assert agent_reply.llm_first_token_ms <= agent_reply.llm_done_ms
    # Never before the first word; the same millisecond is possible when both
    # services are instant.
    assert agent_reply.llm_first_token_ms <= agent_reply.tts_first_audio_ms
    # What the caller waited is when they first heard something.
    assert agent_reply.total_ms == agent_reply.tts_first_audio_ms


async def test_the_clock_starts_when_the_caller_stopped_speaking(
    monkeypatch, sessionmaker_for, seeded
):
    """The detector needs 800 ms of quiet to be sure the caller has finished, and
    they wait through all of it. A timer that started when the server noticed
    would leave that out and flatter the numbers."""
    stt = FakeSTT("I need a dental cleaning next Tuesday evening")
    async with running_call(
        monkeypatch, seeded, client=FakeClient(*OFFER), stt=stt, tts=FakeTTS()
    ) as ws:
        await speak_turn(ws, turn=1)

    done = ws.of("turn_done")[0]
    assert done["endpoint_ms"] == 800
    assert done["stt_final_ms"] >= 800

    _, turns = await stored_turns(sessionmaker_for, ws)
    assert turns[-1].stt_final_ms >= 800


async def test_a_turn_is_stored_as_caller_then_agent_with_the_right_text(
    monkeypatch, sessionmaker_for, seeded
):
    stt = FakeSTT("I need a dental cleaning next Tuesday evening")
    async with running_call(
        monkeypatch, seeded, client=FakeClient(*OFFER), stt=stt, tts=FakeTTS()
    ) as ws:
        await speak_turn(ws, turn=1)

    _, turns = await stored_turns(sessionmaker_for, ws)
    assert [(t.turn_index, t.role) for t in turns] == [
        (0, TurnRole.AGENT),  # the greeting
        (1, TurnRole.CALLER),
        (2, TurnRole.AGENT),
    ]
    assert turns[0].text == "Test Clinic, how can I help?"
    assert turns[1].text == "I need a dental cleaning next Tuesday evening"
    assert "Tuesday at five o'clock" in turns[2].text


async def test_the_greeting_records_time_to_first_audio(monkeypatch, sessionmaker_for, seeded):
    async with running_call(
        monkeypatch, seeded, client=FakeClient(), stt=FakeSTT(), tts=FakeTTS()
    ) as ws:
        await ws.wait_for("listening")

    _, turns = await stored_turns(sessionmaker_for, ws)
    assert turns[0].tts_first_audio_ms is not None
    assert turns[0].total_ms == turns[0].tts_first_audio_ms


async def test_the_model_and_its_token_usage_are_logged(monkeypatch, sessionmaker_for, seeded):
    """The inputs to the cost-per-call figure in the README."""
    stt = FakeSTT("I need a dental cleaning next Tuesday evening")
    async with running_call(
        monkeypatch, seeded, client=FakeClient(*OFFER), stt=stt, tts=FakeTTS()
    ) as ws:
        await speak_turn(ws, turn=1)

    _, turns = await stored_turns(sessionmaker_for, ws)
    reply = turns[-1]
    assert reply.model
    assert reply.input_tokens == 1234
    assert reply.output_tokens == 56


async def test_the_tool_calls_a_turn_made_are_stored_with_their_timings(
    monkeypatch, sessionmaker_for, seeded
):
    stt = FakeSTT("I need a dental cleaning next Tuesday evening")
    async with running_call(
        monkeypatch, seeded, client=FakeClient(*OFFER), stt=stt, tts=FakeTTS()
    ) as ws:
        await speak_turn(ws, turn=1)

    _, turns = await stored_turns(sessionmaker_for, ws)
    (recorded,) = turns[-1].tool_calls["calls"]
    assert recorded["name"] == "check_availability"
    assert recorded["is_error"] is False
    assert recorded["duration_ms"] >= 0
    assert "Free slots" in recorded["output"]


async def test_the_call_itself_is_closed_out(monkeypatch, sessionmaker_for, seeded):
    async with running_call(
        monkeypatch, seeded, client=FakeClient(), stt=FakeSTT(), tts=FakeTTS()
    ) as ws:
        await ws.wait_for("listening")

    call, _ = await stored_turns(sessionmaker_for, ws)
    assert call.status is CallStatus.COMPLETED
    assert call.ended_at is not None
    assert call.duration_ms is not None and call.duration_ms > 0
    assert call.channel.value == "web_demo"


# ---------------------------------------------------------------------------
# The protocol the browser depends on
# ---------------------------------------------------------------------------

async def test_a_turn_sends_its_events_in_order(monkeypatch, seeded):
    stt = FakeSTT("I need a dental cleaning next Tuesday evening")
    async with running_call(
        monkeypatch, seeded, client=FakeClient(*OFFER), stt=stt, tts=FakeTTS()
    ) as ws:
        await speak_turn(ws, turn=1)

    kinds = ws.kinds()
    order = ["ready", "audio_start", "audio_end", "listening", "speech_started",
             "speech_ended", "transcript", "agent_delta", "audio_start", "audio_end",
             "turn_done", "listening"]
    position = -1
    for kind in order:
        position = kinds.index(kind, position + 1)  # raises if out of order


async def test_the_reply_text_streams_to_the_browser_as_it_is_written(monkeypatch, seeded):
    stt = FakeSTT("I need a dental cleaning next Tuesday evening")
    async with running_call(
        monkeypatch, seeded, client=FakeClient(*OFFER), stt=stt, tts=FakeTTS()
    ) as ws:
        await speak_turn(ws, turn=1)

    streamed = "".join(event["text"] for event in ws.of("agent_delta"))
    assert "We have Tuesday at five o'clock." in streamed
    assert "Which would suit you best?" in streamed


async def test_the_reply_is_spoken_a_sentence_at_a_time(monkeypatch, seeded):
    stt, tts = FakeSTT("I need a dental cleaning next Tuesday evening"), FakeTTS()
    async with running_call(monkeypatch, seeded, client=FakeClient(*OFFER), stt=stt, tts=tts) as ws:
        await speak_turn(ws, turn=1)

    assert tts.spoken == [
        "Test Clinic, how can I help?",
        "We have Tuesday at five o'clock.",
        "Which would suit you best?",
    ]
    assert len(ws.audio) > 0


async def test_audio_is_announced_with_its_sample_rate(monkeypatch, seeded):
    async with running_call(
        monkeypatch, seeded, client=FakeClient(), stt=FakeSTT(), tts=FakeTTS()
    ) as ws:
        await ws.wait_for("listening")

    assert ws.of("audio_start")[0]["sample_rate"] == 24_000
    assert ws.of("ready")[0]["sample_rate"] == 16_000


# ---------------------------------------------------------------------------
# What must never reach the model
# ---------------------------------------------------------------------------

async def test_silence_never_reaches_the_transcription_service(monkeypatch, seeded):
    """The reason for the detector: the model answers silence with "Thank you.",
    and an agent that transcribed everything would reply to it."""
    stt, client = FakeSTT("should never be asked for"), FakeClient()
    async with running_call(monkeypatch, seeded, client=client, stt=stt, tts=FakeTTS()) as ws:
        await ws.wait_for("listening")
        ws.feed(silence(1500) + hiss(1500))
        await asyncio.sleep(1.0)

    assert stt.calls == 0
    assert "speech_started" not in ws.kinds()
    assert client.requests == []


async def test_a_transcript_the_service_filters_as_noise_gets_no_reply(
    monkeypatch, sessionmaker_for, seeded
):
    """A breath long enough to trip the detector, transcribed as nothing."""
    stt, client = FakeSTT(""), FakeClient()
    async with running_call(monkeypatch, seeded, client=client, stt=stt, tts=FakeTTS()) as ws:
        await speak_turn(ws, turn=1)

    assert ws.of("transcript") == [{"type": "transcript", "text": "", "discarded": True}]
    assert client.requests == [], "no reply may be generated to noise"
    _, turns = await stored_turns(sessionmaker_for, ws)
    assert len(turns) == 1, "only the greeting should be stored"


async def test_with_barge_in_off_audio_while_the_agent_is_speaking_is_discarded(monkeypatch, seeded):
    """Half duplex, for a setup where the speakers feed the agent its own voice:
    with barge-in switched off the agent talks through whatever it hears."""
    stt = FakeSTT("I need a dental cleaning next Tuesday evening", "this must never be heard")
    tts = FakeTTS(delay=1.0)  # slow enough that the reply is still being made
    async with running_call(
        monkeypatch, seeded, client=FakeClient(*OFFER), stt=stt, tts=tts, barge_in=False
    ) as ws:
        await ws.wait_for("listening")
        ws.feed(tone(700) + silence(1200))
        await ws.wait_for("transcript")
        # The agent is mid-reply. Speech now belongs to nobody.
        ws.feed(tone(700) + silence(1200))
        await ws.wait_for("listening", count=2)

    assert stt.calls == 1


# ---------------------------------------------------------------------------
# Failure
# ---------------------------------------------------------------------------

async def test_a_voice_that_will_not_speak_is_reported_and_logged_honestly(
    monkeypatch, sessionmaker_for, seeded
):
    """Most likely cause in practice: the synthesis model's terms were never
    accepted. The caller hears nothing, so the turn has no response time -- it
    is stored as NULL, not as a plausible-looking number."""
    stt = FakeSTT("I need a dental cleaning next Tuesday evening")
    async with running_call(
        monkeypatch, seeded, client=FakeClient(*OFFER), stt=stt, tts=FakeTTS(fail=True)
    ) as ws:
        await speak_turn(ws, turn=1)

    assert ws.of("tts_error"), "the failure must reach the page"
    assert "terms acceptance" in ws.of("tts_error")[0]["message"]
    assert ws.audio == []

    _, turns = await stored_turns(sessionmaker_for, ws)
    reply = turns[-1]
    assert "Tuesday at five o'clock" in reply.text, "the words are still stored"
    assert reply.tts_first_audio_ms is None
    assert reply.total_ms is None
    # The stages that did happen are still there.
    assert reply.stt_final_ms is not None and reply.llm_first_token_ms is not None


async def test_the_call_survives_a_voice_failure(monkeypatch, seeded):
    async with running_call(
        monkeypatch, seeded, client=FakeClient(*OFFER), stt=FakeSTT("hello there"), tts=FakeTTS(fail=True)
    ) as ws:
        await speak_turn(ws, turn=1)
    assert ws.of("listening"), "still listening after the voice failed"
    assert ws.closed is False


async def test_a_model_that_says_nothing_gets_a_spoken_fallback(monkeypatch, seeded):
    """A phone call must never go silent."""
    client = FakeClient([finish("stop")], [finish("stop")])  # empty, and empty again
    tts = FakeTTS()
    async with running_call(monkeypatch, seeded, client=client, stt=FakeSTT("hello"), tts=tts) as ws:
        await speak_turn(ws, turn=1)

    # Spoken as two sentences, like anything else: the splitter does not know
    # this one was written by us.
    assert "Sorry, I didn't catch that. Could you say it again?" in " ".join(tts.spoken)


# ---------------------------------------------------------------------------
# Handing over
# ---------------------------------------------------------------------------

async def test_a_handoff_ends_the_call_and_records_why(monkeypatch, sessionmaker_for, seeded):
    async with running_call(
        monkeypatch,
        seeded,
        client=FakeClient(*HANDOFF),
        stt=FakeSTT("can I speak to a real person"),
        tts=FakeTTS(),
        hang_up=False,  # the server must end this call by itself
    ) as ws:
        await ws.wait_for("listening")
        ws.feed(tone(700) + silence(1200))
        await ws.wait_for("handoff")

    assert ws.of("handoff")[0]["reason"] == "caller_asked_for_human"
    assert ws.closed is True

    call, turns = await stored_turns(sessionmaker_for, ws)
    assert call.status is CallStatus.HANDED_OFF
    assert call.handoff_reason == "Wants to speak to a person."
    assert "putting you through" in turns[-1].text


async def test_a_model_failure_hands_the_call_over_and_says_why(monkeypatch, sessionmaker_for, seeded):
    """The stored call must say the agent gave up, not that it completed."""
    client = FakeClient(raises=ConnectionError("network unreachable"))
    async with running_call(
        monkeypatch, seeded, client=client, stt=FakeSTT("hello"), tts=FakeTTS(), hang_up=False
    ) as ws:
        await ws.wait_for("listening")
        ws.feed(tone(700) + silence(1200))
        await ws.wait_for("handoff")

    call, _ = await stored_turns(sessionmaker_for, ws)
    assert call.status is CallStatus.HANDED_OFF
    assert "ConnectionError" in call.handoff_reason


# ---------------------------------------------------------------------------
# A real booking by voice
# ---------------------------------------------------------------------------

async def test_a_booking_made_by_voice_lands_in_the_bookings_table(
    monkeypatch, sessionmaker_for, seeded
):
    """The whole chain: speech -> transcript -> agent -> tools -> Postgres."""
    from backend.app.models import Booking

    book = [
        [tool(0, id="c1", name="check_availability", args=CHECK_ARGS), finish("tool_calls")],
        [text("I have five o'clock on Tuesday. Shall I book it?"), finish("stop")],
    ]
    confirm = [
        [
            tool(0, id="b1", name="book_appointment",
                 args='{"slot_id": "s1", "customer_name": "Asha Menon", "customer_phone": null}'),
            finish("tool_calls"),
        ],
        [text("You are booked."), finish("stop")],
    ]
    stt = FakeSTT("I need a dental cleaning next Tuesday evening", "Yes please, this is Asha Menon")
    async with running_call(
        monkeypatch, seeded, client=FakeClient(*book, *confirm), stt=stt, tts=FakeTTS()
    ) as ws:
        await speak_turn(ws, turn=1)
        await speak_turn(ws, turn=2)

    call, _ = await stored_turns(sessionmaker_for, ws)
    async with sessionmaker_for() as db:
        rows = (
            await db.execute(select(Booking).where(Booking.call_id == call.id))
        ).unique().scalars().all()

    assert len(rows) == 1
    assert rows[0].customer_name == "Asha Menon"
    assert rows[0].source.value == "voice"
    assert rows[0].customer_phone_e164 == "+919876543210"


# ---------------------------------------------------------------------------
# Filling the silence while a tool runs
# ---------------------------------------------------------------------------

LOOKUP = '{"service": "Dental cleaning", "when": "next tuesday evening", "staff_name": null}'


async def test_a_tool_turn_fills_the_silence_and_logs_the_stopgap_apart_from_the_answer(
    monkeypatch, sessionmaker_for, seeded
):
    """The model asks for a tool without a word, so the line would be dead for the
    seconds the lookup takes. A stopgap is said at once -- and timed separately,
    because it makes a slow turn feel faster without making it faster."""
    stt, tts = FakeSTT("I need a dental cleaning next Tuesday evening"), FakeTTS()
    async with running_call(
        monkeypatch, seeded, client=FakeClient(*OFFER), stt=stt, tts=tts, tool_filler=True
    ) as ws:
        await speak_turn(ws, turn=1)

    # After the greeting, before the answer, in that order.
    assert tts.spoken[:3] == [
        "Test Clinic, how can I help?",
        "One moment.",
        "We have Tuesday at five o'clock.",
    ]

    _, turns = await stored_turns(sessionmaker_for, ws)
    reply = turns[-1]
    assert reply.filler_ms is not None
    assert reply.filler_ms < reply.total_ms, "the answer arrived after the stopgap"
    # The answer's own timings are still the answer's.
    assert reply.total_ms == reply.tts_first_audio_ms
    assert reply.llm_first_token_ms <= reply.total_ms  # equal when both services are instant
    assert reply.text.startswith("One moment. We have Tuesday")
    assert ws.of("turn_done")[0]["filler_ms"] == reply.filler_ms


async def test_no_stopgap_is_said_when_the_model_already_spoke(monkeypatch, sessionmaker_for, seeded):
    script = [
        [text("Let me check that for you. "),
         tool(0, id="c1", name="check_availability", args=LOOKUP), finish("tool_calls")],
        [text("We have Tuesday at five o'clock. "), finish("stop")],
    ]
    stt, tts = FakeSTT("I need a dental cleaning next Tuesday evening"), FakeTTS()
    async with running_call(
        monkeypatch, seeded, client=FakeClient(*script), stt=stt, tts=tts, tool_filler=True
    ) as ws:
        await speak_turn(ws, turn=1)

    assert "One moment." not in tts.spoken
    _, turns = await stored_turns(sessionmaker_for, ws)
    assert turns[-1].filler_ms is None
    assert turns[-1].total_ms == turns[-1].tts_first_audio_ms


async def test_no_stopgap_is_said_before_a_handoff(monkeypatch, sessionmaker_for, seeded):
    """Handing over has the model's own words; "one moment" ahead of them would
    be a second goodbye."""
    stt, tts = FakeSTT("can I speak to a real person"), FakeTTS()
    async with running_call(
        monkeypatch, seeded, client=FakeClient(*HANDOFF), stt=stt, tts=tts,
        tool_filler=True, hang_up=False,
    ) as ws:
        await ws.wait_for("listening")
        ws.feed(tone(700) + silence(1200))
        await ws.wait_for("handoff")

    assert not [line for line in tts.spoken if line in {"One moment.", "Let me check that.", "Sure, one moment."}]


async def test_the_stopgap_can_be_switched_off(monkeypatch, sessionmaker_for, seeded):
    stt, tts = FakeSTT("I need a dental cleaning next Tuesday evening"), FakeTTS()
    async with running_call(
        monkeypatch, seeded, client=FakeClient(*OFFER), stt=stt, tts=tts, tool_filler=False
    ) as ws:
        await speak_turn(ws, turn=1)

    assert "One moment." not in tts.spoken
    _, turns = await stored_turns(sessionmaker_for, ws)
    assert turns[-1].filler_ms is None


async def test_the_stopgaps_vary_so_a_long_call_does_not_sound_like_a_recording(monkeypatch, seeded):
    again = [
        [tool(0, id="c2", name="check_availability", args=LOOKUP), finish("tool_calls")],
        [text("Wednesday is free too. "), finish("stop")],
    ]
    stt, tts = FakeSTT("tuesday evening", "what about wednesday"), FakeTTS()
    async with running_call(
        monkeypatch, seeded, client=FakeClient(*OFFER, *again), stt=stt, tts=tts, tool_filler=True
    ) as ws:
        await speak_turn(ws, turn=1)
        await speak_turn(ws, turn=2)

    said = [line for line in tts.spoken if line in {"One moment.", "Let me check that.", "Sure, one moment."}]
    assert said == ["One moment.", "Let me check that."]

