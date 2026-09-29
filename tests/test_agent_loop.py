"""Tests for the agent loop, against a scripted model.

The loop does the fiddly part of tool use: tool calls arrive from a streaming API
in fragments (a name in one chunk, the arguments a few characters at a time in the
next), keyed by index, and have to be reassembled before anything can be parsed.
Until now that was only ever exercised live, against one model on one day.

Here a fake client replays exact chunk sequences, so every shape of stream the
loop must survive -- split arguments, interleaved parallel calls, malformed JSON,
a model that says nothing -- is a deterministic test. No network, no database, and
no API quota spent.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import pytest

from backend.app.agent import loop as loop_module
from backend.app.agent.loop import EMPTY_REPLY_NUDGE, MAX_TOOL_ITERATIONS, AgentRunner
from backend.app.agent.session import AgentSession
from backend.app.config import Settings
from tests.fakes import FakeClient, Stream, finish, text, tool, usage  # noqa: F401

NOW = datetime(2026, 9, 28, 10, 0, tzinfo=UTC)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

class ToolLog(list):
    """Every tool call made, as (name, arguments); ``results`` scripts the outcomes."""

    def __init__(self) -> None:
        super().__init__()
        self.results: dict[str, str | Exception] = {}


@pytest.fixture
def calls(monkeypatch) -> ToolLog:
    """Replaces tool execution, recording every call. Returns the log."""
    log = ToolLog()

    async def fake_execute(name, arguments, ctx):
        log.append((name, arguments))
        outcome = log.results.get(name, f"{name} ok")
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    monkeypatch.setattr(loop_module, "execute_tool", fake_execute)
    return log


def make_runner(client, **settings_overrides) -> AgentRunner:
    settings = Settings(
        database_url="postgresql+psycopg://user:pw@localhost/db",
        llm_api_key="test-key",
        llm_reasoning_effort=settings_overrides.pop("llm_reasoning_effort", "low"),
        **settings_overrides,
    )
    session = AgentSession(
        business_id=uuid.uuid4(),
        business_name="Test Clinic",
        tz=ZoneInfo("Asia/Kolkata"),
        caller_phone="+919000000000",
    )
    ctx = SimpleNamespace(session=session, now=NOW)
    return AgentRunner(client=client, settings=settings, system_prompt="You are a receptionist.", ctx=ctx)


async def run(runner: AgentRunner, utterance: str = "hello", on_text=None):
    return await runner.run_turn(utterance, now=NOW, on_text=on_text)


# ---------------------------------------------------------------------------
# Plain text
# ---------------------------------------------------------------------------

async def test_text_is_streamed_to_the_callback_in_order(calls):
    client = FakeClient([text("Sunrise "), text("Dental, "), text("hello."), finish("stop")])
    seen: list[str] = []
    result = await run(make_runner(client), on_text=seen.append)

    assert seen == ["Sunrise ", "Dental, ", "hello."]
    assert result.text == "Sunrise Dental, hello."
    assert result.stop_reason == "stop"
    assert result.handoff is False


async def test_the_reply_is_added_to_history(calls):
    runner = make_runner(FakeClient([text("Hello there."), finish("stop")]))
    await run(runner, "hi")
    assert runner.messages == [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "Hello there."},
    ]


async def test_token_usage_is_counted_from_the_final_chunk(calls):
    client = FakeClient([text("Hello."), finish("stop"), usage(2714, 98)])
    result = await run(make_runner(client))
    assert (result.metrics.input_tokens, result.metrics.output_tokens) == (2714, 98)


async def test_first_token_latency_is_recorded_only_when_words_arrive(calls):
    result = await run(make_runner(FakeClient([text("Hello."), finish("stop")])))
    assert result.metrics.llm_first_token_ms is not None
    assert result.metrics.llm_done_ms is not None
    assert result.metrics.total_ms is not None


# ---------------------------------------------------------------------------
# Tool calls, reassembled from a stream
# ---------------------------------------------------------------------------

async def test_a_tool_call_split_across_chunks_is_reassembled(calls):
    """The name arrives once; the arguments trickle in a few characters at a time."""
    client = FakeClient(
        [
            tool(0, id="call_1", name="check_availability"),
            tool(0, args='{"service": "Dental cl'),
            tool(0, args='eaning", "when": "tomor'),
            tool(0, args='row", "staff_name": null}'),
            finish("tool_calls"),
        ],
        [text("Tomorrow at five."), finish("stop")],
    )
    result = await run(make_runner(client))

    assert calls == [
        ("check_availability", {"service": "Dental cleaning", "when": "tomorrow", "staff_name": None})
    ]
    assert result.text == "Tomorrow at five."
    assert [c["name"] for c in result.tool_calls] == ["check_availability"]


async def test_the_tool_result_is_returned_to_the_model_tied_to_its_call(calls):
    calls.results["check_availability"] = "Free slots: [s1]"
    client = FakeClient(
        [tool(0, id="call_9", name="check_availability", args="{}"), finish("tool_calls")],
        [text("Done."), finish("stop")],
    )
    runner = make_runner(client)
    await run(runner)

    second_request = client.requests[1]["messages"]
    assistant = next(m for m in second_request if m.get("tool_calls"))
    assert assistant["tool_calls"][0]["id"] == "call_9"
    assert assistant["tool_calls"][0]["function"]["name"] == "check_availability"
    result_message = next(m for m in second_request if m["role"] == "tool")
    assert result_message == {"role": "tool", "tool_call_id": "call_9", "content": "Free slots: [s1]"}


async def test_parallel_calls_with_interleaved_fragments_are_kept_apart(calls):
    """Two calls streaming at once, distinguished only by index."""
    client = FakeClient(
        [
            tool(0, id="a", name="cancel_appointment"),
            tool(1, id="b", name="handoff_to_human"),
            tool(0, args='{"booking_reference": "ABC123", '),
            tool(1, args='{"reason": "caller_asked_for_human", '),
            tool(1, args='"summary": "wants a person"}'),
            tool(0, args='"customer_phone": null}'),
            finish("tool_calls"),
        ],
        [text("Done."), finish("stop")],
    )
    await run(make_runner(client))

    assert calls == [
        ("cancel_appointment", {"booking_reference": "ABC123", "customer_phone": None}),
        ("handoff_to_human", {"reason": "caller_asked_for_human", "summary": "wants a person"}),
    ]
    results = [m for m in client.requests[1]["messages"] if m["role"] == "tool"]
    assert [m["tool_call_id"] for m in results] == ["a", "b"]


async def test_a_tool_with_no_arguments_gets_an_empty_object(calls):
    client = FakeClient(
        [tool(0, id="x", name="check_availability"), finish("tool_calls")],
        [text("Ok."), finish("stop")],
    )
    await run(make_runner(client))
    assert calls == [("check_availability", {})]


async def test_malformed_arguments_are_reported_to_the_model_not_executed(calls):
    """A truncated JSON blob is a recoverable slip. Raised, it would end the call."""
    client = FakeClient(
        [
            tool(0, id="bad", name="check_availability", args='{"service": "clean'),
            finish("tool_calls"),
        ],
        [text("Sorry, one moment."), finish("stop")],
    )
    result = await run(make_runner(client))

    assert calls == [], "a call with unreadable arguments must not run"
    told = next(m for m in client.requests[1]["messages"] if m["role"] == "tool")
    assert told["content"].startswith("ERROR:")
    assert "valid JSON" in told["content"]
    assert result.tool_calls[0]["is_error"] is True
    assert result.text == "Sorry, one moment."


async def test_arguments_that_are_not_an_object_are_rejected(calls):
    client = FakeClient(
        [tool(0, id="bad", name="check_availability", args="[1, 2]"), finish("tool_calls")],
        [text("Ok."), finish("stop")],
    )
    await run(make_runner(client))
    assert calls == []


async def test_a_crashing_tool_does_not_end_the_call(calls):
    calls.results["check_availability"] = RuntimeError("database went away")
    client = FakeClient(
        [tool(0, id="x", name="check_availability", args="{}"), finish("tool_calls")],
        [text("Sorry, I could not check that."), finish("stop")],
    )
    result = await run(make_runner(client))

    assert result.text == "Sorry, I could not check that."
    assert result.handoff is False
    told = next(m for m in client.requests[1]["messages"] if m["role"] == "tool")
    assert "database went away" in told["content"]
    assert result.tool_calls[0]["is_error"] is True


async def test_every_tool_call_is_logged_with_its_timing(calls):
    client = FakeClient(
        [tool(0, id="x", name="check_availability", args='{"a": 1}'), finish("tool_calls")],
        [text("Ok."), finish("stop")],
    )
    result = await run(make_runner(client))
    entry = result.tool_calls[0]
    assert entry["name"] == "check_availability"
    assert entry["input"] == '{"a": 1}'
    assert entry["output"] == "check_availability ok"
    assert entry["duration_ms"] >= 0


# ---------------------------------------------------------------------------
# Text either side of a tool call
# ---------------------------------------------------------------------------

async def test_words_before_and_after_a_tool_call_are_separated(calls):
    """"Let me check.We have..." would run together, and hide the sentence
    boundary from whatever cuts this stream into speech."""
    client = FakeClient(
        [text("Let me check."), tool(0, id="x", name="check_availability", args="{}"), finish("tool_calls")],
        [text("We have five o'clock."), finish("stop")],
    )
    seen: list[str] = []
    result = await run(make_runner(client), on_text=seen.append)

    assert result.text == "Let me check. We have five o'clock."
    assert "".join(seen) == "Let me check. We have five o'clock."


# ---------------------------------------------------------------------------
# A model that says nothing
# ---------------------------------------------------------------------------

async def test_an_empty_reply_is_retried_once_with_a_nudge(calls):
    client = FakeClient(
        [finish("stop"), usage(100, 60)],  # reasoned, then wrote nothing
        [text("Sorry, could you repeat that?"), finish("stop")],
    )
    result = await run(make_runner(client))

    assert result.text == "Sorry, could you repeat that?"
    assert len(client.requests) == 2
    retry = client.requests[1]["messages"]
    assert retry[-1] == {"role": "system", "content": EMPTY_REPLY_NUDGE}
    first = client.requests[0]["messages"]
    assert all(m["content"] != EMPTY_REPLY_NUDGE for m in first)


async def test_the_nudge_is_never_stored_in_history(calls):
    client = FakeClient([finish("stop")], [text("Hello."), finish("stop")])
    runner = make_runner(client)
    await run(runner)
    assert all(m.get("content") != EMPTY_REPLY_NUDGE for m in runner.messages)


async def test_a_model_that_stays_silent_is_not_retried_forever(calls):
    """One retry, then give up and let the voice layer say something."""
    client = FakeClient([finish("stop")], [finish("stop")])
    result = await run(make_runner(client))
    assert result.text == ""
    assert len(client.requests) == 2
    assert result.handoff is False


async def test_a_nonempty_reply_is_not_retried(calls):
    client = FakeClient([text("Hello."), finish("stop")])
    await run(make_runner(client))
    assert len(client.requests) == 1


async def test_whitespace_alone_counts_as_empty(calls):
    client = FakeClient([text("   "), finish("stop")], [text("Hello."), finish("stop")])
    result = await run(make_runner(client))
    assert result.text == "Hello."
    assert len(client.requests) == 2


# ---------------------------------------------------------------------------
# Failure containment
# ---------------------------------------------------------------------------

async def test_a_failed_request_rewinds_history(calls):
    """A user turn with no reply left behind would make the next request
    malformed, turning one dropped connection into a dead conversation."""
    runner = make_runner(FakeClient(raises=ConnectionError("network unreachable")))
    result = await run(runner, "book me in")

    assert runner.messages == []
    assert result.stop_reason.startswith("error:ConnectionError")
    assert "network unreachable" in result.stop_reason
    assert result.handoff is True
    # And the session says so too, so the stored call is not filed as completed.
    assert runner._ctx.session.handoff.reason == "agent_cannot_proceed"
    assert "ConnectionError" in runner._ctx.session.handoff.summary


async def test_a_conversation_continues_after_a_failed_turn(calls):
    runner = make_runner(FakeClient(raises=ConnectionError("blip")))
    await run(runner, "first")
    runner._client = FakeClient([text("Hello again."), finish("stop")])
    result = await run(runner, "second")

    assert result.text == "Hello again."
    assert [m["role"] for m in runner.messages] == ["user", "assistant"]


async def test_a_model_that_never_stops_calling_tools_is_cut_off(calls):
    endless = [
        [tool(0, id=f"c{i}", name="check_availability", args="{}"), finish("tool_calls")]
        for i in range(MAX_TOOL_ITERATIONS + 2)
    ]
    result = await run(make_runner(FakeClient(*endless)))

    assert result.stop_reason == "max_iterations"
    assert result.handoff is True
    assert len(calls) == MAX_TOOL_ITERATIONS


# ---------------------------------------------------------------------------
# What is sent
# ---------------------------------------------------------------------------

async def test_reasoning_effort_is_sent_when_configured(calls):
    client = FakeClient([text("Hi."), finish("stop")])
    await run(make_runner(client, llm_reasoning_effort="low"))
    assert client.requests[0]["reasoning_effort"] == "low"


async def test_reasoning_effort_is_omitted_when_empty(calls):
    """For models that reject the parameter."""
    client = FakeClient([text("Hi."), finish("stop")])
    await run(make_runner(client, llm_reasoning_effort=""))
    assert "reasoning_effort" not in client.requests[0]


async def test_the_request_asks_for_streamed_usage(calls):
    client = FakeClient([text("Hi."), finish("stop")])
    await run(make_runner(client))
    request = client.requests[0]
    assert request["stream"] is True
    assert request["stream_options"] == {"include_usage": True}
    assert request["tool_choice"] == "auto"


async def test_the_system_message_is_first_and_carries_the_clock(calls):
    client = FakeClient([text("Hi."), finish("stop")])
    await run(make_runner(client))
    messages = client.requests[0]["messages"]
    assert messages[0]["role"] == "system"
    assert "You are a receptionist." in messages[0]["content"]
    assert "Monday 28 September 2026, 03:30 PM" in messages[0]["content"]
    assert [m["role"] for m in messages].count("system") == 1


async def test_the_caller_is_remembered_for_the_name_guard(calls):
    runner = make_runner(FakeClient([text("Hi."), finish("stop")]))
    await run(runner, "My name is Asha Menon")
    assert runner._ctx.session.name_was_spoken("Asha Menon")


async def test_the_tools_are_offered_on_every_request(calls):
    client = FakeClient(
        [tool(0, id="x", name="check_availability", args="{}"), finish("tool_calls")],
        [text("Ok."), finish("stop")],
    )
    await run(make_runner(client))
    for request in client.requests:
        names = [t["function"]["name"] for t in request["tools"]]
        assert "book_appointment" in names and "handoff_to_human" in names


async def test_a_handoff_the_model_already_requested_is_not_overwritten(calls):
    """If the model asked for a human and then something failed, the caller's
    reason is the one worth keeping."""
    runner = make_runner(FakeClient(raises=ConnectionError("blip")))
    from backend.app.agent.session import HandoffRequest

    runner._ctx.session.handoff = HandoffRequest(reason="caller_asked_for_human", summary="wanted a person")
    await run(runner)
    assert runner._ctx.session.handoff.reason == "caller_asked_for_human"
