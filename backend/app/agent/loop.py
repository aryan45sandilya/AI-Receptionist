"""The streaming tool-use loop, against Groq's OpenAI-compatible endpoint.

A manual loop rather than a framework, because of barge-in: when the caller
talks over the agent, playback stops and the in-flight turn is abandoned
mid-stream. That needs the ability to break out of the stream, and to keep the
message history consistent when it does, which means owning the loop. The
latency measurements belong at the same seams.

Interrupting a turn (``run_turn`` cancelled) follows three rules:

*   **A tool in flight is never abandoned.** Speech and generation stop at once,
    but a tool that has begun -- most importantly a booking being written --
    runs to completion, and its result goes into history. A half-written booking
    is worse than one the caller did not hear confirmed.
*   **History stays valid.** Every tool call has its result. If nothing at all
    had happened -- no tool run, no word heard -- the caller's words are taken
    back out of history and handed back (``Interruption.clean``) to be joined to
    whatever they say next, so a pause that cut a sentence in two does not
    become two separate requests.
*   **History says what the caller heard.** The reply that was cut off is
    recorded as the part that got out, not the part the model had planned
    (``settle_interruption``), so it does not believe it said something nobody
    heard. This applies whenever the caller cuts in: while the model is still
    writing, or -- far more often, because a reply is written in under a second
    and takes several to say -- after it has finished and is only being spoken.

Streaming is on from Phase 1 even though a terminal does not need it, because
Phase 2 pipes these same text chunks into TTS. Getting the first sentence to the
speech engine before the model has finished the paragraph is most of the
perceived-latency win in a voice agent.

Nothing here is Groq-specific beyond the base URL in config: the endpoint is
OpenAI-compatible, so another provider is a configuration change.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from openai import AsyncOpenAI

from ..config import Settings
from .prompts import build_system_prompt, current_time_note
from .session import HandoffRequest
from .tools import ToolContext, execute_tool, tool_schemas

# A booking conversation needs a handful of tool calls per turn at most. This cap
# stops a pathological loop, it is not a real working limit.
MAX_TOOL_ITERATIONS = 8

# A reasoning model at low effort now and then spends its whole budget thinking
# and returns neither words nor a tool call. Asking once more almost always
# produces an answer; asking a third time rarely does, and the caller is waiting.
MAX_EMPTY_RETRIES = 1
EMPTY_REPLY_NUDGE = (
    "You have not replied yet. Answer the caller now, in one or two short spoken "
    "sentences, or call the tool that is needed."
)


@dataclass
class TurnMetrics:
    """Per-stage latency for one turn, in milliseconds from the turn anchor.

    Maps onto the nullable columns of ``call_turns``. The STT and TTS fields stay
    None in Phase 1 -- there is no audio -- which is exactly why those columns are
    nullable: a partial turn should still record whatever it managed to measure.
    """

    turn_started_at: datetime
    stt_final_ms: int | None = None
    llm_first_token_ms: int | None = None
    llm_done_ms: int | None = None
    tts_first_audio_ms: int | None = None
    total_ms: int | None = None

    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    model: str | None = None

    # When the caller first heard anything, if that was a stopgap ("one moment")
    # said while a tool ran rather than the answer itself. Set by the voice layer.
    filler_ms: int | None = None

    # True once the stage times above have been moved from "since this turn
    # began" onto the call's clock (the moment the caller stopped talking). Set
    # by the voice layer; stops an interruption that arrives mid-way through
    # reporting from moving them twice.
    rebased: bool = False


@dataclass
class TurnResult:
    text: str
    metrics: TurnMetrics
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    stop_reason: str | None = None
    handoff: bool = False


@dataclass
class Interruption:
    """A turn that was cancelled while the model was still working on it.

    (An interruption that lands after the model has finished has no such record:
    the turn completed, and ``TurnResult`` says what it did.) Left unsettled until
    the speech side says how much of the reply the caller actually heard --
    ``AgentRunner.settle_interruption`` -- because that decides what the history
    should say. Afterwards ``clean`` is True if the turn had done nothing the
    caller could have noticed: no tool run, no word heard. The user's message has
    then been taken back out of history, and ``user_text`` is theirs to hand back:
    the caller was most likely still mid-sentence, and what they say next belongs
    with it, not in a second turn of its own.
    """

    history_mark: int
    user_text: str
    text: str  # what the model had written when it was cut off
    tool_calls: list[dict[str, Any]]
    metrics: TurnMetrics
    clean: bool = False
    settled: bool = False


# What goes into history for a tool the model asked for but the caller's
# interruption stopped from running. Every tool call needs a result or the next
# request is malformed.
SKIPPED_TOOL_RESULT = "SKIPPED: the caller interrupted before this ran."

# Marks a reply the caller cut off, in the model's own history. The system prompt
# explains it.
INTERRUPTED_MARK = "[interrupted]"


class AgentRunner:
    """Runs one conversation. Owns the message history."""

    def __init__(
        self,
        *,
        client: AsyncOpenAI,
        settings: Settings,
        system_prompt: str,
        ctx: ToolContext,
    ) -> None:
        self._client = client
        self._settings = settings
        self._ctx = ctx
        self._tools = tool_schemas()
        self._system_prompt = system_prompt
        # Only the conversation itself. The system message is rebuilt per
        # request so it can carry the current time.
        self._messages: list[dict[str, Any]] = []
        # Set when a turn was cancelled while the model was still on it; cleared
        # when the next turn starts.
        self.interruption: Interruption | None = None
        # The latest turn's place in history, kept after it completes so that an
        # interruption during its *speech* can still be settled.
        self._open_turn: int | None = None

    @property
    def messages(self) -> list[dict[str, Any]]:
        return self._messages

    def settle_interruption(self, heard: str) -> bool:
        """Put the history in order after the caller cut the latest turn short.

        True if the turn was clean. ``heard`` is what the caller got to hear of
        the reply, which is not what the model wrote: it was cut off, and may
        never have started. If nothing happened -- no tool ran and nothing was
        heard -- the turn is erased, so the caller's words can be merged with
        what they say next. Otherwise it stays, its reply replaced by what was
        heard and a mark saying the caller cut in.

        Works wherever the interruption landed: mid-generation (the reply is not
        in history yet, so it is appended) or after the model finished (the whole
        planned reply is in history, so it is trimmed to what was heard). Settles
        a turn once; a second call changes nothing.
        """
        mark, self._open_turn = self._open_turn, None
        interruption = self.interruption
        if mark is None:
            return bool(interruption and interruption.settled and interruption.clean)

        heard = heard.strip()
        # Any tool message after the user's own is a tool that ran (or, for one
        # the interruption stopped, a note that it did not).
        tools_ran = any(m["role"] == "tool" for m in self._messages[mark:])
        clean = not tools_ran and not heard
        if clean:
            del self._messages[mark:]
        else:
            content = f"{heard} {INTERRUPTED_MARK}" if heard else INTERRUPTED_MARK
            last = self._messages[-1] if len(self._messages) > mark + 1 else None
            if last is not None and last["role"] == "assistant" and not last.get("tool_calls"):
                # The model's whole planned reply is here; the caller heard some of it.
                self._messages[-1] = {"role": "assistant", "content": content}
            else:
                self._messages.append({"role": "assistant", "content": content})

        if interruption is not None:
            interruption.clean, interruption.settled = clean, True
        return clean

    def _hand_off(self, why: str) -> None:
        """Record that the agent is giving the call up, if nothing else has.

        The loop returns ``handoff=True`` when it cannot continue (a failed
        request, a model that will not settle). Recording it in the session as
        well is what makes the stored call say so: without it the call is filed
        as completed, with no reason, and the owner reading the dashboard cannot
        tell a finished booking from a call the agent dropped.
        """
        session = self._ctx.session
        if session.handoff is None:
            session.handoff = HandoffRequest(reason="agent_cannot_proceed", summary=why)

    def _request_messages(self, now: datetime, nudge: str | None = None) -> list[dict[str, Any]]:
        """History with a freshly-timestamped system message in front.

        The clock lives here rather than inside the stored history so that the
        prompt has exactly one system message and the time is never stale. It is
        rebuilt each request, which costs nothing on Groq -- there is no prompt
        cache to invalidate.
        """
        clock = now.astimezone(self._ctx.session.tz).strftime("%A %d %B %Y, %I:%M %p")
        system = f"{self._system_prompt}\n\n{current_time_note(clock)}"
        messages = [{"role": "system", "content": system}, *self._messages]
        if nudge:
            # For this request only: never stored, so it cannot pile up in history.
            messages.append({"role": "system", "content": nudge})
        return messages

    async def run_turn(
        self,
        user_text: str,
        *,
        now: datetime,
        on_text: Callable[[str], None] | None = None,
        on_tool_call: Callable[[str], None] | None = None,
    ) -> TurnResult:
        """Handle one caller utterance, running tools until the model is done.

        ``on_text`` receives text as it streams. In the terminal that prints it;
        from Phase 2 it feeds the speech engine. ``on_tool_call`` is told the name
        of the first tool the model asks for in the turn, the moment it asks --
        well before the tool has run -- so something can fill the silence.
        """
        anchor = time.perf_counter()
        metrics = TurnMetrics(turn_started_at=now, model=self._settings.llm_model)
        self._ctx.now = now
        self.interruption = None

        # Where this turn's messages start. If the turn dies partway -- a dropped
        # connection, a timeout -- history is rewound to here. Leaving a user
        # turn with no reply behind would make the *next* request malformed,
        # turning one transient failure into a dead conversation.
        history_mark = len(self._messages)
        self._messages.append({"role": "user", "content": user_text})
        self._open_turn = history_mark
        self._ctx.session.note_caller_speech(user_text)

        # Owned here rather than in _drive so the error path can still report
        # which tools ran before the turn fell over.
        collected_text: list[str] = []
        tool_log: list[dict[str, Any]] = []

        try:
            return await self._drive(
                anchor=anchor,
                metrics=metrics,
                now=now,
                collected_text=collected_text,
                tool_log=tool_log,
                on_text=on_text,
                on_tool_call=on_tool_call,
            )
        except asyncio.CancelledError:
            # The caller talked over the agent. Not an error, so nothing here
            # apologises or hands over: what happened is recorded, and the
            # cancellation carries on for whoever cancelled to act on. History is
            # left alone until settle_interruption, which knows what was heard.
            self.interruption = Interruption(
                history_mark=history_mark,
                user_text=user_text,
                text="".join(collected_text).strip(),
                tool_calls=tool_log,
                metrics=_finish(metrics, anchor),
            )
            raise
        except Exception as exc:  # noqa: BLE001 - one bad turn must not end the call
            del self._messages[history_mark:]
            self._open_turn = None
            # The provider's own message is kept, not just the exception class.
            # A bare "NotFoundError" hides the one useful fact -- which model
            # was rejected -- and this lands in call_turns.text for the
            # dashboard, where a class name on its own explains nothing.
            detail = str(exc).strip().replace("\n", " ")[:300]
            self._hand_off(f"The assistant failed mid-turn: {type(exc).__name__}.")
            return TurnResult(
                text="Sorry, I lost that for a moment. Let me put you through to a colleague.",
                metrics=_finish(metrics, anchor),
                tool_calls=tool_log,
                stop_reason=f"error:{type(exc).__name__}: {detail}" if detail
                else f"error:{type(exc).__name__}",
                handoff=True,
            )

    async def _drive(
        self,
        *,
        anchor: float,
        metrics: TurnMetrics,
        now: datetime,
        collected_text: list[str],
        tool_log: list[dict[str, Any]],
        on_text: Callable[[str], None] | None,
        on_tool_call: Callable[[str], None] | None = None,
    ) -> TurnResult:
        """The tool-use loop. Split out so ``run_turn`` owns the history rewind."""
        finish_reason: str | None = None
        empty_retries = 0
        nudge: str | None = None
        tool_announced = False

        for _ in range(MAX_TOOL_ITERATIONS):
            first_token_at: float | None = None
            text_parts: list[str] = []
            # Tool calls arrive split across chunks: the name in one, the
            # arguments a few characters at a time in later ones, keyed by index.
            # They have to be reassembled before anything can be parsed.
            partial_calls: dict[int, dict[str, Any]] = {}

            request: dict[str, Any] = {
                "model": self._settings.llm_model,
                "messages": self._request_messages(now, nudge),
                "tools": self._tools,
                "tool_choice": "auto",
                "max_tokens": self._settings.llm_max_tokens,
                "temperature": self._settings.llm_temperature,
                "stream": True,
                # Usage is omitted from streamed responses unless asked for, and
                # without it the Phase 7 cost-per-call estimate has no inputs.
                "stream_options": {"include_usage": True},
            }
            if self._settings.llm_reasoning_effort:
                # A reasoning model spends hidden tokens before its first word.
                # Low effort is what keeps that under a natural pause.
                request["reasoning_effort"] = self._settings.llm_reasoning_effort
            stream = await self._client.chat.completions.create(**request)

            try:
                async for chunk in stream:
                    if chunk.usage is not None:
                        metrics.input_tokens += chunk.usage.prompt_tokens or 0
                        metrics.output_tokens += chunk.usage.completion_tokens or 0
                    if not chunk.choices:
                        # The usage-only chunk carries no choices.
                        continue

                    choice = chunk.choices[0]
                    if choice.finish_reason:
                        finish_reason = choice.finish_reason

                    delta = choice.delta
                    if delta is None:
                        continue

                    if delta.content:
                        if first_token_at is None:
                            first_token_at = time.perf_counter()
                            # Recorded now, not after the stream: a turn the
                            # caller cuts off never reaches the end of it, and
                            # its first-word latency is still worth logging.
                            if metrics.llm_first_token_ms is None:
                                metrics.llm_first_token_ms = _ms(anchor, first_token_at)
                        text_parts.append(delta.content)
                        collected_text.append(delta.content)
                        if on_text is not None:
                            on_text(delta.content)

                    for fragment in delta.tool_calls or ():
                        call = partial_calls.setdefault(
                            fragment.index, {"id": None, "name": "", "arguments": ""}
                        )
                        if fragment.id:
                            call["id"] = fragment.id
                        if fragment.function is not None:
                            if fragment.function.name:
                                call["name"] = fragment.function.name
                                if on_tool_call is not None and not tool_announced:
                                    tool_announced = True
                                    on_tool_call(fragment.function.name)
                            if fragment.function.arguments:
                                call["arguments"] += fragment.function.arguments
            finally:
                # Closing the response is what stops the provider generating (and
                # counting) tokens nobody will hear, when the caller interrupts.
                # A no-op once the stream has been read to its end.
                close = getattr(stream, "close", None)
                if close is not None:
                    with contextlib.suppress(Exception):
                        await close()

            metrics.llm_done_ms = _ms(anchor, time.perf_counter())

            assistant_text = "".join(text_parts)
            if not partial_calls:
                if assistant_text.strip():
                    self._messages.append({"role": "assistant", "content": assistant_text})
                    break
                if empty_retries < MAX_EMPTY_RETRIES:
                    empty_retries += 1
                    nudge = EMPTY_REPLY_NUDGE
                    continue
                break
            nudge = None

            # The assistant turn that requested the tools has to go into history
            # before the results, or the tool_call_ids reference nothing.
            self._messages.append(
                {
                    "role": "assistant",
                    "content": assistant_text or None,
                    "tool_calls": [
                        {
                            "id": call["id"],
                            "type": "function",
                            "function": {"name": call["name"], "arguments": call["arguments"]},
                        }
                        for call in _ordered(partial_calls)
                    ],
                }
            )

            await self._run_tools(_ordered(partial_calls), tool_log)

            # Text from the next model call continues this reply. Without a
            # separator "Let me check.We have..." runs together, which reads
            # badly and, more to the point, hides the sentence boundary from
            # anything cutting the stream into speech.
            if collected_text and not collected_text[-1].endswith((" ", "\n")):
                collected_text.append(" ")
                if on_text is not None:
                    on_text(" ")
        else:
            self._hand_off("The assistant could not settle on a reply and gave up.")
            return TurnResult(
                text="I am having trouble with that. Let me put you through to a colleague.",
                metrics=_finish(metrics, anchor),
                tool_calls=tool_log,
                stop_reason="max_iterations",
                handoff=True,
            )

        return TurnResult(
            text="".join(collected_text).strip(),
            metrics=_finish(metrics, anchor),
            tool_calls=tool_log,
            stop_reason=finish_reason,
            handoff=self._ctx.session.handoff is not None,
        )

    async def _run_tools(
        self, calls: list[dict[str, Any]], tool_log: list[dict[str, Any]]
    ) -> None:
        """Run the requested tools in order, putting each result into history.

        Each runs as its own task behind a shield. If the turn is cancelled while
        one is running -- the caller interrupted -- the cancellation waits: the
        tool finishes, its result is recorded, the tools that had not started are
        recorded as skipped, and only then does the cancellation carry on. A
        booking is either made and remembered or not made; it is never made and
        forgotten.
        """
        done = 0
        try:
            for call in calls:
                task = asyncio.ensure_future(self._run_one_tool(call, tool_log))
                try:
                    output = await asyncio.shield(task)
                except asyncio.CancelledError:
                    output = await task
                    self._append_tool_result(call, output)
                    done += 1
                    raise
                # One message per tool result, each tied back to its call by
                # tool_call_id. Results must not be merged into one message.
                self._append_tool_result(call, output)
                done += 1
        except asyncio.CancelledError:
            for call in calls[done:]:
                self._append_tool_result(call, SKIPPED_TOOL_RESULT)
            raise

    def _append_tool_result(self, call: dict[str, Any], output: str) -> None:
        self._messages.append({"role": "tool", "tool_call_id": call["id"], "content": output})

    async def _run_one_tool(
        self, call: dict[str, Any], tool_log: list[dict[str, Any]]
    ) -> str:
        started = time.perf_counter()
        raw = call["arguments"] or "{}"
        try:
            arguments = json.loads(raw)
            if not isinstance(arguments, dict):
                raise ValueError("tool arguments must be a JSON object")
        except (json.JSONDecodeError, ValueError) as exc:
            # A truncated or malformed argument blob. Told plainly, the model
            # retries; raised, it would end the call over a recoverable slip.
            output, is_error = (
                f"ERROR: could not read the arguments to {call['name']} ({exc}). "
                "Call it again with valid JSON.",
                True,
            )
        else:
            try:
                output = await execute_tool(call["name"], arguments, self._ctx)
                is_error = output.startswith("ERROR:")
            except Exception as exc:  # noqa: BLE001 - a tool crash must not kill the call
                output = (
                    f"The {call['name']} tool failed: {exc}. Apologise briefly and "
                    "offer to have a colleague call back."
                )
                is_error = True

        tool_log.append(
            {
                "name": call["name"],
                "input": raw,
                "output": output,
                "duration_ms": _ms(started, time.perf_counter()),
                "is_error": is_error,
            }
        )
        return output


def _ordered(partial_calls: dict[int, dict[str, Any]]) -> list[dict[str, Any]]:
    """Tool calls in the order the model emitted them, by streaming index."""
    return [partial_calls[index] for index in sorted(partial_calls)]


def _ms(start: float, end: float) -> int:
    return int((end - start) * 1000)


def _finish(metrics: TurnMetrics, anchor: float) -> TurnMetrics:
    # In Phase 1 the turn ends when the text is complete. From Phase 2, total_ms
    # is measured to first audio out, which is what the caller actually waits for.
    metrics.total_ms = _ms(anchor, time.perf_counter())
    return metrics


async def build_runner(*, client: AsyncOpenAI, settings: Settings, ctx: ToolContext) -> AgentRunner:
    """An agent for this conversation, its prompt built from the call's snapshot.

    The one place that turns a tool context into a running agent, so the
    terminal demo, the smoke script and the voice socket cannot drift apart.
    Loads the snapshot if it is not already there, which is why it is async.
    """
    snapshot = await ctx.get_snapshot()
    return AgentRunner(
        client=client,
        settings=settings,
        ctx=ctx,
        # Sorted by name: the system prompt should be identical run to run, and
        # an unstable service order would make it differ for no reason.
        system_prompt=build_system_prompt(
            snapshot.business, sorted(snapshot.services, key=lambda service: service.name)
        ),
    )
