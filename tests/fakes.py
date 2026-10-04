"""A scripted stand-in for the streaming chat API.

Replays one exact chunk sequence per request and records what was asked, so the
agent loop -- and anything built on it -- can be tested against every shape of
stream a real model produces, deterministically, without a network or API quota.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace


def text(content: str):
    delta = SimpleNamespace(content=content, tool_calls=None)
    return SimpleNamespace(usage=None, choices=[SimpleNamespace(finish_reason=None, delta=delta)])


def tool(index: int, *, id: str | None = None, name: str | None = None, args: str | None = None):
    fragment = SimpleNamespace(
        index=index, id=id, function=SimpleNamespace(name=name, arguments=args)
    )
    delta = SimpleNamespace(content=None, tool_calls=[fragment])
    return SimpleNamespace(usage=None, choices=[SimpleNamespace(finish_reason=None, delta=delta)])


def finish(reason: str):
    delta = SimpleNamespace(content=None, tool_calls=None)
    return SimpleNamespace(usage=None, choices=[SimpleNamespace(finish_reason=reason, delta=delta)])


def usage(prompt: int, completion: int):
    return SimpleNamespace(
        usage=SimpleNamespace(prompt_tokens=prompt, completion_tokens=completion), choices=[]
    )


class Stream:
    def __init__(self, chunks):
        self._chunks = list(chunks)

    def __aiter__(self):
        return self

    async def __anext__(self):
        if not self._chunks:
            raise StopAsyncIteration
        return self._chunks.pop(0)


class HangingStream(Stream):
    """Yields its chunks and then waits forever: a model in the middle of a sentence.

    For testing what happens when the caller cuts in. ``reached_the_end`` is set
    once every chunk has been read and the stream is waiting; ``closed`` records
    whether the consumer closed the response, which is what stops a real provider
    generating tokens nobody will hear.
    """

    def __init__(self, chunks=()):
        super().__init__(chunks)
        self.closed = False
        self.reached_the_end = asyncio.Event()

    async def __anext__(self):
        if self._chunks:
            return self._chunks.pop(0)
        self.reached_the_end.set()
        await asyncio.Event().wait()

    async def close(self):
        self.closed = True


class FakeClient:
    """Replays one scripted chunk sequence per request, and records the requests.

    A script is a list of chunks, or a ready-made stream (for ``HangingStream``).
    """

    def __init__(self, *scripts, raises: Exception | None = None):
        self.scripts = list(scripts)
        self.requests: list[dict] = []
        self._raises = raises
        self.chat = SimpleNamespace(completions=self)

    async def create(self, **kwargs):
        # Snapshot the messages: the runner mutates the same list afterwards.
        kwargs["messages"] = [dict(m) for m in kwargs["messages"]]
        self.requests.append(kwargs)
        if self._raises is not None:
            raise self._raises
        if not self.scripts:
            raise AssertionError("the loop made more requests than the test scripted")
        script = self.scripts.pop(0)
        return script if isinstance(script, Stream) else Stream(script)
