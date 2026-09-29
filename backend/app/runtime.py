"""Event-loop setup, which on Windows is not optional.

psycopg's async mode cannot run on Windows' default ``ProactorEventLoop`` and
raises ``InterfaceError`` on the first connection attempt. It needs a
``SelectorEventLoop``.

The obvious fix -- ``asyncio.set_event_loop_policy(WindowsSelectorEventLoopPolicy())``
-- is deprecated in Python 3.14 and removed in 3.16, so this passes a
``loop_factory`` to ``asyncio.run`` instead. That is the supported route and it
sets nothing global, so it cannot surprise anything else in the process.

Every asyncio entry point in the project goes through ``run_async``. The test
suite does the equivalent by overriding pytest-asyncio's ``_asyncio_loop_factory``
fixture in ``tests/conftest.py``.

Worth knowing: a SelectorEventLoop on Windows is limited to 512 sockets and
cannot spawn subprocesses. Neither matters here, and this is a development
platform detail -- deployment targets Linux, where the default loop is already
compatible and ``loop_factory`` stays None.
"""

from __future__ import annotations

import asyncio
import sys
from collections.abc import Callable, Coroutine
from typing import Any, TypeVar

T = TypeVar("T")

# None on Linux and macOS: the default loop already works with psycopg, and
# asyncio.run(loop_factory=None) is just asyncio.run.
LOOP_FACTORY: Callable[[], asyncio.AbstractEventLoop] | None = (
    asyncio.SelectorEventLoop if sys.platform == "win32" else None
)


def use_utf8_console() -> None:
    """Make stdout UTF-8, whatever the console's codepage is.

    Language models write real typographic punctuation -- curly apostrophes, en
    dashes -- and a Windows console defaulting to cp1252 renders those as
    mojibake ("I'm" becomes "I?m"). Harmless for speech, but the terminal demo
    is the first thing anyone sees, and a transcript full of replacement
    characters reads like a bug.
    """
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            # errors="replace" so an odd byte never crashes a live call.
            reconfigure(encoding="utf-8", errors="replace")


def selector_loop_factory() -> asyncio.AbstractEventLoop:
    """A zero-argument loop factory, in the shape uvicorn's ``loop`` option takes.

    uvicorn's own default on Windows is the Proactor loop, which psycopg cannot
    use -- so the server has to be told, explicitly, to build a selector loop.
    A custom import string is used as the factory as-is, which is why this takes
    no arguments and always returns a selector loop, on every platform.
    """
    return asyncio.SelectorEventLoop()


def run_async(coro: Coroutine[Any, Any, T]) -> T:
    """``asyncio.run``, on an event loop psycopg can actually use."""
    use_utf8_console()
    return asyncio.run(coro, loop_factory=LOOP_FACTORY)
