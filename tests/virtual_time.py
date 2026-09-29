"""An event loop whose clock is virtual.

``asyncio.sleep(10)`` completes instantly and ``loop.time()`` reports that ten
seconds passed. Tests of anything timing-dependent -- pacing, overlap, latency
measurement -- can then assert exact numbers instead of sleeping for real and
allowing for jitter. A test that has to be tolerant of scheduling noise is a
test that quietly stops checking anything.

Works by replacing the selector's ``select``: the loop calls it with "how long
until the next timer", and instead of waiting that long it advances the virtual
clock by that amount and returns straight away.
"""

from __future__ import annotations

import asyncio
from collections.abc import Coroutine
from typing import Any, TypeVar

T = TypeVar("T")


class VirtualTimeLoop(asyncio.SelectorEventLoop):
    def __init__(self) -> None:
        super().__init__()
        self._virtual_now = 0.0
        real = self._selector
        outer = self

        class _AdvancingSelector:
            def __getattr__(self, name: str) -> Any:
                return getattr(real, name)

            def select(self, timeout: float | None = None):
                if timeout is None:
                    # Nothing is ready and nothing is scheduled. A real loop
                    # would wait for I/O; here there is none, so this is a
                    # deadlock, and failing loudly beats hanging the suite.
                    raise RuntimeError("virtual-time loop is idle: nothing left to run")
                if timeout > 0:
                    outer._virtual_now += timeout
                return real.select(0)

        self._selector = _AdvancingSelector()

    def time(self) -> float:
        return self._virtual_now


def run_virtual(coro: Coroutine[Any, Any, T]) -> T:
    """Run a coroutine to completion on a fresh virtual-time loop."""
    loop = VirtualTimeLoop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()
