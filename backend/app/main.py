"""The web application: the voice sockets (browser and Twilio) and the demo page.

Run it through ``python -m backend.app.serve`` rather than ``uvicorn`` directly.
On Windows uvicorn defaults to an event loop psycopg cannot use, and the launcher
is what tells it otherwise.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles

from .api.twilio import router as twilio_router
from .api.voice import router as voice_router
from .db import dispose_engine

STATIC_DIR = Path(__file__).parent / "static"


class RevalidatingStaticFiles(StaticFiles):
    """Static files that the browser must check with the server before reusing.

    With no cache header at all, a browser guesses how long a file stays fresh
    from how old it is, so a script edited a few minutes ago can be served stale
    for minutes more -- the page runs yesterday's code and today's bug report
    makes no sense. ``no-cache`` does not disable caching: the browser still
    keeps the file and asks "has this changed?", and StaticFiles answers 304 from
    the ETag when it has not. Cheap, and never stale.
    """

    def file_response(self, *args, **kwargs):
        response = super().file_response(*args, **kwargs)
        response.headers["Cache-Control"] = "no-cache"
        return response


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    yield
    await dispose_engine()


app = FastAPI(title="AI Receptionist", lifespan=lifespan)
app.include_router(voice_router)
app.include_router(twilio_router)


@app.get("/health")
async def health() -> dict[str, str]:
    """Liveness only: the process is up. Deliberately does not touch the
    database, so a slow query cannot make the platform restart a healthy app."""
    return {"status": "ok"}


@app.get("/", include_in_schema=False)
async def root() -> RedirectResponse:
    return RedirectResponse("/demo/")


# html=True serves index.html for the directory, so /demo/ works as a page.
app.mount("/demo", RevalidatingStaticFiles(directory=STATIC_DIR, html=True), name="demo")
