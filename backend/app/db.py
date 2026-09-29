"""Async engine and session factory.

Async from the start, deliberately. From Phase 2 on, a request holds a WebSocket
open while audio streams both ways; a blocking database call inside that handler
would stall the caller's audio. Retrofitting async later would touch every
module in ``domain/``, so it is cheaper to pay the small ceremony cost now.

The ``postgresql+psycopg`` dialect covers both sync and async, so Alembic and the
application read the same DATABASE_URL with no rewriting.

Both factories are lazy. Building the engine at import time would make this
module unimportable without DATABASE_URL set, which turns a missing environment
variable into a confusing ImportError in unrelated places.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from functools import lru_cache

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from .config import get_settings


@lru_cache
def get_engine() -> AsyncEngine:
    settings = get_settings()
    return create_async_engine(
        settings.database_url,
        # Free tiers on both Neon and Supabase suspend an idle database, so a
        # pooled connection can be dead by the time we reuse it. pool_pre_ping
        # costs one cheap round trip and turns a confusing mid-call crash into a
        # silent reconnect.
        pool_pre_ping=True,
        pool_size=5,
        max_overflow=5,
        # Disables prepared statements when the URL is a transaction pooler.
        connect_args=settings.db_connect_args,
        echo=False,
    )


@lru_cache
def get_session_factory() -> async_sessionmaker[AsyncSession]:
    # expire_on_commit=False: after committing a booking we still want to read
    # its reference and slot time back to the caller without a second SELECT.
    return async_sessionmaker(get_engine(), class_=AsyncSession, expire_on_commit=False)


async def get_session() -> AsyncIterator[AsyncSession]:
    """FastAPI dependency (Phase 2+), and a convenient context manager."""
    async with get_session_factory()() as session:
        yield session


async def dispose_engine() -> None:
    """Close pooled connections. Call once on shutdown."""
    if get_engine.cache_info().currsize:
        await get_engine().dispose()
