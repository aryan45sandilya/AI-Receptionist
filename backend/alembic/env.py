"""Alembic environment.

Migrations run synchronously even though the application is async: DDL is a
one-shot administrative task with nothing to overlap, and the sync path keeps
the migration scripts plain. The ``postgresql+psycopg`` dialect serves both, so
there is only ever one DATABASE_URL.
"""

from __future__ import annotations

from logging.config import fileConfig

from alembic import context
from sqlalchemy import create_engine, pool

from backend.app.config import get_settings
from backend.app.models import Base

config = context.config
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata


def _url() -> str:
    return get_settings().database_url


def _connect_args() -> dict[str, object]:
    return get_settings().db_connect_args


def run_migrations_offline() -> None:
    context.configure(
        url=_url(),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    engine = create_engine(
        _url(), poolclass=pool.NullPool, connect_args=_connect_args()
    )
    with engine.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            compare_type=True,
        )
        with context.begin_transaction():
            context.run_migrations()
    engine.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
