"""Tests for connection-string handling.

Needs no database -- it is string parsing. Worth testing because the failure it
prevents is intermittent and its error message points somewhere else entirely:
psycopg prepares statements by default, a transaction pooler routes the next
statement to a different backend, and you get "prepared statement _pg3_0 already
exists" at random under load.
"""

from __future__ import annotations

import pytest

from backend.app.config import uses_transaction_pooling

# Real shapes from each provider's dashboard, with credentials replaced.
SUPABASE_SESSION = (
    "postgresql+psycopg://postgres.abcdefghijklmnop:pw"
    "@aws-0-ap-south-1.pooler.supabase.com:5432/postgres"
)
SUPABASE_TRANSACTION = (
    "postgresql+psycopg://postgres.abcdefghijklmnop:pw"
    "@aws-0-ap-south-1.pooler.supabase.com:6543/postgres"
)
SUPABASE_DIRECT = (
    "postgresql+psycopg://postgres:pw@db.abcdefghijklmnop.supabase.co:5432/postgres"
)
NEON_DIRECT = (
    "postgresql+psycopg://user:pw@ep-cool-lab-12345678.ap-southeast-1.aws.neon.tech"
    "/neondb?sslmode=require"
)
NEON_POOLED = (
    "postgresql+psycopg://user:pw@ep-cool-lab-12345678-pooler.ap-southeast-1"
    ".aws.neon.tech/neondb?sslmode=require"
)


@pytest.mark.parametrize(
    "url,pooled",
    [
        (SUPABASE_SESSION, False),
        (SUPABASE_TRANSACTION, True),
        (SUPABASE_DIRECT, False),
        (NEON_DIRECT, False),
        (NEON_POOLED, True),
    ],
)
def test_transaction_pooling_detection(url, pooled):
    assert uses_transaction_pooling(url) is pooled


def test_supabase_session_and_transaction_differ_only_by_port():
    """The two Supabase URLs are identical apart from the port, which is exactly
    why the port is what we key on."""
    assert SUPABASE_SESSION.replace(":5432", ":6543") == SUPABASE_TRANSACTION
    assert uses_transaction_pooling(SUPABASE_SESSION) is False
    assert uses_transaction_pooling(SUPABASE_TRANSACTION) is True


def test_a_password_containing_an_at_sign_still_parses():
    """urlsplit splits the userinfo on the last '@', so the host is still found.
    An unencoded '@' is a common mistake and must not be read as a pooled host."""
    url = "postgresql+psycopg://postgres.ref:p@ssw0rd@aws-0-ap-south-1.pooler.supabase.com:6543/postgres"
    assert uses_transaction_pooling(url) is True


def test_a_malformed_port_does_not_raise():
    """A bad URL should fail at connect time with a useful driver error, not here."""
    assert uses_transaction_pooling("postgresql+psycopg://u:p@host:not-a-port/db") is False


@pytest.mark.parametrize("url", ["", "not a url at all", "postgresql+psycopg://"])
def test_degenerate_urls_are_treated_as_unpooled(url):
    assert uses_transaction_pooling(url) is False


def test_connect_args_disable_prepared_statements_only_when_pooled(monkeypatch):
    from backend.app.config import Settings

    pooled = Settings(database_url=SUPABASE_TRANSACTION)
    session_mode = Settings(database_url=SUPABASE_SESSION)

    # None, not 0: psycopg treats 0 as "prepare immediately", which is the
    # opposite of what we want here.
    assert pooled.db_connect_args == {"prepare_threshold": None}
    assert session_mode.db_connect_args == {}
