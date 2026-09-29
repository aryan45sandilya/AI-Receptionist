"""Start the web server.

    python -m backend.app.serve
    python -m backend.app.serve --port 8080 --reload

A launcher instead of a bare ``uvicorn backend.app.main:app`` because uvicorn on
Windows builds a Proactor event loop, which psycopg's async mode refuses to run
on. Passing a loop factory by import string is the supported way to override it,
and it is the same in every environment, so production behaves like development.
"""

from __future__ import annotations

import argparse

import uvicorn


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--reload", action="store_true", help="restart on code changes")
    args = parser.parse_args()

    uvicorn.run(
        "backend.app.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        loop="backend.app.runtime:selector_loop_factory",
        # A call is one long-lived socket; keep it open through quiet stretches
        # without the server deciding the client has gone.
        ws_ping_interval=20,
        ws_ping_timeout=20,
    )


if __name__ == "__main__":
    main()
