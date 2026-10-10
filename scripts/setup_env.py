"""Interactive helper used by setup.ps1 and setup.sh.

    python -m scripts.setup_env deps     # print the dependencies from pyproject.toml
    python -m scripts.setup_env env      # ask for your API keys and write .env

Secrets are read without echo and are written only to .env, which is git-ignored.
"""

from __future__ import annotations

import getpass
import re
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ENV = ROOT / ".env"
EXAMPLE = ROOT / ".env.example"

# (variable, prompt, required, secret)
PROMPTS = [
    ("DATABASE_URL", "Supabase Session pooler connection string", True, True),
    ("LLM_API_KEY", "Groq API key (console.groq.com/keys)", True, True),
    ("DEEPGRAM_API_KEY", "Deepgram API key (Enter to skip)", False, True),
    ("TWILIO_ACCOUNT_SID", "Twilio Account SID (Enter to skip)", False, False),
    ("TWILIO_AUTH_TOKEN", "Twilio Auth Token (Enter to skip)", False, True),
    ("TWILIO_PHONE_NUMBER", "Twilio phone number, e.g. +15551234567 (Enter to skip)", False, False),
]


def deps() -> None:
    data = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    project = data["project"]
    names = list(project["dependencies"]) + list(project["optional-dependencies"]["dev"])
    print("\n".join(names))


def current(text: str, key: str) -> str:
    match = re.search(rf"^{key}=(.*)$", text, re.MULTILINE)
    return match.group(1).strip() if match else ""


def put(text: str, key: str, value: str) -> str:
    line = f"{key}={value}"
    if re.search(rf"^{key}=.*$", text, re.MULTILINE):
        return re.sub(rf"^{key}=.*$", lambda _: line, text, count=1, flags=re.MULTILINE)
    return text.rstrip("\n") + f"\n{line}\n"


def normalise_database_url(url: str) -> str:
    # Supabase hands out the plain scheme; SQLAlchemy needs the psycopg driver named.
    if url.startswith("postgresql://"):
        return "postgresql+psycopg://" + url[len("postgresql://"):]
    if url.startswith("postgres://"):
        return "postgresql+psycopg://" + url[len("postgres://"):]
    return url


def env() -> None:
    if not ENV.exists():
        ENV.write_text(EXAMPLE.read_text(encoding="utf-8"), encoding="utf-8")
        print("Created .env from .env.example")
    text = ENV.read_text(encoding="utf-8")

    print("\nEnter your own keys. They are saved to .env only. Press Enter to keep an existing value.\n")
    for key, label, required, secret in PROMPTS:
        existing = current(text, key)
        has_value = bool(existing) and "PROJECTREF" not in existing and existing != "gsk_..."
        hint = " [already set]" if has_value else ""
        ask = getpass.getpass if secret else input
        while True:
            value = ask(f"{label}{hint}: ").strip()
            if value or has_value or not required:
                break
            print("  This one is required.")
        if not value:
            continue
        if key == "DATABASE_URL":
            value = normalise_database_url(value)
        text = put(text, key, value)

    ENV.write_text(text, encoding="utf-8")
    print("\n.env written.")


if __name__ == "__main__":
    command = sys.argv[1] if len(sys.argv) > 1 else ""
    if command == "deps":
        deps()
    elif command == "env":
        env()
    else:
        sys.exit(__doc__)
