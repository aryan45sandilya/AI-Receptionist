"""Ring a phone and connect the call to the agent: the real thing, end to end.

Twilio places a call from your Twilio number to the phone you name. When it is
answered, Twilio asks this server what to do -- the same webhook a call to your
number would -- and the agent takes over. It is the only way to test a real phone
call without an Indian number to dial, and you are not charged to receive it.

Before running it:

  1. The server is running:          python -m backend.app.serve
  2. It is reachable from the internet. For development a free tunnel does it:
                                      cloudflared tunnel --url http://localhost:8000
     Put the address it prints in .env as PUBLIC_BASE_URL and restart the server.
  3. .env has TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN and TWILIO_PHONE_NUMBER (the
     Twilio number the call is placed from).
  4. On a trial account the phone you call must be a verified caller ID in the
     Twilio console, and the business's number in the database must be your
     Twilio number:                   python -m scripts.seed_demo --reset --phone <it>

    python -m scripts.twilio_call +91XXXXXXXXXX
"""

from __future__ import annotations

import argparse

from backend.app.config import get_settings
from backend.app.runtime import run_async, use_utf8_console
from backend.app.telephony.twilio import TwilioRest


async def run(to: str) -> None:
    settings = get_settings()
    missing = [
        name for name, value in (
            ("TWILIO_ACCOUNT_SID", settings.twilio_account_sid),
            ("TWILIO_AUTH_TOKEN", settings.twilio_auth_token),
            ("TWILIO_PHONE_NUMBER", settings.twilio_phone_number),
            ("PUBLIC_BASE_URL", settings.public_base_url),
        ) if not value.strip()
    ]
    if missing:
        raise SystemExit("Not set in .env: " + ", ".join(missing))
    if not settings.public_base_url.startswith("https://"):
        raise SystemExit(
            "PUBLIC_BASE_URL must be the https address Twilio can reach "
            f"(got {settings.public_base_url!r}). Start a tunnel, see this file's docstring."
        )

    rest = TwilioRest(settings.twilio_account_sid.strip(), settings.twilio_auth_token.strip())
    try:
        call = await rest.create_call(
            to=to,
            from_=settings.twilio_phone_number.strip(),
            url=f"{settings.public_base_url.rstrip('/')}/twilio/voice",
        )
    finally:
        await rest.close()
    print(f"Twilio is ringing {to} from {settings.twilio_phone_number}. Call {call.get('sid')} ({call.get('status')}).")
    print("Answer it. On a trial account Twilio plays a short notice first (and may ask you to")
    print("press a key); then the agent greets you.")


def main() -> None:
    use_utf8_console()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("to", help="the phone to ring, in E.164 form, e.g. +919876543210")
    args = parser.parse_args()
    run_async(run(args.to))


if __name__ == "__main__":
    main()
