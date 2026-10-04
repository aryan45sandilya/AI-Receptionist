"""Seed one demo business so the agent has something real to book against.

Idempotent: running it twice leaves one business, not two. Pass --reset to wipe
the demo business and everything cascading from it (bookings included) and start
clean, which is the usual thing to do between eval runs.

    python -m scripts.seed_demo
    python -m scripts.seed_demo --reset
    python -m scripts.seed_demo --reset --phone +15551234567 --handoff-phone +919876543210

``--phone`` is the number callers dial: an inbound phone call finds its business
by it, so for a real call it must be your Twilio number. It defaults to
TWILIO_PHONE_NUMBER from .env when that is set. ``--handoff-phone`` is where the
agent puts a caller through to when they ask for a person.
"""

from __future__ import annotations

import argparse
from datetime import time

from sqlalchemy import delete, or_, select

from backend.app.config import get_settings
from backend.app.db import dispose_engine, get_session_factory
from backend.app.runtime import run_async
from backend.app.models import AvailabilityRule, Business, Service, Staff

DEMO_NAME = "Sunrise Dental"
DEMO_PHONE = "+918000000000"
DEMO_HANDOFF_PHONE = "+918000000001"

# Mon-Sat, morning and evening surgery, closed Sunday (weekday 6).
# Monday=0, matching availability_rules.weekday.
OPENING = [
    (weekday, start, end)
    for weekday in range(0, 6)
    for start, end in [(time(9, 0), time(13, 0)), (time(17, 0), time(20, 0))]
]


async def seed(reset: bool, phone: str, handoff_phone: str) -> None:
    async with get_session_factory()() as session:
        # The demo business, whichever number it was last given.
        existing = (
            await session.execute(
                select(Business).where(
                    or_(
                        Business.phone_e164.in_({DEMO_PHONE, phone}),
                        Business.name == DEMO_NAME,
                    )
                )
            )
        ).scalars().first()

        if existing and not reset:
            print(f"Demo business already present: {existing.name} ({existing.id})")
            print("Pass --reset to wipe it and reseed.")
            return

        if existing:
            # ON DELETE CASCADE takes services, staff, rules, calls and bookings.
            await session.execute(delete(Business).where(Business.id == existing.id))
            await session.flush()
            print("Removed the previous demo business.")

        business = Business(
            name=DEMO_NAME,
            timezone="Asia/Kolkata",
            phone_e164=phone,
            handoff_phone_e164=handoff_phone,
            greeting=f"{DEMO_NAME}, how can I help?",
        )
        session.add(business)
        await session.flush()

        services = [
            Service(business_id=business.id, name="Consultation", duration_min=15,
                    price_paise=30000),
            Service(business_id=business.id, name="Dental cleaning", duration_min=30,
                    price_paise=80000),
            Service(business_id=business.id, name="Root canal", duration_min=60,
                    price_paise=450000),
        ]
        staff = [
            Staff(business_id=business.id, name="Dr Rao"),
            Staff(business_id=business.id, name="Dr Menon"),
        ]
        session.add_all(services + staff)
        await session.flush()

        # Per-staff rules rather than business-wide ones, so the schedule has two
        # independent resources and the double-booking behaviour is observable.
        session.add_all(
            [
                AvailabilityRule(
                    business_id=business.id,
                    staff_id=member.id,
                    weekday=weekday,
                    start_time=start,
                    end_time=end,
                )
                for member in staff
                for weekday, start, end in OPENING
            ]
        )
        await session.commit()

        print(f"Seeded {business.name}  id={business.id}")
        print(f"  timezone   {business.timezone}")
        print(f"  number     {business.phone_e164}   (the number callers dial)")
        print(f"  handoff    {business.handoff_phone_e164}   (where a person is reached)")
        print(f"  services   {', '.join(s.name for s in services)}")
        print(f"  staff      {', '.join(s.name for s in staff)}")
        print("  hours      Mon-Sat 09:00-13:00 and 17:00-20:00, closed Sunday")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reset", action="store_true", help="delete the demo business first"
    )
    parser.add_argument("--phone", help="the number callers dial (default: TWILIO_PHONE_NUMBER, else a placeholder)")
    parser.add_argument("--handoff-phone", default=DEMO_HANDOFF_PHONE, help="where a caller is put through to a person")
    args = parser.parse_args()
    phone = (args.phone or get_settings().twilio_phone_number or DEMO_PHONE).strip()
    run_async(_run(args.reset, phone, args.handoff_phone.strip()))


async def _run(reset: bool, phone: str, handoff_phone: str) -> None:
    try:
        await seed(reset, phone, handoff_phone)
    finally:
        await dispose_engine()


if __name__ == "__main__":
    main()
