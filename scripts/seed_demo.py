"""Seed one demo business so the agent has something real to book against.

Idempotent: running it twice leaves one business, not two. Pass --reset to wipe
the demo business and everything cascading from it (bookings included) and start
clean, which is the usual thing to do between eval runs.

    python -m scripts.seed_demo
    python -m scripts.seed_demo --reset
"""

from __future__ import annotations

import argparse
from datetime import time

from sqlalchemy import delete, select

from backend.app.db import dispose_engine, get_session_factory
from backend.app.runtime import run_async
from backend.app.models import AvailabilityRule, Business, Service, Staff

DEMO_PHONE = "+918000000000"

# Mon-Sat, morning and evening surgery, closed Sunday (weekday 6).
# Monday=0, matching availability_rules.weekday.
OPENING = [
    (weekday, start, end)
    for weekday in range(0, 6)
    for start, end in [(time(9, 0), time(13, 0)), (time(17, 0), time(20, 0))]
]


async def seed(reset: bool) -> None:
    async with get_session_factory()() as session:
        existing = (
            await session.execute(select(Business).where(Business.phone_e164 == DEMO_PHONE))
        ).scalar_one_or_none()

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
            name="Sunrise Dental",
            timezone="Asia/Kolkata",
            phone_e164=DEMO_PHONE,
            handoff_phone_e164="+918000000001",
            greeting="Sunrise Dental, how can I help?",
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
        print(f"  number     {business.phone_e164}")
        print(f"  services   {', '.join(s.name for s in services)}")
        print(f"  staff      {', '.join(s.name for s in staff)}")
        print("  hours      Mon-Sat 09:00-13:00 and 17:00-20:00, closed Sunday")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reset", action="store_true", help="delete the demo business first"
    )
    args = parser.parse_args()
    run_async(_run(args.reset))


async def _run(reset: bool) -> None:
    try:
        await seed(reset)
    finally:
        await dispose_engine()


if __name__ == "__main__":
    main()
