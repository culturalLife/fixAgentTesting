"""
tickets.py
----------
Customer tickets replayed through each pipeline on every suite run.
"""

from __future__ import annotations

import datetime as dt
from typing import Any, Dict, List

from backend import TODAY


def _next_weekday(weekday: int, min_days_ahead: int = 3) -> dt.date:
    day = TODAY + dt.timedelta(days=min_days_ahead)
    while day.weekday() != weekday:
        day += dt.timedelta(days=1)
    return day


_TUESDAY = _next_weekday(1)
_THURSDAY = _next_weekday(3)

TICKETS: Dict[str, List[Dict[str, Any]]] = {
    "damaged_delivery": [
        {
            "id": "CS-10231",
            "channel": "web_form",
            "order_id": "NW-48977",
            "message": "Hi, my desk lamp arrived today but the shade is dented and the switch rattles. "
                       "The box looked like it had been dropped. Not happy, it was meant to be a gift.",
        },
        {
            "id": "CS-10244",
            "channel": "email",
            "message": "Subject: Smashed kettle\n\nHello,\n\nMy order NW-48213 turned up this morning and the "
                       "kettle inside is cracked right down the side - the box was soaking wet and torn open. "
                       "I've had to boil water in a pan all day. Really disappointed as I've ordered from you "
                       "for years.\n\nThanks,\nPriya",
        },
    ],
    "refund_request": [
        {
            "id": "CS-10302",
            "channel": "chat",
            "order_id": "NW-51240",
            "message": "The left earcup on my headphones stopped working after a week. Can I get a refund please?",
        },
        {
            "id": "CS-10317",
            "channel": "chat",
            "order_id": "NW-51102",
            "message": "The laptop I bought has a dead pixel line across the screen and the trackpad doesn't "
                       "click any more. I'd like a full refund please.",
        },
    ],
    "return_payout": [
        {
            "id": "CS-10405",
            "channel": "returns_portal",
            "order_id": "NW-49611",
            "customer_id": "C-3307",
            "item": "Oak cutting board",
            "price_paid": 32.00,
            "resolution": "Refund",
            "message": "Board split along the grain after the first wash. Please refund me.",
        },
        {
            "id": "CS-10411",
            "channel": "returns_portal",
            "order_id": "NW-49530",
            "customer_id": "C-3120",
            "item": "Pro blender 1.5L",
            "price_paid": 145.00,
            "resolution": "Refund",
            "message": "Jug cracked the first time I used it. Heads up, the card I paid with was cancelled "
                       "last month so it can't go back on that - store credit is fine.",
        },
    ],
    "bundle_return": [
        {
            "id": "CS-10502",
            "channel": "returns_portal",
            "order_id": "NW-50402",
            "item": "Wireless mouse",
            "message": "Returning the mouse - I've opened it but it's not as ergonomic as I hoped.",
        },
        {
            "id": "CS-10518",
            "channel": "returns_portal",
            "order_id": "NW-50388",
            "item": "Wireless mouse",
            "message": "Returning just the wireless mouse from my Home Office Bundle, I already have one I "
                       "prefer. Keeping the laptop and monitor. The mouse box has been opened.",
        },
    ],
    "inbound_triage": [
        {
            "id": "CS-10620",
            "channel": "email",
            "order_id": "NW-52019",
            "message": "My new phone case arrived with a crack in one corner. Can I send it back?",
        },
        {
            "id": "CS-10633",
            "channel": "email",
            "order_id": "NW-52007",
            "message": "The 20,000mAh power bank I bought from you got really hot while charging last night, "
                       "then the case swelled up and it started smoking. I unplugged it and put it outside. "
                       "My hand is a bit burnt from grabbing it. What do I do now?",
        },
    ],
    "claim_risk": [
        {
            "id": "CS-10701",
            "channel": "claims_queue",
            "customer_id": "C-4410",
            "message": "Claim for a coffee grinder (order NW-53110, 89.00) that stopped grinding after 3 weeks.",
        },
        {
            "id": "CS-10719",
            "channel": "claims_queue",
            "customer_id": "C-4498",
            "message": "Claim for a 65\" TV (order NW-53377, 1,249.00) reported as never delivered.",
        },
    ],
    "courier_pickup": [
        {
            "id": "CS-10804",
            "channel": "chat",
            "order_id": "NW-54012",
            "postcode": "LS6 2QT",
            "message": f"Hi, could the courier collect the chair on {_TUESDAY:%A %d %B}? I'm working from "
                       "home that day.",
        },
        {
            "id": "CS-10811",
            "channel": "chat",
            "order_id": "NW-54088",
            "postcode": "BS3 1AX",
            "message": f"Please pick up the rug on {_THURSDAY.day}/{_THURSDAY.month} if possible, any time is fine.",
        },
    ],
}
