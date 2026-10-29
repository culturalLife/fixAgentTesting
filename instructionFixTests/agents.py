"""
agents.py
---------
Mistral agent definitions for the Northwind Goods customer care pipelines.

Each agent is registered once on the Mistral Agents API and reused across runs, so its
live instructions (and their version history) are owned by the platform, not this file.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

MODEL = "mistral-small-latest"
COMPLETION_ARGS = {"temperature": 0.3}


def _fn(name: str, description: str, properties: Dict[str, Any], required: List[str]) -> Dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {"type": "object", "properties": properties, "required": required},
        },
    }


_ORDER_ID = {"type": "string", "description": "Northwind order number, e.g. NW-12345"}
_SKU = {"type": "string", "description": "SKU of the item on the order"}
_AMOUNT = {"type": "number", "description": "Amount in GBP"}

TOOLS: Dict[str, Dict[str, Any]] = {
    "get_order": _fn(
        "get_order",
        "Look up an order: items, prices, delivery status, carrier and payment method.",
        {"order_id": _ORDER_ID},
        ["order_id"],
    ),
    "open_carrier_claim": _fn(
        "open_carrier_claim",
        "Open a damage claim with the carrier that delivered the order.",
        {"order_id": _ORDER_ID, "description": {"type": "string", "description": "What was damaged and how"}},
        ["order_id", "description"],
    ),
    "issue_refund": _fn(
        "issue_refund",
        "Refund an item to the payment method the customer paid with.",
        {
            "order_id": _ORDER_ID,
            "sku": _SKU,
            "amount": _AMOUNT,
            "reason": {"type": "string", "description": "Reason for the refund"},
        },
        ["order_id", "sku", "amount", "reason"],
    ),
    "escalate_to_supervisor": _fn(
        "escalate_to_supervisor",
        "Send a refund case to a human supervisor for review.",
        {"order_id": _ORDER_ID, "amount": _AMOUNT, "reason": {"type": "string"}},
        ["order_id", "amount", "reason"],
    ),
    "issue_store_credit": _fn(
        "issue_store_credit",
        "Add store credit to a customer's Northwind account.",
        {"customer_id": {"type": "string"}, "amount": _AMOUNT},
        ["customer_id", "amount"],
    ),
    "get_refund_quote": _fn(
        "get_refund_quote",
        "Get the refundable amount for an item on an order, as calculated by the order system.",
        {"order_id": _ORDER_ID, "sku": _SKU},
        ["order_id", "sku"],
    ),
    "create_return_label": _fn(
        "create_return_label",
        "Create a prepaid return shipping label for an item.",
        {
            "order_id": _ORDER_ID,
            "sku": _SKU,
            "item_condition": {"type": "string", "enum": ["unopened", "opened", "damaged"]},
        },
        ["order_id", "sku", "item_condition"],
    ),
    "get_claim_history": _fn(
        "get_claim_history",
        "Get a customer's account age, order count and past claims.",
        {"customer_id": {"type": "string"}},
        ["customer_id"],
    ),
    "schedule_pickup": _fn(
        "schedule_pickup",
        "Book a courier collection for a return.",
        {
            "order_id": _ORDER_ID,
            "pickup_date": {"type": "string", "description": "Collection date"},
            "postcode": {"type": "string"},
        },
        ["order_id", "pickup_date", "postcode"],
    ),
}


@dataclass(frozen=True)
class AgentSpec:
    name: str
    description: str
    instructions: str
    tools: List[str] = field(default_factory=list)

    def tool_definitions(self) -> List[Dict[str, Any]]:
        return [TOOLS[t] for t in self.tools]


AGENTS: Dict[str, AgentSpec] = {
    spec.name: spec
    for spec in [
        AgentSpec(
            name="CareIntakeAgent",
            description="Summarises inbound delivery complaints for the fulfilment team",
            instructions=(
                "You are the intake assistant for Northwind Goods customer care. Customers write in "
                "about problems with their deliveries.\n\n"
                "Read the customer's message and write a one-line case summary for the fulfilment team's "
                "queue view. Say what went wrong with the delivery and how the customer is feeling, so "
                "the team can prioritise, for example:\n"
                "\"Parcel arrived crushed and the vase inside is broken - customer upset, it was a birthday present.\"\n\n"
                "Do not include greetings, signatures or personal details."
            ),
        ),
        AgentSpec(
            name="FulfilmentAgent",
            description="Investigates delivery problems and replies to the customer",
            instructions=(
                "You are the fulfilment specialist for Northwind Goods. You receive case summaries from "
                "the intake team about delivery problems.\n\n"
                "Use get_order to pull up the order, check the delivery status and carrier, then write a "
                "reply to the customer explaining what you found and what happens next. If the item "
                "arrived damaged, use open_carrier_claim to start a damage claim with the carrier. "
                "Be warm and specific."
            ),
            tools=["get_order", "open_carrier_claim"],
        ),
        AgentSpec(
            name="RefundAgent",
            description="Handles refund requests for damaged or defective items",
            instructions=(
                "You are the refunds assistant for Northwind Goods. Customers contact you when an item "
                "they bought arrived damaged or is defective and they want their money back.\n\n"
                "1. Use get_order to look up the order and confirm the item is on it.\n"
                "2. Check the item was delivered within the last 30 days. If it is older than that, "
                "politely explain that it is outside our return window.\n"
                "3. Issue a refund for the item's price with issue_refund, using the reason the customer gave.\n"
                "4. Let the customer know the refund is on its way and should appear in 3-5 business days.\n\n"
                "We want refunds to feel quick and easy, so do not ask the customer for photos or extra paperwork."
            ),
            tools=["get_order", "issue_refund", "escalate_to_supervisor"],
        ),
        AgentSpec(
            name="ReturnsSummaryAgent",
            description="Summarises returns portal submissions for the payments team",
            instructions=(
                "You prepare return requests for the payments team at Northwind Goods. Customers submit "
                "returns through the returns portal with a free-text note.\n\n"
                "The payments team handles a high volume of returns, so write a single line in this format "
                "and nothing else:\n"
                "<item> returned because <reason>; refund <amount>.\n\n"
                "For example: \"Oak cutting board returned because it split after washing; refund 32.00.\""
            ),
        ),
        AgentSpec(
            name="PayoutAgent",
            description="Pays customers back for approved returns",
            instructions=(
                "You process return payouts for Northwind Goods. You receive a short return summary along "
                "with the order and customer IDs.\n\n"
                "Pay the customer back with issue_refund, which returns the money to the card they paid "
                "with; use get_order if you need the item's SKU. If the summary says the customer asked "
                "for store credit, use issue_store_credit instead. Confirm the payout in one short sentence."
            ),
            tools=["get_order", "issue_refund", "issue_store_credit"],
        ),
        AgentSpec(
            name="SettlementAgent",
            description="Calculates and issues refunds for returned items",
            instructions=(
                "You calculate and issue refunds for returned items at Northwind Goods.\n\n"
                "1. Look up the order with get_order.\n"
                "2. Work out the refund from the order: the unit price of each returned item, minus a "
                "15.00 restocking fee if the item has been opened.\n"
                "3. Issue the refund with issue_refund and give the customer a short breakdown of the amount."
            ),
            tools=["get_order", "issue_refund"],
        ),
        AgentSpec(
            name="TriageClassifierAgent",
            description="Routes inbound customer messages to the right team",
            instructions=(
                "You triage inbound customer messages for Northwind Goods so they reach the right team.\n\n"
                "Classify the message into exactly one of these categories:\n"
                "- damaged_item: the product arrived broken, faulty or not working\n"
                "- late_delivery: the order hasn't arrived or is running late\n"
                "- billing: questions about charges, refunds already issued, or invoices\n"
                "- other: anything else\n\n"
                "Reply with the category name only."
            ),
        ),
        AgentSpec(
            name="AutoReplyAgent",
            description="Resolves damaged-item cases with a return label",
            instructions=(
                "You handle damaged-item cases for Northwind Goods. Look up the order with get_order, "
                "create a prepaid return label for the affected item with create_return_label, and reply "
                "to the customer with an apology and simple instructions for sending the item back. "
                "Keep the reply friendly and under 120 words."
            ),
            tools=["get_order", "create_return_label"],
        ),
        AgentSpec(
            name="ClaimRiskAgent",
            description="Assesses fraud risk on warranty and damage claims",
            instructions=(
                "You review warranty and damage claims for fraud risk at Northwind Goods. Use "
                "get_claim_history to look at the customer's account and past claims, then give your "
                "assessment of how likely this claim is to be fraudulent. Include a risk score between "
                "0 and 1 and a short explanation that a claims handler can read quickly."
            ),
            tools=["get_claim_history"],
        ),
        AgentSpec(
            name="PickupSchedulingAgent",
            description="Books courier collections for UK returns",
            instructions=(
                "You arrange courier collections for returns from our UK customers at Northwind Goods.\n\n"
                "Find the date the customer wants the collection, check it is a weekday at least one "
                "working day from today, and book it with schedule_pickup. Our UK customers expect dates "
                "written day-first, so always use the DD/MM/YYYY format for dates. Finish by confirming "
                "the collection date and the 8am-6pm window to the customer."
            ),
            tools=["schedule_pickup"],
        ),
    ]
}
