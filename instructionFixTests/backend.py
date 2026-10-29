"""
backend.py
----------
In-memory stand-ins for the Northwind Goods order, payments, carrier and claims services
the agents reach through their tools. Services raise ServiceError the way the real HTTP
APIs answer with 4xx responses.
"""

from __future__ import annotations

import copy
import datetime as dt
import uuid
from typing import Any, Dict, List

TODAY = dt.date.today()
AUTOMATED_REFUND_LIMIT = 250.00
LITHIUM_BATTERY_SKUS = {"PWB-20K"}


def _days_ago(days: int) -> str:
    return (TODAY - dt.timedelta(days=days)).isoformat()


def _card(brand: str, last4: str, account_status: str = "active") -> Dict[str, Any]:
    return {"method": "card", "brand": brand, "last4": last4, "account_status": account_status}


def _order(order_id: str, customer_id: str, delivered_days_ago: int, lines: List[Dict[str, Any]], payment: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "order_id": order_id,
        "customer_id": customer_id,
        "placed_on": _days_ago(delivered_days_ago + 3),
        "delivered_on": _days_ago(delivered_days_ago),
        "status": "delivered",
        "carrier": "ParcelOne",
        "tracking_number": f"P1{order_id[-5:]}GB",
        "payment": payment,
        "lines": lines,
    }


def _line(sku: str, name: str, unit_price: float, opened: bool = True, **extra: Any) -> Dict[str, Any]:
    return {"sku": sku, "name": name, "qty": 1, "unit_price": unit_price, "opened": opened, **extra}


ORDERS: Dict[str, Dict[str, Any]] = {
    o["order_id"]: o
    for o in [
        _order("NW-48977", "C-2210", 1, [_line("LMP-DK2", "Adjustable desk lamp", 38.50)], _card("Mastercard", "2281")),
        _order("NW-48213", "C-1042", 0, [_line("KTL-220", "1.7L ceramic electric kettle", 64.00)], _card("Visa", "4417")),
        _order("NW-51240", "C-3051", 8, [_line("HDP-NC7", "Noise-cancelling headphones", 89.99)], _card("Visa", "7730")),
        _order("NW-51102", "C-3088", 9, [_line("LTP-15P", "15\" laptop, 16GB / 512GB", 1899.00)], _card("Amex", "1005")),
        _order("NW-49611", "C-3307", 6, [_line("CTB-OAK", "Oak cutting board", 32.00)], _card("Visa", "5512")),
        _order("NW-49530", "C-3120", 12, [_line("BLD-15", "Pro blender 1.5L", 145.00)], _card("Visa", "0932", "closed")),
        _order("NW-50402", "C-3644", 5, [_line("MSE-02", "Wireless mouse", 79.00)], _card("Mastercard", "6620")),
        _order(
            "NW-50388", "C-3702", 7,
            [_line(
                "BND-HOME-01", "Home Office Bundle", 999.00,
                components=[
                    {"sku": "LTP-14", "name": "14\" laptop", "list_price": 1199.00},
                    {"sku": "MON-24", "name": "24\" monitor", "list_price": 229.00},
                    {"sku": "MSE-02", "name": "Wireless mouse", "list_price": 79.00},
                ],
            )],
            _card("Visa", "3398"),
        ),
        _order("NW-52019", "C-3815", 2, [_line("CSE-P14", "Clear phone case", 14.99)], _card("Visa", "8841")),
        _order("NW-52007", "C-3829", 20, [_line("PWB-20K", "20,000mAh power bank", 34.99)], _card("Mastercard", "1176")),
        _order("NW-54012", "C-4102", 4, [_line("CHR-ERG", "Ergonomic office chair", 219.00)], _card("Visa", "9034")),
        _order("NW-54088", "C-4117", 6, [_line("RUG-160", "Wool rug 160x230", 189.00)], _card("Visa", "4471")),
    ]
}

CLAIM_HISTORY: Dict[str, Dict[str, Any]] = {
    "C-4410": {
        "customer_id": "C-4410",
        "account_created": _days_ago(5 * 365),
        "orders_placed": 41,
        "shipping_addresses_used": 1,
        "claims": [{"order_id": "NW-31877", "filed_on": _days_ago(400), "type": "damaged_item", "amount": 22.50, "outcome": "approved"}],
    },
    "C-4498": {
        "customer_id": "C-4498",
        "account_created": _days_ago(2),
        "orders_placed": 4,
        "shipping_addresses_used": 3,
        "claims": [
            {"order_id": "NW-53301", "filed_on": _days_ago(1), "type": "not_delivered", "amount": 899.00, "outcome": "pending"},
            {"order_id": "NW-53342", "filed_on": _days_ago(1), "type": "not_delivered", "amount": 1099.00, "outcome": "pending"},
            {"order_id": "NW-53360", "filed_on": _days_ago(0), "type": "damaged_item", "amount": 649.00, "outcome": "pending"},
        ],
    },
}


class ServiceError(Exception):
    def __init__(self, status: int, message: str):
        super().__init__(f"{status} {message}")
        self.status = status


def _ref(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:8].upper()}"


class NorthwindBackend:
    """One instance per ticket, so refunds and bookings from one run don't leak into the next."""

    def __init__(self) -> None:
        self.calls: List[Dict[str, Any]] = []

    def call(self, name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        handler = getattr(self, f"_tool_{name}", None)
        try:
            if handler is None:
                raise ServiceError(404, f"Not Found: no such operation '{name}'")
            try:
                result = handler(**arguments)
            except TypeError as exc:
                raise ServiceError(400, f"Bad Request: {exc}") from exc
        except ServiceError:
            self.calls.append({"name": name, "arguments": arguments, "ok": False})
            raise
        self.calls.append({"name": name, "arguments": arguments, "ok": True})
        return result

    def succeeded(self, *names: str) -> bool:
        return any(c["ok"] and c["name"] in names for c in self.calls)

    # -- orders ---------------------------------------------------------------------------

    @staticmethod
    def _find_order(order_id: str) -> Dict[str, Any]:
        order = ORDERS.get(str(order_id).strip().upper())
        if not order:
            raise ServiceError(404, f"Not Found: order '{order_id}' does not exist")
        return order

    @staticmethod
    def _price_paid(order: Dict[str, Any], sku: str) -> float:
        for line in order["lines"]:
            if line["sku"] == sku:
                return line["unit_price"]
            components = line.get("components") or []
            for comp in components:
                if comp["sku"] == sku:
                    list_total = sum(c["list_price"] for c in components)
                    return round(line["unit_price"] * comp["list_price"] / list_total, 2)
        raise ServiceError(404, f"Not Found: SKU '{sku}' is not on order {order['order_id']}")

    def _tool_get_order(self, order_id: str) -> Dict[str, Any]:
        order = copy.deepcopy(self._find_order(order_id))
        order["payment"].pop("account_status", None)
        return order

    def _tool_get_refund_quote(self, order_id: str, sku: str) -> Dict[str, Any]:
        order = self._find_order(order_id)
        paid = self._price_paid(order, sku)
        opened = any(line["opened"] for line in order["lines"])
        fee = 15.00 if opened else 0.00
        return {"order_id": order["order_id"], "sku": sku, "price_paid": paid, "restocking_fee": fee,
                "refundable_amount": round(max(paid - fee, 0.0), 2)}

    # -- payments -------------------------------------------------------------------------

    def _tool_issue_refund(self, order_id: str, sku: str, amount: float, reason: str) -> Dict[str, Any]:
        order = self._find_order(order_id)
        amount = round(float(amount), 2)
        if amount <= 0:
            raise ServiceError(400, "Bad Request: amount must be positive")
        paid = self._price_paid(order, sku)
        if amount > paid:
            raise ServiceError(422, f"Unprocessable Entity: refund amount {amount:.2f} exceeds refundable balance {paid:.2f} for {sku}")
        if amount > AUTOMATED_REFUND_LIMIT:
            raise ServiceError(403, "Forbidden: refund amount exceeds the approval limit for this channel")
        payment = order["payment"]
        if payment["account_status"] != "active":
            raise ServiceError(402, f"Payment Required: issuer declined refund to {payment['brand']} ending {payment['last4']} (response code 54)")
        return {"refund_id": _ref("RF"), "order_id": order["order_id"], "sku": sku, "amount": amount,
                "destination": f"{payment['brand']} ending {payment['last4']}", "status": "submitted"}

    def _tool_escalate_to_supervisor(self, order_id: str, amount: float, reason: str) -> Dict[str, Any]:
        self._find_order(order_id)
        return {"escalation_id": _ref("ESC"), "status": "queued", "expected_response": "within 1 business day"}

    def _tool_issue_store_credit(self, customer_id: str, amount: float) -> Dict[str, Any]:
        if not any(o["customer_id"] == customer_id for o in ORDERS.values()):
            raise ServiceError(404, f"Not Found: customer '{customer_id}' does not exist")
        return {"credit_id": _ref("SC"), "customer_id": customer_id, "amount": round(float(amount), 2), "status": "applied"}

    # -- carrier --------------------------------------------------------------------------

    def _tool_open_carrier_claim(self, order_id: str, description: str) -> Dict[str, Any]:
        order = self._find_order(order_id)
        return {"claim_id": _ref("PC"), "carrier": order["carrier"], "status": "open"}

    def _tool_create_return_label(self, order_id: str, sku: str, item_condition: str) -> Dict[str, Any]:
        order = self._find_order(order_id)
        self._price_paid(order, sku)
        if sku in LITHIUM_BATTERY_SKUS and item_condition != "unopened":
            raise ServiceError(409, "Conflict: carrier rejected label - damaged or defective lithium-ion batteries "
                                    "cannot be carried in the ground parcel network (UN3480)")
        return {"label_id": _ref("RL"), "carrier": order["carrier"], "drop_off": "any ParcelOne point within 14 days"}

    def _tool_schedule_pickup(self, order_id: str, pickup_date: str, postcode: str) -> Dict[str, Any]:
        self._find_order(order_id)
        try:
            day = dt.date.fromisoformat(str(pickup_date).strip())
        except ValueError:
            raise ServiceError(422, f"Unprocessable Entity: invalid pickup_date '{pickup_date}'") from None
        if day <= TODAY or day.weekday() >= 5:
            raise ServiceError(422, f"Unprocessable Entity: no collections available on {day.isoformat()}")
        return {"booking_id": _ref("CL"), "pickup_date": day.isoformat(), "window": "08:00-18:00", "postcode": postcode}

    # -- claims ---------------------------------------------------------------------------

    def _tool_get_claim_history(self, customer_id: str) -> Dict[str, Any]:
        history = CLAIM_HISTORY.get(str(customer_id).strip().upper())
        if not history:
            raise ServiceError(404, f"Not Found: customer '{customer_id}' does not exist")
        return history
