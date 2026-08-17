"""Order Fulfillment & Financial Settlement workflow.

Automates multi-step e-commerce order validation, inventory allocation,
promotional discount calculation, tax/shipping computation, and payment gateway settlement.
"""

from __future__ import annotations

import logging
from datetime import timedelta
from typing import Any, Dict, List, Optional
import mistralai.workflows as workflows
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


def _get_tracer():
    from telemetry import get_telemetry_tracer_instance
    return get_telemetry_tracer_instance("order_processing_worker")



# ─── MODELS ───


class OrderItem(BaseModel):
    item_id: str
    sku: str
    quantity: int
    unit_price: float
    is_tax_exempt: bool = False
    category_code: str = "GEN"


class OrderProcessingInput(BaseModel):
    order_id: str = "ORD-99482"
    customer_id: str = "CUST-88301"
    items: List[OrderItem] = Field(default_factory=list)
    promo_code: Optional[str] = "SUMMER_SALE_20"
    shipping_country: str = "US"


class OrderProcessingResult(BaseModel):
    order_id: str
    status: str
    subtotal: float
    discount_total: float
    tax_total: float
    shipping_total: float
    grand_total: float
    settlement_reference: str


# ─── ACTIVITIES ───


@workflows.activity(
    name="validate_order_details",
    retry_policy_max_attempts=1,
    start_to_close_timeout=timedelta(seconds=30),
)
async def validate_order_details(order_id: str, customer_id: str, items: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Activity 1: Validate customer status, inventory reservation, and item line prices."""
    from telemetry import record_span_exception, get_current_execution_id

    tracer = _get_tracer()
    execution_id = get_current_execution_id()

    with tracer.start_as_current_span("validate_order_details_activity_span") as span:
        span.set_attribute("gen_ai.workflow.name", "order-processing-pipeline")
        span.set_attribute("gen_ai.workflow.execution_id", execution_id)
        span.set_attribute("gen_ai.activity.name", "validate_order_details")
        span.set_attribute("gen_ai.agent.name", "order_validation_agent")
        span.set_attribute("order.id", order_id)
        span.set_attribute("customer.id", customer_id)

        try:
            logger.info(f"[{order_id}] Validating order details for customer {customer_id}")
            if not items:
                raise ValueError("Order contains no items")

            validated_items = []
            subtotal = 0.0
            for item in items:
                line_total = item["quantity"] * item["unit_price"]
                subtotal += line_total
                validated_items.append({**item, "line_total": line_total})

            result = {
                "order_id": order_id,
                "customer_id": customer_id,
                "validated_items": validated_items,
                "subtotal": subtotal,
                "inventory_reserved": True,
            }
            span.set_attribute("gen_ai.activity.status", "SUCCESS")
            span.set_attribute("order.subtotal", subtotal)
            return result
        except Exception as exc:
            record_span_exception(span, exc)
            span.set_attribute("gen_ai.activity.status", "FAILED")
            raise exc


@workflows.activity(
    name="apply_promotional_discount",
    retry_policy_max_attempts=1,
    start_to_close_timeout=timedelta(seconds=30),
)
async def apply_promotional_discount(validation_result: Dict[str, Any], promo_code: Optional[str]) -> Dict[str, Any]:
    """Activity 2: Calculate promotional discounts and adjust line item balances."""
    from telemetry import record_span_exception, get_current_execution_id

    tracer = _get_tracer()
    execution_id = get_current_execution_id()

    with tracer.start_as_current_span("apply_promotional_discount_activity_span") as span:
        order_id = validation_result.get("order_id", "UNKNOWN")
        span.set_attribute("gen_ai.workflow.name", "order-processing-pipeline")
        span.set_attribute("gen_ai.workflow.execution_id", execution_id)
        span.set_attribute("gen_ai.activity.name", "apply_promotional_discount")
        span.set_attribute("gen_ai.agent.name", "promotions_engine_agent")
        span.set_attribute("order.id", order_id)
        span.set_attribute("promo_code", promo_code or "NONE")

        try:
            logger.info(f"[{order_id}] Calculating promo discount for code: {promo_code}")
            subtotal = validation_result["subtotal"]
            discount = 0.0

            if promo_code == "SUMMER_SALE_20":
                discount = round(subtotal * 0.20, 2)
            elif promo_code == "WELCOME10":
                discount = round(subtotal * 0.10, 2)

            adjusted_subtotal = max(0.0, subtotal - discount)
            res = {
                **validation_result,
                "discount_total": discount,
                "adjusted_subtotal": adjusted_subtotal,
                "promo_applied": bool(discount > 0),
            }
            span.set_attribute("gen_ai.activity.status", "SUCCESS")
            span.set_attribute("order.discount_total", discount)
            return res
        except Exception as exc:
            record_span_exception(span, exc)
            span.set_attribute("gen_ai.activity.status", "FAILED")
            raise exc


@workflows.activity(
    name="calculate_tax_and_shipping",
    retry_policy_max_attempts=1,
    start_to_close_timeout=timedelta(seconds=30),
)
async def calculate_tax_and_shipping(discount_result: Dict[str, Any], shipping_country: str) -> Dict[str, Any]:
    """Activity 3: Compute regional sales tax rates, currency adjustments, and shipping fees."""
    from telemetry import record_span_exception, get_current_execution_id

    tracer = _get_tracer()
    execution_id = get_current_execution_id()

    with tracer.start_as_current_span("calculate_tax_and_shipping_activity_span") as span:
        order_id = discount_result.get("order_id", "UNKNOWN")
        span.set_attribute("gen_ai.workflow.name", "order-processing-pipeline")
        span.set_attribute("gen_ai.workflow.execution_id", execution_id)
        span.set_attribute("gen_ai.activity.name", "calculate_tax_and_shipping")
        span.set_attribute("gen_ai.agent.name", "tax_shipping_agent")
        span.set_attribute("order.id", order_id)
        span.set_attribute("shipping_country", shipping_country)

        try:
            logger.info(f"[{order_id}] Computing tax and shipping for destination: {shipping_country}")
            items = discount_result.get("validated_items", [])

            # Compute line item regional tax allocation
            tax_total = 0.0
            non_exempt_count = sum(1 for item in items if not item.get("is_tax_exempt", False))

            # Accidental runtime issue: if all items are marked tax-exempt or weighting ratio encounters
            # zero non-exempt count when dividing regional tax surcharge pool, a ZeroDivisionError occurs!
            regional_tax_pool = 15.50
            tax_per_item = regional_tax_pool / non_exempt_count  # <-- Crashes if non_exempt_count == 0

            for item in items:
                if not item.get("is_tax_exempt", False):
                    tax_total += tax_per_item

            shipping_total = 12.00 if shipping_country == "US" else 35.00

            res = {
                **discount_result,
                "tax_total": round(tax_total, 2),
                "shipping_total": shipping_total,
                "tax_calculated": True,
            }
            span.set_attribute("gen_ai.activity.status", "SUCCESS")
            return res
        except Exception as exc:
            record_span_exception(span, exc)
            span.set_attribute("gen_ai.activity.status", "FAILED")
            logger.error(f"[{order_id}] Tax and shipping calculation failed: {exc}")
            raise exc


@workflows.activity(
    name="process_payment_settlement",
    retry_policy_max_attempts=1,
    start_to_close_timeout=timedelta(seconds=30),
)
async def process_payment_settlement(tax_result: Dict[str, Any]) -> Dict[str, Any]:
    """Activity 4: Settle final authorization against payment gateway."""
    from telemetry import record_span_exception, get_current_execution_id

    tracer = _get_tracer()
    execution_id = get_current_execution_id()

    with tracer.start_as_current_span("process_payment_settlement_activity_span") as span:
        order_id = tax_result.get("order_id", "UNKNOWN")
        span.set_attribute("gen_ai.workflow.name", "order-processing-pipeline")
        span.set_attribute("gen_ai.workflow.execution_id", execution_id)
        span.set_attribute("gen_ai.activity.name", "process_payment_settlement")
        span.set_attribute("gen_ai.agent.name", "payment_settlement_agent")

        try:
            grand_total = (
                tax_result["adjusted_subtotal"] + tax_result["tax_total"] + tax_result["shipping_total"]
            )
            res = {
                **tax_result,
                "grand_total": round(grand_total, 2),
                "settlement_reference": f"SETTLE-TXN-{order_id}-OK",
                "payment_cleared": True,
            }
            span.set_attribute("gen_ai.activity.status", "SUCCESS")
            return res
        except Exception as exc:
            record_span_exception(span, exc)
            span.set_attribute("gen_ai.activity.status", "FAILED")
            raise exc


# ─── WORKFLOW DEFINITION ───


@workflows.workflow.define(
    name="order-processing-pipeline",
    workflow_display_name="Order Fulfillment & Financial Settlement",
    workflow_description="Automates order validation, promotional discount calculation, tax/shipping computation, and payment gateway settlement.",
)
class OrderProcessingWorkflow:
    @workflows.workflow.entrypoint
    async def run(self, input: OrderProcessingInput) -> OrderProcessingResult:
        items_dict = [item.model_dump() for item in input.items]

        # Step 1: Validate Order
        val_res = await validate_order_details(
            order_id=input.order_id,
            customer_id=input.customer_id,
            items=items_dict,
        )

        # Step 2: Apply Promo Discount
        disc_res = await apply_promotional_discount(
            validation_result=val_res,
            promo_code=input.promo_code,
        )

        # Step 3: Compute Tax & Shipping (will encounter tax allocation issue if items are tax-exempt)
        tax_res = await calculate_tax_and_shipping(
            discount_result=disc_res,
            shipping_country=input.shipping_country,
        )

        # Step 4: Settle Payment
        pay_res = await process_payment_settlement(tax_result=tax_res)

        return OrderProcessingResult(
            order_id=pay_res["order_id"],
            status="completed",
            subtotal=pay_res["subtotal"],
            discount_total=pay_res["discount_total"],
            tax_total=pay_res["tax_total"],
            shipping_total=pay_res["shipping_total"],
            grand_total=pay_res["grand_total"],
            settlement_reference=pay_res["settlement_reference"],
        )
