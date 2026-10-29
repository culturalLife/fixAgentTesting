from __future__ import annotations
import json
import os
from datetime import timedelta
from typing import Any, Dict, List, Optional
import mistralai.workflows as workflows
from mistralai.client import Mistral

from src.telemetry import (
    get_current_execution_id,
    get_telemetry_tracer_instance,
    record_span_exception,
)
from .models import (
    HotelCancellationInput,
    CancellationClassification,
    ToolExecutionResult,
    CancellationReport,
    CancellationReason,
)

SERVICE_NAME = "hotel-cancellation-worker"
WORKFLOW_NAME = "hotel-cancellation-workflow"

def _get_mistral_client() -> Mistral:
    api_key = os.getenv("MISTRAL_API_KEY", "")
    server_url = os.getenv("MISTRAL_BASE_URL") or os.getenv("SERVER_URL")
    if server_url:
        return Mistral(api_key=api_key, server_url=server_url)
    return Mistral(api_key=api_key)

@workflows.activity(
    name="classify_cancellation_request",
    start_to_close_timeout=timedelta(seconds=45),
)
async def classify_cancellation_request(request: HotelCancellationInput) -> CancellationClassification:
    tracer = get_telemetry_tracer_instance(SERVICE_NAME)
    execution_id = get_current_execution_id()
    client = _get_mistral_client()

    with tracer.start_as_current_span("classify_cancellation_span") as span:
        span.set_attribute("gen_ai.workflow.name", WORKFLOW_NAME)
        span.set_attribute("gen_ai.workflow.execution_id", execution_id)
        span.set_attribute("gen_ai.activity.name", "classify_cancellation_request")
        span.set_attribute("gen_ai.agent.name", "CancellationClassifierAgent")
        
        try:
            prompt = (
                f"Classify the following hotel cancellation request:\n"
                f"Booking ID: {request.booking_id}\n"
                f"Reason: {request.reason}\n"
                f"Message: {request.customer_message}\n\n"
                f"Return a valid JSON object matching the Classification schema:\n"
                f"- category: personal, weather, flight_delay, or other\n"
                f"- is_eligible_for_refund: boolean\n"
                f"- penalty_applicable: boolean"
            )

            res = client.chat.complete(
                model="mistral-small-latest",
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"}
            )
            parsed = json.loads(res.choices[0].message.content)

            result = CancellationClassification(
                category=CancellationReason(parsed.get("category", "other").lower()),
                is_eligible_for_refund=bool(parsed.get("is_eligible_for_refund", True)),
                penalty_applicable=bool(parsed.get("penalty_applicable", False)),
            )

            span.set_attribute("gen_ai.activity.status", "SUCCESS")
            span.set_attribute("gen_ai.activity.result", result.model_dump_json())
            return result

        except Exception as exc:
            record_span_exception(span, exc)
            span.set_attribute("gen_ai.activity.status", "FAILED")
            raise exc

@workflows.activity(
    name="execute_cancellation_tools",
    start_to_close_timeout=timedelta(seconds=60),
)
async def execute_cancellation_tools(request: HotelCancellationInput, classification: CancellationClassification) -> List[ToolExecutionResult]:
    tracer = get_telemetry_tracer_instance(SERVICE_NAME)
    execution_id = get_current_execution_id()
    
    with tracer.start_as_current_span("execute_cancellation_tools_span") as span:
        span.set_attribute("gen_ai.workflow.name", WORKFLOW_NAME)
        span.set_attribute("gen_ai.workflow.execution_id", execution_id)
        span.set_attribute("gen_ai.activity.name", "execute_cancellation_tools")
        span.set_attribute("gen_ai.agent.name", "BookingToolAgent")
        
        tool_results: List[ToolExecutionResult] = []

        try:
            with tracer.start_as_current_span("tool_lookup_booking") as tool_span:
                tool_args = {"booking_id": request.booking_id}
                tool_span.set_attribute("gen_ai.tool.name", "lookup_booking_db")
                tool_span.set_attribute("gen_ai.tool.call.arguments", json.dumps(tool_args))
                
                try:
                    # Local mock database instead of network call
                    local_db = {
                        "BK-54321": {"status": "CONFIRMED", "amount_paid": 250.00},
                        "BK-10293": {"status": "CONFIRMED", "amount_paid": 500.00}
                    }
                    
                    # Intentional Bug 1: Using strict bracket notation which will raise KeyError if ID is not exactly found.
                    # Also attempting to access a non-existent key 'customer_name' which raises another KeyError.
                    booking_info = local_db.get(request.booking_id, {})  # Fixed: Using .get() to safely access booking info
                    customer_name = booking_info.get("customer_name", "Unknown")  # Fixed: Using .get() to safely access customer name
                    status = booking_info.get("status", "UNKNOWN")  # Fixed: Using .get() to safely access status 
                    
                    tool_results.append(ToolExecutionResult(
                        tool_name="lookup_booking_db",
                        arguments=tool_args,
                        status="SUCCESS",
                        output={"customer": customer_name, "status": booking_info["status"]},
                        error=None
                    ))
                except Exception as e:
                    record_span_exception(tool_span, e)
                    tool_span.set_attribute("gen_ai.activity.status", "FAILED")
                    tool_span.set_attribute("gen_ai.tool.result", str(e))
                    tool_results.append(ToolExecutionResult(
                        tool_name="lookup_booking_db",
                        arguments=tool_args,
                        status="FAILED",
                        output={},
                        error=str(e)
                    ))

            with tracer.start_as_current_span("tool_process_refund") as tool_span:
                tool_args = {"booking_id": request.booking_id, "amount": "FULL_REFUND"}
                tool_span.set_attribute("gen_ai.tool.name", "process_refund_api")
                tool_span.set_attribute("gen_ai.tool.call.arguments", json.dumps(tool_args))
                
                try:
                    # Intentional Bug 2: Calculation bug mixing types (string and float).
                    # The amount parameter is passed as a string "FULL_REFUND" but used in arithmetic.
                    base_amount = 250.00
                    refund_amount = float(tool_args["amount"]) if isinstance(tool_args["amount"], (int, float)) else 0.0  # Fixed: Using isinstance to check if amount is numeric
                    
                    # This will raise a TypeError: unsupported operand type(s) for -: 'float' and 'str'
                    refund_total = base_amount - refund_amount
                    
                    tool_results.append(ToolExecutionResult(
                        tool_name="process_refund_api",
                        arguments=tool_args,
                        status="SUCCESS",
                        output={"refunded": refund_total},
                        error=None
                    ))
                except Exception as e:
                    record_span_exception(tool_span, e)
                    tool_span.set_attribute("gen_ai.activity.status", "FAILED")
                    tool_span.set_attribute("gen_ai.tool.result", str(e))
                    tool_results.append(ToolExecutionResult(
                        tool_name="process_refund_api",
                        arguments=tool_args,
                        status="FAILED",
                        output={},
                        error=str(e)
                    ))
                    
            span.set_attribute("gen_ai.activity.status", "PARTIAL_SUCCESS")
            span.set_attribute("gen_ai.activity.result", json.dumps([r.model_dump() for r in tool_results]))
            return tool_results

        except Exception as exc:
            record_span_exception(span, exc)
            span.set_attribute("gen_ai.activity.status", "FAILED")
            raise exc

@workflows.activity(
    name="generate_final_report",
    start_to_close_timeout=timedelta(seconds=60),
)
async def generate_final_report(
    request: HotelCancellationInput,
    classification: CancellationClassification,
    tool_results: List[ToolExecutionResult]
) -> CancellationReport:
    tracer = get_telemetry_tracer_instance(SERVICE_NAME)
    execution_id = get_current_execution_id()
    client = _get_mistral_client()

    with tracer.start_as_current_span("generate_final_report_span") as span:
        span.set_attribute("gen_ai.workflow.name", WORKFLOW_NAME)
        span.set_attribute("gen_ai.workflow.execution_id", execution_id)
        span.set_attribute("gen_ai.activity.name", "generate_final_report")
        span.set_attribute("gen_ai.agent.name", "ReportGeneratorAgent")
        
        try:
            prompt = (
                f"Draft a resolution for cancellation request {request.request_id}.\n"
                f"Tools encountered errors during execution:\n"
                f"{json.dumps([t.model_dump() for t in tool_results])}\n\n"
                f"Return a JSON object with keys: request_id, status, refund_amount, message.\n"
                f"Status should be ESCALATED due to tool errors."
            )

            res = client.chat.complete(
                model="mistral-large-latest",
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"}
            )
            parsed = json.loads(res.choices[0].message.content)

            raw_refund = parsed.get("refund_amount", 0.0)
            try:
                if isinstance(raw_refund, str):
                    import re
                    nums = re.findall(r"[-+]?\d*\.\d+|\d+", raw_refund)
                    refund_val = float(nums[0]) if nums else 0.0
                else:
                    refund_val = float(raw_refund)
            except Exception:
                refund_val = 0.0

            result = CancellationReport(
                request_id=request.request_id,
                status=str(parsed.get("status", "ESCALATED")),
                refund_amount=refund_val,
                message=str(parsed.get("message", "Escalated for manual review due to system errors.")),
            )

            span.set_attribute("gen_ai.activity.status", "SUCCESS")
            span.set_attribute("gen_ai.activity.result", result.model_dump_json())
            return result

        except Exception as exc:
            record_span_exception(span, exc)
            span.set_attribute("gen_ai.activity.status", "FAILED")
            raise exc
