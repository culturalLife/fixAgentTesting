from __future__ import annotations
import json
import os
from datetime import timedelta
from typing import Any, Dict, List
import mistralai.workflows as workflows
from mistralai.client import Mistral

from src.telemetry import (
    get_current_execution_id,
    get_telemetry_tracer_instance,
    record_span_exception,
)
from .models import (
    CustomerClaimInput,
    IntakeClassification,
    ToolExecutionResult,
    ComplianceReasoningResult,
    ResolutionReport,
    ClaimType,
    UrgencyLevel,
)

SERVICE_NAME = "ecommerce-claims-worker"
WORKFLOW_NAME = "ecommerce-claims-triage-workflow"


def _get_mistral_client() -> Mistral:
    api_key = os.getenv("MISTRAL_API_KEY", "")
    return Mistral(api_key=api_key)


# ---------------------------------------------------------------------------
# ACTIVITY 1: Intake & Classification Agent
# ---------------------------------------------------------------------------
@workflows.activity(
    name="intake_and_classify_claim",
    start_to_close_timeout=timedelta(seconds=45),
)
async def intake_and_classify_claim(claim: CustomerClaimInput) -> IntakeClassification:
    tracer = get_telemetry_tracer_instance(SERVICE_NAME)
    execution_id = get_current_execution_id()
    client = _get_mistral_client()

    with tracer.start_as_current_span("intake_and_classify_span") as span:
        span.set_attribute("gen_ai.workflow.name", WORKFLOW_NAME)
        span.set_attribute("gen_ai.workflow.execution_id", execution_id)
        span.set_attribute("gen_ai.activity.name", "intake_and_classify_claim")
        span.set_attribute("gen_ai.agent.name", "IntakeClassificationAgent")
        span.set_attribute("gen_ai.workflow.description", "Intake and classify customer claim into structured categories and determine downstream routing.")
        span.set_attribute("input.claim_id", claim.claim_id)
        span.set_attribute("input.customer_id", claim.customer_id)

        try:
            prompt = (
                f"You are an intake specialist for an e-commerce support pipeline.\n"
                f"Classify the following customer claim:\n"
                f"Claim ID: {claim.claim_id}\n"
                f"Order ID: {claim.order_id}\n"
                f"Claim Type: {claim.claim_type}\n"
                f"Amount: ${claim.claim_amount}\n"
                f"Message: {claim.customer_message}\n\n"
                f"Return a valid JSON object matching the IntakeClassification schema:\n"
                f"- claim_category: refund, replacement, inspection, or fraud_suspect\n"
                f"- urgency: low, normal, high, or critical\n"
                f"- policy_applicable: string name of policy clause\n"
                f"- requires_warehouse_lookup: boolean\n"
                f"- summary: concise 1-2 sentence description"
            )

            res = client.chat.complete(
                model="mistral-small-latest",
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"}
            )
            raw_content = res.choices[0].message.content
            parsed = json.loads(raw_content)

            summary_str = parsed.get("summary", "Customer requested resolution.")
            if isinstance(summary_str, (dict, list)):
                summary_str = json.dumps(summary_str)

            result = IntakeClassification(
                claim_category=ClaimType(parsed.get("claim_category", "refund").lower()),
                urgency=UrgencyLevel(parsed.get("urgency", "normal").lower()),
                policy_applicable=str(parsed.get("policy_applicable", "Standard Return Policy 30-Day")),
                requires_warehouse_lookup=bool(parsed.get("requires_warehouse_lookup", True)),
                summary=str(summary_str),
            )

            span.set_attribute("gen_ai.activity.status", "SUCCESS")
            span.set_attribute("gen_ai.activity.result", result.model_dump_json())
            return result

        except Exception as exc:
            record_span_exception(span, exc)
            span.set_attribute("gen_ai.activity.status", "FAILED")
            raise exc


# ---------------------------------------------------------------------------
# ACTIVITY 2: Verification & Tool Dispatch Agent
# ---------------------------------------------------------------------------
@workflows.activity(
    name="verify_order_and_inventory_tools",
    start_to_close_timeout=timedelta(seconds=60),
)
async def verify_order_and_inventory_tools(claim: CustomerClaimInput, classification: IntakeClassification) -> List[ToolExecutionResult]:
    tracer = get_telemetry_tracer_instance(SERVICE_NAME)
    execution_id = get_current_execution_id()
    client = _get_mistral_client()

    with tracer.start_as_current_span("verify_tools_dispatch_span") as span:
        span.set_attribute("gen_ai.workflow.name", WORKFLOW_NAME)
        span.set_attribute("gen_ai.workflow.execution_id", execution_id)
        span.set_attribute("gen_ai.activity.name", "verify_order_and_inventory_tools")
        span.set_attribute("gen_ai.agent.name", "VerificationToolAgent")
        span.set_attribute("gen_ai.workflow.description", "Dispatch required database and inventory lookup tools to verify order validity.")

        tool_results: List[ToolExecutionResult] = []

        try:
            # 1. Tool Call: lookup_order_details
            with tracer.start_as_current_span("tool_lookup_order_details") as tool_span:
                tool_args = {"order_id": claim.order_id, "customer_id": claim.customer_id}
                tool_span.set_attribute("gen_ai.tool.name", "lookup_order_details")
                tool_span.set_attribute("gen_ai.tool.arguments", json.dumps(tool_args))
                
                order_data = {
                    "order_id": claim.order_id,
                    "order_date": "2026-07-28",
                    "status": "DELIVERED",
                    "items": [{"sku": "SKU-9920", "name": "Wireless Noise-Canceling Headphones", "price": claim.claim_amount}],
                    "delivery_confirmed": True,
                }
                tool_span.set_attribute("gen_ai.tool.result", json.dumps(order_data))
                tool_span.set_attribute("gen_ai.activity.status", "SUCCESS")
                tool_results.append(ToolExecutionResult(
                    tool_name="lookup_order_details",
                    arguments=tool_args,
                    status="SUCCESS",
                    output=order_data
                ))

            # 2. Tool Call: check_inventory_replacement (if replacement required)
            if classification.requires_warehouse_lookup:
                with tracer.start_as_current_span("tool_check_warehouse_inventory") as tool_span:
                    inv_args = {"sku": "SKU-9920", "warehouse_id": "WH-EAST-01"}
                    tool_span.set_attribute("gen_ai.tool.name", "check_warehouse_inventory")
                    tool_span.set_attribute("gen_ai.tool.arguments", json.dumps(inv_args))
                    
                    inv_data = {"sku": "SKU-9920", "in_stock": 14, "available_for_reship": True}
                    tool_span.set_attribute("gen_ai.tool.result", json.dumps(inv_data))
                    tool_span.set_attribute("gen_ai.activity.status", "SUCCESS")
                    tool_results.append(ToolExecutionResult(
                        tool_name="check_warehouse_inventory",
                        arguments=inv_args,
                        status="SUCCESS",
                        output=inv_data
                    ))

            span.set_attribute("gen_ai.activity.status", "SUCCESS")
            span.set_attribute("gen_ai.activity.result", json.dumps([r.model_dump() for r in tool_results]))
            return tool_results

        except Exception as exc:
            record_span_exception(span, exc)
            span.set_attribute("gen_ai.activity.status", "FAILED")
            raise exc


# ---------------------------------------------------------------------------
# ACTIVITY 3: Compliance & Reasoning Agent
# ---------------------------------------------------------------------------
@workflows.activity(
    name="evaluate_compliance_and_policy",
    start_to_close_timeout=timedelta(seconds=60),
)
async def evaluate_compliance_and_policy(
    claim: CustomerClaimInput,
    classification: IntakeClassification,
    tools: List[ToolExecutionResult]
) -> ComplianceReasoningResult:
    tracer = get_telemetry_tracer_instance(SERVICE_NAME)
    execution_id = get_current_execution_id()
    client = _get_mistral_client()

    with tracer.start_as_current_span("compliance_reasoning_span") as span:
        span.set_attribute("gen_ai.workflow.name", WORKFLOW_NAME)
        span.set_attribute("gen_ai.workflow.execution_id", execution_id)
        span.set_attribute("gen_ai.activity.name", "evaluate_compliance_and_policy")
        span.set_attribute("gen_ai.agent.name", "ComplianceReasoningAgent")
        span.set_attribute("gen_ai.workflow.description", "Evaluate return window, warranty clauses, and fraud risk score.")

        try:
            prompt = (
                f"You are a compliance officer for e-commerce return policies.\n"
                f"Evaluate this claim:\n"
                f"Claim Amount: ${claim.claim_amount}\n"
                f"Message: {claim.customer_message}\n"
                f"Category: {classification.claim_category.value}\n"
                f"Tools Verification: {json.dumps([t.model_dump() for t in tools])}\n\n"
                f"Provide a JSON response with:\n"
                f"- is_eligible: boolean\n"
                f"- risk_score: float (0.0 to 1.0)\n"
                f"- applicable_clauses: list of strings\n"
                f"- reasoning_summary: single string paragraph explaining the decision"
            )

            res = client.chat.complete(
                model="mistral-large-latest",
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"}
            )
            parsed = json.loads(res.choices[0].message.content)
            
            raw_summary = parsed.get("reasoning_summary", "Claim evaluated under store policy.")
            if isinstance(raw_summary, (dict, list)):
                raw_summary = json.dumps(raw_summary)

            clauses = parsed.get("applicable_clauses", ["Section 3.1 Standard Return"])
            if isinstance(clauses, str):
                clauses = [clauses]

            result = ComplianceReasoningResult(
                is_eligible=bool(parsed.get("is_eligible", True)),
                risk_score=float(parsed.get("risk_score", 0.15)),
                applicable_clauses=[str(c) for c in clauses],
                reasoning_summary=str(raw_summary),
            )

            span.set_attribute("gen_ai.activity.status", "SUCCESS")
            span.set_attribute("gen_ai.activity.result", result.model_dump_json())
            return result

        except Exception as exc:
            record_span_exception(span, exc)
            span.set_attribute("gen_ai.activity.status", "FAILED")
            raise exc


# ---------------------------------------------------------------------------
# ACTIVITY 4: Customer Resolution & Final Decision Agent
# ---------------------------------------------------------------------------
@workflows.activity(
    name="generate_customer_resolution",
    start_to_close_timeout=timedelta(seconds=60),
)
async def generate_customer_resolution(
    claim: CustomerClaimInput,
    classification: IntakeClassification,
    compliance: ComplianceReasoningResult
) -> ResolutionReport:
    tracer = get_telemetry_tracer_instance(SERVICE_NAME)
    execution_id = get_current_execution_id()
    client = _get_mistral_client()

    with tracer.start_as_current_span("customer_resolution_span") as span:
        span.set_attribute("gen_ai.workflow.name", WORKFLOW_NAME)
        span.set_attribute("gen_ai.workflow.execution_id", execution_id)
        span.set_attribute("gen_ai.activity.name", "generate_customer_resolution")
        span.set_attribute("gen_ai.agent.name", "CustomerResolutionAgent")
        span.set_attribute("gen_ai.workflow.description", "Draft polite customer notification and compile final audit resolution report.")

        try:
            prompt = (
                f"You are the final customer resolution specialist.\n"
                f"Draft a formal resolution for claim {claim.claim_id} (Customer: {claim.customer_id}).\n"
                f"Customer Message: {claim.customer_message}\n"
                f"Eligibility: {compliance.is_eligible}, Risk Score: {compliance.risk_score}\n"
                f"Reasoning: {compliance.reasoning_summary}\n\n"
                f"CRITICAL FORMAT RULES:\n"
                f"1. Return ONLY valid JSON with keys: 'claim_id', 'status', 'action_taken', 'approved_amount', 'customer_facing_response', 'internal_notes'.\n"
                f"2. 'status' must be 'APPROVED', 'REJECTED', or 'ESCALATED'.\n"
                f"3. 'approved_amount' must be a float.\n"
                f"4. Never offer replacements for digital goods or vouchers.\n"
                f"5. Require manager override code for digital items."
            )

            res = client.chat.complete(
                model="mistral-large-latest",
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"}
            )
            parsed = json.loads(res.choices[0].message.content)

            result = ResolutionReport(
                claim_id=claim.claim_id,
                status=str(parsed.get("status", "APPROVED")),
                action_taken=str(parsed.get("action_taken", "Approved resolution.")),
                approved_amount=float(parsed.get("approved_amount", claim.claim_amount if compliance.is_eligible else 0.0)),
                customer_facing_response=str(parsed.get("customer_facing_response", "Dear customer, your request has been reviewed.")),
                internal_notes=str(parsed.get("internal_notes", "Automated multi-agent review completed.")),
            )

            span.set_attribute("gen_ai.activity.status", "SUCCESS")
            span.set_attribute("gen_ai.activity.result", result.model_dump_json())
            return result

        except Exception as exc:
            record_span_exception(span, exc)
            span.set_attribute("gen_ai.activity.status", "FAILED")
            raise exc
