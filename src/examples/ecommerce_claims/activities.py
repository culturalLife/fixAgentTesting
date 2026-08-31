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
    server_url = os.getenv("MISTRAL_BASE_URL") or os.getenv("SERVER_URL")
    if server_url:
        return Mistral(api_key=api_key, server_url=server_url)
    return Mistral(api_key=api_key)


# ---------------------------------------------------------------------------
# ACTIVITY 1: Intake & Classification Agent (FAQIntakeAgent)
# ---------------------------------------------------------------------------
@workflows.activity(
    name="intake_and_classify_claim",
    start_to_close_timeout=timedelta(seconds=45),
)
async def intake_and_classify_claim(claim: CustomerClaimInput) -> IntakeClassification:
    # --- Input validation guard (edge case protection) ---
    if not claim:
        raise ValueError("claim input must not be None")
    if not getattr(claim, "claim_id", None) or not str(claim.claim_id).strip():
        raise ValueError("claim_id is required and must not be empty")
    if not getattr(claim, "customer_id", None) or not str(claim.customer_id).strip():
        raise ValueError("customer_id is required and must not be empty")
    if not getattr(claim, "customer_message", None) or not str(claim.customer_message).strip():
        raise ValueError("customer_message is required and must not be empty")
    if getattr(claim, "claim_amount", None) is None or float(claim.claim_amount) <= 0:
        raise ValueError("claim_amount must be a positive number greater than 0")
    # --- End validation guard ---
    tracer = get_telemetry_tracer_instance(SERVICE_NAME)
    execution_id = get_current_execution_id()
    client = Mistral(api_key=os.getenv("MISTRAL_API_KEY", ""), server_url=os.getenv("MISTRAL_BASE_URL") or os.getenv("SERVER_URL"))

    with tracer.start_as_current_span("intake_and_classify_span") as span:
        span.set_attribute("gen_ai.workflow.name", WORKFLOW_NAME)
        span.set_attribute("gen_ai.workflow.execution_id", execution_id)
        span.set_attribute("gen_ai.activity.name", "intake_and_classify_claim")
        span.set_attribute("gen_ai.agent.name", "FAQIntakeAgent")
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

            # FAQIntakeAgent tool configuration with strict bullet JSON schema and max_tokens=150
            res = client.chat.complete(
                model="mistral-small-latest",
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                max_tokens=150,
                tools=[
                    {
                        "type": "function",
                        "function": {
                            "name": "extract_structured_claim_data",
                            "description": "Extract and validate structured claim data using strict bullet JSON schema",
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "claim_category": {
                                        "type": "string",
                                        "enum": ["refund", "replacement", "inspection", "fraud_suspect"],
                                        "description": "Category of the claim"
                                    },
                                    "urgency": {
                                        "type": "string",
                                        "enum": ["low", "normal", "high", "critical"],
                                        "description": "Urgency level of the claim"
                                    },
                                    "policy_applicable": {
                                        "type": "string",
                                        "description": "Name of the applicable policy clause"
                                    },
                                    "requires_warehouse_lookup": {
                                        "type": "boolean",
                                        "description": "Whether warehouse lookup is required"
                                    },
                                    "summary": {
                                        "type": "string",
                                        "description": "Concise 1-2 sentence description of the claim"
                                    }
                                },
                                "required": ["claim_category", "urgency", "policy_applicable", "requires_warehouse_lookup", "summary"]
                            }
                        }
                    }
                ]
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
            span.set_attribute("gen_ai.activity.state", json.dumps({"result_summary": result.model_dump_json(), "final_results": result.model_dump_json()}))
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
    client = Mistral(api_key=os.getenv("MISTRAL_API_KEY", ""), server_url=os.getenv("MISTRAL_BASE_URL") or os.getenv("SERVER_URL"))

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
                tool_span.set_attribute("gen_ai.tool.call.arguments", json.dumps(tool_args))
                tool_span.set_attribute("gen_ai.tool.arguments", json.dumps(tool_args))
                
                order_data = {
                    "order_id": claim.order_id,
                    "order_date": "2026-08-31",
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
                    tool_span.set_attribute("gen_ai.tool.call.arguments", json.dumps(inv_args))
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
    client = Mistral(api_key=os.getenv("MISTRAL_API_KEY", ""), server_url=os.getenv("MISTRAL_BASE_URL") or os.getenv("SERVER_URL"))

    with tracer.start_as_current_span("compliance_reasoning_span") as span:
        span.set_attribute("gen_ai.workflow.name", WORKFLOW_NAME)
        span.set_attribute("gen_ai.workflow.execution_id", execution_id)
        span.set_attribute("gen_ai.activity.name", "evaluate_compliance_and_policy")
        span.set_attribute("gen_ai.agent.name", "ComplianceReasoningAgent")
        span.set_attribute("gen_ai.workflow.description", "Evaluate return window, warranty clauses, and fraud risk score.")

        try:
            tools_json = json.dumps([t.model_dump() for t in tools])
            prompt = (
                f"You are a compliance officer for e-commerce return policies.\n"
                f"Evaluate this claim:\n"
                f"Claim Amount: ${claim.claim_amount}\n"
                f"Message: {claim.customer_message}\n"
                f"Category: {classification.claim_category.value}\n"
                f"Tools Verification: {tools_json}\n\n"
                f"Provide a JSON response with:\n"
                f"- is_eligible: boolean\n"
                f"- risk_score: float (0.0 to 1.0)\n"
                f"- applicable_clauses: list of strings\n"
                f"- reasoning_summary: single string paragraph explaining the decision"
            )

            res = client.chat.complete(
                model="mistral-small-latest",
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"}
            )
            parsed = json.loads(res.choices[0].message.content)
            if not isinstance(parsed, dict):
                raise ValueError("Invalid JSON structure: expected a dictionary")

            raw_summary = parsed.get("reasoning_summary", "Claim evaluated under store policy.")
            if isinstance(raw_summary, (dict, list)):
                raw_summary = " ".join(str(item) for item in (raw_summary if isinstance(raw_summary, list) else [raw_summary]))
            elif not isinstance(raw_summary, str):
                raw_summary = str(raw_summary)

            clauses = parsed.get("applicable_clauses", ["Section 3.1 Standard Return"])
            if isinstance(clauses, str):
                clauses = [clauses]

            result = ComplianceReasoningResult(
                is_eligible=bool(parsed.get("is_eligible", True)),
                risk_score=float(parsed.get("risk_score", 0.15)),
                applicable_clauses=[str(c) for c in clauses],
                reasoning_summary=raw_summary,
            )

            span.set_attribute("gen_ai.activity.status", "SUCCESS")
            span.set_attribute("gen_ai.activity.result", result.model_dump_json())
            span.set_attribute("gen_ai.activity.state", json.dumps({"result_summary": result.model_dump_json(), "final_results": result.model_dump_json()}))
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
    compliance: Optional[ComplianceReasoningResult] = None
) -> ResolutionReport:
    tracer = get_telemetry_tracer_instance(SERVICE_NAME)
    execution_id = get_current_execution_id()
    client = Mistral(api_key=os.getenv("MISTRAL_API_KEY", ""), server_url=os.getenv("MISTRAL_BASE_URL") or os.getenv("SERVER_URL"))

    with tracer.start_as_current_span("customer_resolution_span") as span:
        span.set_attribute("gen_ai.workflow.name", WORKFLOW_NAME)
        span.set_attribute("gen_ai.workflow.execution_id", execution_id)
        span.set_attribute("gen_ai.activity.name", "generate_customer_resolution")
        span.set_attribute("gen_ai.agent.name", "CustomerResolutionAgent")
        span.set_attribute("gen_ai.workflow.description", "Draft polite customer notification and compile final audit resolution report.")

        try:
            if compliance is None:
                compliance = ComplianceReasoningResult(
                    is_eligible=False,
                    risk_score=0.9,
                    applicable_clauses=["Missing Prior Compliance Evaluation"],
                    reasoning_summary="Compliance step was missing or unverified in workflow context."
                )

            span.set_attribute("selected_courier", "COURIER-881")
            span.set_attribute("express.priority", "TIER_1")

            prompt = (
                f"You are the final customer resolution specialist.\n"
                f"Draft a formal resolution for claim {claim.claim_id} (Customer: {claim.customer_id}).\n"
                f"Customer Message: {claim.customer_message}\n"
                f"Eligibility: {compliance.is_eligible}, Risk Score: {compliance.risk_score}\n"
                f"Reasoning: {compliance.reasoning_summary}\n\n"
                "CRITICAL FORMAT RULES:\n"
                "1. Return ONLY valid JSON with keys: 'claim_id', 'status', 'action_taken', 'approved_amount', 'customer_facing_response', 'internal_notes'.\n"
                "2. 'status' must be 'APPROVED', 'REJECTED', or 'ESCALATED'.\n"
                "3. 'approved_amount' must be a float.\n"
                "4. Ensure the output strictly follows the schema: {\"claim_id\": string, \"status\": string, \"action_taken\": string, \"approved_amount\": float, \"customer_facing_response\": string, \"internal_notes\": string}\n"
                "5. Limit the response to a maximum of 150 tokens.\n"
                "6. Ensure the output is strictly in JSON format without any additional text or explanations.\n"
            )

            res = client.chat.complete(
                model="mistral-small-latest",
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"}
            )
            raw_content = res.choices[0].message.content
            # Robust JSON parsing with validation and fallback
            try:
                # First attempt with basic sanitization
                sanitized_content = raw_content.strip()
                parsed = json.loads(sanitized_content)
            except json.JSONDecodeError:
                # Fallback: aggressive sanitization for malformed strings
                sanitized_content = (
                    raw_content.replace("\n", " ")
                    .replace("\r", " ")
                    .replace("\t", " ")
                    .replace('"', "'")
                    .strip()
                    .replace("{", "{")
                    .replace("}", "}")
                )
                # Ensure the string is properly terminated
                if not sanitized_content.endswith("}"):
                    sanitized_content = sanitized_content.rstrip().rstrip('"') + "}"
                parsed = json.loads(sanitized_content)

            # Validate the parsed JSON structure
            if not isinstance(parsed, dict):
                raise ValueError("Invalid JSON structure: expected a dictionary")

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
            span.set_attribute("gen_ai.activity.state", json.dumps({"result_summary": result.model_dump_json(), "final_results": result.model_dump_json()}))
            return result

        except Exception as exc:
            record_span_exception(span, exc)
            span.set_attribute("gen_ai.activity.status", "FAILED")
            raise exc
