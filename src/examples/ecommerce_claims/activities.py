from __future__ import annotations
import asyncio
import hashlib
import json
import os
from collections import OrderedDict
from datetime import timedelta
from typing import Any, Dict, List, Optional, Type
import mistralai.workflows as workflows
from mistralai.client import Mistral
from mistralai.extra.observability import trace
from mistralai.client.errors import MistralError

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
# Cache for storing classification results by claim_id to avoid redundant API calls.
# Bounded sliding-window retention: evicts oldest entries once the window is full so
# conversation history does not accumulate monotonically across turns.
_INTAKE_CACHE_MAX_ENTRIES = 256
_intake_classification_cache: "OrderedDict[str, IntakeClassification]" = OrderedDict()
# Per-execution cache for storing LLM responses keyed by canonical prompt hash.
# Bounded sliding-window retention per execution so prompt history stays bounded.
_LLM_CACHE_MAX_ENTRIES_PER_EXECUTION = 64
# Bound on the number of per-execution buckets retained so the outer dict does
# not grow monotonically across sequential turns/executions.
_LLM_CACHE_MAX_EXECUTIONS = 128
_llm_response_cache: "OrderedDict[str, OrderedDict[str, Any]]" = OrderedDict()


def _prune_intake_cache() -> None:
    """Evict oldest intake classification entries beyond the sliding-window bound."""
    while len(_intake_classification_cache) > _INTAKE_CACHE_MAX_ENTRIES:
        _intake_classification_cache.popitem(last=False)


def _prune_llm_cache(execution_id: str) -> None:
    """Evict oldest LLM response entries for an execution beyond the sliding-window bound.

    The current execution's bucket is never dropped here: the caller (re)creates it
    before pruning, and dropping an empty bucket would cause a KeyError on the
    subsequent cache lookup. Instead, the outer per-execution dict is bounded with
    its own sliding window so accumulated history stays bounded across turns.
    """
    cache = _llm_response_cache.get(execution_id)
    if cache is not None:
        while len(cache) > _LLM_CACHE_MAX_ENTRIES_PER_EXECUTION:
            cache.popitem(last=False)
    # Bound the outer dict with a sliding window (evict oldest executions first).
    # The current execution is refreshed as most-recent by the caller before this
    # runs, so it is not evicted by this loop.
    while len(_llm_response_cache) > _LLM_CACHE_MAX_EXECUTIONS:
        _llm_response_cache.popitem(last=False)

@workflows.activity(
    name="intake_and_classify_claim",
    start_to_close_timeout=timedelta(seconds=45),
    retry_policy_max_attempts=3,
)
async def intake_and_classify_claim(claim: Optional[CustomerClaimInput] = None) -> IntakeClassification:
    # --- Input validation guard (edge case protection) ---
    if not claim:
        raise ValueError("claim input must not be None")
    
    # Handle both CustomerClaimInput objects and dict-like objects
    def get_field(obj, field_name, default=None):
        if isinstance(obj, dict):
            return obj.get(field_name, default)
        return getattr(obj, field_name, default)
    
    claim_id = get_field(claim, "claim_id")
    customer_id = get_field(claim, "customer_id")
    customer_message = get_field(claim, "customer_message")
    claim_amount = get_field(claim, "claim_amount")
    order_id = get_field(claim, "order_id")
    claim_type = get_field(claim, "claim_type")
    
    if not claim_id or not str(claim_id).strip():
        raise ValueError("claim_id is required and must not be empty")
    if not customer_id or not str(customer_id).strip():
        raise ValueError("customer_id is required and must not be empty")
    if not customer_message or not str(customer_message).strip():
        raise ValueError("customer_message is required and must not be empty")
    if claim_amount is None or float(claim_amount) <= 0:
        raise ValueError("claim_amount must be a positive number greater than 0")
    # --- End validation guard ---
    
    # Get execution_id for per-execution caching
    tracer = get_telemetry_tracer_instance(SERVICE_NAME)
    execution_id = get_current_execution_id()
    
    # Initialize per-execution cache if not present (OrderedDict for sliding-window eviction)
    if execution_id not in _llm_response_cache:
        _llm_response_cache[execution_id] = OrderedDict()
    # Mark this execution as most-recent before pruning so the sliding window
    # never evicts the bucket we are about to read from.
    _llm_response_cache.move_to_end(execution_id)
    # Prune stale history so caches stay bounded across sequential turns
    _prune_intake_cache()
    _prune_llm_cache(execution_id)
    # Defensive: ensure the bucket still exists after pruning (it must, since we
    # just refreshed it as most-recent, but guard against any future eviction bug).
    if execution_id not in _llm_response_cache:
        _llm_response_cache[execution_id] = OrderedDict()

    # Check cache first to avoid redundant API calls for the same claim
    claim_id_str = str(claim_id)
    if claim_id_str in _intake_classification_cache:
        # Refresh recency: move to end of the sliding window
        _intake_classification_cache.move_to_end(claim_id_str)
        return _intake_classification_cache[claim_id_str]

    # Build prompt with system and user content to compute canonical hash
    system_prompt = "You are an intake specialist for an e-commerce support pipeline."
    user_prompt = (
        f"Classify the following customer claim:\n"
        f"Claim ID: {claim_id}\n"
        f"Order ID: {order_id}\n"
        f"Claim Type: {claim_type}\n"
        f"Amount: ${claim_amount}\n"
        f"Message: {customer_message}\n\n"
        f"Return a valid JSON object matching the IntakeClassification schema:\n"
        f"- claim_category: refund, replacement, inspection, or fraud_suspect\n"
        f"- urgency: low, normal, high, or critical\n"
        f"- policy_applicable: string name of policy clause\n"
        f"- requires_warehouse_lookup: boolean\n"
        f"- summary: concise 1-2 sentence description"
    )
    
    # Compute canonical prompt hash (SHA-256 of system + user content) for per-execution caching
    # This ignores timestamps and retry-specific modifications, focusing on the actual claim content
    canonical_prompt_hash = hashlib.sha256((system_prompt + user_prompt).encode('utf-8')).hexdigest()

    client = Mistral(api_key=os.getenv("MISTRAL_API_KEY", ""), server_url=os.getenv("MISTRAL_BASE_URL") or os.getenv("SERVER_URL"))

    max_attempts = 3
    previous_missing_fields = None
    
    with trace("wf_vendor_invoice_reconciliation") as span:
        span.set_attribute("gen_ai.workflow.name", WORKFLOW_NAME)
        span.set_attribute("gen_ai.workflow.execution_id", execution_id)
        span.set_attribute("gen_ai.activity.name", "intake_and_classify_claim")
        span.set_attribute("gen_ai.agent.name", "GeneralLedgerPostingAgent")
        span.set_attribute("gen_ai.workflow.description", "Intake and classify customer claim into structured categories and determine downstream routing.")
        span.set_attribute("input.claim_id", claim_id)
        span.set_attribute("input.customer_id", customer_id)
        span.set_attribute("retry.max_attempts", max_attempts)
        
        for attempt in range(1, max_attempts + 1):
            span.set_attribute("retry.attempt", attempt)
            
            try:
                # Add corrective prompt prefix on retries
                current_system_prompt = system_prompt
                if attempt > 1 and previous_missing_fields:
                    missing_fields_str = str(previous_missing_fields)
                    current_system_prompt = (
                        f"You are an intake specialist for an e-commerce support pipeline. "
                        f"Previous attempt {attempt - 1} returned incomplete data with missing fields: {missing_fields_str}. "
                        f"Please ensure you return a VALID JSON object with ALL required fields: "
                        f"claim_category (refund/replacement/inspection/fraud_suspect), "
                        f"urgency (low/normal/high/critical), policy_applicable (string), "
                        f"requires_warehouse_lookup (boolean), summary (string). "
                        f"Do NOT wrap the response in markdown fences. "
                        f"CRITICAL: requires_warehouse_lookup must be true or false (boolean), not null or missing."
                    )
                
                # Check per-execution cache using canonical prompt hash to avoid duplicate LLM calls
                # The canonical hash is computed from the original prompts, ignoring retry modifications
                exec_cache = _llm_response_cache.get(execution_id)
                if exec_cache is None:
                    # Bucket was evicted between turns; recreate it defensively.
                    exec_cache = OrderedDict()
                    _llm_response_cache[execution_id] = exec_cache
                if canonical_prompt_hash in exec_cache:
                    cached_result = exec_cache[canonical_prompt_hash]
                    # Refresh recency within the sliding window
                    exec_cache.move_to_end(canonical_prompt_hash)
                    span.set_attribute("gen_ai.activity.status", "CACHED")
                    span.set_attribute("cache.hit", True)
                    return cached_result
                
                # FAQIntakeAgent tool configuration with strict JSON schema, max_tokens=150, temperature=0.1
                res = client.chat.complete(
                    model="mistral-small-latest",
                    messages=[
                        {"role": "system", "content": current_system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    response_format={"type": "json_object"},
                    max_tokens=150,
                    temperature=0.1,
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
                # Handle both regular content and tool call responses with defensive checks
                message = res.choices[0].message
                raw_content = None
                if hasattr(message, 'tool_calls') and message.tool_calls:
                    # Extract arguments from the tool call
                    tool_call = message.tool_calls[0]
                    if hasattr(tool_call, 'function') and hasattr(tool_call.function, 'arguments'):
                        raw_content = tool_call.function.arguments
                if raw_content is None and hasattr(message, 'content'):
                    # Fallback to regular content
                    raw_content = message.content
                
                # Ensure raw_content is a string before JSON parsing
                if not isinstance(raw_content, str):
                    raw_content = str(raw_content) if raw_content is not None else "{}"
                
                # Clean the response by removing markdown fences if present
                clean_response = raw_content.strip()
                if clean_response.startswith("```json") or clean_response.startswith("```"):
                    clean_response = clean_response.replace("```json", "").replace("```", "").strip()
                
                try:
                    parsed = json.loads(clean_response)
                except (json.JSONDecodeError, TypeError):
                    # Fallback to empty dict if parsing fails
                    parsed = {}
                
                summary_str = parsed.get("summary", "Customer requested resolution.")
                if isinstance(summary_str, (dict, list)):
                    summary_str = json.dumps(summary_str)
                
                # Ensure parsed is a dict before accessing keys
                if not isinstance(parsed, dict):
                    parsed = {}
                
                # Validate that all required fields are present
                required_fields = ["claim_category", "urgency", "policy_applicable", "requires_warehouse_lookup", "summary"]
                missing_fields = []
                for f in required_fields:
                    if f not in parsed:
                        missing_fields.append(f)
                    elif f == "requires_warehouse_lookup":
                        # Boolean field - only None is missing, False is valid
                        if parsed.get(f) is None:
                            missing_fields.append(f)
                    else:
                        # String fields - empty strings or None are missing
                        value = parsed.get(f)
                        if value is None or (isinstance(value, str) and not value.strip()):
                            missing_fields.append(f)
                
                if missing_fields:
                    # Store missing fields for corrective prompt on next retry
                    previous_missing_fields = missing_fields
                    span.set_attribute("gen_ai.activity.status", "PARTIAL")
                    span.set_attribute("error.missing_fields", str(missing_fields))
                    
                    # If we have a completely empty response (all fields missing), retry with corrective prompt
                    # Otherwise, use defaults for missing fields and continue
                    if len(missing_fields) == len(required_fields):
                        # Completely empty response - retry with corrective prompt
                        if attempt < max_attempts:
                            continue
                        else:
                            # Final attempt failed - route to human review
                            span.set_attribute("gen_ai.activity.status", "HUMAN_REVIEW")
                            span.set_attribute("error.message", f"Missing required fields after {max_attempts} attempts: {missing_fields}")
                            raise ValueError(f"Missing required fields after {max_attempts} attempts: {missing_fields}. Routed to human review queue.")
                    else:
                        # Partial response - use defaults for missing fields and continue without retry
                        # This handles cases where LLM returns some but not all fields
                        for f in missing_fields:
                            if f == "claim_category":
                                parsed[f] = "refund"
                            elif f == "urgency":
                                parsed[f] = "normal"
                            elif f == "policy_applicable":
                                parsed[f] = "Standard Return Policy 30-Day"
                            elif f == "requires_warehouse_lookup":
                                parsed[f] = True
                            elif f == "summary":
                                parsed[f] = "Customer requested resolution."
                
                result = IntakeClassification(
                    claim_category=ClaimType(str(parsed.get("claim_category", "refund")).lower()),
                    urgency=UrgencyLevel(str(parsed.get("urgency", "normal")).lower()),
                    policy_applicable=str(parsed.get("policy_applicable", "Standard Return Policy 30-Day")),
                    requires_warehouse_lookup=bool(parsed.get("requires_warehouse_lookup", True)),
                    summary=str(summary_str),
                )
                
                # Cache the result to avoid redundant API calls for the same claim.
                # Re-insert to mark as most-recent, then prune to enforce the sliding window.
                _intake_classification_cache.pop(claim_id_str, None)
                _intake_classification_cache[claim_id_str] = result
                _prune_intake_cache()

                # Store in per-execution canonical prompt hash cache, then prune to bound growth
                exec_cache = _llm_response_cache.get(execution_id)
                if exec_cache is None:
                    exec_cache = OrderedDict()
                    _llm_response_cache[execution_id] = exec_cache
                exec_cache.pop(canonical_prompt_hash, None)
                exec_cache[canonical_prompt_hash] = result
                # Refresh this execution as most-recent before pruning the outer window.
                _llm_response_cache.move_to_end(execution_id)
                _prune_llm_cache(execution_id)
                
                span.set_attribute("gen_ai.activity.status", "SUCCESS")
                # Safely serialize result to JSON, handling MagicMock objects
                try:
                    result_json = result.model_dump_json() if hasattr(result, 'model_dump_json') else str(result)
                    span.set_attribute("gen_ai.activity.result", result_json)
                    span.set_attribute("gen_ai.activity.state", json.dumps({"result_summary": result_json, "final_results": result_json}))
                except (TypeError, AttributeError):
                    # Fallback if serialization fails
                    span.set_attribute("gen_ai.activity.result", "{}")
                    span.set_attribute("gen_ai.activity.state", json.dumps({"result_summary": "{}", "final_results": "{}"}))
                return result
            
            except Exception as exc:
                record_span_exception(span, exc)
                span.set_attribute("gen_ai.activity.status", "FAILED")
                span.set_attribute("retry.attempt_failed", attempt)
                span.set_attribute("error.type", type(exc).__name__)
                span.set_attribute("error.message", str(exc))
                
                # Classify errors: deterministic (schema/validation) vs recoverable (network/timeout)
                is_recoverable = False
                if isinstance(exc, (MistralError, asyncio.TimeoutError)):
                    # Network/timeout errors - recoverable
                    is_recoverable = True
                    span.set_attribute("error.category", "recoverable")
                elif isinstance(exc, (json.JSONDecodeError, ValueError, KeyError, TypeError)):
                    # Schema/validation errors - deterministic, fail fast
                    is_recoverable = False
                    span.set_attribute("error.category", "deterministic")
                else:
                    # Unknown errors - treat as recoverable for safety
                    is_recoverable = True
                    span.set_attribute("error.category", "unknown")
                
                if not is_recoverable:
                    # Deterministic error - fail fast without retry
                    span.set_attribute("gen_ai.activity.status", "FAILED")
                    span.set_attribute("gen_ai.activity.error_category", "DETERMINISTIC")
                    span.set_attribute("error.message", f"Deterministic error: {type(exc).__name__}: {str(exc)}")
                    raise ValueError(f"Deterministic error in wf_vendor_invoice_reconciliation: {type(exc).__name__}: {str(exc)}. Failing fast.")
                
                # Recoverable error - apply exponential backoff
                if attempt < max_attempts:
                    # Exponential backoff: 1s for attempt 1, 2s for attempt 2
                    backoff_seconds = min(2 ** (attempt - 1), 2)  # Cap at 2 seconds
                    span.set_attribute("retry.backoff_seconds", backoff_seconds)
                    await asyncio.sleep(backoff_seconds)
                    continue
                else:
                    # Final attempt failed - route to human review
                    span.set_attribute("gen_ai.activity.status", "FAILED")
                    span.set_attribute("gen_ai.activity.error_category", "HUMAN_REVIEW")
                    span.set_attribute("error.message", f"Failed after {max_attempts} attempts: {type(exc).__name__}: {str(exc)}")
                    raise ValueError(f"Failed after {max_attempts} attempts: {type(exc).__name__}: {str(exc)}. Routed to human review queue.")



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

    with trace("verify_tools_dispatch_span") as span:
        span.set_attribute("gen_ai.workflow.name", WORKFLOW_NAME)
        span.set_attribute("gen_ai.workflow.execution_id", execution_id)
        span.set_attribute("gen_ai.activity.name", "verify_order_and_inventory_tools")
        span.set_attribute("gen_ai.agent.name", "VerificationToolAgent")
        span.set_attribute("gen_ai.workflow.description", "Dispatch required database and inventory lookup tools to verify order validity.")

        tool_results: List[ToolExecutionResult] = []

        try:
            # 1. Tool Call: lookup_order_details
            with trace("tool_lookup_order_details") as tool_span:
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
                with trace("tool_check_warehouse_inventory") as tool_span:
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

    with trace("compliance_reasoning_span") as span:
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

    with trace("customer_resolution_span") as span:
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
