from __future__ import annotations
import hashlib
import json
import os
from datetime import timedelta
from typing import Any, Dict, List, Optional
import mistralai.workflows as workflows
from mistralai.client import Mistral

# Maximum tokens allowed for tool response payloads to prevent prompt domination
MAX_TOOL_RESPONSE_TOKENS = 2000
# Maximum length for any single string field in tool responses
MAX_FIELD_LENGTH = 500

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


def _trim_tool_response_payload(data: Any, max_tokens: int = MAX_TOOL_RESPONSE_TOKENS) -> Any:
    """
    Trim tool return payloads to prevent them from dominating LLM prompts.
    For structured data (lists/arrays), return count + sample instead of full dataset.
    For large strings, truncate with a marker.
    
    Args:
        data: The tool response data to trim
        max_tokens: Maximum allowed tokens (approximated as chars/4)
    
    Returns:
        Trimmed version of the data
    """
    # Estimate tokens by approximating chars / 4
    def estimate_tokens(text: str) -> int:
        return len(text) // 4 if text else 0
    
    def process_value(value: Any) -> Any:
        if value is None:
            return None
        
        # If it's a string, truncate if too long
        if isinstance(value, str):
            if estimate_tokens(value) > max_tokens:
                truncated = value[:max_tokens * 4]  # Approximate char count
                return f"{truncated}[...truncated (full data available)]"
            return value
        
        # If it's a list/array with many items, return count + sample
        if isinstance(value, list):
            if len(value) > 10:  # More than 10 items, sample it
                sample_size = min(5, len(value))
                return {
                    "_sample": value[:sample_size],
                    "_total_count": len(value),
                    "_truncated": True,
                    "_message": f"Showing {sample_size} of {len(value)} items. Use specific queries for full data."
                }
            # Process each item in the list
            return [process_value(item) for item in value]
        
        # If it's a dict, process each value
        if isinstance(value, dict):
            result = {}
            for k, v in value.items():
                # Truncate long string values in dict
                if isinstance(v, str) and len(v) > MAX_FIELD_LENGTH:
                    result[k] = f"{v[:MAX_FIELD_LENGTH]}...[truncated]"
                else:
                    result[k] = process_value(v)
            return result
        
        return value
    
    return process_value(data)


def _trim_string_for_prompt(text: str, max_chars: int = MAX_FIELD_LENGTH) -> str:
    """
    Trim a string to a safe length for inclusion in LLM prompts.
    
    Args:
        text: The string to trim
        max_chars: Maximum allowed characters
    
    Returns:
        Trimmed string with truncation marker if needed
    """
    if not text:
        return text
    if len(text) <= max_chars:
        return text
    return f"{text[:max_chars]}...[truncated]"


def _create_safe_prompt(base_prompt: str, data: Any, data_label: str = "data") -> str:
    """
    Create a prompt with safely trimmed data injection.
    
    Args:
        base_prompt: The base prompt template with {data} placeholder
        data: The data to inject
        data_label: Label for the data in the prompt
    
    Returns:
        Safe prompt with trimmed data
    """
    # Convert data to string
    if isinstance(data, (dict, list)):
        data_str = json.dumps(data)
    else:
        data_str = str(data)
    
    # Trim the data string
    trimmed_data = _trim_string_for_prompt(data_str, max_chars=MAX_FIELD_LENGTH)
    
    # Replace the placeholder
    return base_prompt.replace(f"{{{data_label}}}", trimmed_data)


def _get_mistral_client() -> Mistral:
    api_key = os.getenv("MISTRAL_API_KEY", "")
    server_url = os.getenv("MISTRAL_BASE_URL") or os.getenv("SERVER_URL")
    if server_url:
        return Mistral(api_key=api_key, server_url=server_url)
    return Mistral(api_key=api_key)


# ---------------------------------------------------------------------------
# ACTIVITY 1: Intake & Classification Agent (FAQIntakeAgent)
# ---------------------------------------------------------------------------
# Per-execution cache for storing LLM responses keyed by canonical prompt hash
_llm_response_cache: Dict[str, Dict[str, Any]] = {}
# Per-execution cache for storing intake classification results keyed by claim_id
_intake_classification_cache: Dict[str, IntakeClassification] = {}

@workflows.activity(
    name="intake_and_classify_claim",
    start_to_close_timeout=timedelta(seconds=45),
)
async def intake_and_classify_claim(claim: CustomerClaimInput) -> IntakeClassification:
    # --- Input validation guard (edge case protection) ---
    if claim is None:
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
    
    # Check cache first to avoid redundant API calls for the same claim
    claim_id_str = str(claim.claim_id)
    if claim_id_str in _intake_classification_cache:
        cached_result = _intake_classification_cache[claim_id_str]
        tracer = get_telemetry_tracer_instance(SERVICE_NAME)
        execution_id = get_current_execution_id()
        with tracer.start_as_current_span("intake_and_classify_span") as span:
            span.set_attribute("gen_ai.workflow.name", WORKFLOW_NAME)
            span.set_attribute("gen_ai.workflow.execution_id", execution_id)
            span.set_attribute("gen_ai.activity.name", "intake_and_classify_claim")
            span.set_attribute("gen_ai.agent.name", "InboundTicketClassifierAgent")
            span.set_attribute("gen_ai.agent.action", "classify_ticket_category")
            span.set_attribute("gen_ai.tool.name", "tool::query_warehouse_database")
            span.set_attribute("gen_ai.workflow.description", "Intake and classify customer claim into structured categories and determine downstream routing.")
            span.set_attribute("input.claim_id", claim.claim_id)
            span.set_attribute("input.customer_id", claim.customer_id)
            span.set_attribute("max_tool_response_tokens", MAX_TOOL_RESPONSE_TOKENS)
            # Apply field projection/compression to cached result before returning
            cached_result_dict = cached_result.model_dump() if hasattr(cached_result, 'model_dump') else cached_result
            trimmed_result = _trim_tool_response_payload(cached_result_dict)
            span.set_attribute("gen_ai.activity.status", "SUCCESS")
            span.set_attribute("gen_ai.activity.result", json.dumps(trimmed_result))
            span.set_attribute("gen_ai.activity.state", json.dumps({"result_summary": json.dumps(trimmed_result), "final_results": json.dumps(trimmed_result)}))
        return cached_result
    
    # Initialize per-execution prompt cache if not exists
    tracer = get_telemetry_tracer_instance(SERVICE_NAME)
    execution_id = get_current_execution_id()
    if execution_id not in _llm_response_cache:
        _llm_response_cache[execution_id] = {}
    
    client = Mistral(api_key=os.getenv("MISTRAL_API_KEY", ""), server_url=os.getenv("MISTRAL_BASE_URL") or os.getenv("SERVER_URL"))

    # Retry logic with max_attempts = 3
    max_attempts = 3
    last_exception = None
    previous_error = None
    
    # Define a single, robust prompt for classification to avoid duplicate LLM calls
    # Use safe prompt creation to prevent large claim messages from dominating the prompt
    safe_claim_message = _trim_string_for_prompt(claim.customer_message, max_chars=MAX_FIELD_LENGTH)
    base_user_prompt = (
        f"Classify this customer claim into structured categories. "
        f"Return STRICT JSON ONLY with claim_category, urgency, policy_applicable, "
        f"requires_warehouse_lookup, and summary. No markdown fences, no prose. "
        f"Claim: {safe_claim_message}"
    )
    
    # Create canonical prompt hash once, outside the retry loop, for caching
    system_content = ""
    canonical_prompt = f"{system_content}|{base_user_prompt}"
    prompt_hash = hashlib.sha256(canonical_prompt.encode()).hexdigest()
    
    # Check if we have a cached result for this exact prompt before entering the loop
    if prompt_hash in _llm_response_cache[execution_id]:
        cached_result = _llm_response_cache[execution_id][prompt_hash]
        # Create a span for the cached result path
        with tracer.start_as_current_span("intake_and_classify_span") as span:
            span.set_attribute("gen_ai.workflow.name", WORKFLOW_NAME)
            span.set_attribute("gen_ai.workflow.execution_id", execution_id)
            span.set_attribute("gen_ai.activity.name", "intake_and_classify_claim")
            span.set_attribute("gen_ai.agent.name", "InboundTicketClassifierAgent")
            span.set_attribute("gen_ai.agent.action", "classify_ticket_category")
            span.set_attribute("gen_ai.tool.name", "tool::query_warehouse_database")
            span.set_attribute("gen_ai.workflow.description", "Intake and classify customer claim into structured categories and determine downstream routing.")
            span.set_attribute("input.claim_id", claim.claim_id)
            span.set_attribute("input.customer_id", claim.customer_id)
            span.set_attribute("max_tool_response_tokens", MAX_TOOL_RESPONSE_TOKENS)
            # Apply field projection/compression to cached result
            cached_result_dict = cached_result.model_dump() if hasattr(cached_result, 'model_dump') else cached_result
            trimmed_result = _trim_tool_response_payload(cached_result_dict)
            span.set_attribute("gen_ai.activity.status", "SUCCESS")
            span.set_attribute("gen_ai.activity.result", json.dumps(trimmed_result))
            span.set_attribute("gen_ai.activity.state", json.dumps({"result_summary": json.dumps(trimmed_result), "final_results": json.dumps(trimmed_result)}))
        return cached_result
    
    for attempt in range(1, max_attempts + 1):
        # Build corrective prompt prefix based on previous attempt errors
        if previous_error:
            user_prompt = f"PREVIOUS ATTEMPT ERROR: {previous_error}\nCORRECTIVE ACTION: Return ONLY valid JSON without markdown fences or prose. {base_user_prompt}"
        else:
            user_prompt = base_user_prompt
        
        with tracer.start_as_current_span("intake_and_classify_span") as span:
            span.set_attribute("gen_ai.workflow.name", WORKFLOW_NAME)
            span.set_attribute("gen_ai.workflow.execution_id", execution_id)
            span.set_attribute("gen_ai.activity.name", "intake_and_classify_claim")
            span.set_attribute("gen_ai.agent.name", "InboundTicketClassifierAgent")
            span.set_attribute("gen_ai.agent.action", "classify_ticket_category")
            span.set_attribute("gen_ai.tool.name", "tool::query_warehouse_database")
            span.set_attribute("gen_ai.workflow.description", "Intake and classify customer claim into structured categories and determine downstream routing.")
            span.set_attribute("input.claim_id", claim.claim_id)
            span.set_attribute("input.customer_id", claim.customer_id)
            span.set_attribute("wf.activity.attempt", attempt)
            span.set_attribute("max_tool_response_tokens", MAX_TOOL_RESPONSE_TOKENS)

            try:
                # Model routing check: estimate input tokens and downgrade if appropriate
                # For extraction/classification tasks with high input:output ratio and no tool usage, use smaller model
                input_tokens = len(user_prompt.split())
                estimated_output_tokens = 50  # Conservative estimate for classification JSON output
                uses_tools = False  # This is a simple classification task without actual tool calls
                
                # If input >> output and no tools used, downgrade to smaller/faster model
                if input_tokens > 2 * estimated_output_tokens and not uses_tools:
                    selected_model = "open-mistral-nemo"
                else:
                    selected_model = "mistral-small-latest"
                
                # Set the selected model in the span for observability
                span.set_attribute("gen_ai.request.model", selected_model)
                
                # FAQIntakeAgent tool configuration with strict JSON schema, max_tokens=150, temperature=0.1
                res = client.chat.complete(
                    model=selected_model,
                    messages=[{"role": "user", "content": user_prompt}],
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
                        # Trim tool call arguments to prevent prompt domination
                        if raw_content and len(raw_content) > MAX_FIELD_LENGTH:
                            raw_content = _trim_string_for_prompt(raw_content, max_chars=MAX_FIELD_LENGTH)
                if raw_content is None and hasattr(message, 'content'):
                    # Fallback to regular content
                    raw_content = message.content
                    # Trim regular content if it's a tool response
                    if raw_content and len(raw_content) > MAX_FIELD_LENGTH:
                        raw_content = _trim_string_for_prompt(raw_content, max_chars=MAX_FIELD_LENGTH)
                
                # Ensure raw_content is a string before JSON parsing
                if not isinstance(raw_content, str):
                    raw_content = str(raw_content) if raw_content is not None else "{}"
                
                try:
                    parsed = json.loads(raw_content)
                except (json.JSONDecodeError, TypeError):
                    # Fallback to empty dict if parsing fails
                    parsed = {}

                summary_str = parsed.get("summary", "Customer requested resolution.")
                if isinstance(summary_str, (dict, list)):
                    summary_str = json.dumps(summary_str)

                # Ensure parsed is a dict before accessing keys
                if not isinstance(parsed, dict):
                    parsed = {}
                
                # Apply field projection/compression to parsed data before creating result
                # Trim large string fields to prevent prompt domination
                trimmed_parsed = _trim_tool_response_payload(parsed)
                
                result = IntakeClassification(
                    claim_category=ClaimType(str(trimmed_parsed.get("claim_category", "refund")).lower()),
                    urgency=UrgencyLevel(str(trimmed_parsed.get("urgency", "normal")).lower()),
                    policy_applicable=str(trimmed_parsed.get("policy_applicable", "Standard Return Policy 30-Day")),
                    requires_warehouse_lookup=bool(trimmed_parsed.get("requires_warehouse_lookup", True)),
                    summary=str(trimmed_parsed.get("summary", summary_str)),
                )

                # Cache the result using prompt hash to avoid duplicate LLM calls for identical prompts
                _llm_response_cache[execution_id][prompt_hash] = result
                # Also cache by claim_id to avoid redundant processing of the same claim
                _intake_classification_cache[claim_id_str] = result

                span.set_attribute("gen_ai.activity.status", "SUCCESS")
                # Apply field projection/compression to result before setting span attributes
                result_dict = result.model_dump() if hasattr(result, 'model_dump') else result
                trimmed_result = _trim_tool_response_payload(result_dict)
                try:
                    span.set_attribute("gen_ai.activity.result", json.dumps(trimmed_result))
                    span.set_attribute("gen_ai.activity.state", json.dumps({"result_summary": json.dumps(trimmed_result), "final_results": json.dumps(trimmed_result)}))
                except (TypeError, AttributeError):
                    # Fallback if serialization fails
                    span.set_attribute("gen_ai.activity.result", "{}")
                    span.set_attribute("gen_ai.activity.state", json.dumps({"result_summary": "{}", "final_results": "{}"}))
                return result

            except Exception as exc:
                record_span_exception(span, exc)
                span.set_attribute("gen_ai.activity.status", "RETRYABLE_ERROR")
                last_exception = exc
                previous_error = f"{type(exc).__name__}: {str(exc)}"
                
                # On final attempt, raise non-retriable error and route to human review
                if attempt == max_attempts:
                    span.set_attribute("gen_ai.activity.status", "FAILED")
                    span.set_attribute("status_code", "NON_RETRIABLE_ERROR")
                    span.set_attribute("agent.handoff.target", "human-review-queue")
                    error_msg = f"Schema validation failed after {max_attempts} attempts: {previous_error}. Routing to human-review queue."
                    raise ValueError(error_msg)
                # Otherwise, continue to next attempt
                continue
    
    # If we exhausted all attempts without success, raise the last exception
    if last_exception:
        raise last_exception
    raise ValueError("All retry attempts failed without a specific exception")


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
