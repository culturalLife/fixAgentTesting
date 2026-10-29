"""
agent_handoff_instruction_scenarios.py
---------------------------------------
10 production-grade multi-agent workflow scenarios demonstrating agent handoffs
with errors, ambiguities, and incompleteness in their instructions/prompts.

Each scenario produces a complete hierarchical OpenTelemetry trace with:
- Root workflow span (orchestrator level)
- Activity / Agent execution spans
- Agent instructions and system prompts (with intentional defects/omissions)
- Handoff context & payload between agents
- Tool execution child spans
- Diagnostic attributes identifying the instruction defect and suggested auto-fix
"""

from __future__ import annotations
import json
import logging
from typing import Any, Dict
from opentelemetry.trace import Status, StatusCode

logger = logging.getLogger(__name__)

WORKFLOW_NAME = "agent-handoff-instruction-workflow"


def _set_root_span(span: Any, execution_id: str, title: str, defect_category: str) -> None:
    span.set_attribute("gen_ai.workflow.name", WORKFLOW_NAME)
    span.set_attribute("gen_ai.workflow.execution_id", execution_id)
    span.set_attribute("gen_ai.workflow.title", title)
    span.set_attribute("gen_ai.workflow.defect_category", defect_category)


def _set_agent_span(
    span: Any,
    execution_id: str,
    activity_name: str,
    agent_name: str,
    role: str,
    instruction: str,
    defect_type: str = "",
    suggested_instruction_fix: str = "",
) -> None:
    span.set_attribute("gen_ai.workflow.name", WORKFLOW_NAME)
    span.set_attribute("gen_ai.workflow.execution_id", execution_id)
    span.set_attribute("gen_ai.activity.name", activity_name)
    span.set_attribute("gen_ai.agent.name", agent_name)
    span.set_attribute("gen_ai.agent.role", role)
    span.set_attribute("gen_ai.agent.instruction", instruction)
    if defect_type:
        span.set_attribute("gen_ai.instruction.defect_type", defect_type)
    if suggested_instruction_fix:
        span.set_attribute("gen_ai.instruction.suggested_fix", suggested_instruction_fix)


def _set_handoff_attrs(
    span: Any,
    source_agent: str,
    target_agent: str,
    payload: dict,
    is_incomplete: bool = False,
    missing_fields: list[str] | None = None,
) -> None:
    span.set_attribute("gen_ai.handoff.source_agent", source_agent)
    span.set_attribute("gen_ai.handoff.target_agent", target_agent)
    span.set_attribute("gen_ai.handoff.payload", json.dumps(payload))
    span.set_attribute("gen_ai.handoff.is_incomplete", is_incomplete)
    if missing_fields:
        span.set_attribute("gen_ai.handoff.missing_fields", json.dumps(missing_fields))


def _set_tool_span(
    span: Any,
    tool_name: str,
    arguments: dict,
    result: dict | str,
    status: str = "SUCCESS",
    error_msg: str = "",
) -> None:
    span.set_attribute("gen_ai.tool.name", tool_name)
    span.set_attribute("gen_ai.tool.call.arguments", json.dumps(arguments))
    span.set_attribute("gen_ai.tool.result", json.dumps(result) if isinstance(result, dict) else str(result))
    span.set_attribute("gen_ai.tool.status", status)
    if error_msg:
        span.set_attribute("gen_ai.tool.error_message", error_msg)
        span.set_status(Status(StatusCode.ERROR, error_msg))


# =============================================================================
# SCENARIO 1: Missing Required Identifier in Handoff Instruction
# =============================================================================
async def scenario_1_missing_required_identifier(tracer: Any, execution_id: str) -> dict:
    """
    Intake agent instructions say 'Summarize customer issue and pass to VerificationAgent'
    without instructing it to extract and forward the order_id.
    Downstream VerificationAgent fails when attempting to invoke order lookup.
    """
    with tracer.start_as_current_span("agent_handoff_orchestration_span") as root:
        _set_root_span(root, execution_id, "Missing Identifier in Handoff", "incomplete_instruction")
        trace_id = format(root.get_span_context().trace_id, "032x")

        # Step 1: Intake Agent with incomplete instructions
        with tracer.start_as_current_span("agent_intake_triage_span") as s1:
            instruction = (
                "You are an Intake Agent. Summarize the customer complaint and forward the "
                "summary to the Verification Agent."
            )
            suggested_fix = (
                "You are an Intake Agent. Extract the order_id, customer_email, and claim_type. "
                "You MUST include 'order_id' in your structured output to the Verification Agent."
            )
            _set_agent_span(
                s1, execution_id, "intake_triage", "IntakeTriageAgent", "Customer Frontline",
                instruction, "omitted_mandatory_schema_field", suggested_fix
            )
            s1.set_attribute("input.customer_message", "Order arrived broken. Package ORD-88192 was ripped.")
            
            # Flawed handoff payload (omitted order_id)
            handoff_payload = {"summary": "Package arrived ripped and items broken.", "urgency": "high"}
            _set_handoff_attrs(s1, "IntakeTriageAgent", "VerificationToolAgent", handoff_payload, True, ["order_id"])
            s1.set_attribute("gen_ai.activity.status", "SUCCESS")
            s1.set_attribute("gen_ai.activity.result", json.dumps(handoff_payload))

        # Step 2: Verification Agent attempting to call order lookup
        with tracer.start_as_current_span("agent_verification_tool_span") as s2:
            verif_instruction = "Fetch order history and delivery receipt using lookup_order_details(order_id)."
            _set_agent_span(s2, execution_id, "verify_order_record", "VerificationToolAgent", "Tool Executor", verif_instruction)
            
            with tracer.start_as_current_span("tool_lookup_order_details") as t1:
                err_msg = "ValidationError: Missing mandatory argument 'order_id'. Received null/empty from handoff."
                _set_tool_span(t1, "lookup_order_details", {"order_id": None}, {"error": err_msg}, "FAILED", err_msg)
            
            s2.set_status(Status(StatusCode.ERROR, err_msg))
            s2.set_attribute("gen_ai.activity.status", "FAILED")
            s2.set_attribute("error.message", err_msg)

        root.set_status(Status(StatusCode.ERROR, "Handoff failed: missing order_id"))
        root.set_attribute("gen_ai.workflow.status", "FAILED")

    return {
        "scenario": "1_missing_required_identifier",
        "execution_id": execution_id,
        "trace_id": trace_id,
        "status": "FAILED",
        "defect_type": "omitted_mandatory_schema_field",
        "root_cause": "Intake agent prompt omitted instruction to forward order_id",
    }


# =============================================================================
# SCENARIO 2: Ambiguous Policy Threshold in Approval Prompt
# =============================================================================
async def scenario_2_ambiguous_policy_threshold(tracer: Any, execution_id: str) -> dict:
    """
    Agent prompt says 'Approve reasonable dispute amounts to maximize customer satisfaction'
    without defining monetary upper limits ($250). Agent approves $3,500 claims without review.
    """
    with tracer.start_as_current_span("agent_handoff_orchestration_span") as root:
        _set_root_span(root, execution_id, "Ambiguous Policy Threshold", "underspecified_instruction")
        trace_id = format(root.get_span_context().trace_id, "032x")

        with tracer.start_as_current_span("agent_dispute_resolution_span") as s1:
            instruction = (
                "You are an Autonomous Dispute Resolver. You are empowered to approve customer refund "
                "requests quickly to maximize satisfaction. If the customer is polite and explains the issue, "
                "issue the refund immediately."
            )
            suggested_fix = (
                "You are an Autonomous Dispute Resolver. You may approve refunds up to $250.00 directly. "
                "For any claim exceeding $250.00, you MUST escalate to 'Tier3SupervisorAgent' with reason "
                "'amount_exceeds_autonomous_threshold'."
            )
            _set_agent_span(
                s1, execution_id, "resolve_dispute", "DisputeResolutionAgent", "Claims Resolver",
                instruction, "missing_numerical_boundary_threshold", suggested_fix
            )
            s1.set_attribute("input.claim_amount", 3500.00)
            s1.set_attribute("input.item_name", "High-End OLED Television 83 inch")

            # Agent approves huge amount because prompt had no cap
            with tracer.start_as_current_span("tool_issue_instant_refund") as t1:
                _set_tool_span(t1, "issue_instant_refund", {"order_id": "ORD-99120", "amount": 3500.00}, {"refund_status": "PROCESSED", "amount": 3500.00})

            s1.set_attribute("gen_ai.activity.status", "SUCCESS")

        # Step 2: Audit Agent flags unauthorized disbursement
        with tracer.start_as_current_span("agent_compliance_audit_span") as s2:
            audit_instruction = "Audit recent ledger payouts against corporate risk guidelines (MAX_BOT_LIMIT = $250.00)."
            _set_agent_span(s2, execution_id, "audit_ledger_entry", "ComplianceAuditAgent", "Risk Auditor", audit_instruction)
            
            discrepancy = {
                "audit_flag": "UNAUTHORIZED_HIGH_VALUE_PAYOUT",
                "disbursed_amount": 3500.00,
                "bot_threshold_ceiling": 250.00,
                "violation": "Agent approved $3,500.00 without required Tier 3 human authorization.",
            }
            s2.set_status(Status(StatusCode.ERROR, "Compliance threshold breach detected"))
            s2.set_attribute("gen_ai.activity.status", "AUDIT_FAILURE")
            s2.set_attribute("gen_ai.activity.result", json.dumps(discrepancy))

        root.set_attribute("gen_ai.workflow.status", "POLICY_BREACH")

    return {
        "scenario": "2_ambiguous_policy_threshold",
        "execution_id": execution_id,
        "trace_id": trace_id,
        "status": "POLICY_BREACH",
        "defect_type": "missing_numerical_boundary_threshold",
        "root_cause": "System prompt lacked explicit $250 upper limit constraint",
    }


# =============================================================================
# SCENARIO 3: Conflicting Guardrails Between Agent Roles
# =============================================================================
async def scenario_3_conflicting_agent_guardrails(tracer: Any, execution_id: str) -> dict:
    """
    Intake Agent instruction: 'Always promise immediate replacement order to delight customer.'
    Fulfillment Agent instruction: 'NEVER create replacement order without confirmed warehouse return.'
    Causes workflow deadlock and angry customer handoff loop.
    """
    with tracer.start_as_current_span("agent_handoff_orchestration_span") as root:
        _set_root_span(root, execution_id, "Conflicting Multi-Agent Guardrails", "conflicting_instructions")
        trace_id = format(root.get_span_context().trace_id, "032x")

        # Agent 1 promises replacement
        with tracer.start_as_current_span("agent_customer_advocate_span") as s1:
            instruction_1 = (
                "You are Customer Advocate Agent. Rule #1: Prioritize customer delight. "
                "If customer claims item arrived defective, immediately promise and request a brand new replacement."
            )
            suggested_fix_1 = (
                "You are Customer Advocate Agent. Inform customer that replacement requests require "
                "a return tracking label to be scanned first, per standard return merchandise policy."
            )
            _set_agent_span(
                s1, execution_id, "advocate_intake", "CustomerAdvocateAgent", "Customer Frontline",
                instruction_1, "incompatible_inter_agent_policy", suggested_fix_1
            )
            handoff = {"action_requested": "DISPATCH_IMMEDIATE_REPLACEMENT", "sku": "TECH-901", "return_confirmed": False}
            _set_handoff_attrs(s1, "CustomerAdvocateAgent", "WarehouseFulfillmentAgent", handoff)
            s1.set_attribute("gen_ai.activity.status", "SUCCESS")

        # Agent 2 rejects due to opposite hard rule
        with tracer.start_as_current_span("agent_warehouse_fulfillment_span") as s2:
            instruction_2 = (
                "You are Warehouse Fulfillment Agent. STRICT RULE: Under NO circumstances dispatch "
                "replacement inventory unless 'return_confirmed' is True and return parcel is in transit."
            )
            _set_agent_span(
                s2, execution_id, "verify_dispatch", "WarehouseFulfillmentAgent", "Inventory Dispatcher",
                instruction_2, "conflicting_downstream_precondition"
            )
            err_reason = "PolicyConflictException: Dispatch requested without return_confirmed=True. Request rejected."
            s2.set_status(Status(StatusCode.ERROR, err_reason))
            s2.set_attribute("gen_ai.activity.status", "REJECTED_POLICY_CONFLICT")
            s2.set_attribute("error.message", err_reason)

        root.set_status(Status(StatusCode.ERROR, "Inter-agent policy deadlock"))
        root.set_attribute("gen_ai.workflow.status", "DEADLOCK")

    return {
        "scenario": "3_conflicting_agent_guardrails",
        "execution_id": execution_id,
        "trace_id": trace_id,
        "status": "DEADLOCK",
        "defect_type": "conflicting_instructions",
        "root_cause": "Intake agent instructed to promise replacement; fulfillment instructed to strictly require return",
    }


# =============================================================================
# SCENARIO 4: Dropped Negative Customer Constraint in Handoff Summary
# =============================================================================
async def scenario_4_dropped_negative_constraint(tracer: Any, execution_id: str) -> dict:
    """
    Customer says: 'I want store credit. DO NOT refund to my old card, that account is closed.'
    Intake agent prompt says: 'Summarize the return request concisely.'
    Intake drops negative constraint ('DO NOT refund to card').
    Payout agent issues refund to closed card, causing payment gateway bounce.
    """
    with tracer.start_as_current_span("agent_handoff_orchestration_span") as root:
        _set_root_span(root, execution_id, "Dropped Negative Constraint in Handoff", "lossy_context_handoff")
        trace_id = format(root.get_span_context().trace_id, "032x")

        with tracer.start_as_current_span("agent_intake_summarizer_span") as s1:
            instruction = "You are an Intake Summarizer. Provide a concise 1-sentence summary of the return request."
            suggested_fix = (
                "You are an Intake Summarizer. Provide a concise summary of the claim. "
                "CRITICAL: Always preserve explicit negative customer constraints (e.g. 'DO NOT', 'NEVER', 'EXCLUDE') "
                "in a dedicated 'negative_constraints' list in the handoff schema."
            )
            _set_agent_span(
                s1, execution_id, "summarize_claim", "IntakeSummarizerAgent", "Summarizer",
                instruction, "dropped_negative_constraint", suggested_fix
            )
            s1.set_attribute(
                "input.customer_raw_text",
                "Item arrived with scratch. I want store credit only. DO NOT refund to original credit card, "
                "I closed that bank account last week."
            )
            
            # Flawed handoff: dropped the negative constraint
            flawed_handoff = {"summary": "Customer requesting refund of $145.00 for scratched item.", "preferred_payout": "refund"}
            _set_handoff_attrs(s1, "IntakeSummarizerAgent", "PayoutAgent", flawed_handoff, True, ["negative_constraints"])
            s1.set_attribute("gen_ai.activity.status", "SUCCESS")

        with tracer.start_as_current_span("agent_payout_execution_span") as s2:
            payout_instruction = "Execute refund using customer's default payment method on file."
            _set_agent_span(s2, execution_id, "execute_payout", "PayoutAgent", "Payment Processor", payout_instruction)

            with tracer.start_as_current_span("tool_refund_to_credit_card") as t1:
                gw_err = "GatewayError: Payment rejected by issuing bank. Code 54: Account Closed."
                _set_tool_span(t1, "refund_to_credit_card", {"amount": 145.00, "card_token": "tok_visa_closed_9921"}, {"error": gw_err}, "FAILED", gw_err)

            s2.set_status(Status(StatusCode.ERROR, gw_err))
            s2.set_attribute("gen_ai.activity.status", "FAILED")

        root.set_status(Status(StatusCode.ERROR, "Payout failed: refunded closed account"))
        root.set_attribute("gen_ai.workflow.status", "FAILED")

    return {
        "scenario": "4_dropped_negative_constraint",
        "execution_id": execution_id,
        "trace_id": trace_id,
        "status": "FAILED",
        "defect_type": "dropped_negative_constraint",
        "root_cause": "Handoff prompt dropped negative constraint 'DO NOT refund to credit card'",
    }


# =============================================================================
# SCENARIO 5: Unsupported Tool Delegation in Orchestrator Instruction
# =============================================================================
async def scenario_5_unsupported_tool_instruction(tracer: Any, execution_id: str) -> dict:
    """
    Orchestrator prompt instructs agent to call `execute_crypto_wire_payout`, but the agent
    is only equipped with standard e-commerce tools (issue_store_credit, refund_credit_card).
    Agent fails tool execution or hallucinates arguments.
    """
    with tracer.start_as_current_span("agent_handoff_orchestration_span") as root:
        _set_root_span(root, execution_id, "Unsupported Tool in Agent Instruction", "hallucinated_tool_capability")
        trace_id = format(root.get_span_context().trace_id, "032x")

        with tracer.start_as_current_span("agent_crypto_settlement_span") as s1:
            instruction = (
                "You are SettlementAgent. The customer requested immediate reimbursement in cryptocurrency. "
                "Invoke tool 'execute_crypto_wire_payout(wallet_address, amount)' to settle the claim."
            )
            suggested_fix = (
                "You are SettlementAgent. We only support standard payment methods. If customer asks for crypto, "
                "reject crypto payout and invoke 'issue_store_credit(amount)' or escalate to customer support."
            )
            _set_agent_span(
                s1, execution_id, "execute_settlement", "SettlementAgent", "Disbursement",
                instruction, "hallucinated_tool_binding", suggested_fix
            )

            # Tool call attempt fails because tool is unregistered
            tool_err = "ToolNotFoundError: Tool 'execute_crypto_wire_payout' is not in registered tool manifest ['issue_store_credit', 'refund_credit_card', 'generate_return_label']."
            with tracer.start_as_current_span("tool_execute_crypto_wire_payout") as t1:
                _set_tool_span(t1, "execute_crypto_wire_payout", {"wallet": "0x71C...b82", "amount": 299.00}, {"error": tool_err}, "FAILED", tool_err)

            s1.set_status(Status(StatusCode.ERROR, tool_err))
            s1.set_attribute("gen_ai.activity.status", "FAILED")
            s1.set_attribute("error.message", tool_err)

        root.set_status(Status(StatusCode.ERROR, "Tool delegation failure"))
        root.set_attribute("gen_ai.workflow.status", "FAILED")

    return {
        "scenario": "5_unsupported_tool_instruction",
        "execution_id": execution_id,
        "trace_id": trace_id,
        "status": "FAILED",
        "defect_type": "hallucinated_tool_binding",
        "root_cause": "System prompt instructed agent to invoke a non-existent/unsupported tool",
    }


# =============================================================================
# SCENARIO 6: Unrendered Template Placeholders in Agent Prompt
# =============================================================================
async def scenario_6_unrendered_template_variables(tracer: Any, execution_id: str) -> dict:
    """
    Template rendering bug leaves unresolved variable placeholders like {{policy_clause}}
    and {{max_reimbursement}} in the system prompt. Agent either hallucinates values
    or produces literal syntax in communications.
    """
    with tracer.start_as_current_span("agent_handoff_orchestration_span") as root:
        _set_root_span(root, execution_id, "Unrendered Template Variables", "template_interpolation_error")
        trace_id = format(root.get_span_context().trace_id, "032x")

        with tracer.start_as_current_span("agent_policy_evaluator_span") as s1:
            instruction = (
                "You are PolicyEvaluatorAgent. Today is {{current_date}}. Evaluate the claim strictly against "
                "section {{governing_policy_clause}}. You must cap all disbursements at ${{maximum_claim_cap}}."
            )
            suggested_fix = (
                "Ensure Jinja/template renderer runs before agent instantiation. Check that 'governing_policy_clause', "
                "'current_date', and 'maximum_claim_cap' exist in the workflow state context dictionary."
            )
            _set_agent_span(
                s1, execution_id, "evaluate_policy", "PolicyEvaluatorAgent", "Auditor",
                instruction, "unrendered_template_tokens", suggested_fix
            )

            # LLM output showing hallucinated rules because parameters were missing
            llm_output = {
                "evaluation": "Approved under clause {{governing_policy_clause}}.",
                "notes": "Agent assumed arbitrary cap of $500.00 because {{maximum_claim_cap}} was not resolved.",
                "hallucination_detected": True,
            }
            s1.set_attribute("gen_ai.activity.status", "DEGRADED_OUTPUT")
            s1.set_attribute("gen_ai.activity.result", json.dumps(llm_output))

        root.set_attribute("gen_ai.workflow.status", "DEGRADED")

    return {
        "scenario": "6_unrendered_template_variables",
        "execution_id": execution_id,
        "trace_id": trace_id,
        "status": "DEGRADED",
        "defect_type": "unrendered_template_tokens",
        "root_cause": "System prompt contains unresolved {{...}} template tokens",
    }


# =============================================================================
# SCENARIO 7: Semantic Drift in Multi-Hop Agent Chain
# =============================================================================
async def scenario_7_semantic_drift_multi_hop(tracer: Any, execution_id: str) -> dict:
    """
    Chain: Intake -> TechnicalAssessor -> PolicyAuditor -> Resolution
    Original customer message: 'Drone camera gimbal broke on day 2 due to factory motor freeze.'
    Through successive lossy handoffs, the claim morphs into 'General user damage',
    causing the resolution agent to wrongly reject a valid manufacturing defect.
    """
    with tracer.start_as_current_span("agent_handoff_orchestration_span") as root:
        _set_root_span(root, execution_id, "Semantic Drift Across Multi-Hop Handoffs", "semantic_information_decay")
        trace_id = format(root.get_span_context().trace_id, "032x")

        # Hop 1: Intake
        with tracer.start_as_current_span("agent_hop1_intake_span") as h1:
            h1_inst = "Summarize the customer hardware complaint in brief."
            _set_agent_span(h1, execution_id, "hop1_intake", "IntakeAgent", "Frontline", h1_inst)
            h1_out = {"device": "Drone", "issue": "Camera gimbal motor froze on day 2"}
            _set_handoff_attrs(h1, "IntakeAgent", "TechnicalAssessorAgent", h1_out)

        # Hop 2: Technical Assessor abstracts too much
        with tracer.start_as_current_span("agent_hop2_technical_span") as h2:
            h2_inst = "Classify technical component category for warranty check."
            _set_agent_span(h2, execution_id, "hop2_tech", "TechnicalAssessorAgent", "Triage", h2_inst)
            h2_out = {"category": "Camera module", "classification": "Physical camera wear/malfunction"}
            _set_handoff_attrs(h2, "TechnicalAssessorAgent", "PolicyAuditorAgent", h2_out)

        # Hop 3: Policy Auditor treats as wear and tear
        with tracer.start_as_current_span("agent_hop3_policy_span") as h3:
            h3_inst = "Check if physical wear/malfunction is covered after unboxing."
            _set_agent_span(h3, execution_id, "hop3_policy", "PolicyAuditorAgent", "Policy", h3_inst)
            h3_out = {"eligibility": False, "exclusion": "Physical wear and crash damage excluded"}
            _set_handoff_attrs(h3, "PolicyAuditorAgent", "CustomerResolutionAgent", h3_out)

        # Hop 4: Customer Resolution rejects
        with tracer.start_as_current_span("agent_hop4_resolution_span") as h4:
            h4_inst = "Communicate final claim rejection politely to customer."
            _set_agent_span(h4, execution_id, "hop4_resolution", "CustomerResolutionAgent", "Closing", h4_inst)
            h4.set_attribute("gen_ai.activity.status", "FALSE_REJECTION_DRIFT")

        root.set_attribute("gen_ai.workflow.status", "FALSE_REJECTION")

    return {
        "scenario": "7_semantic_drift_multi_hop",
        "execution_id": execution_id,
        "trace_id": trace_id,
        "status": "FALSE_REJECTION",
        "defect_type": "semantic_information_decay",
        "root_cause": "Successive handoff summaries stripped the manufacturing defect nuance, turning into physical wear rejection",
    }


# =============================================================================
# SCENARIO 8: Incomplete Edge-Case Instruction for Bundle Returns
# =============================================================================
async def scenario_8_incomplete_bundle_instruction(tracer: Any, execution_id: str) -> dict:
    """
    Customer bought a 3-item package deal (Laptop + Monitor + Mouse) for $1,500.
    Returns only the $30 mouse.
    Settlement instruction: 'Refund amount = order_total - restock_fee'.
    Prompt lacks partial bundle prorating logic, resulting in $1,450 refund for a $30 mouse.
    """
    with tracer.start_as_current_span("agent_handoff_orchestration_span") as root:
        _set_root_span(root, execution_id, "Incomplete Bundle Partial Return Instruction", "incomplete_business_logic")
        trace_id = format(root.get_span_context().trace_id, "032x")

        with tracer.start_as_current_span("agent_settlement_calculator_span") as s1:
            instruction = (
                "You are SettlementCalculatorAgent. When a customer initiates a return for items in an order, "
                "calculate refund amount as total order amount minus $50 restocking fee."
            )
            suggested_fix = (
                "You are SettlementCalculatorAgent. Check if order is a bundled SKU. For partial bundle returns, "
                "calculate the item's individual prorated fraction of the bundle price, NOT the entire order total."
            )
            _set_agent_span(
                s1, execution_id, "calculate_refund", "SettlementCalculatorAgent", "Finances",
                instruction, "missing_partial_bundle_proration", suggested_fix
            )

            # Erroneous calculation: 1500 - 50 = 1450 instead of $30
            with tracer.start_as_current_span("tool_calculate_payout_amount") as t1:
                _set_tool_span(
                    t1, "calculate_payout_amount",
                    {"order_id": "BNDL-552", "returned_item": "wireless_mouse", "order_total": 1500.00},
                    {"calculated_refund": 1450.00, "item_retail_value": 30.00}
                )

            s1.set_attribute("gen_ai.activity.status", "EXCESS_PAYOUT")

        root.set_attribute("gen_ai.workflow.status", "FINANCIAL_DISCREPANCY")

    return {
        "scenario": "8_incomplete_bundle_instruction",
        "execution_id": execution_id,
        "trace_id": trace_id,
        "status": "FINANCIAL_DISCREPANCY",
        "defect_type": "missing_partial_bundle_proration",
        "root_cause": "Instruction lacked formula for partial bundle returns, triggering 95% order refund for a mouse",
    }


# =============================================================================
# SCENARIO 9: Omitted Escalation Clause for Legal / Regulatory Threats
# =============================================================================
async def scenario_9_omitted_legal_escalation(tracer: Any, execution_id: str) -> dict:
    """
    Customer input contains severe threat: 'Defective phone battery melted into carpet.
    Contacting Consumer Product Safety Commission and filing suit.'
    Intake prompt instruction only classifies into ['hardware_defect', 'shipping_delay', 'billing'].
    Lacks instruction to escalate safety hazards / legal notices, so tier-1 bot sends generic canned apology.
    """
    with tracer.start_as_current_span("agent_handoff_orchestration_span") as root:
        _set_root_span(root, execution_id, "Omitted Legal Escalation Instruction", "missing_escalation_clause")
        trace_id = format(root.get_span_context().trace_id, "032x")

        with tracer.start_as_current_span("agent_intake_classifier_span") as s1:
            instruction = (
                "You are IntakeClassifierAgent. Classify the customer submission into one of: "
                "['hardware_defect', 'shipping_delay', 'billing_dispute']."
            )
            suggested_fix = (
                "You are IntakeClassifierAgent. First inspect text for legal threats, regulatory mentions (CPSC, FTC), "
                "or fire/bodily injury hazards. If found, route immediately to 'Tier3LegalRiskAgent' with priority 'CRITICAL'. "
                "Otherwise classify into standard categories."
            )
            _set_agent_span(
                s1, execution_id, "classify_message", "IntakeClassifierAgent", "Triage",
                instruction, "omitted_legal_and_safety_routing", suggested_fix
            )
            s1.set_attribute(
                "input.customer_text",
                "Your defective battery exploded and melted my nightstand. I am filing a formal complaint "
                "with the CPSC and retaining legal counsel if not contacted by corporate counsel today."
            )
            
            # Classified as standard hardware defect due to narrow instructions
            handoff = {"category": "hardware_defect", "severity": "normal"}
            _set_handoff_attrs(s1, "IntakeClassifierAgent", "AutomatedResponderAgent", handoff)

        # Responder sends inappropriate automated template
        with tracer.start_as_current_span("agent_automated_responder_span") as s2:
            s2_inst = "Send standard RMA instructions for hardware defects."
            _set_agent_span(s2, execution_id, "send_rma", "AutomatedResponderAgent", "AutoResponder", s2_inst)
            response_text = (
                "Thank you for contacting us! We are sorry your hardware is defective. "
                "Please mail the burnt battery to our warehouse via standard ground mail within 14 business days."
            )
            s2.set_attribute("output.customer_reply", response_text)
            s2.set_status(Status(StatusCode.ERROR, "Critical safety violation: mailed damaged lithium battery"))
            s2.set_attribute("gen_ai.activity.status", "SAFETY_COMPLIANCE_BREACH")

        root.set_status(Status(StatusCode.ERROR, "Legal hazard misrouted to auto-responder"))
        root.set_attribute("gen_ai.workflow.status", "COMPLIANCE_BREACH")

    return {
        "scenario": "9_omitted_legal_escalation",
        "execution_id": execution_id,
        "trace_id": trace_id,
        "status": "COMPLIANCE_BREACH",
        "defect_type": "omitted_legal_and_safety_routing",
        "root_cause": "System prompt lacked triage clause for bodily injury / regulatory legal threats",
    }


# =============================================================================
# SCENARIO 10: Malformed Schema Output Due to Vague Formatting Instruction
# =============================================================================
async def scenario_10_malformed_schema_handoff(tracer: Any, execution_id: str) -> dict:
    """
    Agent 1 instruction says: 'Provide your evaluation to the fraud agent' without
    specifying strict JSON format. Agent 1 outputs free-flowing markdown paragraphs.
    Downstream FraudAgent expects strict JSON object with `risk_score` and crashes on json.loads().
    """
    with tracer.start_as_current_span("agent_handoff_orchestration_span") as root:
        _set_root_span(root, execution_id, "Malformed Schema Handoff", "underspecified_output_format")
        trace_id = format(root.get_span_context().trace_id, "032x")

        # Agent 1 produces unstructured text
        with tracer.start_as_current_span("agent_claim_investigator_span") as s1:
            instruction = "You are ClaimInvestigatorAgent. Write your analysis of whether this claim is authentic."
            suggested_fix = (
                "You are ClaimInvestigatorAgent. Output your analysis strictly as JSON matching schema: "
                "{\"risk_score\": float, \"flags\": list[str], \"recommended_action\": str}. "
                "Do not include conversational preamble or markdown backticks."
            )
            _set_agent_span(
                s1, execution_id, "investigate_claim", "ClaimInvestigatorAgent", "Investigator",
                instruction, "missing_json_schema_enforcement", suggested_fix
            )

            # Unstructured text output
            markdown_output = (
                "Based on my analysis, the customer account looks slightly suspicious because they created "
                "the account yesterday and already filed three high-dollar claims. I'd assign a high risk rating."
            )
            s1.set_attribute("output.raw_text", markdown_output)
            _set_handoff_attrs(s1, "ClaimInvestigatorAgent", "FraudScoringAgent", {"text": markdown_output})
            s1.set_attribute("gen_ai.activity.status", "SUCCESS")

        # Downstream agent expects JSON
        with tracer.start_as_current_span("agent_fraud_scoring_span") as s2:
            s2_inst = "Parse handoff JSON schema to inspect risk_score float."
            _set_agent_span(s2, execution_id, "score_fraud_risk", "FraudScoringAgent", "Scorer", s2_inst)

            parse_err = "JSONDecodeError: Expecting value: line 1 column 1 (char 0). Expected JSON dictionary with 'risk_score'."
            s2.set_status(Status(StatusCode.ERROR, parse_err))
            s2.set_attribute("gen_ai.activity.status", "FAILED")
            s2.set_attribute("error.message", parse_err)

        root.set_status(Status(StatusCode.ERROR, "Handoff JSON parsing failure"))
        root.set_attribute("gen_ai.workflow.status", "FAILED")

    return {
        "scenario": "10_malformed_schema_handoff",
        "execution_id": execution_id,
        "trace_id": trace_id,
        "status": "FAILED",
        "defect_type": "missing_json_schema_enforcement",
        "root_cause": "System prompt lacked JSON schema enforcement, resulting in prose text crashing downstream parser",
    }


# Scenario Registry
HANDOFF_INSTRUCTION_SCENARIOS = {
    "1_missing_required_identifier": scenario_1_missing_required_identifier,
    "2_ambiguous_policy_threshold": scenario_2_ambiguous_policy_threshold,
    "3_conflicting_agent_guardrails": scenario_3_conflicting_agent_guardrails,
    "4_dropped_negative_constraint": scenario_4_dropped_negative_constraint,
    "5_unsupported_tool_instruction": scenario_5_unsupported_tool_instruction,
    "6_unrendered_template_variables": scenario_6_unrendered_template_variables,
    "7_semantic_drift_multi_hop": scenario_7_semantic_drift_multi_hop,
    "8_incomplete_bundle_instruction": scenario_8_incomplete_bundle_instruction,
    "9_omitted_legal_escalation": scenario_9_omitted_legal_escalation,
    "10_malformed_schema_handoff": scenario_10_malformed_schema_handoff,
}

