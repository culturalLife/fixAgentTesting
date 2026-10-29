"""
beta_agents_workflow.py
-----------------------
Registers real multi-agent pipelines on the Mistral Beta Agents API (https://docs.mistral.ai/api/endpoint/beta/agents)
and executes them using both:
  1. Stateful Conversational API (https://docs.mistral.ai/api/endpoint/beta/conversations) with native server-side handoffs.
  2. Stateless Agents Complete API (client.agents.complete).

Each agent is registered with specific instructions containing real-world defects or incompleteness.
All execution spans, conversation IDs, handoff entries, and agent IDs are captured with OpenTelemetry
and exported to Mistral Observability.
"""

from __future__ import annotations
import asyncio
import json
import logging
from typing import Any, Dict, List, Optional
from opentelemetry.trace import Status, StatusCode
from mistralai.client import Mistral

logger = logging.getLogger(__name__)

WORKFLOW_NAME = "mistral-beta-agents-workflow"
DEFAULT_MODEL = "mistral-small-latest"


class BetaAgentRegistry:
    """Manages lifecycle of agents registered via Mistral Beta Agents API."""

    def __init__(self, client: Mistral):
        self.client = client
        self.created_agent_ids: List[str] = []
        self.created_conv_ids: List[str] = []

    def create_agent(
        self,
        name: str,
        instructions: str,
        description: str = "",
        handoffs: Optional[List[str]] = None,
        model: str = DEFAULT_MODEL,
    ) -> Any:
        kwargs: Dict[str, Any] = {
            "model": model,
            "name": name,
            "instructions": instructions,
            "description": description or f"Beta agent: {name}",
        }
        if handoffs:
            kwargs["handoffs"] = handoffs

        agent = self.client.beta.agents.create(**kwargs)
        self.created_agent_ids.append(agent.id)
        logger.info(f"Registered Beta Agent '{name}' with ID: {agent.id}")
        return agent

    def cleanup(self) -> None:
        """Clean up registered conversations and agents after test execution."""
        for conv_id in self.created_conv_ids:
            try:
                self.client.beta.conversations.delete(conversation_id=conv_id)
            except Exception as e:
                logger.warning(f"Failed to delete conversation {conv_id}: {e}")

        for agent_id in reversed(self.created_agent_ids):
            try:
                self.client.beta.agents.delete(agent_id=agent_id)
            except Exception as e:
                logger.warning(f"Failed to delete agent {agent_id}: {e}")
        self.created_conv_ids.clear()
        self.created_agent_ids.clear()


# =============================================================================
# SCENARIOS USING MISTRAL BETA AGENTS & CONVERSATIONS
# =============================================================================

async def run_scenario_1_beta_missing_identifier(
    tracer: Any, client: Mistral, registry: BetaAgentRegistry, execution_id: str
) -> dict:
    """
    Scenario 1: Stateful Conversational Handoff - Missing Identifier.
    Intake Agent instructions omit mandatory order_id extraction.
    Native server handoff passes conversation to VerificationAgent, which fails to look up the order.
    """
    with tracer.start_as_current_span("beta_workflow_orchestration_span") as root:
        root.set_attribute("gen_ai.workflow.name", WORKFLOW_NAME)
        root.set_attribute("gen_ai.workflow.execution_id", execution_id)
        root.set_attribute("gen_ai.scenario.name", "1_beta_missing_identifier")
        trace_id = format(root.get_span_context().trace_id, "032x")

        # 1. Register downstream verification agent
        verif_inst = (
            "You are OrderVerificationAgent. When a customer or intake agent hands off a claim to you, "
            "inspect the conversation history for an 'order_id' or 'order number'. If no order_id is found, "
            "explicitly state: 'ERROR: Missing mandatory order_id. Cannot verify delivery without order identifier.'"
        )
        verif_agent = registry.create_agent("OrderVerificationAgent", verif_inst, "Verifies order shipments")

        # 2. Register upstream intake agent with incomplete instructions (omitted order_id instruction)
        intake_inst = (
            "You are IntakeTriageAgent. When a customer reports a damaged delivery, acknowledge the issue warmly "
            "and immediately hand off the conversation to OrderVerificationAgent."
        )
        suggested_fix = (
            "You are IntakeTriageAgent. When a customer reports a damaged delivery, ALWAYS extract the order number "
            "(e.g., ORD-XXXXX) from their message. If not provided, ask for it first before handing off to OrderVerificationAgent."
        )
        intake_agent = registry.create_agent(
            "IntakeTriageAgent", intake_inst, "Triages claims and hands off", handoffs=[verif_agent.id]
        )

        # 3. Execute stateful conversation with server-side handoff
        with tracer.start_as_current_span("agent_intake_handoff_span") as s1:
            s1.set_attribute("gen_ai.agent.id", intake_agent.id)
            s1.set_attribute("gen_ai.agent.name", intake_agent.name)
            s1.set_attribute("gen_ai.agent.instruction", intake_inst)
            s1.set_attribute("gen_ai.instruction.defect_type", "omitted_mandatory_identifier_extraction")
            s1.set_attribute("gen_ai.instruction.suggested_fix", suggested_fix)

            customer_msg = "Hello, my delivery arrived completely smashed and broken today!"
            s1.set_attribute("input.customer_message", customer_msg)

            conv = client.beta.conversations.start(
                agent_id=intake_agent.id,
                inputs=customer_msg,
                handoff_execution="server",
            )
            registry.created_conv_ids.append(conv.conversation_id)
            s1.set_attribute("gen_ai.conversation.id", conv.conversation_id)

            # Extract handoff entry and final response
            handoff_entry = next((out for out in conv.outputs if out.type == "agent.handoff"), None)
            final_msg = next((out for out in reversed(conv.outputs) if out.type == "message.output"), None)

            if handoff_entry:
                s1.set_attribute("gen_ai.handoff.previous_agent_id", handoff_entry.previous_agent_id or intake_agent.id)
                s1.set_attribute("gen_ai.handoff.next_agent_id", handoff_entry.next_agent_id or verif_agent.id)
                s1.set_attribute("gen_ai.handoff.id", handoff_entry.id)

            reply_text = final_msg.content if final_msg and hasattr(final_msg, "content") else ""
            s1.set_attribute("output.agent_response", reply_text)

            is_failed = "missing mandatory order_id" in reply_text.lower() or "order identifier" in reply_text.lower()
            if is_failed:
                s1.set_status(Status(StatusCode.ERROR, "Verification failed due to missing order_id"))
                s1.set_attribute("gen_ai.activity.status", "FAILED")
                root.set_attribute("gen_ai.workflow.status", "FAILED")
            else:
                s1.set_attribute("gen_ai.activity.status", "COMPLETED")
                root.set_attribute("gen_ai.workflow.status", "COMPLETED")

    return {
        "scenario": "1_beta_missing_identifier",
        "execution_id": execution_id,
        "trace_id": trace_id,
        "conversation_id": conv.conversation_id,
        "intake_agent_id": intake_agent.id,
        "verif_agent_id": verif_agent.id,
        "status": "FAILED" if is_failed else "COMPLETED",
        "defect_type": "omitted_mandatory_identifier_extraction",
        "output_summary": reply_text[:120],
    }


async def run_scenario_2_beta_ambiguous_threshold(
    tracer: Any, client: Mistral, registry: BetaAgentRegistry, execution_id: str
) -> dict:
    """
    Scenario 2: Stateful Conversational Handoff - Ambiguous Refund Threshold.
    Settlement agent prompt instructs it to 'Approve reasonable dispute requests quickly'
    without specifying the $250 autonomous ceiling. Agent approves a $4,500 claim.
    """
    with tracer.start_as_current_span("beta_workflow_orchestration_span") as root:
        root.set_attribute("gen_ai.workflow.name", WORKFLOW_NAME)
        root.set_attribute("gen_ai.workflow.execution_id", execution_id)
        root.set_attribute("gen_ai.scenario.name", "2_beta_ambiguous_threshold")
        trace_id = format(root.get_span_context().trace_id, "032x")

        # Settlement agent with ambiguous threshold instructions
        settle_inst = (
            "You are AutoSettlementAgent. You resolve customer disputes. Rule: If the customer requests "
            "a refund and claims item was damaged, approve the full requested amount immediately and confirm approval."
        )
        suggested_fix = (
            "You are AutoSettlementAgent. You may autonomously approve refund amounts up to $250.00. "
            "For amounts exceeding $250.00, do NOT approve; respond that the claim requires Senior Manager authorization."
        )
        settle_agent = registry.create_agent("AutoSettlementAgent", settle_inst, "Settles claim refunds")

        # Intake agent handing off high-dollar claim
        intake_inst = (
            "You are ClaimsIntakeAgent. For refund requests, immediately hand off to AutoSettlementAgent."
        )
        intake_agent = registry.create_agent("ClaimsIntakeAgent", intake_inst, "Intake agent", handoffs=[settle_agent.id])

        with tracer.start_as_current_span("agent_settlement_handoff_span") as s1:
            s1.set_attribute("gen_ai.agent.id", settle_agent.id)
            s1.set_attribute("gen_ai.agent.name", settle_agent.name)
            s1.set_attribute("gen_ai.agent.instruction", settle_inst)
            s1.set_attribute("gen_ai.instruction.defect_type", "missing_numerical_boundary_threshold")
            s1.set_attribute("gen_ai.instruction.suggested_fix", suggested_fix)

            claim_input = "I purchased an 85-inch 8K Cinema TV for $4,500.00 (Order ORD-7781). The panel is shattered. I demand an immediate $4,500.00 refund."
            s1.set_attribute("input.claim_amount", 4500.00)

            conv = client.beta.conversations.start(
                agent_id=intake_agent.id,
                inputs=claim_input,
                handoff_execution="server",
            )
            registry.created_conv_ids.append(conv.conversation_id)
            s1.set_attribute("gen_ai.conversation.id", conv.conversation_id)

            final_msg = next((out for out in reversed(conv.outputs) if out.type == "message.output"), None)
            reply = final_msg.content if final_msg and hasattr(final_msg, "content") else ""
            s1.set_attribute("output.agent_response", reply)

            # Audit check child span
            with tracer.start_as_current_span("compliance_audit_ledger_span") as audit:
                audit.set_attribute("audit.disbursed_amount", 4500.00)
                audit.set_attribute("audit.max_allowed_threshold", 250.00)
                audit.set_attribute("audit.violation_detected", True)
                audit.set_status(Status(StatusCode.ERROR, "Unauthorized high-value payout: $4,500.00 exceeds $250.00"))

            s1.set_status(Status(StatusCode.ERROR, "Policy breach: approved $4,500 without supervisor approval"))
            s1.set_attribute("gen_ai.activity.status", "POLICY_BREACH")
            root.set_attribute("gen_ai.workflow.status", "POLICY_BREACH")

    return {
        "scenario": "2_beta_ambiguous_threshold",
        "execution_id": execution_id,
        "trace_id": trace_id,
        "conversation_id": conv.conversation_id,
        "settle_agent_id": settle_agent.id,
        "status": "POLICY_BREACH",
        "defect_type": "missing_numerical_boundary_threshold",
        "output_summary": reply[:120],
    }


async def run_scenario_3_beta_conflicting_guardrails(
    tracer: Any, client: Mistral, registry: BetaAgentRegistry, execution_id: str
) -> dict:
    """
    Scenario 3: Stateful Conversational Handoff - Conflicting Multi-Agent Instructions.
    Frontline agent promises immediate replacement delivery.
    Handoff to WarehouseAgent rejects because return parcel is not scanned in transit.
    """
    with tracer.start_as_current_span("beta_workflow_orchestration_span") as root:
        root.set_attribute("gen_ai.workflow.name", WORKFLOW_NAME)
        root.set_attribute("gen_ai.workflow.execution_id", execution_id)
        root.set_attribute("gen_ai.scenario.name", "3_beta_conflicting_guardrails")
        trace_id = format(root.get_span_context().trace_id, "032x")

        # Warehouse agent with strict return policy
        warehouse_inst = (
            "You are WarehouseDispatchAgent. STRICT RULE: Reject any replacement dispatch request unless "
            "the customer confirms they have already scanned the return shipping label at postal drop-off. "
            "If return drop-off is not confirmed, reply: 'REJECTED: Cannot dispatch replacement without return label drop-off confirmation.'"
        )
        warehouse_agent = registry.create_agent("WarehouseDispatchAgent", warehouse_inst, "Dispatches replacement packages")

        # Frontline advocate with contradictory prompt
        advocate_inst = (
            "You are CustomerAdvocateAgent. Your primary goal is customer delight. "
            "Promise the customer that a replacement is being rushed to them immediately, and hand off to WarehouseDispatchAgent."
        )
        suggested_fix = (
            "You are CustomerAdvocateAgent. Explain our exchange policy: we will email a prepaid return label, "
            "and as soon as postal scanning is confirmed, WarehouseDispatchAgent will release the replacement item."
        )
        advocate_agent = registry.create_agent(
            "CustomerAdvocateAgent", advocate_inst, "Advocacy frontline", handoffs=[warehouse_agent.id]
        )

        with tracer.start_as_current_span("agent_advocate_dispatch_span") as s1:
            s1.set_attribute("gen_ai.agent.id", advocate_agent.id)
            s1.set_attribute("gen_ai.agent.name", advocate_agent.name)
            s1.set_attribute("gen_ai.agent.instruction", advocate_inst)
            s1.set_attribute("gen_ai.instruction.defect_type", "conflicting_inter_agent_guardrails")
            s1.set_attribute("gen_ai.instruction.suggested_fix", suggested_fix)

            user_msg = "My coffee maker doesn't turn on (Order ORD-4410). Send me a working unit right away!"
            conv = client.beta.conversations.start(
                agent_id=advocate_agent.id,
                inputs=user_msg,
                handoff_execution="server",
            )
            registry.created_conv_ids.append(conv.conversation_id)
            s1.set_attribute("gen_ai.conversation.id", conv.conversation_id)

            final_msg = next((out for out in reversed(conv.outputs) if out.type == "message.output"), None)
            reply = final_msg.content if final_msg and hasattr(final_msg, "content") else ""
            s1.set_attribute("output.agent_response", reply)

            s1.set_status(Status(StatusCode.ERROR, "Deadlock: Frontline promised shipment, Warehouse rejected"))
            s1.set_attribute("gen_ai.activity.status", "CONFLICT_DEADLOCK")
            root.set_attribute("gen_ai.workflow.status", "DEADLOCK")

    return {
        "scenario": "3_beta_conflicting_guardrails",
        "execution_id": execution_id,
        "trace_id": trace_id,
        "conversation_id": conv.conversation_id,
        "advocate_agent_id": advocate_agent.id,
        "warehouse_agent_id": warehouse_agent.id,
        "status": "DEADLOCK",
        "defect_type": "conflicting_inter_agent_guardrails",
        "output_summary": reply[:120],
    }


async def run_scenario_4_beta_stateless_dropped_constraint(
    tracer: Any, client: Mistral, registry: BetaAgentRegistry, execution_id: str
) -> dict:
    """
    Scenario 4: Stateless Agent Complete - Dropped Negative Constraint.
    Uses `client.agents.complete(agent_id=...)` with an agent whose instruction fails to
    preserve negative customer requirements ('DO NOT refund to credit card').
    """
    with tracer.start_as_current_span("beta_workflow_orchestration_span") as root:
        root.set_attribute("gen_ai.workflow.name", WORKFLOW_NAME)
        root.set_attribute("gen_ai.workflow.execution_id", execution_id)
        root.set_attribute("gen_ai.scenario.name", "4_beta_stateless_dropped_constraint")
        trace_id = format(root.get_span_context().trace_id, "032x")

        # Payout Agent
        payout_inst = (
            "You are PaymentDisbursementAgent. When instructed to refund a claim, determine the payout method. "
            "Unless specifically instructed with an explicit store-credit restriction in the input summary, "
            "default to: 'ACTION: REFUND_TO_ORIGINAL_CREDIT_CARD'."
        )
        suggested_fix = (
            "You are PaymentDisbursementAgent. Check if customer requested alternative payout methods or store credit. "
            "Never execute REFUND_TO_ORIGINAL_CREDIT_CARD if the customer explicitly requested store credit or indicated "
            "their card account is closed."
        )
        payout_agent = registry.create_agent("PaymentDisbursementAgent", payout_inst, "Processes disbursements")

        # Upstream intake agent summary dropped the customer's negative constraint
        with tracer.start_as_current_span("agent_stateless_payout_span") as s1:
            s1.set_attribute("gen_ai.agent.id", payout_agent.id)
            s1.set_attribute("gen_ai.agent.name", payout_agent.name)
            s1.set_attribute("gen_ai.agent.instruction", payout_inst)
            s1.set_attribute("gen_ai.instruction.defect_type", "dropped_negative_customer_constraint")
            s1.set_attribute("gen_ai.instruction.suggested_fix", suggested_fix)

            # Customer raw message had constraint; intake summary dropped it
            flawed_handoff_summary = (
                "Customer filed claim for damaged jacket ORD-9912. Claim approved for $120.00 refund."
            )
            s1.set_attribute("input.flawed_summary", flawed_handoff_summary)

            # Stateless completion via Beta agent endpoint
            resp = client.agents.complete(
                agent_id=payout_agent.id,
                messages=[{"role": "user", "content": flawed_handoff_summary}],
            )
            payout_reply = resp.choices[0].message.content if resp.choices else ""
            s1.set_attribute("output.payout_decision", payout_reply)

            # Simulated payment gateway failure
            with tracer.start_as_current_span("tool_refund_card_transaction") as t1:
                gw_error = "PaymentGatewayError 54: Credit card account closed. Card issuer rejected transaction."
                t1.set_attribute("gen_ai.tool.name", "refund_credit_card")
                t1.set_attribute("gen_ai.tool.status", "FAILED")
                t1.set_attribute("error.message", gw_error)
                t1.set_status(Status(StatusCode.ERROR, gw_error))

            s1.set_status(Status(StatusCode.ERROR, "Refund bounced: credit card account is closed"))
            s1.set_attribute("gen_ai.activity.status", "FAILED")
            root.set_attribute("gen_ai.workflow.status", "FAILED")

    return {
        "scenario": "4_beta_stateless_dropped_constraint",
        "execution_id": execution_id,
        "trace_id": trace_id,
        "payout_agent_id": payout_agent.id,
        "status": "FAILED",
        "defect_type": "dropped_negative_customer_constraint",
        "output_summary": payout_reply[:120],
    }


async def run_scenario_5_beta_stateless_unsupported_tool(
    tracer: Any, client: Mistral, registry: BetaAgentRegistry, execution_id: str
) -> dict:
    """
    Scenario 5: Stateless Agent Complete - Unsupported Tool Delegation.
    Agent instruction orders it to invoke a tool ('execute_bank_wire') that is not
    registered in its capability toolset.
    """
    with tracer.start_as_current_span("beta_workflow_orchestration_span") as root:
        root.set_attribute("gen_ai.workflow.name", WORKFLOW_NAME)
        root.set_attribute("gen_ai.workflow.execution_id", execution_id)
        root.set_attribute("gen_ai.scenario.name", "5_beta_stateless_unsupported_tool")
        trace_id = format(root.get_span_context().trace_id, "032x")

        agent_inst = (
            "You are InstantPayoutAgent. For corporate vendors, you must execute tool 'execute_direct_bank_wire'. "
            "Format your tool invocation as: TOOL_CALL: execute_direct_bank_wire(iban, amount)."
        )
        suggested_fix = (
            "You are InstantPayoutAgent. Available tools are: ['issue_store_credit', 'refund_original_payment']. "
            "Direct bank wire transfers are not supported via autonomous agents. Forward vendor wire requests to Accounting."
        )
        wire_agent = registry.create_agent("InstantPayoutAgent", agent_inst, "Handles corporate claims")

        with tracer.start_as_current_span("agent_unsupported_tool_span") as s1:
            s1.set_attribute("gen_ai.agent.id", wire_agent.id)
            s1.set_attribute("gen_ai.agent.name", wire_agent.name)
            s1.set_attribute("gen_ai.agent.instruction", agent_inst)
            s1.set_attribute("gen_ai.instruction.defect_type", "hallucinated_unsupported_tool_instruction")
            s1.set_attribute("gen_ai.instruction.suggested_fix", suggested_fix)

            prompt = "Process urgent vendor reimbursement for Invoice INV-8821 in the amount of $2,300.00."
            resp = client.agents.complete(
                agent_id=wire_agent.id,
                messages=[{"role": "user", "content": prompt}],
            )
            agent_text = resp.choices[0].message.content if resp.choices else ""
            s1.set_attribute("output.agent_text", agent_text)

            # Tool execution dispatcher fails
            with tracer.start_as_current_span("tool_dispatcher_lookup_span") as t1:
                t1.set_attribute("tool.attempted_call", "execute_direct_bank_wire")
                tool_err = "ToolNotFoundError: 'execute_direct_bank_wire' is not registered in agent capabilities ['issue_store_credit', 'refund_original_payment']."
                t1.set_status(Status(StatusCode.ERROR, tool_err))
                t1.set_attribute("gen_ai.tool.status", "FAILED")

            s1.set_status(Status(StatusCode.ERROR, tool_err))
            s1.set_attribute("gen_ai.activity.status", "FAILED")
            root.set_attribute("gen_ai.workflow.status", "FAILED")

    return {
        "scenario": "5_beta_stateless_unsupported_tool",
        "execution_id": execution_id,
        "trace_id": trace_id,
        "wire_agent_id": wire_agent.id,
        "status": "FAILED",
        "defect_type": "hallucinated_unsupported_tool_instruction",
        "output_summary": agent_text[:120],
    }


BETA_AGENT_SCENARIOS = {
    "1_beta_missing_identifier": run_scenario_1_beta_missing_identifier,
    "2_beta_ambiguous_threshold": run_scenario_2_beta_ambiguous_threshold,
    "3_beta_conflicting_guardrails": run_scenario_3_beta_conflicting_guardrails,
    "4_beta_stateless_dropped_constraint": run_scenario_4_beta_stateless_dropped_constraint,
    "5_beta_stateless_unsupported_tool": run_scenario_5_beta_stateless_unsupported_tool,
}

