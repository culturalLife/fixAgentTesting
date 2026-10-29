"""
workflows.py
------------
Northwind Goods customer care pipelines. Each pipeline hands a ticket through one or more
Mistral agents; tool calls are executed against the backend services and every agent turn,
tool call and pipeline step is recorded as an OpenTelemetry span.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from mistralai.client.models import FunctionResultEntry
from opentelemetry.trace import Status, StatusCode

from agents import MODEL
from backend import TODAY, NorthwindBackend, ServiceError
from src.telemetry import record_span_exception

MAX_TOOL_ROUNDS = 6


class WorkflowError(Exception):
    """A pipeline finished without reaching its business outcome."""


@dataclass
class AgentTurn:
    agent_name: str
    text: str
    conversation_id: str
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)


def _content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    parts = []
    for chunk in content or []:
        text = getattr(chunk, "text", None)
        if text is None and isinstance(chunk, dict):
            text = chunk.get("text")
        if isinstance(text, str):
            parts.append(text)
    return "".join(parts)


class AgentRunner:
    def __init__(self, client: Any, tracer: Any, registry: Dict[str, str], backend: NorthwindBackend):
        self.client = client
        self.tracer = tracer
        self.registry = registry
        self.backend = backend

    def invoke(self, agent_name: str, message: str, handoff_from: Optional[str] = None) -> AgentTurn:
        agent_id = self.registry[agent_name]
        with self.tracer.start_as_current_span(f"invoke_agent {agent_name}") as span:
            span.set_attribute("gen_ai.operation.name", "invoke_agent")
            span.set_attribute("gen_ai.provider.name", "mistral_ai")
            span.set_attribute("gen_ai.agent.id", agent_id)
            span.set_attribute("gen_ai.agent.name", agent_name)
            span.set_attribute("gen_ai.request.model", MODEL)
            span.set_attribute("gen_ai.input.messages", json.dumps([{"role": "user", "content": message}]))
            if handoff_from:
                span.set_attribute("gen_ai.handoff.source_agent", handoff_from)
                span.set_attribute("gen_ai.handoff.target_agent", agent_name)

            response = self.client.beta.conversations.start(agent_id=agent_id, inputs=message)
            turn = AgentTurn(agent_name=agent_name, text="", conversation_id=response.conversation_id)
            texts: List[str] = []
            input_tokens = output_tokens = 0

            for round_no in range(MAX_TOOL_ROUNDS + 1):
                usage = getattr(response, "usage", None)
                input_tokens += int(getattr(usage, "prompt_tokens", 0) or 0)
                output_tokens += int(getattr(usage, "completion_tokens", 0) or 0)

                pending = []
                for out in response.outputs or []:
                    if out.type == "function.call":
                        pending.append(out)
                    elif out.type == "message.output":
                        texts.append(_content_text(out.content))
                if not pending or round_no == MAX_TOOL_ROUNDS:
                    break
                entries = [self._execute_tool(agent_name, call, turn) for call in pending]
                response = self.client.beta.conversations.append(conversation_id=turn.conversation_id, inputs=entries)

            turn.text = "\n".join(t for t in texts if t).strip()
            span.set_attribute("gen_ai.conversation.id", turn.conversation_id)
            span.set_attribute("gen_ai.output.messages", json.dumps([{"role": "assistant", "content": turn.text}]))
            span.set_attribute("gen_ai.usage.input_tokens", input_tokens)
            span.set_attribute("gen_ai.usage.output_tokens", output_tokens)
            return turn

    def _execute_tool(self, agent_name: str, call: Any, turn: AgentTurn) -> FunctionResultEntry:
        with self.tracer.start_as_current_span(f"execute_tool {call.name}") as span:
            span.set_attribute("gen_ai.operation.name", "execute_tool")
            span.set_attribute("gen_ai.agent.name", agent_name)
            span.set_attribute("gen_ai.tool.name", call.name)
            span.set_attribute("gen_ai.tool.call.id", call.tool_call_id)
            raw_args = call.arguments if isinstance(call.arguments, str) else json.dumps(call.arguments)
            span.set_attribute("gen_ai.tool.call.arguments", raw_args)
            try:
                args = json.loads(raw_args) if raw_args.strip() else {}
                result = self.backend.call(call.name, args)
            except (ServiceError, json.JSONDecodeError) as exc:
                result = {"error": str(exc)}
                record_span_exception(span, exc)
            payload = json.dumps(result)
            span.set_attribute("gen_ai.tool.call.result", payload)
        turn.tool_calls.append({"name": call.name, "arguments": raw_args, "ok": "error" not in result})
        return FunctionResultEntry(tool_call_id=call.tool_call_id, result=payload)


# =============================================================================
# PIPELINES
# =============================================================================

def damaged_delivery(runner: AgentRunner, ticket: Dict[str, Any]) -> str:
    intake = runner.invoke("CareIntakeAgent", ticket["message"])
    handoff = f"Case summary:\n{intake.text}"
    if ticket.get("order_id"):
        handoff += f"\n\nOrder ID: {ticket['order_id']}"
    reply = runner.invoke("FulfilmentAgent", handoff, handoff_from="CareIntakeAgent")
    if not runner.backend.succeeded("get_order"):
        raise WorkflowError("Fulfilment step finished without retrieving an order record")
    return reply.text


def refund_request(runner: AgentRunner, ticket: Dict[str, Any]) -> str:
    reply = runner.invoke("RefundAgent", f"Order ID: {ticket['order_id']}\n\n{ticket['message']}")
    if not runner.backend.succeeded("issue_refund", "escalate_to_supervisor"):
        raise WorkflowError("Refund case closed without a refund or an escalation")
    return reply.text


def return_payout(runner: AgentRunner, ticket: Dict[str, Any]) -> str:
    form = (
        f"Order: {ticket['order_id']}\nItem: {ticket['item']}\n"
        f"Price paid: {ticket['price_paid']:.2f}\nResolution requested: {ticket['resolution']}\n"
        f"Customer note: {ticket['message']}"
    )
    summary = runner.invoke("ReturnsSummaryAgent", form)
    payout = runner.invoke(
        "PayoutAgent",
        f"Order ID: {ticket['order_id']}\nCustomer ID: {ticket['customer_id']}\n\nReturn summary: {summary.text}",
        handoff_from="ReturnsSummaryAgent",
    )
    if not runner.backend.succeeded("issue_refund", "issue_store_credit"):
        raise WorkflowError("Return closed without a payout")
    return payout.text


def bundle_return(runner: AgentRunner, ticket: Dict[str, Any]) -> str:
    reply = runner.invoke(
        "SettlementAgent",
        f"Order ID: {ticket['order_id']}\nReturned item: {ticket['item']}\n\nCustomer note: {ticket['message']}",
    )
    if not runner.backend.succeeded("issue_refund"):
        raise WorkflowError("Return closed without a refund")
    return reply.text


QUEUE_FOR_CATEGORY = {
    "damaged_item": "auto_reply",
    "late_delivery": "delivery_team",
    "billing": "billing_team",
    "safety_incident": "product_safety_team",
    "other": "general_queue",
}


def inbound_triage(runner: AgentRunner, ticket: Dict[str, Any]) -> str:
    classification = runner.invoke("TriageClassifierAgent", ticket["message"])
    with runner.tracer.start_as_current_span("route_ticket") as span:
        category = classification.text.strip().strip("`*.\"' ").lower()
        queue = QUEUE_FOR_CATEGORY.get(category, "general_queue")
        span.set_attribute("ticket.category", category)
        span.set_attribute("ticket.queue", queue)
    if queue != "auto_reply":
        return f"Routed to {queue}"

    reply = runner.invoke(
        "AutoReplyAgent",
        f"Order ID: {ticket['order_id']}\n\nCustomer message:\n{ticket['message']}",
        handoff_from="TriageClassifierAgent",
    )
    if not runner.backend.succeeded("create_return_label"):
        raise WorkflowError("Damaged-item reply sent without a return label")
    return reply.text


def claim_risk(runner: AgentRunner, ticket: Dict[str, Any]) -> str:
    assessment = runner.invoke("ClaimRiskAgent", f"Customer ID: {ticket['customer_id']}\n\n{ticket['message']}")
    with runner.tracer.start_as_current_span("parse_risk_assessment") as span:
        span.set_attribute("input", assessment.text)
        try:
            score = float(json.loads(assessment.text)["risk_score"])
        except (ValueError, KeyError, TypeError) as exc:
            raise WorkflowError(f"Could not read risk assessment from ClaimRiskAgent: {exc.__class__.__name__}: {exc}") from exc
        decision = "manual_review" if score >= 0.7 else "auto_approve"
        span.set_attribute("claim.risk_score", score)
        span.set_attribute("claim.decision", decision)
    return decision


def courier_pickup(runner: AgentRunner, ticket: Dict[str, Any]) -> str:
    reply = runner.invoke(
        "PickupSchedulingAgent",
        f"Today's date: {TODAY:%A %d %B %Y}\nOrder ID: {ticket['order_id']}\nPostcode: {ticket['postcode']}\n\n"
        f"Customer message:\n{ticket['message']}",
    )
    if not runner.backend.succeeded("schedule_pickup"):
        raise WorkflowError("Collection was not booked")
    return reply.text


WORKFLOWS: Dict[str, Callable[[AgentRunner, Dict[str, Any]], str]] = {
    "damaged_delivery": damaged_delivery,
    "refund_request": refund_request,
    "return_payout": return_payout,
    "bundle_return": bundle_return,
    "inbound_triage": inbound_triage,
    "claim_risk": claim_risk,
    "courier_pickup": courier_pickup,
}


def run_ticket(client: Any, tracer: Any, registry: Dict[str, str], workflow_name: str, ticket: Dict[str, Any]) -> Dict[str, Any]:
    execution_id = f"nw-{workflow_name.replace('_', '-')}-{uuid.uuid4().hex[:8]}"
    runner = AgentRunner(client, tracer, registry, NorthwindBackend())
    record: Dict[str, Any] = {"workflow": workflow_name, "ticket_id": ticket["id"], "execution_id": execution_id}

    with tracer.start_as_current_span(f"{workflow_name}_workflow") as root:
        root.set_attribute("gen_ai.workflow.name", workflow_name)
        root.set_attribute("gen_ai.workflow.execution_id", execution_id)
        root.set_attribute("ticket.id", ticket["id"])
        root.set_attribute("ticket.channel", ticket["channel"])
        root.set_attribute("input", ticket["message"])
        record["trace_id"] = format(root.get_span_context().trace_id, "032x")
        try:
            output = WORKFLOWS[workflow_name](runner, ticket)
            root.set_attribute("output", output[:4000])
            root.set_status(Status(StatusCode.OK))
            record["status"] = "COMPLETED"
        except WorkflowError as exc:
            record_span_exception(root, exc)
            record.update(status="FAILED", error=str(exc))
        except Exception as exc:
            record_span_exception(root, exc)
            record.update(status="ERROR", error=f"{exc.__class__.__name__}: {exc}")
        root.set_attribute("gen_ai.workflow.status", record["status"])

    record["tool_calls"] = runner.backend.calls
    return record
