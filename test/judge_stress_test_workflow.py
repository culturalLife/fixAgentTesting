"""
judge_stress_test_workflow.py
------------------------------
Adversarial test harness for the Mistral Observability judge suite.

Every activity below is a deliberately engineered "planted defect." None of
it is real production logic. Each trap targets one specific judge label, or
one known fragility in trace_eval_service.py, and is built so that when the
judge suite fails to catch it, you know exactly which rubric or which line
of code is wrong.

Workflow:
  1. Run this workflow (or a subset of scenarios) to emit real traces.
  2. Run trace_eval_service.scan_and_remediate_trace_batch() against them.
  3. Compare batch_trace_reports[i]["trigger_type"] against the EXPECTED
     column in STRESS_TEST_MATRIX.md.
  4. Any mismatch is either a judge rubric gap or an engine bug -- the
     matrix tells you which.

Deviation from telemetry_implementation_guide.md, on purpose:
  Rule 2 says activities must re-raise exceptions so Temporal's retry
  policy can take over. Trap 5 catches and records the exception on the
  span but does NOT re-raise, purely so the rest of the stress suite can
  keep running in one pass. Do not copy that pattern into real activities.
"""

from __future__ import annotations

import asyncio
import os
from contextlib import contextmanager
from datetime import timedelta
from typing import Any

import mistralai.workflows as workflows
from pydantic import BaseModel

SERVICE_NAME = "judge-stress-test-worker"
WORKFLOW_NAME = "judge-stress-test"

from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode


class StressTestInput(BaseModel):
    scenario: str


class StressTestResult(BaseModel):
    scenario: str
    execution_id: str
    outcome: str


def _new_execution_id(scenario: str) -> str:
    # Isolated execution_id per scenario so group_spans_by_execution_id
    # treats each trap as its own Workflow Execution Unit, independent
    # of every other trap in the suite.
    from temporalio import workflow
    return f"stress-{scenario}-{workflow.uuid4().hex[:8]}"


@contextmanager
def stress_span(
    span_name: str,
    execution_id: str,
    activity_name: str,
    agent_name: str,
    description: str | None = None,
):
    """Standardizes the gen_ai.* attributes every trap needs, per
    telemetry_implementation_guide.md section 2."""
    span = trace.get_current_span()
    span.set_attribute("gen_ai.workflow.name", WORKFLOW_NAME)
    span.set_attribute("gen_ai.workflow.execution_id", execution_id)
    span.set_attribute("gen_ai.activity.name", activity_name)
    span.set_attribute("gen_ai.agent.name", agent_name)
    if description:
        span.set_attribute("gen_ai.workflow.description", description)
    yield span


# ---------------------------------------------------------------------------
# TRAP 1 -- Instruction Adherence -> LOW_QUALITY_SCORE
# Format contract explicitly stated, explicitly violated. No tool calls,
# no exceptions -- only a content-reading judge can catch this.
# ---------------------------------------------------------------------------
@workflows.activity(name="trap_instruction_violation", start_to_close_timeout=timedelta(seconds=30))
async def trap_instruction_violation(execution_id: str) -> dict:
    description = (
        'Return ONLY a JSON object: {"status": string, "total": float}. '
        "No prose. No markdown fences. No disclaimers or hedging."
    )
    with stress_span(
        "trap_instruction_violation_span", execution_id,
        "generate_invoice_summary", "SummaryAgent", description,
    ) as span:
        bad_output = (
            "```\nI think the total is roughly correct, but please double check "
            "with finance before relying on this.\nstatus: ok, total: about 42\n```"
        )
        span.set_attribute("gen_ai.activity.status", "SUCCESS")
        span.set_attribute("gen_ai.activity.result", bad_output)
        return {"scenario": "instruction_violation", "execution_id": execution_id}


# ---------------------------------------------------------------------------
# TRAP 2 -- Tool-Call Judge -> INVALID_ARGUMENTS
# Tool call "succeeds" (no exception) but arguments are fabricated/malformed.
# ---------------------------------------------------------------------------
@workflows.activity(name="trap_tool_invalid_arguments", start_to_close_timeout=timedelta(seconds=30))
async def trap_tool_invalid_arguments(execution_id: str) -> dict:
    description = (
        "Process a refund for order ORD-8841. Call issue_refund with a valid "
        "order_id (string) and amount (float, USD)."
    )
    with stress_span(
        "trap_tool_invalid_args_span", execution_id,
        "process_refund", "RefundAgent", description,
    ) as span:
        bad_args = {"order_id": None, "amount": "approximately forty dollars ish"}
        span.set_attribute("gen_ai.tool.name", "issue_refund")
        span.set_attribute("gen_ai.tool.call.arguments", str(bad_args))
        span.set_attribute("gen_ai.tool.result", "Error: order_id required, amount must be numeric.")
        span.set_attribute("gen_ai.activity.status", "SUCCESS")
        span.set_attribute("gen_ai.activity.result", "Attempted refund with malformed arguments.")
        return {"scenario": "tool_invalid_arguments", "execution_id": execution_id}


# ---------------------------------------------------------------------------
# TRAP 3 -- Tool-Call Judge -> WRONG_TOOL_SELECTED
# THE key fragility probe. WRONG_TOOL_SELECTED contains neither "INVALID"
# nor "UNNECESSARY", so trace_eval_service.py line 517 mislabels it as
# TOOL_UNNECESSARY_CALL. This trap should surface that bug directly.
# ---------------------------------------------------------------------------
@workflows.activity(name="trap_wrong_tool_selected", start_to_close_timeout=timedelta(seconds=30))
async def trap_wrong_tool_selected(execution_id: str) -> dict:
    description = (
        "User asked: 'What is the delivery status of order ORD-2291?' "
        "Available tools: get_order_status, cancel_subscription, issue_refund."
    )
    with stress_span(
        "trap_wrong_tool_span", execution_id,
        "handle_order_status_query", "SupportAgent", description,
    ) as span:
        span.set_attribute("gen_ai.tool.name", "cancel_subscription")
        span.set_attribute("gen_ai.tool.call.arguments", str({"subscription_id": "SUB-5521"}))
        span.set_attribute("gen_ai.tool.result", "Subscription SUB-5521 cancelled.")
        span.set_attribute("gen_ai.activity.status", "SUCCESS")
        span.set_attribute(
            "gen_ai.activity.result",
            "Cancelled subscription SUB-5521 in response to a delivery status question.",
        )
        return {"scenario": "wrong_tool_selected", "execution_id": execution_id}


# ---------------------------------------------------------------------------
# TRAP 4 -- Tool-Call Judge -> UNNECESSARY_CALL (control case)
# Verdict contains "UNNECESSARY" -> correctly maps to TOOL_UNNECESSARY_CALL.
# Run this next to Trap 3 to prove the labeling bug is specific to
# WRONG_TOOL_SELECTED, not a blanket failure of the tool-trigger branch.
# ---------------------------------------------------------------------------
@workflows.activity(name="trap_tool_unnecessary_call", start_to_close_timeout=timedelta(seconds=30))
async def trap_tool_unnecessary_call(execution_id: str) -> dict:
    description = (
        "User said: 'Thanks, that answers my question, no further action needed.' "
        "Available tools: send_followup_email, get_order_status."
    )
    with stress_span(
        "trap_unnecessary_call_span", execution_id,
        "handle_closing_message", "SupportAgent", description,
    ) as span:
        span.set_attribute("gen_ai.tool.name", "send_followup_email")
        span.set_attribute("gen_ai.tool.call.arguments", str({"template": "closing_confirmation"}))
        span.set_attribute("gen_ai.tool.result", "Email sent.")
        span.set_attribute("gen_ai.activity.status", "SUCCESS")
        span.set_attribute(
            "gen_ai.activity.result",
            "Sent a followup email even though the user indicated no further action was needed.",
        )
        return {"scenario": "tool_unnecessary_call", "execution_id": execution_id}


# ---------------------------------------------------------------------------
# TRAP 5 -- SPAN_EXCEPTION_ERROR / GOAL_FAILED precedence
# Real unhandled exception, span marked FAILED with exception attributes.
# Tests whether has_error_status and a GOAL_FAILED verdict from the Goal
# judge compose the way you expect, since has_error_status only overwrites
# primary_trigger_type if it's still at the default value.
# ---------------------------------------------------------------------------
@workflows.activity(name="trap_unhandled_exception", start_to_close_timeout=timedelta(seconds=30))
async def trap_unhandled_exception(execution_id: str) -> dict:
    description = "Calculate per-item shipping surcharge for a fully tax-exempt order of 12 items."
    with stress_span(
        "trap_unhandled_exception_span", execution_id,
        "calculate_shipping_surcharge", "BillingAgent", description,
    ) as span:
        try:
            non_exempt_count = 0
            regional_pool = 84.00
            _ = regional_pool / non_exempt_count  # deliberate ZeroDivisionError
        except Exception as exc:
            span.record_exception(exc)
            span.set_status(Status(StatusCode.ERROR, str(exc)))
            span.set_attribute("exception.type", exc.__class__.__name__)
            span.set_attribute("exception.message", str(exc))
            span.set_attribute("gen_ai.activity.status", "FAILED")
            span.set_attribute("gen_ai.activity.result", "N/A")
            # NOTE: normally you `raise exc` here per the telemetry guide.
            # Not re-raising here is a deliberate harness-only deviation --
            # see the module docstring.
            return {"scenario": "unhandled_exception", "execution_id": execution_id, "error": str(exc)}
        return {"scenario": "unhandled_exception", "execution_id": execution_id}


# ---------------------------------------------------------------------------
# TRAP 6 -- Workflow Goal Judge -> LOOP_THRASH_DETECTED
# Five handoffs alternating between two agents, identical step_name each
# time, every span reports SUCCESS. Only a judge reading the full handoff
# sequence -- not any single span -- can catch this.
# ---------------------------------------------------------------------------
@workflows.activity(name="trap_loop_thrash", start_to_close_timeout=timedelta(seconds=30))
async def trap_loop_thrash(execution_id: str) -> dict:
    description = (
        "Reconcile a $312.40 discrepancy between BillingAgent's ledger and "
        "LedgerAgent's statement for account ACC-7734."
    )
    agents = ["BillingAgent", "LedgerAgent"]
    for i in range(5):
        agent = agents[i % 2]
        with stress_span(
            f"trap_loop_thrash_span_{i}", execution_id,
            "reconcile_discrepancy", agent,
            description if i == 0 else None,
        ) as span:
            span.set_attribute("gen_ai.activity.status", "SUCCESS")
            span.set_attribute(
                "gen_ai.activity.result",
                f"Handoff {i + 1}: {agent} re-attempted reconciliation, "
                "discrepancy still unresolved, passing back.",
            )
    return {"scenario": "loop_thrash", "execution_id": execution_id}


# ---------------------------------------------------------------------------
# TRAP 7 -- silently swallowed failure, no deterministic signal at all
# Downstream call actually returned 503. The activity falls back to stale
# cached data WITHOUT surfacing that as an error, and reports SUCCESS.
# has_error_status must stay False here -- this is the negative control.
# Only a judge that actually reads gen_ai.tool.result content against the
# stated objective ("current live rate") can catch the mismatch.
# ---------------------------------------------------------------------------
@workflows.activity(name="trap_silent_degraded_success", start_to_close_timeout=timedelta(seconds=30))
async def trap_silent_degraded_success(execution_id: str) -> dict:
    description = "Fetch the current live FX rate for USD to INR and use it to convert a $1,000 invoice."
    with stress_span(
        "trap_silent_degraded_success_span", execution_id,
        "fetch_fx_rate_and_convert", "PricingAgent", description,
    ) as span:
        span.set_attribute("gen_ai.tool.name", "get_fx_rate")
        span.set_attribute("gen_ai.tool.call.arguments", str({"pair": "USD/INR"}))
        span.set_attribute(
            "gen_ai.tool.result",
            "Used cached rate 82.10 (fetched 6 months ago) after provider returned 503.",
        )
        span.set_attribute("gen_ai.activity.status", "SUCCESS")
        span.set_attribute("gen_ai.activity.result", "Converted $1,000 to INR 82,100 using rate 82.10.")
        return {"scenario": "silent_degraded_success", "execution_id": execution_id}


# ---------------------------------------------------------------------------
# TRAP 8 -- phantom success, wrong entity acted on
# Everything is green: no exception, tool call "succeeds," workflow
# completes. But the agent acted on a different order than the one
# requested. Deterministic checks see a clean pass. Only Correctness /
# Completeness / Tool judges reading content against the stated objective
# should catch this.
# ---------------------------------------------------------------------------
@workflows.activity(name="trap_phantom_success", start_to_close_timeout=timedelta(seconds=30))
async def trap_phantom_success(execution_id: str) -> dict:
    description = "User asked to cancel order ORD-1042. Confirm the cancellation."
    with stress_span(
        "trap_phantom_success_span", execution_id,
        "cancel_order", "OrderAgent", description,
    ) as span:
        span.set_attribute("gen_ai.tool.name", "process_refund")
        span.set_attribute("gen_ai.tool.call.arguments", str({"order_id": "ORD-2091", "amount": 58.40}))
        span.set_attribute("gen_ai.tool.result", "Refunded $58.40 for order ORD-2091.")
        span.set_attribute("gen_ai.activity.status", "SUCCESS")
        span.set_attribute("gen_ai.activity.result", "Order ORD-2091 refunded successfully.")
        return {"scenario": "phantom_success", "execution_id": execution_id}


# ---------------------------------------------------------------------------
# TRAP 9 -- all_judges[0]["id"] fragility probe
# Single atomic step, zero tool calls. Both the Tool judge (no tools
# present) and the Goal judge (handoff_count <= 1) get SKIPPED by
# scan_and_remediate_trace_batch's conditional-skip logic. If this trace
# still gets flagged anomalous (via the instruction-quality judge), watch
# what generate_remediation_card() does: it always calls
# evaluate_conversation(judge_id=all_judges[0]["id"], ...) regardless of
# whether that judge said anything about THIS trace, or was skipped on it.
# ---------------------------------------------------------------------------
@workflows.activity(name="trap_atomic_no_tools", start_to_close_timeout=timedelta(seconds=30))
async def trap_atomic_no_tools(execution_id: str) -> dict:
    description = "Summarize the attached contract in exactly 3 bullet points, under 15 words each."
    with stress_span(
        "trap_atomic_no_tools_span", execution_id,
        "summarize_contract", "SummaryAgent", description,
    ) as span:
        bad_output = (
            "This contract is a standard vendor services agreement covering payment terms, "
            "termination clauses, liability caps, and a lengthy indemnification section that "
            "goes well beyond 15 words and is not in bullet points at all."
        )
        span.set_attribute("gen_ai.activity.status", "SUCCESS")
        span.set_attribute("gen_ai.activity.result", bad_output)
        return {"scenario": "atomic_no_tools", "execution_id": execution_id}


STRESS_ACTIVITIES: dict[str, Any] = {
    "instruction_violation": trap_instruction_violation,
    "tool_invalid_arguments": trap_tool_invalid_arguments,
    "wrong_tool_selected": trap_wrong_tool_selected,
    "tool_unnecessary_call": trap_tool_unnecessary_call,
    "unhandled_exception": trap_unhandled_exception,
    "loop_thrash": trap_loop_thrash,
    "silent_degraded_success": trap_silent_degraded_success,
    "phantom_success": trap_phantom_success,
    "atomic_no_tools": trap_atomic_no_tools,
}


@workflows.workflow.define(
    name="judge-stress-test-suite",
    workflow_display_name="Judge Stress Test Suite",
    workflow_description="Runs a single planted-defect scenario to validate the observability judge suite.",
)
class JudgeStressTestSuite:
    @workflows.workflow.entrypoint
    async def run(self, input: StressTestInput) -> StressTestResult:
        activity_fn = STRESS_ACTIVITIES.get(input.scenario)
        if activity_fn is None:
            raise ValueError(
                f"Unknown stress scenario: {input.scenario!r}. "
                f"Valid options: {list(STRESS_ACTIVITIES)}"
            )
        execution_id = _new_execution_id(input.scenario)
        result = await activity_fn(execution_id)
        return StressTestResult(scenario=input.scenario, execution_id=execution_id, outcome=str(result))


# ---------------------------------------------------------------------------
# Worker entrypoint (src/entrypoints/worker.py pattern)
# ---------------------------------------------------------------------------
async def start_worker():
    await workflows.run_worker([JudgeStressTestSuite])


# ---------------------------------------------------------------------------
# Trigger entrypoint -- runs all 9 traps sequentially and prints execution
# ids so you can feed each one into
# trace_eval_service.scan_and_remediate_trace_batch(target_trace_id=...)
# ---------------------------------------------------------------------------
async def run_full_stress_suite() -> dict[str, str]:
    from mistralai.workflows.client import get_mistral_client
    from mistralai.extra.workflows import WorkflowEncodingConfig, configure_workflow_encoding

    client = get_mistral_client(api_key=os.environ.get("MISTRAL_API_KEY"))
    await configure_workflow_encoding(WorkflowEncodingConfig(), client=client)

    execution_ids: dict[str, str] = {}
    for scenario in STRESS_ACTIVITIES:
        result = await client.workflows.execute_workflow_and_wait_async(
            workflow_identifier="judge-stress-test-suite",
            input={"scenario": scenario},
            deployment_name=os.environ.get("DEPLOYMENT_NAME", "default"),
        )
        exec_id = result.get("execution_id") if isinstance(result, dict) else getattr(result, "execution_id", None)
        execution_ids[scenario] = exec_id
        print(f"[{scenario}] execution_id = {exec_id}")

    print(
        "\nCross-reference each execution_id against STRESS_TEST_MATRIX.md, "
        "either one at a time via scan_and_remediate_trace_batch(target_trace_id=...) "
        "or in a single batch scan."
    )
    return execution_ids


if __name__ == "__main__":
    asyncio.run(run_full_stress_suite())
