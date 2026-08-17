"""
judge_mini_stress_test.py
--------------------------
One workflow. One execution. Three deliberately planted defects, run as
three sequential steps in the SAME trace, so the judge suite evaluates
them together the way it would a real multi-agent run with mixed
correct/incorrect steps.

Step 1 -- wrong_tool_selected: probes the trigger_type mislabeling bug.
Step 2 -- unhandled_exception: probes has_error_status vs GOAL_FAILED precedence.
Step 3 -- silent_degraded_success: probes whether judges catch a "clean"
          trace that's substantively wrong.

Run this, then run trace_eval_service.scan_and_remediate_trace_batch()
against the resulting trace. One execution_id, one batch_trace_reports
entry, one trigger_type -- but the evaluation_summary list should show
findings from at least three different judges. Compare against
EXPECTED_RESULTS at the bottom.

Deviation from telemetry_implementation_guide.md, on purpose: step 2
catches its exception and records it on the span but does NOT re-raise,
so steps 3 still runs in the same execution. In real activities, always
re-raise per the guide.
"""

from __future__ import annotations

import asyncio
import os
import uuid
from contextlib import contextmanager
from datetime import timedelta

import mistralai.workflows as workflows
from pydantic import BaseModel

SERVICE_NAME = "judge-mini-stress-test-worker"
WORKFLOW_NAME = "judge-mini-stress-test"

def _get_tracer():
    from telemetry import get_telemetry_tracer_instance
    return get_telemetry_tracer_instance(SERVICE_NAME)



class MiniStressResult(BaseModel):
    execution_id: str
    steps_run: list[str]


from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode

@contextmanager
def stress_span(span_name: str, execution_id: str, activity_name: str, agent_name: str, description: str | None = None):
    span = trace.get_current_span()
    # span.update_name(span_name) # optional: rename to make it look nice in trace UI
    span.set_attribute("gen_ai.workflow.name", WORKFLOW_NAME)
    span.set_attribute("gen_ai.workflow.execution_id", execution_id)
    span.set_attribute("gen_ai.activity.name", activity_name)
    span.set_attribute("gen_ai.agent.name", agent_name)
    if description:
        span.set_attribute("gen_ai.activity.description", description)
    try:
        yield span
    except Exception as exc:
        span.record_exception(exc)
        span.set_status(Status(StatusCode.ERROR, str(exc)))
        span.set_attribute("exception.type", exc.__class__.__name__)
        span.set_attribute("exception.message", str(exc))
        span.set_attribute("gen_ai.activity.status", "FAILED")
        span.set_attribute("gen_ai.activity.result", "N/A")
        raise exc


# --- Step 1: wrong tool selected -------------------------------------------
@workflows.activity(name="step_wrong_tool_selected", start_to_close_timeout=timedelta(seconds=30))
async def step_wrong_tool_selected(execution_id: str) -> None:
    description = (
        "User asked: 'What is the delivery status of order ORD-2291?' "
        "Available tools: get_order_status, cancel_subscription, issue_refund."
    )
    with stress_span("step1_wrong_tool_span", execution_id, "handle_order_status_query", "SupportAgent", description) as span:
        span.set_attribute("gen_ai.tool.name", "cancel_subscription")
        span.set_attribute("gen_ai.tool.call.arguments", str({"subscription_id": "SUB-5521"}))
        span.set_attribute("gen_ai.tool.result", "Subscription SUB-5521 cancelled.")
        span.set_attribute("gen_ai.activity.status", "SUCCESS")
        span.set_attribute(
            "gen_ai.activity.result",
            "Cancelled subscription SUB-5521 in response to a delivery status question.",
        )


# --- Step 2: unhandled exception --------------------------------------------
@workflows.activity(name="step_unhandled_exception", start_to_close_timeout=timedelta(seconds=30))
async def step_unhandled_exception(execution_id: str) -> None:
    description = "Calculate per-item shipping surcharge for a fully tax-exempt order of 12 items."
    with stress_span("step2_exception_span", execution_id, "calculate_shipping_surcharge", "BillingAgent", description) as span:
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
            # Deliberately not re-raising here -- see module docstring.
            return


# --- Step 3: silent degraded success ----------------------------------------
@workflows.activity(name="step_silent_degraded_success", start_to_close_timeout=timedelta(seconds=30))
async def step_silent_degraded_success(execution_id: str) -> None:
    description = "Fetch the current live FX rate for USD to INR and use it to convert a $1,000 invoice."
    with stress_span("step3_degraded_span", execution_id, "fetch_fx_rate_and_convert", "PricingAgent", description) as span:
        span.set_attribute("gen_ai.tool.name", "get_fx_rate")
        span.set_attribute("gen_ai.tool.call.arguments", str({"pair": "USD/INR"}))
        span.set_attribute(
            "gen_ai.tool.result",
            "Used cached rate 82.10 (fetched 6 months ago) after provider returned 503.",
        )
        span.set_attribute("gen_ai.activity.status", "SUCCESS")
        span.set_attribute("gen_ai.activity.result", "Converted $1,000 to INR 82,100 using rate 82.10.")


@workflows.workflow.define(
    name="judge-mini-stress-test",
    workflow_display_name="Judge Mini Stress Test",
    workflow_description="Three planted defects in one execution, to validate the observability judge suite.",
)
class JudgeMiniStressTest:
    @workflows.workflow.entrypoint
    async def run(self) -> MiniStressResult:
        from temporalio import workflow
        execution_id = f"mini-stress-{workflow.uuid4().hex[:8]}"
        await step_wrong_tool_selected(execution_id)
        await step_unhandled_exception(execution_id)
        await step_silent_degraded_success(execution_id)
        return MiniStressResult(
            execution_id=execution_id,
            steps_run=["wrong_tool_selected", "unhandled_exception", "silent_degraded_success"],
        )


async def start_worker():
    await workflows.run_worker([JudgeMiniStressTest])


async def trigger_run() -> str:
    from mistralai.workflows.client import get_mistral_client
    from mistralai.extra.workflows import WorkflowEncodingConfig, configure_workflow_encoding

    client = get_mistral_client(api_key=os.environ.get("MISTRAL_API_KEY"))
    await configure_workflow_encoding(WorkflowEncodingConfig(), client=client)

    result = await client.workflows.execute_workflow_and_wait_async(
        workflow_identifier="judge-mini-stress-test",
        input={},
        deployment_name=os.environ.get("DEPLOYMENT_NAME", "default"),
    )
    execution_id = result.get("execution_id") if isinstance(result, dict) else getattr(result, "execution_id", None)
    print(f"execution_id = {execution_id}")
    print("Feed this into scan_and_remediate_trace_batch(target_trace_id=<execution_id>)")
    return execution_id


if __name__ == "__main__":
    asyncio.run(trigger_run())


# -----------------------------------------------------------------------------
# EXPECTED_RESULTS -- one execution, one batch_trace_reports entry
# -----------------------------------------------------------------------------
# is_anomaly:          True
# trigger_type:        depends on judge execution order and which fires first
#                       (this is exactly what step 2 is designed to expose --
#                       see STRESS_TEST_MATRIX.md if you want the full breakdown)
# evaluation_summary:  should contain at minimum
#   - Tool judge verdict on step 1 -> expect "WRONG_TOOL_SELECTED"
#     -> check whether trigger_type shows TOOL_UNNECESSARY_CALL instead
#        (that's the mislabeling bug, not a judge failure)
#   - Goal/Workflow judge verdict on the full 3-step sequence -> expect
#     GOAL_FAILED given step 2's crash
#   - Correctness/Completeness judge verdict on step 3 -> if this does NOT
#     flag anything, that's a real gap: nothing else in the trace will
#     surface the stale-FX-rate problem, since step 3's span reports clean
#     SUCCESS with no exception
