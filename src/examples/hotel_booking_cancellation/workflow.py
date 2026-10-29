from __future__ import annotations
import mistralai.workflows as workflows
from mistralai.workflows import workflow

with workflow.unsafe.imports_passed_through():
    from .activities import (
        classify_cancellation_request,
        execute_cancellation_tools,
        generate_final_report,
    )

from .models import (
    HotelCancellationInput,
    CancellationReport,
)

from datetime import timedelta
import temporalio.workflow as temporal_workflow

@workflows.workflow.define(
    name="hotel-cancellation-workflow",
    workflow_display_name="Hotel Booking Cancellation Workflow with Tool Issues",
    workflow_description=(
        "Processes hotel cancellations but intentionally encounters issues "
        "in tool execution to demonstrate error tracing."
    ),
)
class HotelCancellationWorkflow:
    @workflows.workflow.entrypoint
    async def run(self, request: HotelCancellationInput) -> CancellationReport:
        classification = await classify_cancellation_request(request)

        tool_results = await execute_cancellation_tools(request, classification)

        report = await generate_final_report(request, classification, tool_results)

        return report
