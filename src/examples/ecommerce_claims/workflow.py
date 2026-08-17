from __future__ import annotations
import mistralai.workflows as workflows
from mistralai.workflows import workflow

with workflow.unsafe.imports_passed_through():
    from .activities import (
        intake_and_classify_claim,
        verify_order_and_inventory_tools,
        evaluate_compliance_and_policy,
        generate_customer_resolution,
    )

from .models import (
    CustomerClaimInput,
    ResolutionReport,
)


@workflows.workflow.define(
    name="ecommerce-claims-triage-workflow",
    workflow_display_name="E-commerce Claims Triage and Support Multi-Agent Workflow",
    workflow_description=(
        "Triages customer e-commerce claims: multi-agent intake classification, "
        "order and warehouse inventory tool dispatch, compliance reasoning, "
        "and automated final resolution drafting."
    ),
)
class EcommerceClaimsTriageWorkflow:
    @workflows.workflow.entrypoint
    async def run(self, claim: CustomerClaimInput) -> ResolutionReport:
        # Step 1: Intake & Classification Agent
        classification = await workflow.execute_activity(
            intake_and_classify_claim,
            claim,
            start_to_close_timeout=intake_and_classify_claim.start_to_close_timeout,
        )

        # Step 2: Verification & Tool Dispatch Agent
        tool_results = await workflow.execute_activity(
            verify_order_and_inventory_tools,
            claim,
            classification,
            start_to_close_timeout=verify_order_and_inventory_tools.start_to_close_timeout,
        )

        # Step 3: Compliance & Reasoning Agent
        compliance = await workflow.execute_activity(
            evaluate_compliance_and_policy,
            claim,
            classification,
            tool_results,
            start_to_close_timeout=evaluate_compliance_and_policy.start_to_close_timeout,
        )

        # Step 4: Customer Resolution & Final Decision Agent
        resolution = await workflow.execute_activity(
            generate_customer_resolution,
            claim,
            classification,
            compliance,
            start_to_close_timeout=generate_customer_resolution.start_to_close_timeout,
        )

        return resolution
