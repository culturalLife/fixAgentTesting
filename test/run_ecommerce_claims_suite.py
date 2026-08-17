"""
run_ecommerce_claims_suite.py
------------------------------
Executes the multi-agent Ecommerce Claims Triage Workflow across 4 realistic
production scenarios and logs their Mistral Observability OpenTelemetry trace IDs.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import uuid
from pathlib import Path
from dotenv import load_dotenv

# Path setups
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))
load_dotenv(ROOT_DIR / ".env")

from src.telemetry import setup_telemetry_client
from src.examples.ecommerce_claims.models import CustomerClaimInput
from src.examples.ecommerce_claims.activities import (
    intake_and_classify_claim,
    verify_order_and_inventory_tools,
    evaluate_compliance_and_policy,
    generate_customer_resolution,
)

from opentelemetry import trace


async def run_claim_scenario(scenario_file: Path, client, tracer) -> dict:
    with open(scenario_file, "r", encoding="utf-8-sig") as f:
        claim_data = json.load(f)

    claim = CustomerClaimInput(**claim_data)
    scenario_name = scenario_file.stem
    execution_id = f"ecom-{scenario_name}-{uuid.uuid4().hex[:8]}"

    print("\n" + "=" * 80)
    print(f">> EXECUTING SCENARIO: {scenario_name}")
    print(f"  Execution ID : {execution_id}")
    print(f"  Claim ID     : {claim.claim_id} (${claim.claim_amount})")
    print("=" * 80)

    # Root workflow span to anchor the OTel trace tree
    with tracer.start_as_current_span("ecommerce_claims_triage_workflow_span") as root_span:
        root_span.set_attribute("gen_ai.workflow.name", "ecommerce-claims-triage-workflow")
        root_span.set_attribute("gen_ai.workflow.execution_id", execution_id)
        root_span.set_attribute(
            "gen_ai.workflow.description",
            "Multi-agent customer support claim triage, inventory lookup, compliance verification, and resolution drafting."
        )

        trace_id_hex = format(root_span.get_span_context().trace_id, "032x")
        print(f"  OTel Trace ID: {trace_id_hex}")

        # Step 1: Intake & Classification
        print("  [Step 1/4] Running Intake & Classification Agent...")
        classification = await intake_and_classify_claim(claim)
        print(f"             Category: {classification.claim_category.value}, Urgency: {classification.urgency.value}")

        # Step 2: Tools Verification
        print("  [Step 2/4] Running Verification & Tool Dispatch Agent...")
        tool_results = await verify_order_and_inventory_tools(claim, classification)
        print(f"             Executed {len(tool_results)} tool calls: {[t.tool_name for t in tool_results]}")

        # Step 3: Compliance & Reasoning
        print("  [Step 3/4] Running Compliance & Policy Reasoning Agent...")
        compliance = await evaluate_compliance_and_policy(claim, classification, tool_results)
        print(f"             Eligible: {compliance.is_eligible}, Risk Score: {compliance.risk_score:.2f}")

        # Step 4: Final Resolution
        print("  [Step 4/4] Running Customer Resolution Agent...")
        resolution = await generate_customer_resolution(claim, classification, compliance)
        print(f"             Final Status: {resolution.status}, Action: {resolution.action_taken[:60]}...")

        root_span.set_attribute("gen_ai.activity.status", "SUCCESS")
        root_span.set_attribute("gen_ai.activity.result", resolution.model_dump_json())

        return {
            "scenario": scenario_name,
            "execution_id": execution_id,
            "trace_id": trace_id_hex,
            "status": resolution.status,
            "approved_amount": resolution.approved_amount,
            "response_preview": resolution.customer_facing_response[:100] + "...",
        }


async def main():
    print("=========================================================================")
    print("  ECOMMERCE MULTI-AGENT CLAIMS WORKFLOW SUITE (MISTRAL TELEMETRY)")
    print("=========================================================================")

    client, tracer = setup_telemetry_client(service_name="ecommerce-claims-worker")
    sample_dir = ROOT_DIR / "src" / "examples" / "ecommerce_claims" / "sample_data"
    scenario_files = sorted(sample_dir.glob("*.json"))

    if not scenario_files:
        print(f"No scenario JSON files found in {sample_dir}")
        return

    results = []
    for s_file in scenario_files:
        try:
            res = await run_claim_scenario(s_file, client, tracer)
            results.append(res)
            # Give short breather between runs for clean telemetry flushing
            await asyncio.sleep(2)
        except Exception as e:
            print(f"  [ERROR in {s_file.stem}]: {e}")

    # Output trace manifest
    output_manifest_path = ROOT_DIR / "test" / "LATEST_ECOMMERCE_TRACE_IDS.json"
    with open(output_manifest_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 80)
    print("  WORKFLOW SUITE COMPLETE -- RECORDED TRACE IDS")
    print("=" * 80)
    print(f"Saved {len(results)} execution trace records to:\n  {output_manifest_path}\n")
    for r in results:
        print(f"* Scenario: {r['scenario']}")
        print(f"  Trace ID    : {r['trace_id']}")
        print(f"  Execution ID: {r['execution_id']}")
        print(f"  Status      : {r['status']}\n")


if __name__ == "__main__":
    asyncio.run(main())
