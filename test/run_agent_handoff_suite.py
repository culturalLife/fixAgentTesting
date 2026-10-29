"""
run_agent_handoff_suite.py
--------------------------
Executes the 10 multi-agent handoff instruction defect scenarios, captures their
hierarchical OpenTelemetry traces, exports them directly to the Mistral Observability
API, and saves the trace IDs to LATEST_AGENT_HANDOFF_TRACE_IDS.json.
"""

from __future__ import annotations
import asyncio
import json
import os
import sys
import uuid
from pathlib import Path

# Configure Mistral Cloud Observability Environment
os.environ["MISTRAL_API_KEY"] = "ftus6TfrrzL6EbJAHCsb3ixd2U5ROYah"
os.environ["MISTRAL_BASE_URL"] = "https://api.f418c72b5f65.dc.mistral.ai"
os.environ["MISTRAL_OTLP_TRACES_ENDPOINT"] = "https://api.f418c72b5f65.dc.mistral.ai/telemetry/v1/traces"

# Path setups
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.telemetry import setup_telemetry_client
from agent_handoff_instruction_scenarios import HANDOFF_INSTRUCTION_SCENARIOS


async def main():
    print("=" * 80)
    print("  MULTI-AGENT HANDOFF INSTRUCTION DEFECT SUITE (MISTRAL OBSERVABILITY)")
    print("=" * 80)
    print(f"Targeting Mistral Endpoint: {os.environ['MISTRAL_OTLP_TRACES_ENDPOINT']}")
    print(f"Number of Scenarios: {len(HANDOFF_INSTRUCTION_SCENARIOS)}\n")

    # Initialize Mistral client & OpenTelemetry tracer
    client, tracer = setup_telemetry_client(service_name="agent-handoff-instruction-worker")

    results = []

    for name, scenario_fn in HANDOFF_INSTRUCTION_SCENARIOS.items():
        execution_id = f"handoff-err-{uuid.uuid4().hex[:8]}"
        print(f">> EXECUTING SCENARIO: {name}")
        try:
            res = await scenario_fn(tracer, execution_id)
            results.append(res)
            print(f"  Trace ID:    {res['trace_id']}")
            print(f"  Status:      {res['status']}")
            print(f"  Defect Type: {res['defect_type']}")
            print(f"  Root Cause:  {res['root_cause']}\n")
            # Brief pause between scenarios for clean span batching
            await asyncio.sleep(0.5)
        except Exception as e:
            print(f"  [ERROR executing {name}]: {e}\n")

    # Write trace manifest
    manifest_path = ROOT_DIR / "test" / "LATEST_AGENT_HANDOFF_TRACE_IDS.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("=" * 80)
    print("  HANDOFF DEFECT SUITE COMPLETE -- RECORDED TRACE IDS")
    print("=" * 80)
    print(f"Saved {len(results)} execution trace records to:\n  {manifest_path}\n")

    # Force flush telemetry before terminating process
    from opentelemetry import trace as otel_trace
    provider = otel_trace.get_tracer_provider()
    if hasattr(provider, "force_flush"):
        print("  Flushing OpenTelemetry trace batches to Mistral cloud...")
        provider.force_flush()
        print("  ✓ All traces successfully flushed to Mistral Observability.")


if __name__ == "__main__":
    asyncio.run(main())

