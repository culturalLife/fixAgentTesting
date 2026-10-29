"""
run_ecommerce_stress_suite.py
------------------------------
Executes all 10 planted-defect scenarios in the E-Commerce Claims stress test suite
and logs their trace IDs for the observability judge to evaluate.
"""

from __future__ import annotations
import asyncio
import json
import sys
import uuid
import os
from pathlib import Path

os.environ["MISTRAL_API_KEY"] = "ftus6TfrrzL6EbJAHCsb3ixd2U5ROYah"
os.environ["MISTRAL_BASE_URL"] = "https://api.f418c72b5f65.dc.mistral.ai"
os.environ["MISTRAL_OTLP_TRACES_ENDPOINT"] = "https://api.f418c72b5f65.dc.mistral.ai/telemetry/v1/traces"

# Path setups
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.telemetry import setup_telemetry_client
from ecommerce_stress_scenarios import STRESS_SCENARIOS


async def main():
    print("=========================================================================")
    print("  ECOMMERCE CLAIMS STRESS TEST SUITE (MISTRAL TELEMETRY)")
    print("=========================================================================")

    # Use the existing telemetry setup (which sets up tracer based on env vars)
    client, tracer = setup_telemetry_client(service_name="ecommerce-claims-stress-worker")
    
    results = []
    
    for scenario_name, scenario_fn in STRESS_SCENARIOS.items():
        execution_id = f"ecom-stress-{uuid.uuid4().hex[:8]}"
        if "collision" in scenario_name:
            execution_id = "ecom-stress-00a1b2c3"
            
        print(f"\n>> EXECUTING SCENARIO: {scenario_name}")
        try:
            res = await scenario_fn(tracer, execution_id)
            results.append(res)
            print(f"  Trace ID: {res['trace_id']}")
            print(f"  Status: {res['status']}")
            # Give short breather between runs for clean telemetry flushing
            await asyncio.sleep(1)
        except Exception as e:
            print(f"  [ERROR in {scenario_name}]: {e}")

    # Output trace manifest
    output_manifest_path = ROOT_DIR / "test" / "LATEST_ECOMMERCE_STRESS_TRACE_IDS.json"
    with open(output_manifest_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 80)
    print("  STRESS SUITE COMPLETE -- RECORDED TRACE IDS")
    print("=" * 80)
    print(f"Saved {len(results)} execution trace records to:\n  {output_manifest_path}\n")

    # Force flush telemetry before exiting
    from opentelemetry import trace as otel_trace
    provider = otel_trace.get_tracer_provider()
    if hasattr(provider, "force_flush"):
        provider.force_flush()
        print("  Telemetry traces force flushed to Mistral cloud.")

if __name__ == "__main__":
    asyncio.run(main())
