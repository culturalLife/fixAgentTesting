"""
run_beta_agents_suite.py
------------------------
Executes the Mistral Beta Agents & Conversations test suite against your dedicated
Mistral environment.

1. Dynamically registers agents via client.beta.agents.create()
2. Wires handoffs across agents
3. Executes stateful conversations (client.beta.conversations.start with handoff_execution='server')
4. Executes stateless agent calls (client.agents.complete)
5. Captures OpenTelemetry traces and exports them to Mistral Observability
6. Cleans up test agents and conversations
7. Outputs manifest to LATEST_BETA_AGENTS_TRACE_IDS.json
"""

from __future__ import annotations
import asyncio
import json
import os
import sys
import uuid
from pathlib import Path

# Force UTF-8 on Windows stdout if supported
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Dedicated Mistral Environment Configuration
os.environ["MISTRAL_API_KEY"] = "ftus6TfrrzL6EbJAHCsb3ixd2U5ROYah"
os.environ["MISTRAL_BASE_URL"] = "https://api.f418c72b5f65.dc.mistral.ai"
os.environ["MISTRAL_OTLP_TRACES_ENDPOINT"] = "https://api.f418c72b5f65.dc.mistral.ai/telemetry/v1/traces"

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.telemetry import setup_telemetry_client
from beta_agents_workflow import BetaAgentRegistry, BETA_AGENT_SCENARIOS


async def main():
    print("=" * 80)
    print("  MISTRAL BETA AGENTS & CONVERSATIONS SUITE (MISTRAL OBSERVABILITY)")
    print("=" * 80)
    print(f"Targeting Mistral Dedicated Endpoint: {os.environ['MISTRAL_BASE_URL']}")
    print(f"OTLP Telemetry Ingestion:            {os.environ['MISTRAL_OTLP_TRACES_ENDPOINT']}")
    print(f"Number of Scenarios:                 {len(BETA_AGENT_SCENARIOS)}\n")

    # Initialize Mistral client & OpenTelemetry tracer
    client, tracer = setup_telemetry_client(service_name="mistral-beta-agents-worker")
    registry = BetaAgentRegistry(client)

    results = []

    try:
        for name, scenario_fn in BETA_AGENT_SCENARIOS.items():
            execution_id = f"beta-agent-{uuid.uuid4().hex[:8]}"
            print(f">> EXECUTING SCENARIO: {name}")
            try:
                res = await scenario_fn(tracer, client, registry, execution_id)
                results.append(res)
                print(f"  Trace ID:        {res['trace_id']}")
                print(f"  Status:          {res['status']}")
                print(f"  Defect Type:     {res['defect_type']}")
                if "conversation_id" in res:
                    print(f"  Conversation ID: {res['conversation_id']}")
                if "intake_agent_id" in res:
                    print(f"  Intake Agent:    {res['intake_agent_id']}")
                print(f"  Output Summary:  {res['output_summary']}\n")
                await asyncio.sleep(1)
            except Exception as e:
                print(f"  [ERROR executing {name}]: {e}\n")

    finally:
        print("Cleaning up registered test agents and conversations...")
        registry.cleanup()
        print("  [OK] Cleanup complete.")

    # Write trace manifest
    manifest_path = ROOT_DIR / "test" / "LATEST_BETA_AGENTS_TRACE_IDS.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 80)
    print("  BETA AGENTS SUITE COMPLETE -- RECORDED TRACE IDS")
    print("=" * 80)
    print(f"Saved {len(results)} execution trace records to:\n  {manifest_path}\n")

    # Force flush OpenTelemetry trace batches
    from opentelemetry import trace as otel_trace
    provider = otel_trace.get_tracer_provider()
    if hasattr(provider, "force_flush"):
        print("  Flushing OpenTelemetry trace batches to Mistral cloud...")
        provider.force_flush()
        print("  [OK] All traces successfully flushed to Mistral Observability.")


if __name__ == "__main__":
    asyncio.run(main())
