"""
run_cost_suite.py
-----------------
CLI Runner for Cost & Token Behavior Multi-Agent Workflow Suite.

Executes realistic enterprise agent workflows containing organic real-world defects
(schema mismatches, redundant queries, missing keys, unpruned context stuffing,
oversized model tiers, monotonic context growth, tool output amplification,
and prefix-cache invalidation), collecting OpenTelemetry spans and uploading
complete traces directly to the Mistral Observability Traces API:
  https://api.mistral.ai/telemetry/v1/traces (or MISTRAL_OTLP_TRACES_ENDPOINT)

Usage:
  python run_cost_suite.py                             # Run all 9 workflows
  python run_cost_suite.py --scenario retries          # Run Vendor Invoice Reconciliation (RUNAWAY_RETRIES)
  python run_cost_suite.py --scenario redundant        # Run Dual Compliance Screening (REDUNDANT_CALLS)
  python run_cost_suite.py --scenario failed           # Run Mortgage Underwriting Pipeline (FAILED_EXECUTIONS)
  python run_cost_suite.py --scenario expensive        # Run Commercial Lease Audit (EXPENSIVE_PROMPT)
  python run_cost_suite.py --scenario oversize         # Run Support Ticket Triage (OVER_PROVISIONED_MODEL)
  python run_cost_suite.py --scenario leak             # Run VIP Travel Concierge Dialogue (CONTEXT_GROWTH_LEAK)
  python run_cost_suite.py --scenario amplification    # Run Retail Supply Chain Audit (TOOL_CALL_AMPLIFICATION)
  python run_cost_suite.py --scenario collapse         # Run Portfolio Performance Reporting (CACHE_COLLAPSE)
  python run_cost_suite.py --scenario baseline         # Run High-Efficiency Claims Baseline (CLEAN_BASELINE)
  python run_cost_suite.py --from-scenario 4           # Start from scenario #4 onwards
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Setup paths and environment
CURRENT_DIR = Path(__file__).resolve().parent
ROOT_DIR = CURRENT_DIR.parent
sys.path.insert(0, str(CURRENT_DIR))
sys.path.insert(0, str(ROOT_DIR))

from dotenv import load_dotenv
env_file = ROOT_DIR / ".env"
if env_file.exists():
    load_dotenv(env_file)
else:
    load_dotenv()

from cost_telemetry import setup_cost_tracer, flush_all_traces
from agent_handoff_scenarios import (
    run_vendor_invoice_reconciliation_scenario,
    run_dual_compliance_screening_scenario,
    run_mortgage_underwriting_settlement_scenario,
    run_commercial_lease_due_diligence_scenario,
    run_support_ticket_triage_scenario,
    run_concierge_service_dialogue_scenario,
    run_warehouse_inventory_audit_scenario,
    run_portfolio_performance_reporting_scenario,
    run_clean_optimized_claims_scenario,
    ScenarioResult,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("cost_suite")

SCENARIOS_MAP = {
    "retries": ("Vendor Invoice Reconciliation", "RUNAWAY_RETRIES", run_vendor_invoice_reconciliation_scenario),
    "redundant": ("Dual Compliance Screening", "REDUNDANT_CALLS", run_dual_compliance_screening_scenario),
    "failed": ("Mortgage Underwriting & Settlement", "FAILED_EXECUTIONS", run_mortgage_underwriting_settlement_scenario),
    "expensive": ("Commercial Lease Due Diligence", "EXPENSIVE_PROMPT", run_commercial_lease_due_diligence_scenario),
    "oversize": ("Customer Support Ticket Triage", "OVER_PROVISIONED_MODEL", run_support_ticket_triage_scenario),
    "leak": ("VIP Travel Concierge Dialogue", "CONTEXT_GROWTH_LEAK", run_concierge_service_dialogue_scenario),
    "amplification": ("Retail Supply Chain Inventory Audit", "TOOL_CALL_AMPLIFICATION", run_warehouse_inventory_audit_scenario),
    "collapse": ("Quantitative Portfolio Strategy Reporting", "CACHE_COLLAPSE", run_portfolio_performance_reporting_scenario),
    "baseline": ("High-Efficiency Claims Baseline", "CLEAN_OPTIMAL_BASELINE", run_clean_optimized_claims_scenario),
}


async def main():
    parser = argparse.ArgumentParser(description="Cost & Token Behavior Multi-Agent Workflow Benchmark Suite")
    parser.add_argument(
        "--scenario",
        type=str,
        choices=list(SCENARIOS_MAP.keys()) + ["all"],
        default="all",
        help="Specific scenario key to execute or 'all' (default: all)",
    )
    parser.add_argument(
        "--from-scenario",
        type=int,
        default=None,
        help="Start execution from 1-indexed scenario number (e.g. 4)",
    )
    args = parser.parse_args()

    endpoint = os.getenv("MISTRAL_OTLP_TRACES_ENDPOINT", "https://api.mistral.ai/telemetry/v1/traces")
    print("=" * 88)
    print("       MISTRAL OBSERVABILITY -- COST & TOKEN BEHAVIOR WORKFLOW SUITE")
    print("=" * 88)
    print(f" Target OTLP Endpoint : {endpoint}")
    print(f" Service Name          : agent-cost-token-suite")
    print(f" Mode                  : Standalone Agent Handoffs & Production Workflows")
    if args.from_scenario:
        print(f" Filter                : Running from Scenario #{args.from_scenario} onwards")
    print("=" * 88 + "\n")

    client, tracer = setup_cost_tracer(service_name="agent-cost-token-suite")

    all_items = list(SCENARIOS_MAP.items())
    if args.from_scenario is not None:
        start_idx = max(0, args.from_scenario - 1)
        selected_scenarios = all_items[start_idx:]
    elif args.scenario != "all":
        selected_scenarios = [(args.scenario, SCENARIOS_MAP[args.scenario])]
    else:
        selected_scenarios = all_items

    results: list[ScenarioResult] = []

    for key, (display_name, anomaly_type, runner_fn) in selected_scenarios:
        idx = [k for k, _ in all_items].index(key) + 1
        print(f">> [RUNNING #{idx}] {display_name} ({key})")
        print(f"   Target Anomaly : {anomaly_type}")
        try:
            res: ScenarioResult = await runner_fn(client, tracer)
            results.append(res)
            print(f"   [DONE]  Execution ID : {res.execution_id}")
            print(f"           OTel Trace ID: {res.trace_id}")
            print(f"           Status       : {res.status}")
            print(f"           Duration     : {res.total_duration_ms:.1f} ms ({res.total_duration_ms/1000:.2f}s)")
            print(f"           Summary      : {res.summary}\n")
            # Brief pause between scenario runs for clear telemetry ingestion boundary
            await asyncio.sleep(1.0)
        except Exception as exc:
            logger.error(f"Error executing scenario '{key}': {exc}", exc_info=True)

    # Synchronously flush all traces to the Mistral OTLP endpoint
    print(">> Synchronously flushing and uploading OpenTelemetry traces to Mistral Observability...")
    flush_success = flush_all_traces(tracer, timeout_ms=30000)
    if flush_success:
        print("   [OK] All spans successfully uploaded to Mistral Observability API.")
    else:
        print("   [WARNING] Telemetry flush completed with fallback.")

    # Save / merge manifest of trace IDs
    manifest_path = CURRENT_DIR / "LATEST_COST_TRACE_IDS.json"
    existing_manifest = []
    if manifest_path.exists():
        try:
            with open(manifest_path, "r", encoding="utf-8") as fp:
                existing_manifest = json.load(fp)
                if not isinstance(existing_manifest, list):
                    existing_manifest = []
        except Exception:
            existing_manifest = []

    existing_trace_ids = {entry.get("trace_id") for entry in existing_manifest if isinstance(entry, dict)}
    new_entries = []
    for r in results:
        entry = {
            "scenario_name": r.scenario_name,
            "anomaly_type": r.anomaly_type,
            "execution_id": r.execution_id,
            "trace_id": r.trace_id,
            "status": r.status,
            "duration_ms": round(r.total_duration_ms, 2),
            "step_count": r.step_count,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "summary": r.summary,
            "details": r.details,
        }
        if r.trace_id not in existing_trace_ids:
            new_entries.append(entry)
        else:
            # Update existing entry with latest run details
            for idx, existing_entry in enumerate(existing_manifest):
                if existing_entry.get("trace_id") == r.trace_id:
                    existing_manifest[idx] = entry
                    break

    combined_manifest = existing_manifest + new_entries
    with open(manifest_path, "w", encoding="utf-8") as fp:
        json.dump(combined_manifest, fp, indent=2)
    print(f">> Manifest updated: {manifest_path} ({len(combined_manifest)} total traces recorded)\n")

    # Display Execution Summary Table
    print("=" * 88)
    print("                         WORKFLOW EXECUTION SUMMARY")
    print("=" * 88)
    print(f"{'#':<3} {'Scenario Name':<34} {'Anomaly Type':<24} {'Status':<8} {'Trace ID'}")
    print("-" * 88)
    for i, r in enumerate(results, 1):
        print(f"{i:<3} {r.scenario_name[:33]:<34} {r.anomaly_type[:23]:<24} {r.status:<8} {r.trace_id}")
    print("=" * 88)
    print("\nDeep Dive in Observability UI:")
    for r in results:
        print(f"  • {r.scenario_name}:")
        print(f"    - Cost Deep Dive : http://localhost:5173/cost?trace_id={r.trace_id}")
        print(f"    - Token Deep Dive: http://localhost:5173/tokens?trace_id={r.trace_id}")
    print("=" * 88 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
