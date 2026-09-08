"""
run_latency_suite.py
--------------------
CLI Runner for Agent Handoff Latency Error Benchmarks.

Executes 7 distinct agent handoff latency scenarios, collects OpenTelemetry spans,
and synchronously uploads traces to the Mistral Observability Traces API:
  https://api.mistral.ai/telemetry/v1/traces (or MISTRAL_OTLP_TRACES_ENDPOINT)

Usage:
  python run_latency_suite.py                     # Run all 7 scenarios
  python run_latency_suite.py --scenario cascade  # Run specific scenario
  python run_latency_suite.py --scenario loop
  python run_latency_suite.py --scenario tool
  python run_latency_suite.py --scenario bloat
  python run_latency_suite.py --scenario fanout
  python run_latency_suite.py --scenario retry
  python run_latency_suite.py --scenario baseline
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import time
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

from latency_telemetry import setup_latency_tracer, flush_all_traces
from agent_handoff_scenarios import (
    run_cascading_delay_scenario,
    run_ping_pong_loop_scenario,
    run_blocking_tool_stall_scenario,
    run_token_bloat_scenario,
    run_fanout_straggler_scenario,
    run_retry_storm_scenario,
    run_clean_baseline_scenario,
    run_model_oversizing_scenario,
    run_unconstrained_generation_scenario,
    run_sequential_io_waterfall_scenario,
    run_uncached_repeated_io_scenario,
    run_heavy_vision_payload_scenario,
    ScenarioResult,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("latency_suite")


SCENARIOS_MAP = {
    "cascade": ("Cascading Handoff Delay", run_cascading_delay_scenario),
    "loop": ("Ping-Pong Loop Thrash", run_ping_pong_loop_scenario),
    "tool": ("Blocking Tool Stall", run_blocking_tool_stall_scenario),
    "bloat": ("Token Bloat Latency", run_token_bloat_scenario),
    "fanout": ("Fanout Straggler Delay", run_fanout_straggler_scenario),
    "retry": ("Retry Storm Backoff", run_retry_storm_scenario),
    "oversize": ("Model Oversizing LLM Delay", run_model_oversizing_scenario),
    "unconstrained": ("Unconstrained Generation Drag", run_unconstrained_generation_scenario),
    "waterfall": ("Sequential I/O Waterfall", run_sequential_io_waterfall_scenario),
    "uncached": ("Uncached Repeated I/O Lookup", run_uncached_repeated_io_scenario),
    "vision_io": ("Heavy Vision Payload I/O", run_heavy_vision_payload_scenario),
    "baseline": ("Clean Optimal Baseline", run_clean_baseline_scenario),
}


async def main():
    parser = argparse.ArgumentParser(description="Agent Handoff Latency Error Test Suite")
    parser.add_argument(
        "--scenario",
        type=str,
        choices=list(SCENARIOS_MAP.keys()) + ["all"],
        default="all",
        help="Specific scenario to execute or 'all' (default: all)",
    )
    parser.add_argument(
        "--from-scenario",
        type=int,
        default=None,
        help="Start execution from 1-indexed scenario number (e.g. 8)",
    )
    args = parser.parse_args()

    endpoint = os.getenv("MISTRAL_OTLP_TRACES_ENDPOINT", "https://api.mistral.ai/telemetry/v1/traces")
    print("=" * 85)
    print("      MISTRAL OBSERVABILITY -- AGENT HANDOFF LATENCY BENCHMARK SUITE")
    print("=" * 85)
    print(f" Target OTLP Endpoint : {endpoint}")
    print(f" Service Name          : agent-handoff-latency-suite")
    print(f" Mode                  : Standalone Agent Handoffs (Zero Workflow Engine Dependency)")
    if args.from_scenario:
        print(f" Filter                : Running from Scenario #{args.from_scenario} onwards")
    print("=" * 85 + "\n")

    client, tracer = setup_latency_tracer(service_name="agent-handoff-latency-suite")

    all_items = list(SCENARIOS_MAP.items())
    if args.from_scenario is not None:
        start_idx = max(0, args.from_scenario - 1)
        selected_scenarios = all_items[start_idx:]
    elif args.scenario != "all":
        selected_scenarios = [(args.scenario, SCENARIOS_MAP[args.scenario])]
    else:
        selected_scenarios = all_items

    results: list[ScenarioResult] = []

    for key, (display_name, runner_fn) in selected_scenarios:
        # Find 1-indexed index in full scenario map
        idx = [k for k, _ in all_items].index(key) + 1
        print(f">> [RUNNING #{idx}] {display_name} ({key})...")
        try:
            res: ScenarioResult = await runner_fn(client, tracer)
            results.append(res)
            print(f"   [DONE]  Execution ID : {res.execution_id}")
            print(f"           OTel Trace ID: {res.trace_id}")
            print(f"           Duration     : {res.total_duration_ms:.1f} ms ({res.total_duration_ms/1000:.2f}s)")
            print(f"           Latency Type : {res.latency_type}")
            print(f"           Summary      : {res.summary}\n")
            # Brief pause between scenario batches for clean trace packet boundary
            await asyncio.sleep(1.0)
        except Exception as exc:
            logger.error(f"Error executing scenario '{key}': {exc}", exc_info=True)

    # Synchronously flush all traces to the Mistral OTLP endpoint
    print(">> Synchronously flushing and uploading OpenTelemetry traces to Mistral...")
    flush_success = flush_all_traces(tracer, timeout_ms=30000)
    if flush_success:
        print("   [OK] All spans successfully uploaded to Mistral Observability.")
    else:
        print("   [WARNING] Telemetry flush completed with fallback.")

    # Save / merge manifest of trace IDs
    manifest_path = CURRENT_DIR / "LATEST_LATENCY_TRACE_IDS.json"
    existing_manifest = []
    if manifest_path.exists():
        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                existing_manifest = json.load(f)
        except Exception:
            existing_manifest = []

    new_entries = [
        {
            "scenario_name": r.scenario_name,
            "latency_type": r.latency_type,
            "execution_id": r.execution_id,
            "trace_id": r.trace_id,
            "total_duration_ms": round(r.total_duration_ms, 2),
            "status": r.status,
            "step_count": r.step_count,
            "summary": r.summary,
            "details": r.details,
        }
        for r in results
    ]

    # Merge: update existing by scenario_name or append
    merged_dict = {m["scenario_name"]: m for m in existing_manifest}
    for entry in new_entries:
        merged_dict[entry["scenario_name"]] = entry
    merged_list = list(merged_dict.values())

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(merged_list, f, indent=2)

    print(f"\n>> Saved/Updated trace manifest to: {manifest_path}\n")

    # Display comparison table
    print("=" * 110)
    print(f"{'#':<3} | {'Scenario Name':<32} | {'Latency Type':<26} | {'Duration':<10} | {'Trace ID':<32}")
    print("-" * 110)
    for idx, r in enumerate(results, start=1):
        dur_str = f"{r.total_duration_ms/1000:.2f}s"
        print(f"{idx:<3} | {r.scenario_name:<32} | {r.latency_type:<26} | {dur_str:<10} | {r.trace_id:<32}")
    print("=" * 110 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
