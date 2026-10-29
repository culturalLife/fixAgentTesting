"""
run_instruction_fix_suite.py
----------------------------
Runs the Northwind Goods customer care pipelines against registered Mistral agents and
exports the traces to Mistral Observability.

Agents are created on the first run and reused afterwards (ids in agent_registry.json),
so changes made to their instructions on the platform carry over to later runs.

Usage:
  python run_instruction_fix_suite.py                          # run every pipeline
  python run_instruction_fix_suite.py --workflow refund_request
  python run_instruction_fix_suite.py --repeat 3               # replay each ticket 3 times
  python run_instruction_fix_suite.py --reset-instructions     # push agents.py definitions back to the platform
  python run_instruction_fix_suite.py --teardown               # delete the registered agents
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path
from typing import Any, Dict

CURRENT_DIR = Path(__file__).resolve().parent
ROOT_DIR = CURRENT_DIR.parent
sys.path.insert(0, str(CURRENT_DIR))
sys.path.insert(0, str(ROOT_DIR))

from dotenv import load_dotenv

load_dotenv(ROOT_DIR / ".env")

from agents import AGENTS, COMPLETION_ARGS, MODEL, AgentSpec
from src.telemetry import setup_telemetry_client
from tickets import TICKETS
from workflows import WORKFLOWS, run_ticket

REGISTRY_PATH = CURRENT_DIR / "agent_registry.json"
MANIFEST_PATH = CURRENT_DIR / "LATEST_INSTRUCTION_FIX_TRACE_IDS.json"


def _load_registry() -> Dict[str, Dict[str, Any]]:
    if REGISTRY_PATH.exists():
        return json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    return {}


def _save_registry(registry: Dict[str, Dict[str, Any]]) -> None:
    REGISTRY_PATH.write_text(json.dumps(registry, indent=2), encoding="utf-8")


def _create_agent(client: Any, spec: AgentSpec) -> str:
    kwargs: Dict[str, Any] = {
        "model": MODEL,
        "name": spec.name,
        "description": spec.description,
        "instructions": spec.instructions,
        "completion_args": COMPLETION_ARGS,
    }
    if spec.tools:
        kwargs["tools"] = spec.tool_definitions()
    return client.beta.agents.create(**kwargs).id


def ensure_agents(client: Any, reset_instructions: bool) -> Dict[str, str]:
    registry = _load_registry()
    for name, spec in AGENTS.items():
        entry = registry.get(name)
        if entry:
            try:
                live = client.beta.agents.get(agent_id=entry["id"])
            except Exception as exc:
                print(f"  {name}: registered agent {entry['id']} unavailable ({exc}); creating a new one")
                entry = None
            else:
                if reset_instructions:
                    client.beta.agents.update(
                        agent_id=entry["id"], description=spec.description,
                        instructions=spec.instructions, tools=spec.tool_definitions(),
                    )
                    print(f"  {name}: reset to agents.py ({entry['id']})")
                else:
                    print(f"  {name}: reusing {entry['id']} (version {getattr(live, 'version', '?')})")
        if not entry:
            agent_id = _create_agent(client, spec)
            registry[name] = {"id": agent_id, "created_at": dt.datetime.now().isoformat(timespec="seconds")}
            print(f"  {name}: created {agent_id}")
    _save_registry(registry)
    return {name: entry["id"] for name, entry in registry.items()}


def teardown(client: Any) -> None:
    registry = _load_registry()
    for name, entry in registry.items():
        try:
            client.beta.agents.delete(agent_id=entry["id"])
            print(f"  deleted {name} ({entry['id']})")
        except Exception as exc:
            print(f"  could not delete {name} ({entry['id']}): {exc}")
    REGISTRY_PATH.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--workflow", choices=sorted(WORKFLOWS), action="append")
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--reset-instructions", action="store_true")
    parser.add_argument("--teardown", action="store_true")
    args = parser.parse_args()

    client, tracer = setup_telemetry_client(service_name="northwind-care-worker")

    if args.teardown:
        teardown(client)
        return

    print("Agents:")
    registry = ensure_agents(client, args.reset_instructions)

    results = []
    for workflow_name in args.workflow or list(WORKFLOWS):
        for ticket in TICKETS[workflow_name]:
            for _ in range(max(1, args.repeat)):
                record = run_ticket(client, tracer, registry, workflow_name, ticket)
                results.append(record)
                line = f"  {workflow_name:<17} {ticket['id']}  {record['status']:<9}  trace={record['trace_id']}"
                if record.get("error"):
                    line += f"\n{'':22}{record['error']}"
                print(line)

    manifest = {
        "generated_at": dt.datetime.now().isoformat(timespec="seconds"),
        "agents": registry,
        "runs": results,
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"\nSaved {len(results)} runs to {MANIFEST_PATH}")

    if flush_traces(client):
        print("Traces flushed to Mistral Observability.")


def flush_traces(client: Any) -> bool:
    """Flushes the Mistral client's dedicated tracer provider, falling back to the global one."""
    from opentelemetry import trace as otel_trace

    hooks = getattr(client.sdk_configuration, "_hooks", None)
    providers = [getattr(h, "tracer_provider", None) for h in getattr(hooks, "before_request_hooks", [])]
    providers.append(otel_trace.get_tracer_provider())
    for provider in providers:
        if provider is not None and hasattr(provider, "force_flush"):
            return bool(provider.force_flush())
    return False


if __name__ == "__main__":
    main()
