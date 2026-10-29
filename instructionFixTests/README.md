# Northwind Goods customer care pipelines

Multi-agent customer care pipelines running on Mistral agents, traced to Mistral Observability.

| Pipeline | Agents |
|---|---|
| `damaged_delivery` | CareIntakeAgent → FulfilmentAgent |
| `refund_request` | RefundAgent |
| `return_payout` | ReturnsSummaryAgent → PayoutAgent |
| `bundle_return` | SettlementAgent |
| `inbound_triage` | TriageClassifierAgent → router → AutoReplyAgent |
| `claim_risk` | ClaimRiskAgent → risk parser |
| `courier_pickup` | PickupSchedulingAgent |

- `agents.py`: agent definitions and tool schemas
- `backend.py`: order, payments, carrier and claims services the tools call
- `tickets.py`: tickets replayed on each run
- `workflows.py`: pipelines and tracing
- `run_instruction_fix_suite.py`: registers or reuses the agents, runs the tickets, writes
  `LATEST_INSTRUCTION_FIX_TRACE_IDS.json`

```
.venv/Scripts/python.exe instructionFixTests/run_instruction_fix_suite.py [--workflow NAME] [--repeat N]
```

Agents are created once and their ids are kept in `agent_registry.json`. Later runs use the
agents' live instructions from the platform. `--reset-instructions` pushes the ones in
`agents.py` back. `--teardown` deletes the agents.

Each agent span carries `gen_ai.agent.id`, `gen_ai.agent.name`, `gen_ai.input.messages` and
`gen_ai.output.messages`. Each tool span carries `gen_ai.tool.name`, `gen_ai.tool.call.arguments`
and `gen_ai.tool.call.result`. Service errors are recorded as span exceptions.
