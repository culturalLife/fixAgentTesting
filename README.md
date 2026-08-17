# Mistral Workflows & Observability Testing Harness

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Mistral AI](https://img.shields.io/badge/Mistral%20AI-Workflows%20%26%20Telemetry-purple.svg)](https://docs.mistral.ai/)
[![OpenTelemetry](https://img.shields.io/badge/OpenTelemetry-OTLP%20Tracing-orange.svg)](https://opentelemetry.io/)
[![Package Manager](https://img.shields.io/badge/uv-Package%20Manager-blueviolet.svg)](https://github.com/astral-sh/uv)

A production-grade collection of **Mistral AI Workflows**, multi-agent orchestration pipelines, OpenTelemetry (OTel) telemetry integrations, and an **adversarial judge stress-testing harness** designed to validate, benchmark, and evaluate AI agent behavior and automated remediation engines.

---

## 📑 Table of Contents

- [Overview](#-overview)
- [Architecture & Design](#-architecture--design)
- [Workflows & Examples](#-workflows--examples)
  - [1. E-Commerce Claims Triage Workflow](#1-e-commerce-claims-triage-workflow)
  - [2. Adversarial Judge Stress-Testing Harness](#2-adversarial-judge-stress-testing-harness)
  - [3. Insurance Claims Triage](#3-insurance-claims-triage)
  - [4. Cargo Release Compliance](#4-cargo-release-compliance)
  - [5. Code Modernization Pipeline](#5-code-modernization-pipeline)
  - [6. Linear Summarization](#6-linear-summarization)
- [Telemetry & Observability Standard](#-telemetry--observability-standard)
- [Project Layout](#-project-layout)
- [Quickstart Guide](#-quickstart-guide)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Environment Configuration](#environment-configuration)
- [Usage & Execution](#-usage--execution)
  - [Running the Core Worker](#running-the-core-worker)
  - [Running Example Workflows](#running-example-workflows)
  - [Executing the Test Suites](#executing-the-test-suites)
- [Development & Code Quality](#-development--code-quality)

---

## 🌟 Overview

This repository demonstrates how to build robust, distributed AI agent workflows powered by Mistral AI, Temporal, and OpenTelemetry. It provides:

1. **Multi-Agent Orchestration**: Coordinating specialized LLM agents (intake, verification, compliance reasoning, decision synthesis).
2. **Tool Execution & Validation**: Standardized patterns for calling tools, recording arguments, and handling outputs.
3. **Telemetry & Observability**: OpenTelemetry tracing instrumentation conforming to `gen_ai.*` semantic conventions.
4. **Adversarial Stress Testing**: A deliberate suite of test traps (instruction violations, malformed arguments, loop thrashing, silent degraded states) to evaluate LLM judges and auto-remediation systems.

---

## 🏛 Architecture & Design

```
+-------------------------------------------------------------------------+
|                        Mistral AI Studio / Temporal                     |
+------------------------------------+------------------------------------+
                                     | Dispatches Tasks
                                     v
+------------------------------------+------------------------------------+
|                         Workflow Worker Engine                          |
|  - Auto-discovery of workflow classes (src/workflows/ & src/examples/)  |
|  - Managed state persistence, retries, timeouts, and child workflows   |
+------------------------------------+------------------------------------+
                                     |
               +---------------------+---------------------+
               |                                           |
               v                                           v
+--------------+---------------+            +-------------+--------------+
|   Multi-Agent Activities     |            |  OTel Telemetry Exporter   |
| - LLM Inferences (Mistral)   |            | - Workflow execution spans |
| - Tool Invocations & Output  | ---------> | - Activity & agent metrics |
| - Policy & Rule Reasoning    |            | - Tool call & arg tracing  |
+------------------------------+            +----------------------------+
```

---

## 🚀 Workflows & Examples

### 1. E-Commerce Claims Triage Workflow
*Location: `src/examples/ecommerce_claims/`*

An end-to-end multi-agent pipeline resolving customer e-commerce returns, replacements, and claims:
- **Intake & Classification Agent** (`intake_and_classify_claim`): Categorizes claims (refund, replacement, inspection, fraud) and urgency.
- **Verification & Tool Dispatch Agent** (`verify_order_and_inventory_tools`): Executes simulated database lookups (`lookup_order_details`) and inventory queries (`check_warehouse_inventory`).
- **Compliance & Policy Reasoning Agent** (`evaluate_compliance_and_policy`): Validates return windows, warranty clauses, and calculates fraud risk scores.
- **Customer Resolution Agent** (`generate_customer_resolution`): Formulates structured approval/rejection outcomes and generates polite customer-facing correspondence.

### 2. Adversarial Judge Stress-Testing Harness
*Location: `test/judge_stress_test_workflow.py` & `test/STRESS_TEST_MATRIX.md`*

A purpose-built testing suite featuring deliberate "planted defects" to benchmark evaluation judges:

| Trap # | Scenario | Defect Type | Expected Target Judge | Expected Verdict |
|---|---|---|---|---|
| **Trap 1** | `instruction_violation` | Formats contract explicitly violated (hedging, markdown fences) | Instruction Adherence | `score < 0.70` / `LOW_QUALITY_SCORE` |
| **Trap 2** | `tool_invalid_arguments` | Tool call arguments malformed/null | Tool-Call & Parameter Validity | `INVALID_ARGUMENTS` |
| **Trap 3** | `wrong_tool_selected` | Irrelevant tool called (e.g. cancel sub for order query) | Tool-Call & Parameter Validity | `WRONG_TOOL_SELECTED` |
| **Trap 4** | `tool_unnecessary_call` | Tool invoked when user already resolved | Tool-Call & Parameter Validity | `UNNECESSARY_CALL` |
| **Trap 5** | `unhandled_exception` | Unhandled runtime division by zero with error span | Workflow Goal Attainment | `GOAL_FAILED` / `SPAN_EXCEPTION_ERROR` |
| **Trap 6** | `loop_thrash` | Alternating agent handoffs without progress (5 cycles) | Workflow Goal Attainment | `LOOP_THRASH_DETECTED` |
| **Trap 7** | `silent_degraded_success` | Downstream 503 fallback returning stale data without error flag | Completeness / Correctness | `LOW_QUALITY_SCORE` / `POOR_CORRECTNESS` |
| **Trap 8** | `phantom_success` | Agent acted on wrong entity ID (`ORD-2091` vs `ORD-1042`) | Correctness / Completeness | `POOR_CORRECTNESS` |
| **Trap 9** | `atomic_no_tools` | Single-step execution with no tools invoked | Instruction Adherence | `LOW_QUALITY_SCORE` |

### 3. Insurance Claims Triage
*Location: `src/examples/insurance_claims/`*
- Multi-modal vehicle insurance triage processing damage photos using vision models.
- Parallel damage evaluation, repair cost estimation, policy clause validation, and settlement drafting.

### 4. Cargo Release Compliance
*Location: `src/examples/cargo_release/`*
- Shipping logistics and customs clearance compliance checks.
- Demonstrates Human-in-the-Loop (`wait_for_input()`) approvals and child sub-workflows.

### 5. Code Modernization Pipeline
*Location: `src/examples/code_modernization/`*
- Legacy code refactoring pipeline with fan-out child workflows.
- Sandboxed test verification loops and human review checkpoints.

### 6. Linear Summarization
*Location: `src/examples/linear_summarization/`*
- Connector integration for issue trackers (Linear), parsing updates, deduplicating work items, and generating sprint summaries.

---

## 📡 Telemetry & Observability Standard

All activities and tools export OpenTelemetry spans following standard semantic attributes defined in `telemetry_implementation_guide.md`:

```python
with tracer.start_as_current_span("tool_lookup_order_details") as tool_span:
    tool_args = {"order_id": claim.order_id, "customer_id": claim.customer_id}
    tool_span.set_attribute("gen_ai.tool.name", "lookup_order_details")
    tool_span.set_attribute("gen_ai.tool.call.arguments", json.dumps(tool_args))
    tool_span.set_attribute("gen_ai.tool.result", json.dumps(order_data))
    tool_span.set_attribute("gen_ai.activity.status", "SUCCESS")
```

### Core Semantic Keys:
- `gen_ai.workflow.name`: Identifier of the executing workflow.
- `gen_ai.workflow.execution_id`: Runtime execution ID.
- `gen_ai.activity.name`: Activity/task identifier.
- `gen_ai.agent.name`: Name of the agent performing the step.
- `gen_ai.activity.status`: `"SUCCESS"` or `"FAILED"`.
- `gen_ai.tool.name`: Name of the tool called.
- `gen_ai.tool.call.arguments`: JSON/stringified dictionary of tool input arguments.
- `gen_ai.tool.result`: JSON/stringified tool execution response.

---

## 📁 Project Layout

```
.
├── Makefile                          # Convenient CLI shortcuts for worker & execution
├── README.md                         # Project documentation
├── pyproject.toml                    # UV project configuration and dependencies
├── telemetry_implementation_guide.md # Specification for OTel GenAI telemetry
├── workflow_creation_guide.md        # Best practice guide for developing workflows
├── .env.example                      # Template environment variables
├── src/
│   ├── entrypoints/                  # Worker & execution entrypoint scripts
│   │   ├── worker.py                 # Core worker (auto-discovers src/workflows/)
│   │   ├── start.py                  # CLI trigger for single executions
│   │   └── dev.py                    # Worker with live-reload watching
│   ├── telemetry.py                  # OpenTelemetry tracer singleton & helpers
│   ├── workflows/                    # Base / test workflows
│   │   ├── hello.py                  # Baseline Hello World workflow
│   │   └── failing.py                # Order processing workflow with edge cases
│   └── examples/                     # Workflow cookbooks
│       ├── cargo_release/            # Cargo compliance workflow
│       ├── code_modernization/       # Code refactoring pipeline
│       ├── ecommerce_claims/         # E-commerce multi-agent claims triage
│       ├── insurance_claims/         # Vehicle damage vision & settlement
│       ├── linear_summarization/     # Linear ticket digest workflow
│       └── worker.py                 # Worker dedicated to example workflows
└── test/                             # Test suites & judge benchmarking
    ├── STRESS_TEST_MATRIX.md         # Matrix of stress test traps & expected verdicts
    ├── judge_stress_test_workflow.py # 9 Adversarial judge stress test scenarios
    ├── judge_mini_stress_test.py     # Mini stress test runner
    ├── run_ecommerce_claims_suite.py # 4-scenario runner for e-commerce triage
    └── LATEST_ECOMMERCE_TRACE_IDS.json # Logged OTel trace executions
```

---

## 🛠 Quickstart Guide

### Prerequisites
- Python 3.10 or newer
- [`uv`](https://github.com/astral-sh/uv) (fast Python package manager)
- A Mistral AI API key ([Mistral AI Console](https://console.mistral.ai/))

### Installation

Clone the repository and install all dependencies using `uv`:

```bash
# Clone the repository
git clone https://github.com/culturalLife/fixAgentTesting.git
cd fixAgentTesting

# Sync and install virtual environment
uv sync
```

### Environment Configuration

Copy `.env.example` to `.env` and fill in your API credentials:

```bash
cp .env.example .env
```

Edit `.env`:
```ini
MISTRAL_API_KEY=your_mistral_api_key_here
SERVER_URL=https://api.mistral.ai
DEPLOYMENT_NAME=trial-workflow-worker
MISTRAL_OTLP_TRACES_ENDPOINT=https://api.mistral.ai/telemetry/v1/traces
OTEL_EXPORTER_OTLP_TRACES_ENDPOINT=https://api.mistral.ai/telemetry/v1/traces
MISTRAL_SDK_TELEMETRY=dedicated
```

---

## 💻 Usage & Execution

### Running the Core Worker

Start the worker to register workflows from `src/workflows/` with AI Studio and poll for tasks:

```bash
make start-worker
```

Trigger an execution from another terminal:

```bash
make execute workflow=hello-world input='{"name": "World"}'
```

### Running Example Workflows

Start the examples worker:

```bash
make start-examples
```

Trigger an insurance claim execution:

```bash
make execute-insurance-claims input='{"claim_id":"CLM-001","claimant_name":"Jane","description":"My car was hit.","photos":["src/examples/insurance_claims/sample_data/photos/claim_low_scratch_door.jpg"]}'
```

### Executing the Test Suites

#### 1. E-Commerce Multi-Agent Claims Suite
Runs 4 realistic claim scenarios (`clean_replacement`, `ambiguous_partial`, `policy_contradiction`, `negative_constraint`) and records their OpenTelemetry trace IDs:

```bash
uv run python test/run_ecommerce_claims_suite.py
```

#### 2. Judge Adversarial Stress Tests
Emits traces for the 9 adversarial traps to test judge accuracy and remediation triggers:

```bash
uv run python test/judge_mini_stress_test.py
```

---

## 🧹 Development & Code Quality

Format and lint the codebase using `ruff`:

```bash
# Format code
uv run ruff format .

# Check and fix lint issues
uv run ruff check --fix .
```

---

## 📄 License

This project is licensed under the Apache-2.0 License.
