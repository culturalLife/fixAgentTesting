# Mistral Workflows: Creation & Execution Guide

This guide details the general structure, requirements, and components needed to create and run Mistral workflows based on the standardized `mistralai.workflows` pattern. This pattern is applicable for building robust, observable, and scalable multi-step AI agents and workflows.

## 1. Prerequisites & Requirements

Before building a workflow, ensure the following setup is in place:

*   **Python Version:** Python 3.12 or higher.
*   **Package Manager:** `uv` (recommended for fast dependency resolution) or `pip`.
*   **Environment Variables (`.env` file):**
    *   `MISTRAL_API_KEY`: Required for authenticating with the Mistral API.
    *   `SERVER_URL`: The Mistral API server URL (defaults to `https://api.mistral.ai`).
    *   `DEPLOYMENT_NAME`: A unique identifier for your worker deployment (useful to isolate multiple projects).
*   **Core Dependencies (e.g., in `pyproject.toml`):**
    *   `mistralai-workflows[mistralai]>=3.0.0`
    *   `pydantic` (for robust typing and schema validation)
    *   `python-dotenv` (for loading environment variables)

## 2. Standard Project Structure

A clean, modular structure is critical for separating workflow logic from execution mechanics.

```text
src/
├── entrypoints/          # Runnable modules to start workers or trigger workflows
│   ├── worker.py         # Auto-discovers workflows and starts the polling worker
│   └── start.py          # Script to submit/trigger a workflow execution
├── workflows/            # Contains your workflow classes and activities
│   ├── __init__.py
│   └── my_workflow.py    # The core logic (Activities + Workflow Class)
└── telemetry.py          # (Optional but recommended) Tracing and observability setup
```

## 3. Core Components (The "Agents" & Workflows)

The pattern relies on three main concepts: Models, Activities, and the Workflow definition.

### A. Data Models (Pydantic)
Define strict input and output schemas for your workflow using `pydantic.BaseModel`. This ensures data validation and provides clear interfaces.

```python
from pydantic import BaseModel, Field
from typing import List, Optional

class OrderItem(BaseModel):
    sku: str
    quantity: int

class MyWorkflowInput(BaseModel):
    order_id: str
    items: List[OrderItem]

class MyWorkflowResult(BaseModel):
    status: str
    total_processed: int
```

### B. Activities (The "Agents")
Activities are the granular, individual tasks (or "agents") that perform the actual work (e.g., validating data, calling LLMs, processing payments). 
*   They are wrapped with the `@workflows.activity()` decorator.
*   You can configure `retry_policy_max_attempts` and `start_to_close_timeout`.
*   **Telemetry Integration:** Activities typically initialize a tracer to log their execution, inputs, and handle exceptions.

```python
import mistralai.workflows as workflows
from datetime import timedelta

@workflows.activity(
    name="process_items_activity",
    retry_policy_max_attempts=2,
    start_to_close_timeout=timedelta(seconds=30),
)
async def process_items(order_id: str, items: list) -> dict:
    # 1. (Optional) Setup telemetry/tracing here
    # 2. Perform the actual work
    try:
        # Example logic
        result = {"processed": len(items)}
        return result
    except Exception as exc:
        # Log exception to telemetry
        raise exc
```

### C. The Workflow Class
The workflow orchestrates the activities. It defines the execution graph (linear, parallel, or conditional).
*   Decorated with `@workflows.workflow.define()`.
*   Must contain an entrypoint method decorated with `@workflows.workflow.entrypoint`.

```python
@workflows.workflow.define(
    name="my-custom-pipeline",
    workflow_display_name="Custom Data Pipeline",
    workflow_description="Processes items and returns status.",
)
class MyWorkflow:
    @workflows.workflow.entrypoint
    async def run(self, input: MyWorkflowInput) -> MyWorkflowResult:
        # Step 1: Call activities sequentially or in parallel
        # Note: Activities are called using `await`
        activity_result = await process_items(
            order_id=input.order_id,
            items=[item.model_dump() for item in input.items]
        )
        
        # Step 2: Return final structured result
        return MyWorkflowResult(
            status="completed",
            total_processed=activity_result["processed"]
        )
```

## 4. Execution Mechanics

Running a workflow involves two distinct processes: the **Worker** (which listens for tasks and executes them) and the **Trigger** (which submits the input data).

### A. The Worker (`src/entrypoints/worker.py`)
The worker uses introspection to discover all workflow classes and starts polling the Mistral API for tasks assigned to its deployment.

```python
import asyncio
import mistralai.workflows as mistralai_workflows

async def start_worker():
    # 1. Discover workflows (e.g., by scanning the 'workflows' module)
    discovered = [MyWorkflow] # Alternatively, auto-discover using importlib/pkgutil
    
    # 2. Run the worker
    await mistralai_workflows.run_worker(discovered)

if __name__ == "__main__":
    asyncio.run(start_worker())
```

### B. Triggering the Workflow (`src/entrypoints/start.py`)
A separate client process submits the workflow execution payload and waits for the result.

```python
import asyncio
import os
from mistralai.workflows.client import get_mistral_client
from mistralai.extra.workflows import WorkflowEncodingConfig, configure_workflow_encoding

async def execute():
    client = get_mistral_client(api_key=os.environ.get("MISTRAL_API_KEY"))
    
    # Enable client-side payload encoding
    await configure_workflow_encoding(WorkflowEncodingConfig(), client=client)

    input_data = {
        "order_id": "ORD-123",
        "items": [{"sku": "ITEM-A", "quantity": 1}]
    }

    result = await client.workflows.execute_workflow_and_wait_async(
        workflow_identifier="my-custom-pipeline",
        input=input_data,
        deployment_name=os.environ.get("DEPLOYMENT_NAME", "default"),
    )
    print("Workflow Result:", result)

if __name__ == "__main__":
    asyncio.run(execute())
```

## 5. Summary of Best Practices
1. **Strong Typing:** Always use Pydantic models for the main workflow input and output to ensure strict contracts.
2. **Granular Activities:** Break down complex logic into smaller, independent `@workflows.activity` functions. This makes retries safer and telemetry more precise.
3. **Telemetry & Observability:** Inject OpenTelemetry spans inside your activities. Track `gen_ai.agent.name`, `gen_ai.activity.status`, and input/output attributes to monitor your agents' performance effectively.
4. **Error Handling:** Catch exceptions within activities, record them in your tracer, and re-raise them so the workflow engine's retry policies can take over.
