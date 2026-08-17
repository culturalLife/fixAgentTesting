# Mistral Workflows: Telemetry & Tracing Guide

This document outlines the standard pattern for implementing observability and uploading OpenTelemetry (OTLP) traces within Mistral workflows. Models and developers must strictly follow this pattern to ensure telemetry is properly logged and visible on the platform.

## 1. Setup and Initialization (`telemetry.py`)

Tracing is initialized using the `mistralai.extra.observability` SDK. A helper module (usually `telemetry.py`) manages tracer singletons and environment variables.

### Environment Requirements
*   `MISTRAL_OTLP_TRACES_ENDPOINT`: Where traces are uploaded (defaults to `https://api.mistral.ai/telemetry/v1/traces`).
*   `MISTRAL_SDK_TELEMETRY`: Usually set to `dedicated`.

### Core Setup Pattern
Use this boilerplate to configure and acquire a tracer:

```python
import os
from mistralai.client import Mistral
from mistralai.extra.observability import configure_telemetry, get_telemetry_tracer

def setup_telemetry_client(service_name: str = "workflow-agent-worker"):
    key = os.getenv("MISTRAL_API_KEY", "")
    endpoint = os.getenv("MISTRAL_OTLP_TRACES_ENDPOINT", "https://api.mistral.ai/telemetry/v1/traces")
    os.environ["OTEL_EXPORTER_OTLP_TRACES_ENDPOINT"] = endpoint

    client = Mistral(api_key=key)
    
    # Configure the global provider and extract the tracer
    configure_telemetry(client, provider="dedicated")
    tracer = get_telemetry_tracer(client, name=service_name)
    
    return client, tracer

# Singleton cache wrapper
_TRACER_CACHE = {}
def get_telemetry_tracer_instance(service_name: str):
    if service_name not in _TRACER_CACHE:
        _, tracer = setup_telemetry_client(service_name=service_name)
        _TRACER_CACHE[service_name] = tracer
    return _TRACER_CACHE[service_name]
```

## 2. Emitting Traces inside Activities (Agents)

Inside your `@workflows.activity()` definitions, you must open a span and manually set standardized attributes (`gen_ai.*`).

### Required Attributes
Every span must record:
*   `gen_ai.workflow.name`: The identifier of the workflow.
*   `gen_ai.workflow.execution_id`: The runtime execution ID (retrieved via Temporal context).
*   `gen_ai.activity.name`: The name of the specific step/activity.
*   `gen_ai.agent.name`: The descriptive name of the agent fulfilling the task.
*   `gen_ai.activity.status`: Must be explicitly set to `"SUCCESS"` or `"FAILED"`.

### Implementation Template

```python
import mistralai.workflows as workflows
from telemetry import get_telemetry_tracer_instance, get_current_execution_id, record_span_exception

@workflows.activity(name="example_activity")
async def example_activity(input_data: str) -> str:
    tracer = get_telemetry_tracer_instance("example_worker")
    execution_id = get_current_execution_id()
    
    # 1. Start the span
    with tracer.start_as_current_span("example_activity_span") as span:
        # 2. Attach context
        span.set_attribute("gen_ai.workflow.name", "my-workflow")
        span.set_attribute("gen_ai.workflow.execution_id", execution_id)
        span.set_attribute("gen_ai.activity.name", "example_activity")
        span.set_attribute("gen_ai.agent.name", "example_agent")
        span.set_attribute("input.data", input_data)
        
        try:
            # ---> DO WORK HERE <---
            result = f"Processed {input_data}"
            
            # 3. Mark Success
            span.set_attribute("gen_ai.activity.status", "SUCCESS")
            span.set_attribute("gen_ai.activity.result", result)
            return result
            
        except Exception as exc:
            # 4. Handle Failure (Record exception and mark FAILED)
            record_span_exception(span, exc)
            span.set_attribute("gen_ai.activity.status", "FAILED")
            raise exc  # Important: re-raise for workflow retry policies
```

## 3. Helper Functions

You should maintain these helpers in your `telemetry.py` for consistent error handling and ID generation.

```python
from opentelemetry.trace import Status, StatusCode

def record_span_exception(span, exc: Exception) -> None:
    """Records the exception on the span and marks it as an error."""
    if not span: return
    span.record_exception(exc)
    span.set_status(Status(StatusCode.ERROR, str(exc)))
    span.set_attribute("exception.type", exc.__class__.__name__)
    span.set_attribute("exception.message", str(exc))

def get_current_execution_id() -> str:
    """Extracts the execution ID from the Temporal context."""
    try:
        from temporalio import activity
        if activity.in_activity(): return activity.info().workflow_id
    except Exception: pass
    
    try:
        from temporalio import workflow
        if workflow.in_workflow(): return workflow.info().workflow_id
    except Exception: pass
    
    import os
    return "exec-local-" + os.urandom(4).hex()
```

## 4. Key Rules
- Always use the `with tracer.start_as_current_span(...)` context manager to prevent memory leaks and dangling spans.
- Do not swallow exceptions in the activity. Always `raise exc` after recording it on the span so the workflow engine can retry or fail gracefully.
