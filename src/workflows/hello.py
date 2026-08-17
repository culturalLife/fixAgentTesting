"""Minimal example workflow — edit this file or create new ones."""

import mistralai.workflows as workflows
from pydantic import BaseModel


class HelloInput(BaseModel):
    name: str = "World"


@workflows.activity()
async def greet(name: str) -> str:
    """A simple activity that returns a greeting."""
    from telemetry import setup_telemetry_client, record_span_exception, get_current_execution_id

    client, tracer = setup_telemetry_client(service_name="hello_workflow_activity")

    execution_id = get_current_execution_id()
    with tracer.start_as_current_span("greet_activity_span") as activity_span:
        activity_span.set_attribute("gen_ai.workflow.name", "hello-world")
        activity_span.set_attribute("gen_ai.workflow.execution_id", execution_id)
        activity_span.set_attribute("gen_ai.activity.name", "greet")
        activity_span.set_attribute("gen_ai.agent.name", "greeter_agent")
        activity_span.set_attribute("input.name", name)
        
        try:
            greeting = f"Hello, {name}! Welcome to Mistral Workflows."
            activity_span.set_attribute("gen_ai.activity.status", "SUCCESS")
            activity_span.set_attribute("gen_ai.activity.result", greeting)
            return greeting
        except Exception as e:
            record_span_exception(activity_span, e)
            activity_span.set_attribute("gen_ai.activity.status", "FAILED")
            raise e




@workflows.workflow.define(
    name="hello-world",
    workflow_display_name="Hello World",
    workflow_description="A minimal hello-world workflow.",
)
class HelloWorkflow:
    @workflows.workflow.entrypoint
    async def run(self, input: HelloInput) -> str:
        return await greet(input.name)
