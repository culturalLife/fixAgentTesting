import os
import json
import logging
from typing import Any, Dict, Optional
from opentelemetry.trace import Status, StatusCode
from mistralai.client import Mistral
from mistralai.extra.observability import configure_telemetry, get_telemetry_tracer

logger = logging.getLogger(__name__)


_TRACER_CACHE: Dict[str, Any] = {}
_CLIENT_CACHE: Optional[Mistral] = None


def setup_telemetry_client(
    api_key: Optional[str] = None,
    service_name: str = "workflow-agent-worker"
) -> tuple[Mistral, Any]:
    """
    Initialize a Mistral client with OpenTelemetry tracing configured
    via the mistralai.extra.observability Python SDK module.

    Sends OTLP traces to MISTRAL_OTLP_TRACES_ENDPOINT.
    """
    key = api_key or os.getenv("MISTRAL_API_KEY", "")
    endpoint = os.getenv(
        "MISTRAL_OTLP_TRACES_ENDPOINT", "https://api.mistral.ai/telemetry/v1/traces"
    )
    os.environ["MISTRAL_OTLP_TRACES_ENDPOINT"] = endpoint
    os.environ["OTEL_EXPORTER_OTLP_TRACES_ENDPOINT"] = endpoint

    client = Mistral(api_key=key)

    try:
        mode = os.getenv("MISTRAL_SDK_TELEMETRY", "dedicated")
        configure_telemetry(client, provider=mode)
        tracer = get_telemetry_tracer(client, name=service_name)
        logger.info(f"Mistral Observability Telemetry initialized for '{service_name}' on endpoint: {endpoint}")
    except Exception as e:
        logger.warning(f"Failed to configure Mistral SDK telemetry tracer: {e}")
        from opentelemetry import trace as otel_trace
        tracer = otel_trace.get_tracer(service_name)

    return client, tracer


def get_telemetry_tracer_instance(service_name: str = "workflow-agent-worker") -> Any:
    """Returns a cached singleton tracer instance for the given service_name."""
    global _TRACER_CACHE, _CLIENT_CACHE
    if service_name not in _TRACER_CACHE:
        client, tracer = setup_telemetry_client(service_name=service_name)
        _CLIENT_CACHE = client
        _TRACER_CACHE[service_name] = tracer
    return _TRACER_CACHE[service_name]




def record_span_exception(span: Any, exc: Exception) -> None:
    """Helper to record exception details and set error status on an OTel span."""
    if not span:
        return
    try:
        span.record_exception(exc)
        span.set_status(Status(StatusCode.ERROR, str(exc)))
        span.set_attribute("exception.type", exc.__class__.__name__)
        span.set_attribute("exception.message", str(exc))
    except Exception as err:
        logger.warning(f"Failed to record exception on span: {err}")


def get_current_execution_id() -> str:
    """
    Helper to extract the current Temporal/Workflow execution_id in activity context.
    Returns the workflow_id if inside an activity, or fallback execution ID.
    """
    try:
        from temporalio import activity
        if activity.in_activity():
            info = activity.info()
            return info.workflow_id
    except Exception:
        pass

    try:
        from temporalio import workflow
        if workflow.in_workflow():
            info = workflow.info()
            return info.workflow_id
    except Exception:
        pass

    return "exec-local-" + os.urandom(4).hex()
