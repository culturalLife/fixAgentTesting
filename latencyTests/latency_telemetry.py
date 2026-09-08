"""
latency_telemetry.py
--------------------
Standalone OpenTelemetry tracing setup for Agent Handoff Latency Benchmarks.
Directly instruments agent functions, handoff transitions, tool executions, and LLM calls
without relying on Temporal or Mistral workflow engine components.

Uploads traces to Mistral Observability OTLP Endpoint:
  https://api.mistral.ai/telemetry/v1/traces (or MISTRAL_OTLP_TRACES_ENDPOINT)
"""

from __future__ import annotations

import os
import json
import uuid
import logging
from contextlib import contextmanager
from typing import Any, Dict, Optional, Tuple
from pathlib import Path
from dotenv import load_dotenv

from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode, Span, Tracer
from mistralai.client import Mistral
from mistralai.extra.observability import configure_telemetry, get_telemetry_tracer

logger = logging.getLogger(__name__)

# Ensure .env is loaded from trial-workflow or parent directory
env_path = Path(__file__).resolve().parent.parent / ".env"
if env_path.exists():
    load_dotenv(env_path)
else:
    load_dotenv()


def setup_latency_tracer(
    service_name: str = "agent-handoff-latency-worker",
    api_key: Optional[str] = None,
) -> Tuple[Mistral, Tracer]:
    """
    Initializes Mistral client and OpenTelemetry tracer provider with dedicated
    observability exporter pointing to the configured OTLP traces endpoint.
    """
    key = api_key or os.getenv("MISTRAL_API_KEY", "")
    endpoint = os.getenv(
        "MISTRAL_OTLP_TRACES_ENDPOINT",
        os.getenv("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT", "https://api.mistral.ai/telemetry/v1/traces")
    )
    # Ensure standard OTel and Mistral telemetry env vars are synchronized
    os.environ["MISTRAL_OTLP_TRACES_ENDPOINT"] = endpoint
    os.environ["OTEL_EXPORTER_OTLP_TRACES_ENDPOINT"] = endpoint

    server_url = os.getenv("MISTRAL_BASE_URL") or os.getenv("SERVER_URL")
    client = Mistral(api_key=key, server_url=server_url) if server_url else Mistral(api_key=key)

    try:
        telemetry_mode = os.getenv("MISTRAL_SDK_TELEMETRY", "dedicated")
        configure_telemetry(client, provider=telemetry_mode)
        tracer = get_telemetry_tracer(client, name=service_name)
        logger.info(f"Mistral Latency Telemetry tracer initialized ('{service_name}') -> {endpoint}")
    except Exception as exc:
        logger.warning(f"Failed to initialize Mistral SDK telemetry tracer: {exc}. Using standard OTel fallback.")
        tracer = trace.get_tracer(service_name)

    return client, tracer


def flush_all_traces(tracer: Tracer, timeout_ms: int = 30000) -> bool:
    """
    Forces an immediate synchronous upload of all recorded spans to the Mistral OTLP endpoint.
    """
    flushed = False
    # 1. Flush via tracer span_processor if present (BatchSpanProcessor / MultiSpanProcessor)
    if hasattr(tracer, "span_processor") and tracer.span_processor:
        try:
            tracer.span_processor.force_flush(timeout_millis=timeout_ms)
            flushed = True
        except Exception as exc:
            logger.warning(f"Tracer span_processor force_flush failed: {exc}")

    # 2. Flush via global TracerProvider if available
    try:
        provider = trace.get_tracer_provider()
        if hasattr(provider, "force_flush"):
            provider.force_flush(timeout_millis=timeout_ms)
            flushed = True
    except Exception as exc:
        logger.warning(f"Global TracerProvider force_flush failed: {exc}")

    return flushed


def record_span_error(span: Span, exc: Exception) -> None:
    """Records an exception onto the given span and marks the status as ERROR."""
    if not span:
        return
    try:
        span.record_exception(exc)
        span.set_status(Status(StatusCode.ERROR, str(exc)))
        span.set_attribute("exception.type", exc.__class__.__name__)
        span.set_attribute("exception.message", str(exc))
        span.set_attribute("gen_ai.activity.status", "FAILED")
    except Exception as err:
        logger.warning(f"Failed to record exception on span: {err}")


@contextmanager
def handoff_span(
    tracer: Tracer,
    agent_name: str,
    action_name: str,
    execution_id: str,
    handoff_from: Optional[str] = None,
    handoff_to: Optional[str] = None,
    handoff_reason: Optional[str] = None,
    latency_type: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
):
    """
    Context manager for wrapping an agent execution step and its handoff boundary.
    Captures GenAI agent metadata, correlation execution_id, and latency classification attributes.
    """
    span_name = f"agent_{agent_name.lower()}_{action_name}"
    with tracer.start_as_current_span(span_name) as span:
        span.set_attribute("gen_ai.agent.name", agent_name)
        span.set_attribute("gen_ai.activity.name", action_name)
        span.set_attribute("gen_ai.workflow.execution_id", execution_id)
        span.set_attribute("gen_ai.conversation.id", execution_id)
        span.set_attribute("gen_ai.activity.status", "SUCCESS")

        if handoff_from:
            span.set_attribute("gen_ai.agent.handoff.from", handoff_from)
        if handoff_to:
            span.set_attribute("gen_ai.agent.handoff.to", handoff_to)
        if handoff_reason:
            span.set_attribute("gen_ai.agent.handoff.reason", handoff_reason)
        if latency_type:
            span.set_attribute("gen_ai.latency.type", latency_type)

        if metadata:
            for k, v in metadata.items():
                val_str = json.dumps(v) if isinstance(v, (dict, list)) else str(v)
                span.set_attribute(f"agent.metadata.{k}", val_str)

        try:
            yield span
        except Exception as exc:
            record_span_error(span, exc)
            raise exc


@contextmanager
def tool_span(
    tracer: Tracer,
    tool_name: str,
    arguments: Optional[Dict[str, Any]] = None,
    latency_type: Optional[str] = None,
):
    """
    Context manager for instrumenting tool / external I/O execution as child spans.
    """
    span_name = f"tool_{tool_name}"
    with tracer.start_as_current_span(span_name) as span:
        span.set_attribute("gen_ai.operation.name", "execute_tool")
        span.set_attribute("gen_ai.tool.name", tool_name)
        if latency_type:
            span.set_attribute("gen_ai.latency.type", latency_type)
        if arguments:
            span.set_attribute("gen_ai.tool.call.arguments", json.dumps(arguments))

        def set_result(result_val: Any):
            res_str = json.dumps(result_val) if isinstance(result_val, (dict, list)) else str(result_val)
            span.set_attribute("gen_ai.tool.result", res_str)
            span.set_attribute("gen_ai.activity.status", "SUCCESS")

        try:
            yield set_result
        except Exception as exc:
            record_span_error(span, exc)
            raise exc
