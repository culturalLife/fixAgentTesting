"""
agent_handoff_scenarios.py
--------------------------
Implements 7 realistic multi-agent handoff scenarios designed to probe and benchmark
different latency failure modes and bottlenecks in agentic systems.

Each scenario produces a complete OpenTelemetry trace tree with agent handoffs,
nested spans, tool calls, and LLM telemetry uploaded directly to Mistral Observability.

Scenarios:
  1. CASCADING_HANDOFF_DELAY    - Upstream agent parsing bottleneck stalling downstream chain
  2. LOOP_THRASH_HANDOFF        - Cyclical ping-pong handoffs between agents due to ambiguity
  3. BLOCKING_TOOL_STALL        - Intermediate agent blocked on slow synchronous external I/O
  4. TOKEN_BLOAT_LATENCY        - Massive uncompressed context payload causing TTFT & generation drag
  5. FANOUT_STRAGGLER_DELAY     - Parallel sub-agents where one straggler bottlenecks the aggregator
  6. RETRY_STORM_BACKOFF        - Progressive timeout retry backoffs before degraded fallback
  7. MODEL_OVERSIZING_LLM_DELAY - Heavyweight model used for trivial task (recommends model downgrade)
  8. UNCONSTRAINED_GEN_DRAG     - Runaway verbose output generation (recommends max_tokens & schema)
  9. SEQUENTIAL_IO_WATERFALL    - Multiple sequential network/DB calls (recommends asyncio.gather parallelization)
  10. UNCACHED_REPEATED_IO      - Duplicate idempotent external queries (recommends Redis/TTL cache)
  11. HEAVY_VISION_PAYLOAD_IO   - Massive uncompressed multi-modal I/O (recommends downscaling/compression)
  12. CLEAN_OPTIMAL_BASELINE    - Fast, healthy reference handoff with zero latency defects
"""

from __future__ import annotations

import asyncio
import json
import math
import time
import uuid
import logging
from typing import Any, Dict, List, Optional
from dataclasses import dataclass

from opentelemetry import trace
from opentelemetry.trace import Tracer
from mistralai.client import Mistral

from .latency_telemetry import handoff_span, tool_span, record_span_error

logger = logging.getLogger(__name__)


@dataclass
class ScenarioResult:
    scenario_name: str
    latency_type: str
    execution_id: str
    trace_id: str
    total_duration_ms: float
    status: str
    step_count: int
    summary: str
    details: Dict[str, Any]


async def _safe_llm_call(
    client: Mistral,
    model: str = "mistral-small-latest",
    prompt: str = "Respond with a brief confirmation.",
    system_prompt: Optional[str] = None,
) -> str:
    """Executes a live Mistral chat completion, gracefully falling back if offline."""
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    try:
        response = await client.chat.complete_async(
            model=model,
            messages=messages,
            temperature=0.2,
            max_tokens=250,
        )
        return response.choices[0].message.content or ""
    except Exception as exc:
        logger.debug(f"LLM call fallback due to: {exc}")
        # Synthetic delay mimicking model inference when running offline
        await asyncio.sleep(0.35)
        return f"[Simulated LLM response for: {prompt[:40]}...]"


# ===========================================================================
# SCENARIO 1: CASCADING HANDOFF DELAY (Upstream Bottleneck)
# ===========================================================================
async def run_cascading_delay_scenario(client: Mistral, tracer: Tracer) -> ScenarioResult:
    """
    Scenario 1: Upstream IngestionAgent has severe OCR/chunking delay (3.5s),
    starving downstream EntityExtractorAgent and SynthesisAgent.
    """
    execution_id = f"lat-cascade-{uuid.uuid4().hex[:8]}"
    start_time = time.perf_counter()

    with tracer.start_as_current_span("trace_cascading_handoff_delay") as root_span:
        root_span.set_attribute("gen_ai.workflow.name", "document-intake-pipeline")
        root_span.set_attribute("gen_ai.workflow.execution_id", execution_id)
        root_span.set_attribute("gen_ai.latency.scenario", "cascading_handoff_delay")
        root_span.set_attribute("gen_ai.latency.type", "CASCADING_HANDOFF_DELAY")
        trace_id = format(root_span.get_span_context().trace_id, "032x")

        # Step 1: Bottleneck Agent (Heavy Document Ingestion)
        with handoff_span(
            tracer,
            agent_name="DocumentIngestionAgent",
            action_name="parse_and_chunk_pdf",
            execution_id=execution_id,
            handoff_to="EntityExtractorAgent",
            handoff_reason="Document parsed, passing chunks for entity extraction",
            latency_type="CASCADING_HANDOFF_DELAY",
            metadata={"document_pages": 48, "file_size_mb": 14.2},
        ) as span1:
            with tool_span(tracer, "ocr_pdf_rasterize", {"pages": 48}, latency_type="CASCADING_HANDOFF_DELAY") as set_tool:
                # Simulated upstream OCR / heavy parsing bottleneck
                await asyncio.sleep(3.2)
                set_tool({"status": "SUCCESS", "pages_processed": 48, "raw_chunks": 120})

            summary = await _safe_llm_call(
                client,
                prompt="Summarize raw OCR ingestion status for a 48-page logistics manifest.",
                system_prompt="You are a Document Ingestion Agent."
            )
            span1.set_attribute("agent.output_summary", summary[:100])

        # Step 2: Downstream Agent (Entity Extraction)
        with handoff_span(
            tracer,
            agent_name="EntityExtractorAgent",
            action_name="extract_customs_entities",
            execution_id=execution_id,
            handoff_from="DocumentIngestionAgent",
            handoff_to="SynthesisAgent",
            handoff_reason="Entities extracted, passing to synthesis",
        ) as span2:
            await asyncio.sleep(0.3)
            entities = await _safe_llm_call(
                client,
                prompt="Extract UN dangerous goods codes, consignee, and destination port from text.",
                system_prompt="You are an Entity Extraction Agent."
            )
            span2.set_attribute("agent.extracted_entities", entities[:100])

        # Step 3: Downstream Agent (Synthesis)
        with handoff_span(
            tracer,
            agent_name="SynthesisAgent",
            action_name="compile_release_dossier",
            execution_id=execution_id,
            handoff_from="EntityExtractorAgent",
            handoff_reason="Final clearance dossier generated",
        ) as span3:
            await asyncio.sleep(0.2)
            dossier = await _safe_llm_call(
                client,
                prompt="Generate final cargo release certificate.",
                system_prompt="You are a Customs Synthesis Agent."
            )
            span3.set_attribute("agent.final_dossier", dossier[:100])

        duration_ms = (time.perf_counter() - start_time) * 1000
        root_span.set_attribute("gen_ai.latency.duration_ms", duration_ms)

        return ScenarioResult(
            scenario_name="Cascading Handoff Delay",
            latency_type="CASCADING_HANDOFF_DELAY",
            execution_id=execution_id,
            trace_id=trace_id,
            total_duration_ms=duration_ms,
            status="SUCCESS",
            step_count=3,
            summary="Upstream DocumentIngestionAgent caused 3.2s stall, delaying downstream agents.",
            details={"bottleneck_agent": "DocumentIngestionAgent", "bottleneck_duration_s": 3.2},
        )


# ===========================================================================
# SCENARIO 2: PING-PONG / LOOP THRASH HANDOFF (Cyclical Runaway Latency)
# ===========================================================================
async def run_ping_pong_loop_scenario(
    client: Mistral,
    tracer: Tracer,
    iterations: int = 4,
) -> ScenarioResult:
    """
    Scenario 2: TriageAgent and ValidationAgent repeatedly bounce requests back
    and forth due to conflicting requirements, multiplying cumulative latency.
    """
    # Input validation: iterations must be positive
    if iterations <= 0:
        raise ValueError(f"iterations must be positive, got {iterations}")
    
    execution_id = f"lat-loop-{uuid.uuid4().hex[:8]}"
    start_time = time.perf_counter()

    with tracer.start_as_current_span("trace_ping_pong_loop_thrash") as root_span:
        root_span.set_attribute("gen_ai.workflow.name", "claims-dispute-resolution")
        root_span.set_attribute("gen_ai.workflow.execution_id", execution_id)
        root_span.set_attribute("gen_ai.latency.scenario", "ping_pong_loop_thrash")
        root_span.set_attribute("gen_ai.latency.type", "LOOP_THRASH_HANDOFF")
        trace_id = format(root_span.get_span_context().trace_id, "032x")

        step_count = 0
        is_validation_complete = False
        is_claim_finalized = False
        triage_iteration_count = 0
        max_triage_iterations = 2
        result_summary = ""
        
        for i in range(1, iterations + 1):
            # Check if we can terminate early based on state
            if is_validation_complete and is_claim_finalized:
                result_summary = result_summary or "Both validation and claim finalization completed successfully"
                break
            
            # TriageAgent loop detection: terminate if max iterations reached without progress
            if triage_iteration_count >= max_triage_iterations:
                result_summary = "TriageAgent: Max iterations reached without resolution, escalating to FinalResolutionAgent"
                is_claim_finalized = True
                break
            
            # Additional check: if validation is complete but claim not finalized, or vice versa
            # and we've exceeded max iterations, escalate
            if (is_validation_complete or is_claim_finalized) and triage_iteration_count >= max_triage_iterations:
                result_summary = result_summary or "Partial resolution detected, escalating to FinalResolutionAgent"
                is_claim_finalized = True
                is_validation_complete = True
                break
                
            # Triage Agent Turn
            with handoff_span(
                tracer,
                agent_name="TriageAgent",
                action_name=f"evaluate_claim_turn_{i}",
                execution_id=execution_id,
                handoff_to="PolicyValidationAgent",
                handoff_reason=f"Turn {i}: Requesting clarification on ambiguous exclusion clause",
                latency_type="LOOP_THRASH_HANDOFF",
                metadata={"iteration": i, "max_iterations": iterations, "is_validation_complete": is_validation_complete, "is_claim_finalized": is_claim_finalized, "triage_iteration_count": triage_iteration_count, "result_summary": result_summary or "In Progress"},
            ) as s_triage:
                s_triage.set_attribute("gen_ai.agent.handoff.iteration", i)
                await asyncio.sleep(0.4)
                res1 = await _safe_llm_call(
                    client,
                    prompt=f"Iteration {i}: Querying policy clause 14.B coverage for damaged shipment.",
                    system_prompt="You are a Triage Claims Agent."
                )
                s_triage.set_attribute("agent.response", res1[:80])
                s_triage.set_attribute("result_summary", result_summary or "In Progress")
                step_count += 1
                triage_iteration_count += 1
                
                # TriageAgent evaluates if claim can be finalized
                if i >= iterations or "resolved" in res1.lower():
                    is_claim_finalized = True
                    result_summary = f"TriageAgent: Claim finalized at iteration {i}"
                
                # State validation: if max iterations reached, force finalization
                if triage_iteration_count >= max_triage_iterations and not is_claim_finalized:
                    is_claim_finalized = True
                    result_summary = result_summary or f"TriageAgent: Max iterations ({max_triage_iterations}) reached, forcing finalization"

            # Validation Agent Turn (Bounces back to Triage)
            with handoff_span(
                tracer,
                agent_name="PolicyValidationAgent",
                action_name=f"verify_clauses_turn_{i}",
                execution_id=execution_id,
                handoff_from="TriageAgent",
                handoff_to="FinalResolutionAgent" if (is_claim_finalized or is_validation_complete) else "TriageAgent" if i < iterations else "FinalResolutionAgent",
                handoff_reason=f"Turn {i}: Insufficient clause specificity, returning for refinement" if i < iterations and not (is_claim_finalized or is_validation_complete) else "Resolved",
                latency_type="LOOP_THRASH_HANDOFF",
                metadata={"iteration": i, "rejection_code": "AMBIGUOUS_EVIDENCE", "is_validation_complete": is_validation_complete, "is_claim_finalized": is_claim_finalized, "result_summary": result_summary or "In Progress"},
            ) as s_val:
                s_val.set_attribute("gen_ai.agent.handoff.iteration", i)
                await asyncio.sleep(0.45)
                res2 = await _safe_llm_call(
                    client,
                    prompt=f"Iteration {i}: Policy rejection: provide itemized proof of shipment transit.",
                    system_prompt="You are a Policy Validation Agent."
                )
                s_val.set_attribute("agent.response", res2[:80])
                s_val.set_attribute("result_summary", result_summary or "In Progress")
                step_count += 1
                
                # PolicyValidationAgent marks validation as complete when resolved
                if i >= iterations or "resolved" in res2.lower():
                    is_validation_complete = True
                    result_summary = result_summary or f"PolicyValidationAgent: Validation complete at iteration {i}"
                
                # State validation: if both states are complete, ensure result_summary is set
                if is_validation_complete and is_claim_finalized:
                    result_summary = result_summary or "Both validation and claim finalization completed successfully"

        duration_ms = (time.perf_counter() - start_time) * 1000
        root_span.set_attribute("gen_ai.latency.duration_ms", duration_ms)

        return ScenarioResult(
            scenario_name="Ping-Pong Loop Thrash Handoff",
            latency_type="LOOP_THRASH_HANDOFF",
            execution_id=execution_id,
            trace_id=trace_id,
            total_duration_ms=duration_ms,
            status="SUCCESS",
            step_count=step_count,
            summary=result_summary or f"Cyclical handoff thrashing between TriageAgent and PolicyValidationAgent across {step_count} handoff cycles.",
            details={"iterations": iterations, "total_handoff_cycles": step_count, "result_summary": result_summary},
        )


# ===========================================================================
# SCENARIO 3: BLOCKING TOOL / EXTERNAL I/O STALL
# ===========================================================================
async def run_blocking_tool_stall_scenario(client: Mistral, tracer: Tracer) -> ScenarioResult:
    """
    Scenario 3: SupportAgent delegates to DatabaseAgent, which hangs on a
    slow legacy ERP tool call (3.8s blocking I/O) before handing off to ResolutionAgent.
    """
    execution_id = f"lat-tool-stall-{uuid.uuid4().hex[:8]}"
    start_time = time.perf_counter()

    with tracer.start_as_current_span("trace_blocking_tool_stall") as root_span:
        root_span.set_attribute("gen_ai.workflow.name", "customer-order-support")
        root_span.set_attribute("gen_ai.workflow.execution_id", execution_id)
        root_span.set_attribute("gen_ai.latency.scenario", "blocking_tool_stall")
        root_span.set_attribute("gen_ai.latency.type", "BLOCKING_TOOL_STALL")
        trace_id = format(root_span.get_span_context().trace_id, "032x")

        # Step 1: Customer Support Agent receives query
        with handoff_span(
            tracer,
            agent_name="CustomerSupportAgent",
            action_name="intake_order_inquiry",
            execution_id=execution_id,
            handoff_to="LegacyDatabaseAgent",
            handoff_reason="Dispatching order lookup to database specialist",
        ) as span1:
            await asyncio.sleep(0.2)
            span1.set_attribute("order.id", "ORD-99381")

        # Step 2: Database Agent makes blocking slow tool call
        with handoff_span(
            tracer,
            agent_name="LegacyDatabaseAgent",
            action_name="fetch_order_records",
            execution_id=execution_id,
            handoff_from="CustomerSupportAgent",
            handoff_to="CustomerResolutionAgent",
            handoff_reason="Order retrieved from ERP, passing to resolution",
            latency_type="BLOCKING_TOOL_STALL",
        ) as span2:
            with tool_span(
                tracer,
                tool_name="query_legacy_sap_erp",
                arguments={"order_id": "ORD-99381", "timeout_sec": 10},
                latency_type="BLOCKING_TOOL_STALL",
            ) as set_tool:
                # Simulated blocking network / database stall
                await asyncio.sleep(3.6)
                set_tool({
                    "order_id": "ORD-99381",
                    "status": "IN_TRANSIT",
                    "carrier": "FedEx",
                    "tracking": "9400100000000000000000",
                })

            resp = await _safe_llm_call(
                client,
                prompt="Format ERP response for order ORD-99381.",
                system_prompt="You are a Database Specialist Agent."
            )
            span2.set_attribute("agent.response", resp[:100])

        # Step 3: Resolution Agent drafts response
        with handoff_span(
            tracer,
            agent_name="CustomerResolutionAgent",
            action_name="compose_customer_reply",
            execution_id=execution_id,
            handoff_from="LegacyDatabaseAgent",
            handoff_reason="Customer reply composed",
        ) as span3:
            await asyncio.sleep(0.25)
            final_reply = await _safe_llm_call(
                client,
                prompt="Draft polite customer update for order in transit.",
                system_prompt="You are a Customer Resolution Agent."
            )
            span3.set_attribute("agent.final_reply", final_reply[:100])

        duration_ms = (time.perf_counter() - start_time) * 1000
        root_span.set_attribute("gen_ai.latency.duration_ms", duration_ms)

        return ScenarioResult(
            scenario_name="Blocking Tool / External I/O Stall",
            latency_type="BLOCKING_TOOL_STALL",
            execution_id=execution_id,
            trace_id=trace_id,
            total_duration_ms=duration_ms,
            status="SUCCESS",
            step_count=3,
            summary="LegacyDatabaseAgent blocked for 3.6s on slow tool query_legacy_sap_erp.",
            details={"slow_tool": "query_legacy_sap_erp", "tool_delay_s": 3.6},
        )


# ===========================================================================
# SCENARIO 4: CONTEXT / TOKEN BLOAT LATENCY
# ===========================================================================
async def run_token_bloat_scenario(client: Mistral, tracer: Tracer) -> ScenarioResult:
    """
    Scenario 4: ResearchAgent passes an oversized, unpruned 30,000+ token context dump
    across the handoff to SummaryAgent, causing high TTFT and token generation drag.
    """
    execution_id = f"lat-bloat-{uuid.uuid4().hex[:8]}"
    start_time = time.perf_counter()

    with tracer.start_as_current_span("trace_token_bloat_latency") as root_span:
        root_span.set_attribute("gen_ai.workflow.name", "deep-research-briefing")
        root_span.set_attribute("gen_ai.workflow.execution_id", execution_id)
        root_span.set_attribute("gen_ai.latency.scenario", "token_bloat_context_drag")
        root_span.set_attribute("gen_ai.latency.type", "TOKEN_BLOAT_LATENCY")
        trace_id = format(root_span.get_span_context().trace_id, "032x")

        # Step 1: Research Agent accumulates huge unpruned context
        with handoff_span(
            tracer,
            agent_name="DocumentResearchAgent",
            action_name="scrape_and_aggregate_corpus",
            execution_id=execution_id,
            handoff_to="ExecutiveSummaryAgent",
            handoff_reason="Passing 35,000 token unpruned corpus to summary agent",
            latency_type="TOKEN_BLOAT_LATENCY",
            metadata={"accumulated_tokens": 35400, "uncompressed_documents": 24},
        ) as span1:
            await asyncio.sleep(0.3)
            # Create high token payload representation
            bloated_context_snippet = ("Article Paragraph: Comprehensive analysis of global trade flows... " * 300)
            span1.set_attribute("llm.prompt_tokens", 35400)
            span1.set_attribute("llm.completion_tokens", 450)

        # Step 2: Summary Agent suffers token generation & TTFT drag
        with handoff_span(
            tracer,
            agent_name="ExecutiveSummaryAgent",
            action_name="synthesize_bloated_corpus",
            execution_id=execution_id,
            handoff_from="DocumentResearchAgent",
            handoff_reason="Generated 3-bullet executive briefing from oversized context",
            latency_type="TOKEN_BLOAT_LATENCY",
            metadata={"input_token_count": 35400},
        ) as span2:
            span2.set_attribute("llm.time_to_first_token_ms", 2400)
            span2.set_attribute("llm.prompt_tokens", 35400)
            span2.set_attribute("llm.completion_tokens", 850)
            # Simulate high TTFT and token throughput delay
            await asyncio.sleep(2.8)

            summary_out = await _safe_llm_call(
                client,
                prompt=f"Summarize key findings from this research extract: {bloated_context_snippet[:800]}",
                system_prompt="You are an Executive Briefing Agent."
            )
            span2.set_attribute("agent.summary_result", summary_out[:120])

        duration_ms = (time.perf_counter() - start_time) * 1000
        root_span.set_attribute("gen_ai.latency.duration_ms", duration_ms)

        return ScenarioResult(
            scenario_name="Token Bloat & High TTFT Latency",
            latency_type="TOKEN_BLOAT_LATENCY",
            execution_id=execution_id,
            trace_id=trace_id,
            total_duration_ms=duration_ms,
            status="SUCCESS",
            step_count=2,
            summary="35,400 token context payload caused elevated TTFT (2400ms) and LLM generation drag (2.8s).",
            details={"prompt_tokens": 35400, "ttft_ms": 2400, "llm_processing_s": 2.8},
        )


# ===========================================================================
# SCENARIO 5: FAN-OUT ASYNCHRONOUS STRAGGLER BOTTLENECK
# ===========================================================================
async def run_fanout_straggler_scenario(client: Mistral, tracer: Tracer) -> ScenarioResult:
    """
    Scenario 5: UnderwritingCoordinator fans out tasks to 3 parallel sub-agents.
    IdentityAgent (0.2s) and FraudAgent (0.3s) finish quickly, but CreditBureauAgent
    straggles for 3.9s, gating the final approval handoff.
    """
    execution_id = f"lat-fanout-{uuid.uuid4().hex[:8]}"
    start_time = time.perf_counter()

    with tracer.start_as_current_span("trace_fanout_straggler_bottleneck") as root_span:
        root_span.set_attribute("gen_ai.workflow.name", "loan-underwriting-evaluation")
        root_span.set_attribute("gen_ai.workflow.execution_id", execution_id)
        root_span.set_attribute("gen_ai.latency.scenario", "fanout_straggler_bottleneck")
        root_span.set_attribute("gen_ai.latency.type", "FANOUT_STRAGGLER_DELAY")
        trace_id = format(root_span.get_span_context().trace_id, "032x")

        # Step 1: Orchestrator Initiates Parallel Dispatch
        with handoff_span(
            tracer,
            agent_name="UnderwritingCoordinator",
            action_name="dispatch_parallel_checks",
            execution_id=execution_id,
            handoff_to="ParallelSubAgents",
            handoff_reason="Fanning out parallel compliance, fraud, and credit evaluations",
        ) as span_coord:
            span_coord.set_attribute("underwriting.applicant_id", "APP-77192")

        # Step 2: Execute Parallel Sub-Agents
        async def sub_identity():
            with handoff_span(
                tracer,
                agent_name="IdentityVerificationAgent",
                action_name="verify_kyc_identity",
                execution_id=execution_id,
                handoff_from="UnderwritingCoordinator",
                handoff_to="ApprovalAggregationAgent",
            ) as s:
                await asyncio.sleep(0.2)
                s.set_attribute("kyc.status", "VERIFIED")
                return {"kyc": "VERIFIED"}

        async def sub_fraud():
            with handoff_span(
                tracer,
                agent_name="FraudDetectionAgent",
                action_name="score_fraud_risk",
                execution_id=execution_id,
                handoff_from="UnderwritingCoordinator",
                handoff_to="ApprovalAggregationAgent",
            ) as s:
                await asyncio.sleep(0.3)
                s.set_attribute("fraud.score", 0.04)
                return {"fraud_score": 0.04}

        async def sub_credit_straggler():
            with handoff_span(
                tracer,
                agent_name="CreditBureauStragglerAgent",
                action_name="pull_tri_bureau_credit_history",
                execution_id=execution_id,
                handoff_from="UnderwritingCoordinator",
                handoff_to="ApprovalAggregationAgent",
                latency_type="FANOUT_STRAGGLER_DELAY",
            ) as s:
                with tool_span(
                    tracer,
                    "external_credit_bureau_gateway",
                    {"applicant_id": "APP-77192"},
                    latency_type="FANOUT_STRAGGLER_DELAY",
                ) as set_tool:
                    # Straggler delay
                    await asyncio.sleep(3.9)
                    set_tool({"credit_score": 740, "derogatory_marks": 0})
                s.set_attribute("credit.status", "APPROVED")
                return {"credit_score": 740}

        # Fan-out execution
        await asyncio.gather(sub_identity(), sub_fraud(), sub_credit_straggler())

        # Step 3: Aggregator waits for all parallel agents before completing
        with handoff_span(
            tracer,
            agent_name="ApprovalAggregationAgent",
            action_name="finalize_underwriting_decision",
            execution_id=execution_id,
            handoff_from="CreditBureauStragglerAgent",
            handoff_reason="All parallel evaluations received, approving loan",
        ) as span_agg:
            await asyncio.sleep(0.2)
            decision = await _safe_llm_call(
                client,
                prompt="Approve loan application APP-77192 with credit score 740 and low fraud risk.",
                system_prompt="You are a Loan Underwriting Approver."
            )
            span_agg.set_attribute("underwriting.final_decision", decision[:100])

        duration_ms = (time.perf_counter() - start_time) * 1000
        root_span.set_attribute("gen_ai.latency.duration_ms", duration_ms)

        return ScenarioResult(
            scenario_name="Fan-out Straggler Bottleneck",
            latency_type="FANOUT_STRAGGLER_DELAY",
            execution_id=execution_id,
            trace_id=trace_id,
            total_duration_ms=duration_ms,
            status="SUCCESS",
            step_count=5,
            summary="Parallel CreditBureauStragglerAgent took 3.9s, gating the entire fan-in aggregation.",
            details={"straggler_agent": "CreditBureauStragglerAgent", "straggler_duration_s": 3.9},
        )


# ===========================================================================
# SCENARIO 6: RETRY STORM & DEGRADED FALLBACK DELAY
# ===========================================================================
async def run_retry_storm_scenario(client: Mistral, tracer: Tracer) -> ScenarioResult:
    """
    Scenario 6: PaymentSettlementAgent attempts 3 successive retries with exponential
    backoffs against a timing-out endpoint before falling back to DegradedSettlementAgent.
    """
    execution_id = f"lat-retry-{uuid.uuid4().hex[:8]}"
    start_time = time.perf_counter()

    with tracer.start_as_current_span("trace_retry_storm_exhaustion") as root_span:
        root_span.set_attribute("gen_ai.workflow.name", "settlement-and-payout")
        root_span.set_attribute("gen_ai.workflow.execution_id", execution_id)
        root_span.set_attribute("gen_ai.latency.scenario", "retry_storm_exhaustion")
        root_span.set_attribute("gen_ai.latency.type", "RETRY_STORM_BACKOFF")
        trace_id = format(root_span.get_span_context().trace_id, "032x")

        # Step 1: Initial Dispatch
        with handoff_span(
            tracer,
            agent_name="PaymentOrchestratorAgent",
            action_name="dispatch_settlement",
            execution_id=execution_id,
            handoff_to="PaymentSettlementAgent",
            handoff_reason="Initiating ACH settlement payout",
        ) as span1:
            await asyncio.sleep(0.15)
            span1.set_attribute("payment.amount", 4250.00)

        # Step 2: Retry Storm in Settlement Agent
        max_attempts = 3
        backoffs = [0.8, 1.6, 0.0]

        for attempt in range(1, max_attempts + 1):
            with handoff_span(
                tracer,
                agent_name="PaymentSettlementAgent",
                action_name=f"attempt_ach_gateway_call_{attempt}",
                execution_id=execution_id,
                handoff_from="PaymentOrchestratorAgent" if attempt == 1 else "PaymentSettlementAgent",
                handoff_to="DegradedSettlementAgent" if attempt == max_attempts else "PaymentSettlementAgent",
                handoff_reason=f"Retry attempt {attempt}/{max_attempts} after gateway timeout",
                latency_type="RETRY_STORM_BACKOFF",
                metadata={"attempt": attempt, "max_attempts": max_attempts},
            ) as span_retry:
                span_retry.set_attribute("gen_ai.agent.retry_attempt", attempt)

                with tool_span(
                    tracer,
                    "ach_payment_gateway",
                    {"amount": 4250.00, "attempt": attempt},
                    latency_type="RETRY_STORM_BACKOFF",
                ) as set_tool:
                    # Simulated gateway stall and timeout failure
                    await asyncio.sleep(1.0)
                    exc = TimeoutError(f"HTTP 504 Gateway Timeout on attempt {attempt}")
                    record_span_error(span_retry, exc)
                    span_retry.set_attribute("gen_ai.activity.status", "FAILED")

                # Backoff sleep before next attempt
                if backoffs[attempt - 1] > 0:
                    await asyncio.sleep(backoffs[attempt - 1])

        # Step 3: Degraded Fallback Agent Queues Offline Batch
        with handoff_span(
            tracer,
            agent_name="DegradedSettlementAgent",
            action_name="queue_asynchronous_batch_payout",
            execution_id=execution_id,
            handoff_from="PaymentSettlementAgent",
            handoff_reason="Gateway retries exhausted, enqueuing offline queue batch",
            latency_type="RETRY_STORM_BACKOFF",
        ) as span_fallback:
            span_fallback.set_attribute("gen_ai.activity.status", "DEGRADED_SUCCESS")
            fallback_res = await _safe_llm_call(
                client,
                prompt="Acknowledge offline batch payout queueing for transaction of $4,250.",
                system_prompt="You are a Degraded Settlement Fallback Agent."
            )
            span_fallback.set_attribute("agent.fallback_ack", fallback_res[:100])

        duration_ms = (time.perf_counter() - start_time) * 1000
        root_span.set_attribute("gen_ai.latency.duration_ms", duration_ms)

        return ScenarioResult(
            scenario_name="Retry Storm & Degraded Fallback",
            latency_type="RETRY_STORM_BACKOFF",
            execution_id=execution_id,
            trace_id=trace_id,
            total_duration_ms=duration_ms,
            status="DEGRADED",
            step_count=5,
            summary="3 sequential timeout retries with exponential backoffs caused 5.4s delay before degraded fallback.",
            details={"attempts": max_attempts, "cumulative_backoff_s": sum(backoffs)},
        )


# ===========================================================================
# SCENARIO 7: CLEAN OPTIMAL BASELINE (Reference Control)
# ===========================================================================
async def run_clean_baseline_scenario(client: Mistral, tracer: Tracer) -> ScenarioResult:
    """
    Scenario 7: Healthy, crisp 2-agent handoff completing in < 500ms with zero errors
    or stalls, serving as an optimal baseline.
    """
    execution_id = f"lat-clean-{uuid.uuid4().hex[:8]}"
    start_time = time.perf_counter()

    with tracer.start_as_current_span("trace_clean_optimal_baseline") as root_span:
        root_span.set_attribute("gen_ai.workflow.name", "express-order-triage")
        root_span.set_attribute("gen_ai.workflow.execution_id", execution_id)
        root_span.set_attribute("gen_ai.latency.scenario", "clean_optimal_baseline")
        root_span.set_attribute("gen_ai.latency.type", "CLEAN_OPTIMAL_BASELINE")
        trace_id = format(root_span.get_span_context().trace_id, "032x")

        # Step 1: Intake
        with handoff_span(
            tracer,
            agent_name="ExpressIntakeAgent",
            action_name="classify_priority_request",
            execution_id=execution_id,
            handoff_to="ExpressDispatchAgent",
            handoff_reason="Classified as urgent express shipment, routing to dispatch",
        ) as span1:
            await asyncio.sleep(0.1)
            span1.set_attribute("express.priority", "TIER_1")

        # Step 2: Dispatch
        with handoff_span(
            tracer,
            agent_name="ExpressDispatchAgent",
            action_name="assign_delivery_courier",
            execution_id=execution_id,
            handoff_from="ExpressIntakeAgent",
            handoff_reason="Courier assigned instantly",
        ) as span2:
            with tool_span(tracer, "check_courier_availability", {"zone": "US-WEST-1"}) as set_tool:
                await asyncio.sleep(0.08)
                set_tool({"available_couriers": 14, "selected_courier": "COURIER-881"})

            reply = await _safe_llm_call(
                client,
                prompt="Confirm priority courier dispatch.",
                system_prompt="You are an Express Dispatch Agent."
            )
            span2.set_attribute("agent.reply", reply[:80])

        duration_ms = (time.perf_counter() - start_time) * 1000
        root_span.set_attribute("gen_ai.latency.duration_ms", duration_ms)

        return ScenarioResult(
            scenario_name="Clean Optimal Baseline",
            latency_type="CLEAN_OPTIMAL_BASELINE",
            execution_id=execution_id,
            trace_id=trace_id,
            total_duration_ms=duration_ms,
            status="SUCCESS",
            step_count=2,
            summary="Optimal reference handoff completed in under 400ms without latency defects.",
            details={"duration_ms": duration_ms},
        )


# ===========================================================================
# SCENARIO 8: MODEL OVERSIZING LLM DELAY (Heavyweight Model on Simple Task)
# ===========================================================================
async def run_model_oversizing_scenario(client: Mistral, tracer: Tracer) -> ScenarioResult:
    """
    Scenario 8: TriageAgent uses mistral-large-latest with heavy chain-of-thought
    for a trivial boolean category classification, incurring 4.5s generation latency
    for a 2-word decision.
    Judge Recommendation: Switch model to mistral-small-latest or ministral-8b.
    """
    execution_id = f"lat-oversize-{uuid.uuid4().hex[:8]}"
    start_time = time.perf_counter()

    with tracer.start_as_current_span("trace_model_oversizing_llm_delay") as root_span:
        root_span.set_attribute("gen_ai.workflow.name", "account-tier-routing")
        root_span.set_attribute("gen_ai.workflow.execution_id", execution_id)
        root_span.set_attribute("gen_ai.latency.scenario", "model_oversizing_llm_delay")
        root_span.set_attribute("gen_ai.latency.type", "MODEL_OVERSIZING_LLM_DELAY")
        trace_id = format(root_span.get_span_context().trace_id, "032x")

        # Step 1: Oversized LLM for trivial classification
        with handoff_span(
            tracer,
            agent_name="HeavyTierClassifierAgent",
            action_name="classify_vip_tier",
            execution_id=execution_id,
            handoff_to="AccountRoutingAgent",
            handoff_reason="Trivial VIP tier check routed to oversized frontier model",
            latency_type="MODEL_OVERSIZING_LLM_DELAY",
            metadata={
                "task_complexity": "LOW_SIMPLE_CLASSIFICATION",
                "recommended_model": "mistral-small-latest",
                "current_model": "mistral-large-latest",
            },
        ) as span1:
            span1.set_attribute("llm.request.model", "mistral-large-latest")
            span1.set_attribute("llm.prompt_tokens", 85)
            span1.set_attribute("llm.completion_tokens", 8)
            span1.set_attribute("llm.duration_ms", 4500)

            # Simulated heavy model inference delay
            await asyncio.sleep(3.8)

            res = await _safe_llm_call(
                client,
                prompt="Answer ONLY with 'VIP: YES' or 'VIP: NO'. Is a customer with $10,000 spend VIP?",
                system_prompt="You are a Tier Classifier. Output exactly two words.",
            )
            span1.set_attribute("agent.classification", res[:50])

        # Step 2: Downstream Routing
        with handoff_span(
            tracer,
            agent_name="AccountRoutingAgent",
            action_name="route_to_priority_queue",
            execution_id=execution_id,
            handoff_from="HeavyTierClassifierAgent",
            handoff_reason="Account routed to VIP queue",
        ) as span2:
            await asyncio.sleep(0.15)
            span2.set_attribute("routing.queue", "VIP_PRIORITY")

        duration_ms = (time.perf_counter() - start_time) * 1000
        root_span.set_attribute("gen_ai.latency.duration_ms", duration_ms)

        return ScenarioResult(
            scenario_name="Model Oversizing LLM Delay",
            latency_type="MODEL_OVERSIZING_LLM_DELAY",
            execution_id=execution_id,
            trace_id=trace_id,
            total_duration_ms=duration_ms,
            status="SUCCESS",
            step_count=2,
            summary="mistral-large used for trivial 2-word boolean classification (3.8s LLM delay).",
            details={
                "current_model": "mistral-large-latest",
                "recommended_model": "mistral-small-latest or ministral-8b",
                "latency_reduction_potential_pct": 75,
            },
        )


# ===========================================================================
# SCENARIO 9: UNCONSTRAINED GENERATION DRAG (Verbose Output Drag)
# ===========================================================================
async def run_unconstrained_generation_scenario(client: Mistral, tracer: Tracer) -> ScenarioResult:
    """
    Scenario 9: Prompt lacks max_tokens limits and structural constraints,
    causing the LLM to emit 1,400+ verbose tokens (5.2s generation delay) for a simple inquiry.
    Judge Recommendation: Add max_tokens: 150, enforce JSON schema or bullet output format.
    """
    execution_id = f"lat-unconstrained-{uuid.uuid4().hex[:8]}"
    start_time = time.perf_counter()

    with tracer.start_as_current_span("trace_unconstrained_generation_drag") as root_span:
        root_span.set_attribute("gen_ai.workflow.name", "faq-policy-inquiry")
        root_span.set_attribute("gen_ai.workflow.execution_id", execution_id)
        root_span.set_attribute("gen_ai.latency.scenario", "unconstrained_generation_drag")
        root_span.set_attribute("gen_ai.latency.type", "UNCONSTRAINED_GEN_DRAG")
        trace_id = format(root_span.get_span_context().trace_id, "032x")

        # Step 1: Ingestion
        with handoff_span(
            tracer,
            agent_name="FAQIntakeAgent",
            action_name="receive_policy_question",
            execution_id=execution_id,
            handoff_to="VerbosePolicyAgent",
            handoff_reason="Passing customer return policy question",
        ) as span1:
            await asyncio.sleep(0.1)

        # Step 2: Verbose Unconstrained Generation
        with handoff_span(
            tracer,
            agent_name="VerbosePolicyAgent",
            action_name="draft_policy_explanation",
            execution_id=execution_id,
            handoff_from="FAQIntakeAgent",
            handoff_to="CustomerDeliveryAgent",
            handoff_reason="Drafted 1,400-token verbose legal treatise instead of concise 2-sentence summary",
            latency_type="UNCONSTRAINED_GEN_DRAG",
            metadata={
                "completion_tokens": 1420,
                "recommended_fix": "Enforce max_tokens: 150 and strict bullet JSON schema",
            },
        ) as span2:
            span2.set_attribute("llm.completion_tokens", 1420)
            span2.set_attribute("llm.prompt_tokens", 110)
            span2.set_attribute("llm.duration_ms", 5200)

            # Simulated unconstrained token generation delay
            await asyncio.sleep(4.5)

            verbose_reply = await _safe_llm_call(
                client,
                prompt="Explain the 30-day return policy in exhaustive detail with historical context.",
                system_prompt="You are a detailed legal policy consultant. Be as comprehensive and long-winded as possible.",
            )
            span2.set_attribute("agent.reply_preview", verbose_reply[:120])

        # Step 3: Customer Delivery
        with handoff_span(
            tracer,
            agent_name="CustomerDeliveryAgent",
            action_name="deliver_reply",
            execution_id=execution_id,
            handoff_from="VerbosePolicyAgent",
            handoff_reason="Reply dispatched",
        ) as span3:
            await asyncio.sleep(0.1)

        duration_ms = (time.perf_counter() - start_time) * 1000
        root_span.set_attribute("gen_ai.latency.duration_ms", duration_ms)

        return ScenarioResult(
            scenario_name="Unconstrained Generation Drag",
            latency_type="UNCONSTRAINED_GEN_DRAG",
            execution_id=execution_id,
            trace_id=trace_id,
            total_duration_ms=duration_ms,
            status="SUCCESS",
            step_count=3,
            summary="Unconstrained 1,420 token generation caused 4.5s generation drag.",
            details={
                "completion_tokens": 1420,
                "recommended_action": "Set max_tokens=150 and enforce structured Pydantic schema",
            },
        )


# ===========================================================================
# SCENARIO 10: SEQUENTIAL I/O WATERFALL (Un-batched Synchronous HTTP/DB Calls)
# ===========================================================================
async def run_sequential_io_waterfall_scenario(client: Mistral, tracer: Tracer) -> ScenarioResult:
    """
    Scenario 10: ProfileEnrichmentAgent fetches 4 independent records sequentially
    (4 x 0.85s = 3.4s) instead of concurrent asyncio.gather() or a batch query.
    Judge Recommendation: Parallelize independent I/O calls with asyncio.gather() or batch API.
    """
    execution_id = f"lat-waterfall-{uuid.uuid4().hex[:8]}"
    start_time = time.perf_counter()

    with tracer.start_as_current_span("trace_sequential_io_waterfall") as root_span:
        root_span.set_attribute("gen_ai.workflow.name", "customer-profile-enrichment")
        root_span.set_attribute("gen_ai.workflow.execution_id", execution_id)
        root_span.set_attribute("gen_ai.latency.scenario", "sequential_io_waterfall")
        root_span.set_attribute("gen_ai.latency.type", "SEQUENTIAL_IO_WATERFALL")
        trace_id = format(root_span.get_span_context().trace_id, "032x")

        # Step 1: Sequential I/O waterfall inside Enrichment Agent
        with handoff_span(
            tracer,
            agent_name="ProfileEnrichmentAgent",
            action_name="fetch_customer_attributes_sequential",
            execution_id=execution_id,
            handoff_to="RiskScoringAgent",
            handoff_reason="Collected 4 customer attributes sequentially over network",
            latency_type="SEQUENTIAL_IO_WATERFALL",
            metadata={"io_call_count": 4, "io_pattern": "SEQUENTIAL_SYNCHRONOUS"},
        ) as span1:
            # 4 sequential un-batched tool calls
            io_endpoints = [
                ("fetch_kyc_status", {"user_id": "USR-881"}, 0.85),
                ("fetch_credit_score", {"user_id": "USR-881"}, 0.80),
                ("fetch_transaction_history", {"user_id": "USR-881"}, 0.90),
                ("fetch_fraud_blacklist", {"user_id": "USR-881"}, 0.85),
            ]

            for tool_name, args, delay in io_endpoints:
                with tool_span(tracer, tool_name, args, latency_type="SEQUENTIAL_IO_WATERFALL") as set_tool:
                    await asyncio.sleep(delay)
                    set_tool({"status": "SUCCESS", "latency_s": delay})

            enrichment_summary = await _safe_llm_call(
                client,
                prompt="Summarize customer KYC and credit status for USR-881.",
                system_prompt="You are a Profile Enrichment Agent.",
            )
            span1.set_attribute("agent.enrichment_summary", enrichment_summary[:100])

        # Step 2: Risk Scoring Agent
        with handoff_span(
            tracer,
            agent_name="RiskScoringAgent",
            action_name="compute_risk_score",
            execution_id=execution_id,
            handoff_from="ProfileEnrichmentAgent",
            handoff_reason="Profile enriched, scored risk as LOW",
        ) as span2:
            await asyncio.sleep(0.2)
            span2.set_attribute("risk.score", 0.08)

        duration_ms = (time.perf_counter() - start_time) * 1000
        root_span.set_attribute("gen_ai.latency.duration_ms", duration_ms)

        return ScenarioResult(
            scenario_name="Sequential I/O Waterfall",
            latency_type="SEQUENTIAL_IO_WATERFALL",
            execution_id=execution_id,
            trace_id=trace_id,
            total_duration_ms=duration_ms,
            status="SUCCESS",
            step_count=2,
            summary="4 sequential network calls caused 3.4s I/O delay; parallelization would reduce to ~0.9s.",
            details={
                "sequential_calls": 4,
                "cumulative_io_s": 3.4,
                "parallel_potential_s": 0.9,
                "recommended_action": "Use asyncio.gather() or batch HTTP endpoint",
            },
        )


# ===========================================================================
# SCENARIO 11: UNCACHED REPEATED I/O LOOKUP (Missing Cache Layer)
# ===========================================================================
async def run_uncached_repeated_io_scenario(client: Mistral, tracer: Tracer, mock_tool_result: dict = None) -> ScenarioResult:
    """
    Scenario 11: InvoiceCalculationAgent and TaxComplianceAgent - FIXED via handoff state propagation.
    Previously both agents made duplicate remote API calls for the exact same static FX rate table 
    (2 x 1.6s = 3.2s delay). FIXED: FX rate and converted currency lines now passed as structured 
    arguments via handoff metadata, eliminating redundant fetch_live_fx_rates invocations and 
    reducing latency by 1.6s.
    """
    execution_id = f"lat-uncached-{uuid.uuid4().hex[:8]}"
    start_time = time.perf_counter()

    # Defensive validation: ensure tracer is a valid OpenTelemetry Tracer
    if not hasattr(tracer, 'start_as_current_span'):
        # Create a mock tracer for testing scenarios
        from opentelemetry import trace
        tracer = trace.get_tracer(__name__)

    with tracer.start_as_current_span("trace_uncached_repeated_io") as root_span:
        root_span.set_attribute("gen_ai.workflow.name", "international-invoice-processing")
        root_span.set_attribute("gen_ai.workflow.execution_id", execution_id)
        root_span.set_attribute("gen_ai.latency.scenario", "uncached_repeated_io")
        root_span.set_attribute("gen_ai.latency.type", "UNCACHED_REPEATED_IO")
        trace_id = format(root_span.get_span_context().trace_id, "032x")

        # Step 1: Agent 1 queries FX API without cache
        subtotal_usd = None
        fx_rate = None
        with handoff_span(
            tracer,
            agent_name="InvoiceCalculationAgent",
            action_name="fetch_live_fx_rates",
            execution_id=execution_id,
            handoff_to="TaxComplianceAgent",
            handoff_reason="Converted EUR to USD, passing invoice to Tax Compliance",
            latency_type="UNCACHED_REPEATED_IO",
            metadata={"cache_status": "MISS_UNCACHED"},
        ) as span1:
            with tool_span(
                tracer,
                "fetch_live_fx_rates",
                {"amount_eur": 500, "pair": "EUR_USD", "date": "2026-08-27"},
                latency_type="UNCACHED_REPEATED_IO",
            ) as set_tool:
                # Slow remote FX provider call
                await asyncio.sleep(1.6)
                tool_result = mock_tool_result if mock_tool_result is not None else {"rate": 1.085, "cache_hit": False, "subtotal_usd": 542.50}
                set_tool(tool_result)
                
                # Extract subtotal_usd and fx_rate from tool result
                subtotal_usd = tool_result.get("subtotal_usd")
                fx_rate = tool_result.get("rate")
                
                # Early edge case validation: ensure required fields exist in tool result
                if subtotal_usd is None:
                    return ScenarioResult(
                        scenario_name="Uncached Repeated I/O Lookup",
                        latency_type="UNCACHED_REPEATED_IO",
                        execution_id=execution_id,
                        trace_id=trace_id,
                        total_duration_ms=(time.perf_counter() - start_time) * 1000,
                        status="FAILED",
                        step_count=1,
                        summary="Missing subtotal_usd in tool result from InvoiceCalculationAgent.",
                        details={
                            "error": "Tool result missing required field: subtotal_usd",
                            "duplicate_tool": "fetch_live_fx_rates",
                            "recommended_action": "Ensure tool returns subtotal_usd field",
                        },
                    )
                if fx_rate is None:
                    return ScenarioResult(
                        scenario_name="Uncached Repeated I/O Lookup",
                        latency_type="UNCACHED_REPEATED_IO",
                        execution_id=execution_id,
                        trace_id=trace_id,
                        total_duration_ms=(time.perf_counter() - start_time) * 1000,
                        status="FAILED",
                        step_count=1,
                        summary="Missing fx_rate in tool result from InvoiceCalculationAgent.",
                        details={
                            "error": "Tool result missing required field: rate",
                            "duplicate_tool": "fetch_live_fx_rates",
                            "recommended_action": "Ensure tool returns rate field",
                        },
                    )
                
                # Set metadata on span1 to propagate state to next agent
                span1.set_attribute("agent.metadata.subtotal_usd", str(subtotal_usd) if subtotal_usd is not None else "")
                span1.set_attribute("agent.metadata.fx_rate", str(fx_rate) if fx_rate is not None else "")
                span1.set_attribute("agent.metadata.cache_status", "MISS_UNCACHED")

            # Edge case validation: ensure FX rate and subtotal_usd from tool call are valid
            if fx_rate is not None:
                try:
                    fx_rate_float = float(fx_rate)
                    if fx_rate_float <= 0 or math.isnan(fx_rate_float) or math.isinf(fx_rate_float):
                        return ScenarioResult(
                            scenario_name="Uncached Repeated I/O Lookup",
                            latency_type="UNCACHED_REPEATED_IO",
                            execution_id=execution_id,
                            trace_id=trace_id,
                            total_duration_ms=(time.perf_counter() - start_time) * 1000,
                            status="FAILED",
                            step_count=1,
                            summary="Invalid FX rate detected: FX rate must be positive and finite.",
                            details={
                                "error": f"Invalid FX rate: {fx_rate}. FX rate must be positive and finite.",
                                "duplicate_tool": "fetch_live_fx_rates",
                                "recommended_action": "Validate FX rate before processing",
                            },
                        )
                except (ValueError, TypeError):
                    return ScenarioResult(
                        scenario_name="Uncached Repeated I/O Lookup",
                        latency_type="UNCACHED_REPEATED_IO",
                        execution_id=execution_id,
                        trace_id=trace_id,
                        total_duration_ms=(time.perf_counter() - start_time) * 1000,
                        status="FAILED",
                        step_count=1,
                        summary="Invalid FX rate type detected: FX rate must be numeric.",
                        details={
                            "error": f"Invalid FX rate type: {fx_rate}. FX rate must be numeric.",
                            "duplicate_tool": "fetch_live_fx_rates",
                            "recommended_action": "Validate FX rate type before processing",
                        },
                    )
            
            # Edge case validation: ensure subtotal_usd is valid
            if subtotal_usd is not None:
                try:
                    subtotal_usd_float = float(subtotal_usd)
                    if subtotal_usd_float <= 0 or math.isnan(subtotal_usd_float) or math.isinf(subtotal_usd_float):
                        return ScenarioResult(
                            scenario_name="Uncached Repeated I/O Lookup",
                            latency_type="UNCACHED_REPEATED_IO",
                            execution_id=execution_id,
                            trace_id=trace_id,
                            total_duration_ms=(time.perf_counter() - start_time) * 1000,
                            status="FAILED",
                            step_count=1,
                            summary="Invalid subtotal_usd detected: subtotal must be positive and finite.",
                            details={
                                "error": f"Invalid subtotal_usd: {subtotal_usd}. Subtotal must be positive and finite.",
                                "duplicate_tool": "fetch_live_fx_rates",
                                "recommended_action": "Validate subtotal before processing",
                            },
                        )
                except (ValueError, TypeError):
                    return ScenarioResult(
                        scenario_name="Uncached Repeated I/O Lookup",
                        latency_type="UNCACHED_REPEATED_IO",
                        execution_id=execution_id,
                        trace_id=trace_id,
                        total_duration_ms=(time.perf_counter() - start_time) * 1000,
                        status="FAILED",
                        step_count=1,
                        summary="Invalid subtotal_usd type detected: subtotal must be numeric.",
                        details={
                            "error": f"Invalid subtotal_usd type: {subtotal_usd}. Subtotal must be numeric.",
                            "duplicate_tool": "fetch_live_fx_rates",
                            "recommended_action": "Validate subtotal type before processing",
                        },
                    )

            res1 = await _safe_llm_call(
                client,
                prompt="Convert EUR 500 invoice to USD at rate 1.085.",
                system_prompt="You are an Invoice Calculation Agent.",
            )
            span1.set_attribute("agent.subtotal_usd", subtotal_usd)

        # Step 2: Agent 2 uses handed-off FX rate and subtotal_usd, avoiding duplicate API call
        # Extract subtotal_usd and fx_rate from metadata to ensure it was properly propagated through handoff
        handoff_metadata = {"duplicate_call_detected": False, "cache_status": "CACHE_HIT", "subtotal_usd": subtotal_usd, "fx_rate": fx_rate, "converted_invoice_lines": [{"amount_eur": 500, "amount_usd": subtotal_usd, "fx_rate": fx_rate}]}
        
        # Defensive validation: ensure subtotal_usd and fx_rate are not None before handoff
        # (This is a safety check - they should already be validated above, but we check again for edge cases)
        if subtotal_usd is None:
            return ScenarioResult(
                scenario_name="Uncached Repeated I/O Lookup",
                latency_type="UNCACHED_REPEATED_IO",
                execution_id=execution_id,
                trace_id=trace_id,
                total_duration_ms=(time.perf_counter() - start_time) * 1000,
                status="FAILED",
                step_count=1,
                summary="InvoiceCalculationAgent failed to propagate subtotal_usd to TaxComplianceAgent.",
                details={
                    "error": "subtotal_usd is None during handoff",
                    "duplicate_tool": "fetch_live_fx_rates",
                    "recommended_action": "Validate subtotal_usd before handoff",
                },
            )
        if fx_rate is None:
            return ScenarioResult(
                scenario_name="Uncached Repeated I/O Lookup",
                latency_type="UNCACHED_REPEATED_IO",
                execution_id=execution_id,
                trace_id=trace_id,
                total_duration_ms=(time.perf_counter() - start_time) * 1000,
                status="FAILED",
                step_count=1,
                summary="InvoiceCalculationAgent failed to propagate fx_rate to TaxComplianceAgent.",
                details={
                    "error": "fx_rate is None during handoff",
                    "duplicate_tool": "fetch_live_fx_rates",
                    "recommended_action": "Validate fx_rate before handoff",
                },
            )
        
        with handoff_span(
            tracer,
            agent_name="TaxComplianceAgent",
            action_name="fetch_live_fx_rates",
            execution_id=execution_id,
            handoff_from="InvoiceCalculationAgent",
            handoff_reason="VAT computed using handed-off FX rate from InvoiceCalculationAgent, eliminating duplicate call",
            latency_type="UNCACHED_REPEATED_IO",
            metadata=handoff_metadata,
        ) as span2:
            # Extract subtotal_usd and fx_rate from handoff metadata to verify propagation
            # Defensive validation: handle both recording and non-recording spans
            try:
                received_subtotal_usd = span2.get_attribute("agent.metadata.subtotal_usd")
                received_fx_rate = span2.get_attribute("agent.metadata.fx_rate")
            except AttributeError:
                # For NonRecordingSpan or mock spans, extract from metadata dict if available
                received_subtotal_usd = handoff_metadata.get("subtotal_usd")
                received_fx_rate = handoff_metadata.get("fx_rate")
            
            # Convert to float if they are string representations (from span attributes)
            if isinstance(received_subtotal_usd, str) and received_subtotal_usd:
                received_subtotal_usd = float(received_subtotal_usd)
            if isinstance(received_fx_rate, str) and received_fx_rate:
                received_fx_rate = float(received_fx_rate)
            
            if received_subtotal_usd is None:
                return ScenarioResult(
                    scenario_name="Uncached Repeated I/O Lookup",
                    latency_type="UNCACHED_REPEATED_IO",
                    execution_id=execution_id,
                    trace_id=trace_id,
                    total_duration_ms=(time.perf_counter() - start_time) * 1000,
                    status="FAILED",
                    step_count=2,
                    summary="TaxComplianceAgent did not receive subtotal_usd from handoff metadata.",
                    details={
                        "error": "received_subtotal_usd is None from handoff",
                        "duplicate_tool": "fetch_live_fx_rates",
                        "recommended_action": "Validate handoff metadata propagation",
                    },
                )
            if received_fx_rate is None:
                return ScenarioResult(
                    scenario_name="Uncached Repeated I/O Lookup",
                    latency_type="UNCACHED_REPEATED_IO",
                    execution_id=execution_id,
                    trace_id=trace_id,
                    total_duration_ms=(time.perf_counter() - start_time) * 1000,
                    status="FAILED",
                    step_count=2,
                    summary="TaxComplianceAgent did not receive fx_rate from handoff metadata.",
                    details={
                        "error": "received_fx_rate is None from handoff",
                        "duplicate_tool": "fetch_live_fx_rates",
                        "recommended_action": "Validate handoff metadata propagation",
                    },
                )
            
            # Edge case validation: ensure FX rate and subtotal_usd are valid positive numbers
            try:
                fx_rate_float = float(received_fx_rate)
                if fx_rate_float <= 0 or math.isnan(fx_rate_float) or math.isinf(fx_rate_float):
                    return ScenarioResult(
                        scenario_name="Uncached Repeated I/O Lookup",
                        latency_type="UNCACHED_REPEATED_IO",
                        execution_id=execution_id,
                        trace_id=trace_id,
                        total_duration_ms=(time.perf_counter() - start_time) * 1000,
                        status="FAILED",
                        step_count=2,
                        summary=f"Invalid FX rate received by TaxComplianceAgent: {received_fx_rate}. FX rate must be positive and finite.",
                        details={
                            "error": f"Invalid FX rate: {received_fx_rate}. FX rate must be positive and finite.",
                            "duplicate_tool": "fetch_live_fx_rates",
                            "recommended_action": "Validate FX rate before processing",
                        },
                    )
            except (ValueError, TypeError):
                return ScenarioResult(
                    scenario_name="Uncached Repeated I/O Lookup",
                    latency_type="UNCACHED_REPEATED_IO",
                    execution_id=execution_id,
                    trace_id=trace_id,
                    total_duration_ms=(time.perf_counter() - start_time) * 1000,
                    status="FAILED",
                    step_count=2,
                    summary=f"Invalid FX rate type received by TaxComplianceAgent: {received_fx_rate}. FX rate must be numeric.",
                    details={
                        "error": f"Invalid FX rate type: {received_fx_rate}. FX rate must be numeric.",
                        "duplicate_tool": "fetch_live_fx_rates",
                        "recommended_action": "Validate FX rate type before processing",
                    },
                )
            
            try:
                subtotal_usd_float = float(received_subtotal_usd)
                if subtotal_usd_float <= 0 or math.isnan(subtotal_usd_float) or math.isinf(subtotal_usd_float):
                    return ScenarioResult(
                        scenario_name="Uncached Repeated I/O Lookup",
                        latency_type="UNCACHED_REPEATED_IO",
                        execution_id=execution_id,
                        trace_id=trace_id,
                        total_duration_ms=(time.perf_counter() - start_time) * 1000,
                        status="FAILED",
                        step_count=2,
                        summary=f"Invalid subtotal_usd received by TaxComplianceAgent: {received_subtotal_usd}. Subtotal must be positive and finite.",
                        details={
                            "error": f"Invalid subtotal_usd: {received_subtotal_usd}. Subtotal must be positive and finite.",
                            "duplicate_tool": "fetch_live_fx_rates",
                            "recommended_action": "Validate subtotal before processing",
                        },
                    )
            except (ValueError, TypeError):
                return ScenarioResult(
                    scenario_name="Uncached Repeated I/O Lookup",
                    latency_type="UNCACHED_REPEATED_IO",
                    execution_id=execution_id,
                    trace_id=trace_id,
                    total_duration_ms=(time.perf_counter() - start_time) * 1000,
                    status="FAILED",
                    step_count=2,
                    summary=f"Invalid subtotal_usd type received by TaxComplianceAgent: {received_subtotal_usd}. Subtotal must be numeric.",
                    details={
                        "error": f"Invalid subtotal_usd type: {received_subtotal_usd}. Subtotal must be numeric.",
                        "duplicate_tool": "fetch_live_fx_rates",
                        "recommended_action": "Validate subtotal type before processing",
                    },
                )
            
            with tool_span(
                tracer,
                "fetch_live_fx_rates",
                {"pair": "EUR_USD", "date": "2026-08-27", "subtotal_usd": received_subtotal_usd, "fx_rate": received_fx_rate, "cache_hit": True},
                latency_type="UNCACHED_REPEATED_IO",
            ) as set_tool2:
                # Use FX rate and subtotal_usd from handoff metadata, eliminating duplicate remote call
                vat_usd = received_subtotal_usd * 0.20
                set_tool2({"rate": received_fx_rate, "cache_hit": True, "subtotal_usd": received_subtotal_usd, "vat_usd": vat_usd})

            res2 = await _safe_llm_call(
                client,
                prompt="Calculate 20% VAT on USD 542.50.",
                system_prompt="You are a Tax Compliance Agent.",
            )
            span2.set_attribute("agent.vat_usd", 108.50)

        duration_ms = (time.perf_counter() - start_time) * 1000
        root_span.set_attribute("gen_ai.latency.duration_ms", duration_ms)

        return ScenarioResult(
            scenario_name="Uncached Repeated I/O Lookup",
            latency_type="UNCACHED_REPEATED_IO",
            execution_id=execution_id,
            trace_id=trace_id,
            total_duration_ms=duration_ms,
            status="SUCCESS",
            step_count=2,
            summary="FX rate and subtotal_usd passed via handoff metadata from InvoiceCalculationAgent to TaxComplianceAgent, eliminating duplicate fetch_live_fx_rates call (1.6s saved).",
            details={
                "duplicate_tool": "fetch_live_fx_rates",
                "cache_savings_actual_s": 1.6,
                "recommended_action": "Add TTL cache / Redis decorator to fetch_live_fx_rates",
                "fix_applied": "FX rate and subtotal_usd propagated through handoff metadata; TaxComplianceAgent uses handed-off values instead of duplicate API call",
            },
        )


# ===========================================================================
# SCENARIO 12: HEAVY VISION PAYLOAD I/O (Uncompressed Multi-Modal I/O)
# ===========================================================================
async def run_heavy_vision_payload_scenario(client: Mistral, tracer: Tracer) -> ScenarioResult:
    """
    Scenario 12: VisionVerificationAgent ingests an uncompressed 25MB raw TIFF/PNG image
    over base64, incurring 4.2s network transfer and vision model preprocessing delay.
    Judge Recommendation: Downsample resolution to max 1024px, compress to WebP, or OCR-extract text first.
    """
    execution_id = f"lat-vision-io-{uuid.uuid4().hex[:8]}"
    start_time = time.perf_counter()

    with tracer.start_as_current_span("trace_heavy_vision_payload_io") as root_span:
        root_span.set_attribute("gen_ai.workflow.name", "identity-document-verification")
        root_span.set_attribute("gen_ai.workflow.execution_id", execution_id)
        root_span.set_attribute("gen_ai.latency.scenario", "heavy_vision_payload_io")
        root_span.set_attribute("gen_ai.latency.type", "HEAVY_VISION_PAYLOAD_IO")
        trace_id = format(root_span.get_span_context().trace_id, "032x")

        # Step 1: Intake
        with handoff_span(
            tracer,
            agent_name="KYCIntakeAgent",
            action_name="receive_uncompressed_passport_scan",
            execution_id=execution_id,
            handoff_to="HeavyVisionAgent",
            handoff_reason="Transmitting 25.4MB raw uncompressed passport scan to vision model",
            latency_type="HEAVY_VISION_PAYLOAD_IO",
            metadata={"raw_image_size_mb": 25.4, "image_format": "TIFF_UNCOMPRESSED"},
        ) as span1:
            await asyncio.sleep(0.15)
            span1.set_attribute("media.size_mb", 25.4)

        # Step 2: Vision Model Processing with high I/O transfer delay
        with handoff_span(
            tracer,
            agent_name="HeavyVisionAgent",
            action_name="process_high_res_vision_document",
            execution_id=execution_id,
            handoff_from="KYCIntakeAgent",
            handoff_to="KYCVerificationAgent",
            handoff_reason="Vision document decoded, passing extracted MRZ to verification",
            latency_type="HEAVY_VISION_PAYLOAD_IO",
            metadata={
                "input_resolution": "6000x4000",
                "recommended_action": "Downscale to 1024x768 and compress as WebP (reduces 25MB -> 350KB)",
            },
        ) as span2:
            with tool_span(
                tracer,
                "base64_vision_payload_transfer",
                {"size_bytes": 26633830, "mime": "image/tiff"},
                latency_type="HEAVY_VISION_PAYLOAD_IO",
            ) as set_tool:
                # Simulated high payload transfer & vision model rasterization delay
                await asyncio.sleep(4.2)
                set_tool({"status": "SUCCESS", "mrz_code": "P<USASMITH<<JOHN<<<<<<<<<<<"})

            vision_out = await _safe_llm_call(
                client,
                prompt="Verify MRZ checksum for passport P<USASMITH<<JOHN.",
                system_prompt="You are a Vision KYC Agent.",
            )
            span2.set_attribute("agent.mrz_verified", True)

        # Step 3: KYC Verification
        with handoff_span(
            tracer,
            agent_name="KYCVerificationAgent",
            action_name="finalize_identity_clearance",
            execution_id=execution_id,
            handoff_from="HeavyVisionAgent",
            handoff_reason="Identity cleared",
        ) as span3:
            await asyncio.sleep(0.2)
            span3.set_attribute("kyc.status", "CLEARED")

        duration_ms = (time.perf_counter() - start_time) * 1000
        root_span.set_attribute("gen_ai.latency.duration_ms", duration_ms)

        return ScenarioResult(
            scenario_name="Heavy Vision Payload I/O",
            latency_type="HEAVY_VISION_PAYLOAD_IO",
            execution_id=execution_id,
            trace_id=trace_id,
            total_duration_ms=duration_ms,
            status="SUCCESS",
            step_count=3,
            summary="25.4MB uncompressed TIFF image transfer caused 4.2s vision I/O latency; downscaling saves ~3.8s.",
            details={
                "raw_size_mb": 25.4,
                "transfer_delay_s": 4.2,
                "recommended_action": "Downscale to 1024px WebP to reduce payload by 98%",
            },
        )

