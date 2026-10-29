"""
agent_handoff_scenarios.py
--------------------------
Production-grade multi-agent benchmark suite designed to probe, benchmark,
and evaluate complex latency anomalies and multi-step distributed failure modes.

Scenarios Implemented:
  1. CASCADING_HANDOFF_DELAY    - Cross-Border Trade Customs Clearance & 52-Page Manifest OCR Ingestion Choke
  2. LOOP_THRASH_HANDOFF        - Marine Cargo Insurance Claims Dispute & Multi-Party Arbitration Thrashing
  3. BLOCKING_TOOL_STALL        - Global Treasury Liquidity Sweep & Core Banking Ledger Lock Contention Stall
  4. TOKEN_BLOAT_LATENCY        - Cross-Border M&A Antitrust Due Diligence & 38k Unpruned Context TTFT Drag
  5. FANOUT_STRAGGLER_DELAY     - Commercial Real Estate Loan Syndication & Asynchronous 3D GIS Straggler Bottleneck
  6. RETRY_STORM_BACKOFF        - SWIFT Corporate Wire Transfer Settlement & Cascading Rate-Limit Retry Storm
  7. MODEL_OVERSIZING_LLM_DELAY - Enterprise Helpdesk Support Triage & Frontier Model Compute Overkill
  8. UNCONSTRAINED_GEN_DRAG     - Regulatory Security Incident Disclosure & Unbounded Narrative Output Drag
  9. SEQUENTIAL_IO_WATERFALL    - Defense Contractor Security Clearance & Sequential Multi-Hop Background Check Waterfall
  10. UNCACHED_REPEATED_IO      - Multinational Enterprise Tax Calculation & Uncached Repeated ECB Foreign Exchange Lookups
  11. HEAVY_VISION_PAYLOAD_IO   - Emergency Radiology Diagnostic Triage & 34MB Uncompressed DICOM Image Base64 Drag
  12. CLEAN_OPTIMAL_BASELINE    - High-Throughput E-Commerce Automated Order Fulfillment Reference Pipeline
  13. DISTRIBUTED_LOCK_TIMEOUT  - Institutional FX Cross-Currency Two-Phase Commit Deadlock & Lock Timeout
  14. EXPONENTIAL_CONTEXT_EXPLOSION - Zero-Day Cybersecurity Incident Response & Uncompacted Memory Explosion
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import time
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from opentelemetry import trace
from opentelemetry.trace import Tracer
from mistralai.client import Mistral

try:
    from .latency_telemetry import handoff_span, tool_span, record_span_error
except ImportError:
    from latency_telemetry import handoff_span, tool_span, record_span_error

logger = logging.getLogger(__name__)




# ===========================================================================
# ENTERPRISE DISTRIBUTED SYSTEM FAILURE CLASSES
# ===========================================================================
class DeadlineExceededError(TimeoutError):
    """gRPC / Service Mesh 504 Deadline Exceeded when upstream processing stalls exceed ingress SLA."""
    pass

class MaxHandoffHopsExceededError(RuntimeError):
    """Cyclic agent handoff thrash detected exceeding maximum enterprise conversation hop limits."""
    pass

class DatabaseLockTimeoutError(TimeoutError):
    """HikariCP / SQLAlchemy connection pool timeout & row-level lock contention under concurrent load."""
    pass

class ContextWindowExceededError(ValueError):
    """HTTP 413 Payload Too Large / context quota breach when unpruned documents exceed downstream limits."""
    pass

class AsyncFanoutTimeoutError(TimeoutError):
    """Scatter-gather coordinator join barrier timeout when an async worker branch straggles beyond SLA."""
    pass

class CircuitBreakerOpenException(ConnectionError):
    """Resilience4j / Netflix Hystrix circuit breaker tripped to OPEN state after cascading retry exhaustion."""
    pass

class InferenceQueueTimeoutError(TimeoutError):
    """GPU vLLM / TensorRT-LLM cluster scheduler preemption and KV cache queuing timeout under load."""
    pass

class GatewayReadTimeoutError(TimeoutError):
    """Envoy / Cloudflare / NGINX reverse proxy 504/524 Read Timeout during unconstrained generation."""
    pass

class CumulativeTransactionBudgetExceeded(TimeoutError):
    """Distributed transaction latency budget breached across sequential synchronous microservice hops."""
    pass

class UpstreamRateLimitExceeded(ConnectionError):
    """HTTP 429 Too Many Requests: External provider quota exhausted by unmemoized duplicate queries."""
    pass

class ResourceExhaustedError(ValueError):
    """gRPC StatusCode.RESOURCE_EXHAUSTED: Inbound serialized message payload exceeds 16MB max message length."""
    pass

class TwoPhaseCommitDeadlockException(RuntimeError):
    """Two-Phase Commit (2PC) distributed lock graph cycle detected across cross-region clearing desks."""
    pass

class MemoryAllocationError(MemoryError):
    """Worker memory allocation failure and context buffer explosion during uncompacted binary processing."""
    pass


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
    max_tokens: int = 250,
    temperature: float = 0.2,
    parent_span: Optional[Any] = None,
) -> str:
    """Executes a live Mistral chat completion, syncing telemetry metadata to parent_span."""
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    try:
        response = await client.chat.complete_async(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        content = response.choices[0].message.content or ""
        in_tokens = getattr(response.usage, "prompt_tokens", 0) if hasattr(response, "usage") and response.usage else sum(len(m.get("content", "")) for m in messages) // 4
        out_tokens = getattr(response.usage, "completion_tokens", 0) if hasattr(response, "usage") and response.usage else len(content) // 4

        if parent_span:
            if hasattr(parent_span, "set_token_usage"):
                parent_span.set_token_usage(input_tokens=in_tokens, output_tokens=out_tokens)
            parent_span.set_attribute("gen_ai.request.model", model)
            parent_span.set_attribute("request_model", model)
            parent_span.set_attribute("response_model", model)
            parent_span.set_attribute("gen_ai.response.model", model)
            parent_span.set_attribute("operation_name", "chat")
            msg_json = json.dumps(messages)
            parent_span.set_attribute("input_messages", msg_json)
            parent_span.set_attribute("gen_ai.input.messages", msg_json)

        return content
    except Exception as exc:
        logger.debug(f"LLM call fallback due to: {exc}")
        await asyncio.sleep(0.35)
        fallback_content = f"[Synthesized enterprise completion for: {prompt[:60]}...]"
        in_tok = sum(len(m.get("content", "")) for m in messages) // 4
        out_tok = max(20, len(fallback_content) // 4)

        if parent_span:
            if hasattr(parent_span, "set_token_usage"):
                parent_span.set_token_usage(input_tokens=in_tok, output_tokens=out_tok)
            parent_span.set_attribute("gen_ai.request.model", model)
            parent_span.set_attribute("request_model", model)
            parent_span.set_attribute("response_model", model)
            parent_span.set_attribute("gen_ai.response.model", model)
            parent_span.set_attribute("operation_name", "chat")
            msg_json = json.dumps(messages)
            parent_span.set_attribute("input_messages", msg_json)
            parent_span.set_attribute("gen_ai.input.messages", msg_json)

        return fallback_content


# ===========================================================================
# SCENARIO 1: CASCADING HANDOFF DELAY (Upstream Ingestion Choke)
# ===========================================================================
async def run_cascading_delay_scenario(client: Mistral, tracer: Tracer) -> ScenarioResult:
    """
    Scenario 1: Cross-Border Trade Customs Clearance & Manifest Ingestion Choke.
    Upstream CustomsDocumentParserAgent suffers severe 3.8s OCR rasterization delay
    on a 52-page Bill of Lading, starving downstream Tariff, Sanctions, Duty, and Dossier agents.
    """
    execution_id = f"lat-cascade-{uuid.uuid4().hex[:8]}"
    start_time = time.perf_counter()

    with tracer.start_as_current_span("trace_cascading_handoff_delay") as root_span:
        root_span.set_attribute("gen_ai.workflow.name", "wf_cross_border_customs_clearance")
        root_span.set_attribute("gen_ai.workflow.execution_id", execution_id)
        root_span.set_attribute("gen_ai.latency.scenario", "cascading_handoff_delay")
        root_span.set_attribute("gen_ai.latency.type", "CASCADING_HANDOFF_DELAY")
        trace_id = format(root_span.get_span_context().trace_id, "032x")

        # Step 1: Upstream Bottleneck Agent (Heavy OCR Rasterization & Table Parsing)
        with handoff_span(
            tracer,
            agent_name="CustomsDocumentParserAgent",
            action_name="parse_multipage_bill_of_lading",
            execution_id=execution_id,
            workflow_name="wf_cross_border_customs_clearance",
            handoff_to="TariffClassificationAgent",
            handoff_reason="Raw manifest parsed into line-item chunks for harmonized system classification",
            latency_type="CASCADING_HANDOFF_DELAY",
            metadata={"document_pages": 52, "file_size_mb": 18.6, "customs_port": "USLAX"},
        ) as span1:
            with tool_span(
                tracer,
                tool_name="ocr_pdf_rasterize_and_parse_tables",
                arguments={"document_path": "s3://customs-intake/bol_manifest_52p.pdf", "pages": 52, "dpi": 300},
                latency_type="CASCADING_HANDOFF_DELAY",
                execution_id=execution_id,
            ) as set_tool:
                # Upstream OCR & tabular extraction stall dominating >70% of trace
                await asyncio.sleep(3.8)
                set_tool({"status": "SUCCESS", "pages_processed": 52, "extracted_containers": 18, "line_items": 142})

            summary = await _safe_llm_call(
                client,
                prompt="Summarize cargo manifest extraction status for 18 intermodal shipping containers from Shanghai to Los Angeles.",
                system_prompt="You are a Customs Ingestion Agent. State container counts, seal integrity, and gross tonnage.",
            )
            span1.set_token_usage(input_tokens=1420, output_tokens=110)
            span1.set_attribute("agent.output_summary", summary[:120])

        # Step 2: Downstream Agent 1 (Harmonized Tariff Classification)
        with handoff_span(
            tracer,
            agent_name="TariffClassificationAgent",
            action_name="classify_harmonized_system_codes",
            execution_id=execution_id,
            workflow_name="wf_cross_border_customs_clearance",
            handoff_from="CustomsDocumentParserAgent",
            handoff_to="SanctionsCrossCheckAgent",
            handoff_reason="HS codes assigned, forwarding to sanctions screening",
            latency_type="CASCADING_HANDOFF_DELAY",
        ) as span2:
            with tool_span(tracer, "lookup_hs_tariff_schedule", {"commodity": "lithium_ion_battery_packs"}, execution_id=execution_id) as set_tool2:
                await asyncio.sleep(0.25)
                set_tool2({"hs_code": "8507.60.0000", "duty_rate_pct": 3.4, "antidumping_applicable": False})

            tariff_res = await _safe_llm_call(
                client,
                prompt="Verify HTSUS code 8507.60 for industrial battery storage modules and state documentation rules.",
                system_prompt="You are a Customs Tariff Classification Specialist.",
            )
            span2.set_token_usage(input_tokens=850, output_tokens=95)
            span2.set_attribute("agent.classification", tariff_res[:100])

        # Step 3: Downstream Agent 2 (Sanctions & Trade Embargo Screening)
        with handoff_span(
            tracer,
            agent_name="SanctionsCrossCheckAgent",
            action_name="screen_trade_embargo_registers",
            execution_id=execution_id,
            workflow_name="wf_cross_border_customs_clearance",
            handoff_from="TariffClassificationAgent",
            handoff_to="DutyValuationAgent",
            handoff_reason="Sanctions cleared, proceeding to duty and tax valuation",
            latency_type="CASCADING_HANDOFF_DELAY",
        ) as span3:
            with tool_span(tracer, "query_ofac_bis_consolidated_screening", {"consignee": "PacRim Logistics LLC"}, execution_id=execution_id) as set_tool3:
                await asyncio.sleep(0.2)
                set_tool3({"matches_found": 0, "status": "CLEARED", "verification_hash": "sec-chk-8819a"})

            span3.set_token_usage(input_tokens=620, output_tokens=70)

        # Step 4: Downstream Agent 3 (Duty & Port Harbor Maintenance Valuation)
        with handoff_span(
            tracer,
            agent_name="DutyValuationAgent",
            action_name="compute_tariff_and_vat_liability",
            execution_id=execution_id,
            workflow_name="wf_cross_border_customs_clearance",
            handoff_from="SanctionsCrossCheckAgent",
            handoff_to="ClearanceDossierCompilerAgent",
            handoff_reason="Duty liabilities computed, compiling final clearance dossier",
            latency_type="CASCADING_HANDOFF_DELAY",
        ) as span4:
            with tool_span(tracer, "calculate_customs_valuation", {"declared_fob_usd": 485000, "duty_rate": 0.034}, execution_id=execution_id) as set_tool4:
                await asyncio.sleep(0.18)
                set_tool4({"ad_valorem_duty_usd": 16490.0, "harbor_maintenance_fee": 606.25, "merchandise_processing_fee": 575.35})

            span4.set_token_usage(input_tokens=710, output_tokens=85)

        # Step 5: Downstream Agent 4 (Release Dossier Compilation - Ingress Deadline Exceeded)
        execution_failed = False
        error_msg = ""
        try:
            with handoff_span(
                tracer,
                agent_name="ClearanceDossierCompilerAgent",
                action_name="generate_cargo_release_certificate",
                execution_id=execution_id,
                workflow_name="wf_cross_border_customs_clearance",
                handoff_from="DutyValuationAgent",
                handoff_reason="CBP Form 7501 Entry Summary & Release Certificate finalized",
                latency_type="CASCADING_HANDOFF_DELAY",
            ) as span5:
                await asyncio.sleep(0.22)
                span5.set_token_usage(input_tokens=940, output_tokens=120)
                elapsed_so_far = (time.perf_counter() - start_time) * 1000
                if elapsed_so_far > 4200:
                    raise DeadlineExceededError(
                        f"gRPC StatusCode.DEADLINE_EXCEEDED: Ingress transaction deadline of 4200ms exceeded (elapsed: {elapsed_so_far:.0f}ms); "
                        "upstream CustomsDocumentParserAgent OCR rasterization choke starved downstream pipeline."
                    )
                span5.set_attribute("agent.clearance_status", "RELEASED_ENTRY_SUMMARY_PAID")
        except DeadlineExceededError as exc:
            execution_failed = True
            error_msg = str(exc)
            record_span_error(root_span, exc)
            root_span.set_attribute("gen_ai.activity.status", "FAILED")
            root_span.set_attribute("status_code", "Error")

        duration_ms = (time.perf_counter() - start_time) * 1000
        root_span.set_attribute("gen_ai.latency.duration_ms", duration_ms)

        return ScenarioResult(
            scenario_name="Cascading Handoff Delay",
            latency_type="CASCADING_HANDOFF_DELAY",
            execution_id=execution_id,
            trace_id=trace_id,
            total_duration_ms=duration_ms,
            status="FAILED" if execution_failed else "SUCCESS",
            step_count=5,
            summary="Upstream CustomsDocumentParserAgent suffered 3.8s OCR choke, breaching 4200ms ingress SLA and causing DeadlineExceededError at Dossier compilation.",
            details={"error": error_msg, "bottleneck_agent": "CustomsDocumentParserAgent", "bottleneck_duration_s": 3.8, "downstream_agents_delayed": 4, "sla_limit_ms": 4200},
        )


# ===========================================================================
# SCENARIO 2: PING-PONG / LOOP THRASH HANDOFF (Multi-Party Arbitration)
# ===========================================================================
async def run_ping_pong_loop_scenario(
    client: Mistral,
    tracer: Tracer,
    iterations: int = 4,
) -> ScenarioResult:
    """
    Scenario 2: Marine Cargo Insurance Claims Dispute & Multi-Party Arbitration.
    ClaimsTriageAgent, MarinePolicyAuditorAgent, and CarrierLiabilityAdjusterAgent
    repeatedly bounce damage claims over 8 handoff cycles due to conflicting policy exclusion clauses.
    """
    execution_id = f"lat-loop-{uuid.uuid4().hex[:8]}"
    start_time = time.perf_counter()

    with tracer.start_as_current_span("trace_ping_pong_loop_thrash") as root_span:
        root_span.set_attribute("gen_ai.workflow.name", "wf_freight_claim_arbitration")
        root_span.set_attribute("gen_ai.workflow.execution_id", execution_id)
        root_span.set_attribute("gen_ai.latency.scenario", "ping_pong_loop_thrash")
        root_span.set_attribute("gen_ai.latency.type", "LOOP_THRASH_HANDOFF")
        trace_id = format(root_span.get_span_context().trace_id, "032x")

        step_count = 0

        # Unbounded back-and-forth arbitration loop (4 iterations = 8 alternating handoffs)
        for i in range(1, iterations + 1):
            # Phase A: Claims Triage & Policy Interpretation
            with handoff_span(
                tracer,
                agent_name="ClaimsTriageAgent",
                action_name=f"evaluate_perishable_cargo_loss_turn_{i}",
                execution_id=execution_id,
                workflow_name="wf_freight_claim_arbitration",
                handoff_to="MarinePolicyAuditorAgent",
                handoff_reason=f"Turn {i}: Resubmitting container reefer spoilage claim with supplementary telematics",
                latency_type="LOOP_THRASH_HANDOFF",
                metadata={"iteration": i, "claim_id": "CLM-REEFER-902", "perishable_type": "Biologics"},
            ) as s_triage:
                s_triage.set_attribute("gen_ai.agent.handoff.iteration", i)
                await asyncio.sleep(0.4)
                res1 = await _safe_llm_call(
                    client,
                    prompt=f"Cycle {i}: Requesting coverage confirmation under Cargo Clause A for container MSCU-881923.",
                    system_prompt="You are a Claims Triage Adjuster advocating for policyholder cargo recovery.",
                )
                s_triage.set_token_usage(input_tokens=780 + (i * 90), output_tokens=95)
                s_triage.set_attribute("agent.claim_pleading", res1[:100])
                step_count += 1

            # Phase B: Marine Policy Underwriter Rejection & Counter-Challenge
            with handoff_span(
                tracer,
                agent_name="MarinePolicyAuditorAgent",
                action_name=f"audit_exclusion_clauses_turn_{i}",
                execution_id=execution_id,
                workflow_name="wf_freight_claim_arbitration",
                handoff_from="ClaimsTriageAgent",
                handoff_to="CarrierLiabilityAdjusterAgent" if i < iterations else "GeneralCounselArbitratorAgent",
                handoff_reason=f"Turn {i}: Rejection under Clause 14.B (shore-power interruption); demand carrier affidavit",
                latency_type="LOOP_THRASH_HANDOFF",
                metadata={"iteration": i, "rejection_code": "EXCLUSION_14B_SHORE_POWER", "disputed_amount_usd": 320000},
            ) as s_auditor:
                s_auditor.set_attribute("gen_ai.agent.handoff.iteration", i)
                await asyncio.sleep(0.45)
                res2 = await _safe_llm_call(
                    client,
                    prompt=f"Cycle {i}: Issue formal policy reservation of rights citing carrier shore electrical failure.",
                    system_prompt="You are a Senior Marine Policy Auditor defending underwriter exclusion rights.",
                )
                s_auditor.set_token_usage(input_tokens=910 + (i * 85), output_tokens=105)
                s_auditor.set_attribute("agent.rejection_memo", res2[:100])
                step_count += 1

        # Enterprise Recursion Guardrail Trip: 8-hop cyclical handoff thrash
        execution_failed = False
        error_msg = ""
        try:
            raise MaxHandoffHopsExceededError(
                f"MaxHandoffHopsExceededError: Cyclic agent handoff thrashing detected ({step_count} round-trip handoffs "
                "between ClaimsTriageAgent and MarinePolicyAuditorAgent); exceeded maximum enterprise recursion ceiling of 8 hops."
            )
        except MaxHandoffHopsExceededError as exc:
            execution_failed = True
            error_msg = str(exc)
            record_span_error(root_span, exc)
            root_span.set_attribute("gen_ai.activity.status", "FAILED")
            root_span.set_attribute("status_code", "Error")

        duration_ms = (time.perf_counter() - start_time) * 1000
        root_span.set_attribute("gen_ai.latency.duration_ms", duration_ms)

        return ScenarioResult(
            scenario_name="Ping-Pong Loop Thrash",
            latency_type="LOOP_THRASH_HANDOFF",
            execution_id=execution_id,
            trace_id=trace_id,
            total_duration_ms=duration_ms,
            status="FAILED" if execution_failed else "SUCCESS",
            step_count=step_count,
            summary=f"ClaimsTriageAgent and MarinePolicyAuditorAgent bounced claim across {step_count} cyclic handoffs, tripping MaxHandoffHopsExceededError guardrail.",
            details={"error": error_msg, "iterations": iterations, "total_handoff_cycles": step_count, "recursion_limit": 8},
        )


# ===========================================================================
# SCENARIO 3: BLOCKING TOOL STALL (Distributed Database Lock Contention)
# ===========================================================================
async def run_blocking_tool_stall_scenario(client: Mistral, tracer: Tracer) -> ScenarioResult:
    """
    Scenario 3: Global Treasury Liquidity Sweep & Core Banking Ledger Lock Contention.
    CoreBankingLedgerAgent blocks for 4.2s executing an atomic balance transfer tool due
    to row-level lock contention on the master cash ledger and connection pool exhaustion.
    """
    execution_id = f"lat-tool-stall-{uuid.uuid4().hex[:8]}"
    start_time = time.perf_counter()

    with tracer.start_as_current_span("trace_blocking_tool_stall") as root_span:
        root_span.set_attribute("gen_ai.workflow.name", "wf_treasury_liquidity_sweep")
        root_span.set_attribute("gen_ai.workflow.execution_id", execution_id)
        root_span.set_attribute("gen_ai.latency.scenario", "blocking_tool_stall")
        root_span.set_attribute("gen_ai.latency.type", "BLOCKING_TOOL_STALL")
        trace_id = format(root_span.get_span_context().trace_id, "032x")

        # Step 1: Account Balance Aggregation
        with handoff_span(
            tracer,
            agent_name="AccountBalanceAggregatorAgent",
            action_name="aggregate_subsidiary_balances",
            execution_id=execution_id,
            workflow_name="wf_treasury_liquidity_sweep",
            handoff_to="LiquidityOptimizationAgent",
            handoff_reason="Balances retrieved across 14 European operating entities; passing to optimization",
            latency_type="BLOCKING_TOOL_STALL",
        ) as span1:
            with tool_span(tracer, "query_open_banking_aisp_balances", {"entity_count": 14}, execution_id=execution_id) as set_t1:
                await asyncio.sleep(0.3)
                set_t1({"total_eur_pool": 84200000.0, "deficits_detected": 3, "surpluses_detected": 11})

            span1.set_token_usage(input_tokens=920, output_tokens=80)

        # Step 2: Liquidity Allocation Matrix Computation
        with handoff_span(
            tracer,
            agent_name="LiquidityOptimizationAgent",
            action_name="calculate_target_sweep_allocations",
            execution_id=execution_id,
            workflow_name="wf_treasury_liquidity_sweep",
            handoff_from="AccountBalanceAggregatorAgent",
            handoff_to="CoreBankingLedgerAgent",
            handoff_reason="Intercompany sweep instructions formulated; forwarding to core ledger execution",
            latency_type="BLOCKING_TOOL_STALL",
        ) as span2:
            await asyncio.sleep(0.25)
            allocations = await _safe_llm_call(
                client,
                prompt="Formulate multi-currency zero-balance sweep instructions for EUR 14.5M pooling transfer.",
                system_prompt="You are a Corporate Treasury Liquidity Optimization Engine.",
            )
            span2.set_token_usage(input_tokens=1100, output_tokens=130)
            span2.set_attribute("agent.sweep_instructions", allocations[:100])

        # Step 3: Core Banking Ledger Agent (Severe Row-Lock Tool Stall & Pool Exhaustion)
        execution_failed = False
        error_msg = ""
        try:
            with handoff_span(
                tracer,
                agent_name="CoreBankingLedgerAgent",
                action_name="execute_atomic_ledger_journal_entry",
                execution_id=execution_id,
                workflow_name="wf_treasury_liquidity_sweep",
                handoff_from="LiquidityOptimizationAgent",
                handoff_to="AuditJournalPostingAgent",
                handoff_reason="Ledger accounts locked and transfers committed; dispatching to audit journal",
                latency_type="BLOCKING_TOOL_STALL",
            ) as span3:
                with tool_span(
                    tracer,
                    tool_name="execute_atomic_ledger_journal_entry",
                    arguments={
                        "debit_account": "CORP-EUR-MAIN-001",
                        "credit_account": "SUBS-EUR-DEPO-042",
                        "amount_eur": 14500000.0,
                        "isolation_level": "SERIALIZABLE",
                    },
                    latency_type="BLOCKING_TOOL_STALL",
                    execution_id=execution_id,
                ) as set_tool:
                    # Severe database row-level lock wait & pool starvation (4.2s stall)
                    await asyncio.sleep(4.2)
                    exc = DatabaseLockTimeoutError(
                        "HikariCP-CoreLedgerPool: Connection acquisition timed out after 4200ms. "
                        "Lock wait timeout exceeded on row 'GL_ACCT_EUR_MAIN_001' under SERIALIZABLE isolation; transaction rolled back."
                    )
                    set_tool({
                        "status": "FAILED",
                        "error": str(exc),
                        "lock_wait_ms": 4200,
                    })
                    raise exc
        except DatabaseLockTimeoutError as exc:
            execution_failed = True
            error_msg = str(exc)
            record_span_error(root_span, exc)
            root_span.set_attribute("gen_ai.activity.status", "FAILED")
            root_span.set_attribute("status_code", "Error")

        duration_ms = (time.perf_counter() - start_time) * 1000
        root_span.set_attribute("gen_ai.latency.duration_ms", duration_ms)

        return ScenarioResult(
            scenario_name="Blocking Tool Stall",
            latency_type="BLOCKING_TOOL_STALL",
            execution_id=execution_id,
            trace_id=trace_id,
            total_duration_ms=duration_ms,
            status="FAILED" if execution_failed else "SUCCESS",
            step_count=3,
            summary="CoreBankingLedgerAgent stalled for 4.2s on row lock contention, terminating with DatabaseLockTimeoutError.",
            details={"error": error_msg, "slow_tool": "execute_atomic_ledger_journal_entry", "tool_delay_s": 4.2, "isolation": "SERIALIZABLE"},
        )


# ===========================================================================
# SCENARIO 4: CONTEXT / TOKEN BLOAT LATENCY (M&A Antitrust Due Diligence)
# ===========================================================================
async def run_token_bloat_scenario(client: Mistral, tracer: Tracer) -> ScenarioResult:
    """
    Scenario 4: Cross-Border M&A Antitrust Due Diligence & 38k Unpruned Context TTFT Drag.
    DocumentCorpusCollectorAgent naively injects 38,000+ tokens of raw regulatory filings
    and antitrust transcripts without chunking, triggering high TTFT (2.8s) and LLM generation drag.
    """
    execution_id = f"lat-bloat-{uuid.uuid4().hex[:8]}"
    start_time = time.perf_counter()

    with tracer.start_as_current_span("trace_token_bloat_latency") as root_span:
        root_span.set_attribute("gen_ai.workflow.name", "wf_ma_due_diligence_briefing")
        root_span.set_attribute("gen_ai.workflow.execution_id", execution_id)
        root_span.set_attribute("gen_ai.latency.scenario", "token_bloat_context_drag")
        root_span.set_attribute("gen_ai.latency.type", "TOKEN_BLOAT_LATENCY")
        trace_id = format(root_span.get_span_context().trace_id, "032x")

        # Step 1: Ingestion & Unpruned Corpus Assembly
        with handoff_span(
            tracer,
            agent_name="DocumentCorpusCollectorAgent",
            action_name="aggregate_unpruned_regulatory_filings",
            execution_id=execution_id,
            workflow_name="wf_ma_due_diligence_briefing",
            handoff_to="AntitrustRiskAssessmentAgent",
            handoff_reason="Raw 38,000 token corporate filing corpus injected without semantic pruning",
            latency_type="TOKEN_BLOAT_LATENCY",
            metadata={"accumulated_tokens": 38400, "uncompressed_documents": 28},
        ) as span1:
            with tool_span(
                tracer,
                tool_name="scrape_sec_edgar_and_ftc_depositions",
                arguments={"target_company": "Acme Semiconductor Corp", "filings": ["10-K", "10-Q", "FTC-FTC-8812"]},
                latency_type="TOKEN_BLOAT_LATENCY",
                execution_id=execution_id,
            ) as set_tool:
                await asyncio.sleep(0.35)
                set_tool({"status": "SUCCESS", "documents_aggregated": 28, "total_raw_tokens": 38400})

            span1.set_token_usage(input_tokens=38400, output_tokens=150)
            span1.set_attribute("prompt_tokens", 38400)
            span1.set_attribute("agent.corpus_state", "UNPRUNED_38K")

        # Step 2: Antitrust Risk Assessment (Severe TTFT & Context Allocation Breach)
        execution_failed = False
        error_msg = ""
        try:
            with handoff_span(
                tracer,
                agent_name="AntitrustRiskAssessmentAgent",
                action_name="evaluate_herfindahl_hirschman_index",
                execution_id=execution_id,
                workflow_name="wf_ma_due_diligence_briefing",
                handoff_from="DocumentCorpusCollectorAgent",
                handoff_to="ExecutiveSummaryAgent",
                handoff_reason="Market concentration computed, passing full bloated context to executive briefing",
                latency_type="TOKEN_BLOAT_LATENCY",
                metadata={"ttft_ms": 2800, "prompt_tokens": 38400},
            ) as span2:
                span2.set_attribute("ttft_ms", 2800)
                span2.set_attribute("prompt_tokens", 38400)
                span2.set_attribute("gen_ai.latency.ttft_ms", 2800)

                # Simulated heavy TTFT (Time-To-First-Token) drag on massive prompt
                await asyncio.sleep(2.8)
                span2.set_token_usage(input_tokens=38400, output_tokens=0)
                exc = ContextWindowExceededError(
                    "HTTP 413 Payload Too Large: Prompt payload size 38,400 tokens exceeds microservice ingress allocation limit "
                    "of 16,384 tokens for cluster 'regulatory-nlp-v2'. Context compaction or semantic chunking required."
                )
                record_span_error(span2, exc)
                span2.set_attribute("status", "error")
                span2.set_attribute("status_code", "Error")
                raise exc
        except ContextWindowExceededError as exc:
            execution_failed = True
            error_msg = str(exc)
            record_span_error(root_span, exc)
            root_span.set_attribute("gen_ai.activity.status", "FAILED")
            root_span.set_attribute("status_code", "Error")

        duration_ms = (time.perf_counter() - start_time) * 1000
        root_span.set_attribute("gen_ai.latency.duration_ms", duration_ms)

        return ScenarioResult(
            scenario_name="Token Bloat & High TTFT Latency",
            latency_type="TOKEN_BLOAT_LATENCY",
            execution_id=execution_id,
            trace_id=trace_id,
            total_duration_ms=duration_ms,
            status="FAILED" if execution_failed else "SUCCESS",
            step_count=2,
            summary="38,400 token context payload caused elevated TTFT (2800ms) and breached microservice allocation limit (ContextWindowExceededError).",
            details={"error": error_msg, "prompt_tokens": 38400, "ttft_ms": 2800, "token_limit": 16384},
        )


# ===========================================================================
# SCENARIO 5: FAN-OUT STRAGGLER BOTTLENECK (Asynchronous Concurrency Imbalance)
# ===========================================================================
async def run_fanout_straggler_scenario(client: Mistral, tracer: Tracer) -> ScenarioResult:
    """
    Scenario 5: Commercial Real Estate Loan Syndication & Asynchronous 3D GIS Straggler Bottleneck.
    Coordinator fans out concurrently to 4 underwriting branches. Three finish in < 350ms,
    while CommercialPropertyValuationAgent stalls for 4.2s running 3D spatial GIS raster regression,
    strictly gating the syndicate decision matrix.
    """
    execution_id = f"lat-fanout-{uuid.uuid4().hex[:8]}"
    start_time = time.perf_counter()

    with tracer.start_as_current_span("trace_fanout_straggler_delay") as root_span:
        root_span.set_attribute("gen_ai.workflow.name", "wf_commercial_loan_underwriting")
        root_span.set_attribute("gen_ai.workflow.execution_id", execution_id)
        root_span.set_attribute("gen_ai.latency.scenario", "fanout_straggler_bottleneck")
        root_span.set_attribute("gen_ai.latency.type", "FANOUT_STRAGGLER_DELAY")
        trace_id = format(root_span.get_span_context().trace_id, "032x")

        # Step 1: Loan Origination Coordinator Intake
        with handoff_span(
            tracer,
            agent_name="LoanOriginationCoordinatorAgent",
            action_name="orchestrate_underwriting_intake",
            execution_id=execution_id,
            workflow_name="wf_commercial_loan_underwriting",
            handoff_to="[BranchFanout] 4 Parallel Underwriting Services",
            handoff_reason="Dispatched concurrent underwriting evaluations to 4 specialized agents",
            latency_type="FANOUT_STRAGGLER_DELAY",
            metadata={"requested_loan_usd": 45000000, "property_type": "Class-A Office"},
        ) as coord_span:
            await asyncio.sleep(0.15)
            coord_span.set_token_usage(input_tokens=850, output_tokens=70)

        # Step 2: Concurrent Fan-Out (4 Parallel Underwriting Branches)
        async def run_credit_agent():
            with handoff_span(
                tracer,
                agent_name="BorrowerCreditBureauAgent",
                action_name="audit_corporate_sponsor_credit",
                execution_id=execution_id,
                workflow_name="wf_commercial_loan_underwriting",
                latency_type="FANOUT_STRAGGLER_DELAY",
            ) as s:
                await asyncio.sleep(0.25)
                s.set_token_usage(input_tokens=720, output_tokens=60)
                s.set_attribute("agent.credit_rating", "AAA_SPONSOR")
                return "CREDIT_CLEARED"

        async def run_environmental_agent():
            with handoff_span(
                tracer,
                agent_name="EnvironmentalRiskScreeningAgent",
                action_name="screen_phase_1_environmental_hazards",
                execution_id=execution_id,
                workflow_name="wf_commercial_loan_underwriting",
                latency_type="FANOUT_STRAGGLER_DELAY",
            ) as s:
                await asyncio.sleep(0.32)
                s.set_token_usage(input_tokens=680, output_tokens=55)
                s.set_attribute("agent.environmental_risk", "LOW_NO_SUPERFUND")
                return "ENVIRONMENTAL_CLEARED"

        async def run_title_agent():
            with handoff_span(
                tracer,
                agent_name="TitleDeedVerificationAgent",
                action_name="verify_municipal_liens_and_easements",
                execution_id=execution_id,
                workflow_name="wf_commercial_loan_underwriting",
                latency_type="FANOUT_STRAGGLER_DELAY",
            ) as s:
                await asyncio.sleep(0.28)
                s.set_token_usage(input_tokens=640, output_tokens=50)
                s.set_attribute("agent.title_status", "CLEAN_UNENCUMBERED")
                return "TITLE_CLEARED"

        execution_failed = False
        error_msg = ""

        async def run_valuation_straggler():
            with handoff_span(
                tracer,
                agent_name="CommercialPropertyValuationAgent",
                action_name="perform_3d_spatial_gis_valuation",
                execution_id=execution_id,
                workflow_name="wf_commercial_loan_underwriting",
                latency_type="FANOUT_STRAGGLER_DELAY",
            ) as s:
                with tool_span(
                    tracer,
                    tool_name="render_lidar_spatial_comparable_regression",
                    arguments={"parcel_id": "NYC-MN-4491-002", "radius_miles": 2.5, "lidar_points": 12500000},
                    latency_type="FANOUT_STRAGGLER_DELAY",
                    execution_id=execution_id,
                ) as set_tool:
                    # Severe async straggler delay: 3.5s (breaching 3000ms barrier SLA)
                    await asyncio.sleep(3.5)
                    exc = AsyncFanoutTimeoutError(
                        "AsyncFanoutTimeoutError: Worker 'CommercialPropertyValuationAgent' exceeded barrier sync SLA limit of 3000ms "
                        "(elapsed: 3500ms) on spatial LiDAR regression. Fan-out synchronization barrier collapsed."
                    )
                    set_tool({"status": "TIMEOUT", "error": str(exc), "duration_ms": 3500})
                    record_span_error(s, exc)
                    s.set_attribute("status", "error")
                    s.set_attribute("status_code", "Error")
                    raise exc

        # Concurrently await all 4 branches with error capture
        results = await asyncio.gather(
            run_credit_agent(),
            run_environmental_agent(),
            run_title_agent(),
            run_valuation_straggler(),
            return_exceptions=True,
        )

        for res in results:
            if isinstance(res, Exception):
                execution_failed = True
                error_msg = str(res)
                record_span_error(root_span, res)
                root_span.set_attribute("gen_ai.activity.status", "FAILED")
                root_span.set_attribute("status_code", "Error")

        duration_ms = (time.perf_counter() - start_time) * 1000
        root_span.set_attribute("gen_ai.latency.duration_ms", duration_ms)

        return ScenarioResult(
            scenario_name="Fanout Straggler Delay",
            latency_type="FANOUT_STRAGGLER_DELAY",
            execution_id=execution_id,
            trace_id=trace_id,
            total_duration_ms=duration_ms,
            status="FAILED" if execution_failed else "SUCCESS",
            step_count=5,
            summary="Parallel fan-out barrier collapsed when CommercialPropertyValuationAgent exceeded 3000ms SLA (AsyncFanoutTimeoutError).",
            details={"error": error_msg, "straggler_agent": "CommercialPropertyValuationAgent", "straggler_duration_s": 3.5, "barrier_sla_s": 3.0},
        )


# ===========================================================================
# SCENARIO 6: RETRY STORM & DEGRADED FALLBACK (Payment Gateway Outage)
# ===========================================================================
async def run_retry_storm_scenario(client: Mistral, tracer: Tracer) -> ScenarioResult:
    """
    Scenario 6: SWIFT Corporate Wire Transfer Settlement & Cascading Rate-Limit Retry Storm.
    PaymentProcessingAgent encounters 504 Timeout, 429 Rate Limit, and 503 Service Unavailable
    with escalating exponential backoffs before circuit tripping and falling back to batch ACH.
    """
    execution_id = f"lat-retry-{uuid.uuid4().hex[:8]}"
    start_time = time.perf_counter()

    with tracer.start_as_current_span("trace_retry_storm_backoff") as root_span:
        root_span.set_attribute("gen_ai.workflow.name", "wf_payment_gateway_routing")
        root_span.set_attribute("gen_ai.workflow.execution_id", execution_id)
        root_span.set_attribute("gen_ai.latency.scenario", "retry_storm_backoff")
        root_span.set_attribute("gen_ai.latency.type", "RETRY_STORM_BACKOFF")
        trace_id = format(root_span.get_span_context().trace_id, "032x")

        # Step 1: Pre-Payment Sanctions & Liquidity Check
        with handoff_span(
            tracer,
            agent_name="PaymentValidationAgent",
            action_name="validate_beneficiary_sanctions",
            execution_id=execution_id,
            workflow_name="wf_payment_gateway_routing",
            handoff_to="PaymentProcessingAgent",
            handoff_reason="Sanctions verified, forwarding $185,000 corporate payment to SWIFT gateway",
            latency_type="RETRY_STORM_BACKOFF",
        ) as span1:
            await asyncio.sleep(0.2)
            span1.set_token_usage(input_tokens=680, output_tokens=60)

        # Step 2: Escalating Retry Storm (3 Consecutive Failures with Backoffs)
        backoffs = [0.8, 1.4, 2.0]
        failure_codes = [
            ("504 Gateway Timeout", "TIMEOUT"),
            ("429 Too Many Requests (Rate Limited)", "RATE_LIMITED"),
            ("503 Service Unavailable (Remote Node Offline)", "FAILED"),
        ]

        for attempt, (backoff, (err_msg, err_status)) in enumerate(zip(backoffs, failure_codes), start=1):
            with handoff_span(
                tracer,
                agent_name="PaymentProcessingAgent",
                action_name=f"dispatch_swift_wire_payment_attempt_{attempt}",
                execution_id=execution_id,
                workflow_name="wf_payment_gateway_routing",
                handoff_reason=f"Attempt {attempt}: Escalating retry following remote gateway rejection",
                latency_type="RETRY_STORM_BACKOFF",
                metadata={"attempt": attempt, "backoff_seconds": backoff, "error_expected": err_msg},
            ) as span_retry:
                span_retry.set_attribute("wf.activity.attempt", attempt)
                span_retry.set_attribute("status", "error")
                span_retry.set_attribute("status_code", "Error")

                with tool_span(
                    tracer,
                    tool_name="external_swift_wire_gateway",
                    arguments={"wire_id": "SWIFT-US-8819", "attempt": attempt, "amount": 185000},
                    latency_type="RETRY_STORM_BACKOFF",
                    execution_id=execution_id,
                ) as set_tool:
                    await asyncio.sleep(backoff)
                    set_tool({"status": err_status, "http_status": err_msg, "attempt": attempt})

                span_retry.set_token_usage(input_tokens=940 + (attempt * 60), output_tokens=40)
                record_span_error(span_retry, ConnectionError(f"SWIFT Gateway failure on attempt {attempt}: {err_msg}"))

        # Step 3: Circuit Breaker Trip & Hard Rejection
        execution_failed = False
        error_msg = ""
        try:
            with handoff_span(
                tracer,
                agent_name="PaymentCircuitBreakerAgent",
                action_name="evaluate_circuit_breaker_threshold",
                execution_id=execution_id,
                workflow_name="wf_payment_gateway_routing",
                handoff_from="PaymentProcessingAgent",
                handoff_to="DeadLetterQueueAgent",
                handoff_reason="Circuit breaker tripped after 3 consecutive retry storm failures (504, 429, 503)",
                latency_type="RETRY_STORM_BACKOFF",
            ) as cb_span:
                await asyncio.sleep(0.3)
                exc = CircuitBreakerOpenException(
                    "CircuitBreakerOpenException: Downstream circuit for 'SWIFT-US-8819' tripped OPEN after 3 consecutive failures "
                    "(504 Gateway Timeout, 429 Rate Limit, 503 Unavailable). Escalation budget exhausted; payment rejected to DLQ."
                )
                record_span_error(cb_span, exc)
                cb_span.set_attribute("status", "error")
                cb_span.set_attribute("status_code", "Error")
                raise exc
        except CircuitBreakerOpenException as exc:
            execution_failed = True
            error_msg = str(exc)
            record_span_error(root_span, exc)
            root_span.set_attribute("gen_ai.activity.status", "FAILED")
            root_span.set_attribute("status_code", "Error")

        duration_ms = (time.perf_counter() - start_time) * 1000
        root_span.set_attribute("gen_ai.latency.duration_ms", duration_ms)

        return ScenarioResult(
            scenario_name="Retry Storm Backoff",
            latency_type="RETRY_STORM_BACKOFF",
            execution_id=execution_id,
            trace_id=trace_id,
            total_duration_ms=duration_ms,
            status="FAILED" if execution_failed else "SUCCESS",
            step_count=5,
            summary="PaymentProcessingAgent performed 3 retries with escalating backoffs before tripping CircuitBreakerOpenException.",
            details={"error": error_msg, "retry_attempts": 3, "cumulative_backoff_s": 4.2, "circuit_state": "OPEN"},
        )


# ===========================================================================
# SCENARIO 7: MODEL OVERSIZING LLM DELAY (Frontier Compute Overkill)
# ===========================================================================
async def run_model_oversizing_scenario(client: Mistral, tracer: Tracer) -> ScenarioResult:
    """
    Scenario 7: Enterprise Helpdesk Support Triage & Frontier Model Compute Overkill.
    PriorityClassificationAgent invokes mistral-large-latest with 2,400 prompt tokens
    for a trivial single-token enum classification ('URGENT'), causing 3.8s generation delay.
    """
    execution_id = f"lat-oversize-{uuid.uuid4().hex[:8]}"
    start_time = time.perf_counter()

    with tracer.start_as_current_span("trace_model_oversizing_llm_delay") as root_span:
        root_span.set_attribute("gen_ai.workflow.name", "wf_omnichannel_ticket_routing")
        root_span.set_attribute("gen_ai.workflow.execution_id", execution_id)
        root_span.set_attribute("gen_ai.latency.scenario", "model_oversizing_llm_delay")
        root_span.set_attribute("gen_ai.latency.type", "MODEL_OVERSIZING_LLM_DELAY")
        trace_id = format(root_span.get_span_context().trace_id, "032x")

        # Step 1: Inbound Ticket Normalization
        with handoff_span(
            tracer,
            agent_name="InboundEmailNormalizerAgent",
            action_name="sanitize_incoming_email",
            execution_id=execution_id,
            workflow_name="wf_omnichannel_ticket_routing",
            handoff_to="PriorityClassificationAgent",
            handoff_reason="Sanitized raw customer message, passing to priority classification",
            latency_type="MODEL_OVERSIZING_LLM_DELAY",
        ) as span1:
            await asyncio.sleep(0.18)
            span1.set_token_usage(input_tokens=420, output_tokens=45)

        # Step 2: Priority Classification (Frontier Model Overkill & Inference Queue Timeout)
        execution_failed = False
        error_msg = ""
        try:
            with handoff_span(
                tracer,
                agent_name="PriorityClassificationAgent",
                action_name="classify_ticket_urgency",
                execution_id=execution_id,
                workflow_name="wf_omnichannel_ticket_routing",
                handoff_from="InboundEmailNormalizerAgent",
                handoff_to="QueueDispatcherAgent",
                handoff_reason="Urgency classified as URGENT, dispatching to priority queue",
                latency_type="MODEL_OVERSIZING_LLM_DELAY",
                metadata={"model": "mistral-large-latest", "task_complexity": "LOW_SIMPLE_CLASSIFICATION", "output_tokens": 1},
            ) as span2:
                span2.set_attribute("model", "mistral-large-latest")
                span2.set_attribute("task_complexity", "LOW_SIMPLE_CLASSIFICATION")

                # Oversized model queuing stall: waiting for KV cache allocation on heavy model cluster
                await asyncio.sleep(3.6)
                exc = InferenceQueueTimeoutError(
                    "InferenceQueueTimeoutError: GPU cluster 'vllm-mistral-large' KV-cache allocation queue timeout exceeded (3600ms > 3000ms max_lease). "
                    "Frontier model requested for 1-token output; worker timed out waiting for available tensor-parallel shard."
                )
                span2.set_token_usage(input_tokens=2480, output_tokens=0)
                record_span_error(span2, exc)
                span2.set_attribute("status", "error")
                span2.set_attribute("status_code", "Error")
                raise exc
        except InferenceQueueTimeoutError as exc:
            execution_failed = True
            error_msg = str(exc)
            record_span_error(root_span, exc)
            root_span.set_attribute("gen_ai.activity.status", "FAILED")
            root_span.set_attribute("status_code", "Error")

        duration_ms = (time.perf_counter() - start_time) * 1000
        root_span.set_attribute("gen_ai.latency.duration_ms", duration_ms)

        return ScenarioResult(
            scenario_name="Model Oversizing LLM Delay",
            latency_type="MODEL_OVERSIZING_LLM_DELAY",
            execution_id=execution_id,
            trace_id=trace_id,
            total_duration_ms=duration_ms,
            status="FAILED" if execution_failed else "SUCCESS",
            step_count=2,
            summary="mistral-large-latest requested for 1-token classification timed out in GPU KV-cache queue (InferenceQueueTimeoutError).",
            details={"error": error_msg, "model": "mistral-large-latest", "task_complexity": "LOW", "queue_wait_s": 3.6},
        )


# ===========================================================================
# SCENARIO 8: UNCONSTRAINED GENERATION DRAG (Runaway Narrative Loop)
# ===========================================================================
async def run_unconstrained_generation_scenario(client: Mistral, tracer: Tracer) -> ScenarioResult:
    """
    Scenario 8: Regulatory Security Incident Disclosure & Unbounded Narrative Output Drag.
    ComplianceDisclosureAgent lacks max_tokens limits and schema bounds, emitting over 1,450
    verbose narrative tokens over 4.8s on an otherwise swift reporting pipeline.
    """
    execution_id = f"lat-unconstrained-{uuid.uuid4().hex[:8]}"
    start_time = time.perf_counter()

    with tracer.start_as_current_span("trace_unconstrained_gen_drag") as root_span:
        root_span.set_attribute("gen_ai.workflow.name", "wf_regulatory_incident_disclosure")
        root_span.set_attribute("gen_ai.workflow.execution_id", execution_id)
        root_span.set_attribute("gen_ai.latency.scenario", "unconstrained_gen_drag")
        root_span.set_attribute("gen_ai.latency.type", "UNCONSTRAINED_GEN_DRAG")
        trace_id = format(root_span.get_span_context().trace_id, "032x")

        # Step 1: Security Telemetry Aggregation
        with handoff_span(
            tracer,
            agent_name="SecurityIncidentReporterAgent",
            action_name="compile_threat_telemetry",
            execution_id=execution_id,
            workflow_name="wf_regulatory_incident_disclosure",
            handoff_to="ComplianceDisclosureAgent",
            handoff_reason="Threat vector and affected user counts summarized, drafting SEC 8-K disclosure",
            latency_type="UNCONSTRAINED_GEN_DRAG",
        ) as span1:
            await asyncio.sleep(0.2)
            span1.set_token_usage(input_tokens=750, output_tokens=70)

        # Step 2: Compliance Disclosure Agent (Unconstrained Runaway Generation & Reverse Proxy Read Timeout)
        execution_failed = False
        error_msg = ""
        try:
            with handoff_span(
                tracer,
                agent_name="ComplianceDisclosureAgent",
                action_name="draft_sec_form_8k_narrative",
                execution_id=execution_id,
                workflow_name="wf_regulatory_incident_disclosure",
                handoff_from="SecurityIncidentReporterAgent",
                handoff_to="LegalReviewAgent",
                handoff_reason="Runaway disclosure draft emitted without token constraints",
                latency_type="UNCONSTRAINED_GEN_DRAG",
                metadata={"completion_tokens": 1460, "generation_duration_ms": 4200},
            ) as span2:
                span2.set_attribute("completion_tokens", 1460)

                # High generation delay emitting verbose unconstrained tokens breaching ingress gateway timeout
                await asyncio.sleep(4.2)
                exc = GatewayReadTimeoutError(
                    "GatewayReadTimeoutError: HTTP 504 Envoy reverse proxy stream read timeout after 4200ms. "
                    "Unconstrained token generation (1,460 tokens without schema bounds) breached upstream ingress gateway deadline."
                )
                span2.set_token_usage(input_tokens=1250, output_tokens=1460)
                record_span_error(span2, exc)
                span2.set_attribute("status", "error")
                span2.set_attribute("status_code", "Error")
                raise exc
        except GatewayReadTimeoutError as exc:
            execution_failed = True
            error_msg = str(exc)
            record_span_error(root_span, exc)
            root_span.set_attribute("gen_ai.activity.status", "FAILED")
            root_span.set_attribute("status_code", "Error")

        duration_ms = (time.perf_counter() - start_time) * 1000
        root_span.set_attribute("gen_ai.latency.duration_ms", duration_ms)

        return ScenarioResult(
            scenario_name="Unconstrained Generation Drag",
            latency_type="UNCONSTRAINED_GEN_DRAG",
            execution_id=execution_id,
            trace_id=trace_id,
            total_duration_ms=duration_ms,
            status="FAILED" if execution_failed else "SUCCESS",
            step_count=2,
            summary="ComplianceDisclosureAgent generated unbounded 1,460 tokens, breaching ingress proxy timeout (GatewayReadTimeoutError).",
            details={"error": error_msg, "completion_tokens": 1460, "generation_duration_s": 4.2, "gateway_timeout_s": 4.0},
        )


# ===========================================================================
# SCENARIO 9: SEQUENTIAL I/O WATERFALL (Un-Batched Microservice Calls)
# ===========================================================================
async def run_sequential_io_waterfall_scenario(client: Mistral, tracer: Tracer) -> ScenarioResult:
    """
    Scenario 9: Defense Contractor Security Clearance & Sequential Background Check Waterfall.
    IdentityVerificationAgent performs 5 independent external registry verifications sequentially
    in a loop (5 x ~800ms = 4.1s) instead of parallelizing with asyncio.gather().
    """
    execution_id = f"lat-waterfall-{uuid.uuid4().hex[:8]}"
    start_time = time.perf_counter()

    with tracer.start_as_current_span("trace_sequential_io_waterfall") as root_span:
        root_span.set_attribute("gen_ai.workflow.name", "wf_employee_clearance_pipeline")
        root_span.set_attribute("gen_ai.workflow.execution_id", execution_id)
        root_span.set_attribute("gen_ai.latency.scenario", "sequential_io_waterfall")
        root_span.set_attribute("gen_ai.latency.type", "SEQUENTIAL_IO_WATERFALL")
        trace_id = format(root_span.get_span_context().trace_id, "032x")

        # Step 1: Candidate Intake
        with handoff_span(
            tracer,
            agent_name="ApplicantIntakeAgent",
            action_name="parse_sf86_questionnaire",
            execution_id=execution_id,
            workflow_name="wf_employee_clearance_pipeline",
            handoff_to="IdentityVerificationAgent",
            handoff_reason="SF-86 validated, initiating 5 mandatory federated background checks",
            latency_type="SEQUENTIAL_IO_WATERFALL",
        ) as span1:
            await asyncio.sleep(0.18)
            span1.set_token_usage(input_tokens=850, output_tokens=65)

        # Step 2: Sequential Waterfall Agent (5 Synchronous I/O Calls Breaching Transaction Budget)
        execution_failed = False
        error_msg = ""
        try:
            with handoff_span(
                tracer,
                agent_name="IdentityVerificationAgent",
                action_name="run_federated_background_checks",
                execution_id=execution_id,
                workflow_name="wf_employee_clearance_pipeline",
                handoff_from="ApplicantIntakeAgent",
                handoff_to="ClearanceAdjudicationAgent",
                handoff_reason="Synchronous loop of 5 remote checks",
                latency_type="SEQUENTIAL_IO_WATERFALL",
            ) as span2:
                check_tools = [
                    ("verify_national_identity_registry", 0.82, {"jurisdiction": "US_SSA"}),
                    ("query_fbi_criminal_history", 0.89, {"fingerprint_id": "FP-881920"}),
                    ("validate_accredited_degrees", 0.76, {"institution": "MIT", "degree": "M.S. EECS"}),
                    ("query_credit_scoring_bureau", 0.84, {"ssn_token": "TOK-CRED-099"}),
                    ("verify_prior_defense_employment", 0.81, {"employer": "Lockheed Aerospace"}),
                ]

                elapsed_io = 0.0
                for idx, (tool_name, delay_s, args) in enumerate(check_tools, start=1):
                    with tool_span(tracer, tool_name=tool_name, arguments=args, latency_type="SEQUENTIAL_IO_WATERFALL", execution_id=execution_id) as set_res:
                        await asyncio.sleep(delay_s)
                        elapsed_io += delay_s
                        if elapsed_io > 3.0:
                            exc = CumulativeTransactionBudgetExceeded(
                                f"CumulativeTransactionBudgetExceeded: Synchronous sequential I/O loop breached global transaction budget (3000ms). "
                                f"Terminated at check {idx} ('{tool_name}') with cumulative duration {int(elapsed_io * 1000)}ms. Concurrency refactoring required."
                            )
                            set_res({"status": "BUDGET_EXCEEDED", "elapsed_ms": int(elapsed_io * 1000)})
                            record_span_error(span2, exc)
                            span2.set_attribute("status", "error")
                            span2.set_attribute("status_code", "Error")
                            raise exc
                        set_res({"status": "VERIFIED_CLEAR", "check_duration_ms": int(delay_s * 1000)})

                span2.set_token_usage(input_tokens=1420, output_tokens=110)
                span2.set_attribute("agent.waterfall_checks_count", 5)
        except CumulativeTransactionBudgetExceeded as exc:
            execution_failed = True
            error_msg = str(exc)
            record_span_error(root_span, exc)
            root_span.set_attribute("gen_ai.activity.status", "FAILED")
            root_span.set_attribute("status_code", "Error")

        duration_ms = (time.perf_counter() - start_time) * 1000
        root_span.set_attribute("gen_ai.latency.duration_ms", duration_ms)

        return ScenarioResult(
            scenario_name="Sequential I/O Waterfall",
            latency_type="SEQUENTIAL_IO_WATERFALL",
            execution_id=execution_id,
            trace_id=trace_id,
            total_duration_ms=duration_ms,
            status="FAILED" if execution_failed else "SUCCESS",
            step_count=2,
            summary="Sequential I/O loop breached global transaction budget of 3000ms (CumulativeTransactionBudgetExceeded).",
            details={"error": error_msg, "aborted_at_check": 4, "budget_limit_s": 3.0, "recommended_fix": "asyncio.gather()"},
        )


# ===========================================================================
# SCENARIO 10: UNCACHED REPEATED I/O (Missing Idempotent Memoization)
# ===========================================================================
async def run_uncached_repeated_io_scenario(client: Mistral, tracer: Tracer) -> ScenarioResult:
    """
    Scenario 10: Multinational Enterprise Tax Invoicing & Uncached Repeated FX Lookups.
    Three distinct agents across the workflow make identical remote API queries for static
    ECB daily exchange rates (3 x 1.4s = 4.2s wasted) due to missing TTL memoization.
    """
    execution_id = f"lat-uncached-{uuid.uuid4().hex[:8]}"
    start_time = time.perf_counter()

    with tracer.start_as_current_span("trace_uncached_repeated_io") as root_span:
        root_span.set_attribute("gen_ai.workflow.name", "wf_multinational_tax_invoicing")
        root_span.set_attribute("gen_ai.workflow.execution_id", execution_id)
        root_span.set_attribute("gen_ai.latency.scenario", "uncached_repeated_io")
        root_span.set_attribute("gen_ai.latency.type", "UNCACHED_REPEATED_IO")
        trace_id = format(root_span.get_span_context().trace_id, "032x")

        # Step 1: Invoice Line Item Calculation (First FX Call: 1.4s)
        with handoff_span(
            tracer,
            agent_name="InvoiceCalculationAgent",
            action_name="calculate_line_item_subtotals",
            execution_id=execution_id,
            workflow_name="wf_multinational_tax_invoicing",
            handoff_to="VATComplianceAgent",
            handoff_reason="Subtotals converted to EUR base; forwarding to VAT compliance",
            latency_type="UNCACHED_REPEATED_IO",
        ) as span1:
            with tool_span(
                tracer,
                tool_name="fetch_ecb_fx_rates",
                arguments={"base_currency": "USD", "quote_currency": "EUR", "rate_date": "2026-09-28"},
                latency_type="UNCACHED_REPEATED_IO",
                execution_id=execution_id,
            ) as set_fx1:
                await asyncio.sleep(1.4)
                set_fx1({"pair": "USD_EUR", "rate": 0.924, "source": "ECB_OFFICIAL_DAILY", "cached": False})

            span1.set_token_usage(input_tokens=780, output_tokens=75)

        # Step 2: VAT Compliance Agent (Duplicate Second FX Call: 1.4s)
        with handoff_span(
            tracer,
            agent_name="VATComplianceAgent",
            action_name="determine_reverse_charge_vat",
            execution_id=execution_id,
            workflow_name="wf_multinational_tax_invoicing",
            handoff_from="InvoiceCalculationAgent",
            handoff_to="AuditJournalVerificationAgent",
            handoff_reason="Cross-border reverse charge VAT computed; forwarding to audit ledger",
            latency_type="UNCACHED_REPEATED_IO",
        ) as span2:
            with tool_span(
                tracer,
                tool_name="fetch_ecb_fx_rates",
                arguments={"base_currency": "USD", "quote_currency": "EUR", "rate_date": "2026-09-28"},
                latency_type="UNCACHED_REPEATED_IO",
                execution_id=execution_id,
            ) as set_fx2:
                # Identical redundant lookup
                await asyncio.sleep(1.4)
                set_fx2({"pair": "USD_EUR", "rate": 0.924, "source": "ECB_OFFICIAL_DAILY", "cached": False})

            span2.set_token_usage(input_tokens=840, output_tokens=80)

        # Step 3: Audit Journal Agent (Duplicate Third FX Call triggering Upstream Rate Limit)
        execution_failed = False
        error_msg = ""
        try:
            with handoff_span(
                tracer,
                agent_name="AuditJournalVerificationAgent",
                action_name="verify_cross_currency_ledger",
                execution_id=execution_id,
                workflow_name="wf_multinational_tax_invoicing",
                handoff_from="VATComplianceAgent",
                handoff_reason="Audit ledger reconciled against ECB benchmark rates",
                latency_type="UNCACHED_REPEATED_IO",
            ) as span3:
                with tool_span(
                    tracer,
                    tool_name="fetch_ecb_fx_rates",
                    arguments={"base_currency": "USD", "quote_currency": "EUR", "rate_date": "2026-09-28"},
                    latency_type="UNCACHED_REPEATED_IO",
                    execution_id=execution_id,
                ) as set_fx3:
                    # 3rd identical redundant lookup triggers rate limiter
                    await asyncio.sleep(1.4)
                    exc = UpstreamRateLimitExceeded(
                        "UpstreamRateLimitExceeded: HTTP 429 Too Many Requests from external provider 'api.ecb.europa.eu'. "
                        "Identical unmemoized query payload repeated 3 times in 4.2s window; upstream provider throttled client IP for 300s."
                    )
                    set_fx3({"status": "RATE_LIMITED", "http_status": 429, "error": str(exc)})
                    record_span_error(span3, exc)
                    span3.set_attribute("status", "error")
                    span3.set_attribute("status_code", "Error")
                    raise exc
        except UpstreamRateLimitExceeded as exc:
            execution_failed = True
            error_msg = str(exc)
            record_span_error(root_span, exc)
            root_span.set_attribute("gen_ai.activity.status", "FAILED")
            root_span.set_attribute("status_code", "Error")

        duration_ms = (time.perf_counter() - start_time) * 1000
        root_span.set_attribute("gen_ai.latency.duration_ms", duration_ms)

        return ScenarioResult(
            scenario_name="Uncached Repeated I/O Lookup",
            latency_type="UNCACHED_REPEATED_IO",
            execution_id=execution_id,
            trace_id=trace_id,
            total_duration_ms=duration_ms,
            status="FAILED" if execution_failed else "SUCCESS",
            step_count=3,
            summary="fetch_ecb_fx_rates invoked 3 redundant times, triggering upstream HTTP 429 rate limit (UpstreamRateLimitExceeded).",
            details={"error": error_msg, "duplicate_tool": "fetch_ecb_fx_rates", "call_count": 3, "recommended_cache": "Redis / LRU Cache"},
        )


# ===========================================================================
# SCENARIO 11: HEAVY MULTI-MODAL VISION PAYLOAD I/O
# ===========================================================================
async def run_heavy_vision_payload_scenario(client: Mistral, tracer: Tracer) -> ScenarioResult:
    """
    Scenario 11: Emergency Radiology Diagnostic Triage & 34MB Uncompressed DICOM Image Drag.
    DICOMImageIngestionAgent transmits raw uncompressed 34.2 MB multi-slice DICOM scans
    in base64 format, causing 4.8s JSON serialization and rasterization drag.
    """
    execution_id = f"lat-vision-{uuid.uuid4().hex[:8]}"
    start_time = time.perf_counter()

    with tracer.start_as_current_span("trace_heavy_vision_payload_io") as root_span:
        root_span.set_attribute("gen_ai.workflow.name", "wf_radiology_diagnostic_triage")
        root_span.set_attribute("gen_ai.workflow.execution_id", execution_id)
        root_span.set_attribute("gen_ai.latency.scenario", "heavy_vision_payload_io")
        root_span.set_attribute("gen_ai.latency.type", "HEAVY_VISION_PAYLOAD_IO")
        trace_id = format(root_span.get_span_context().trace_id, "032x")

        # Step 1: PACS DICOM Ingestion & Volume Rasterization (34.2 MB triggers gRPC RESOURCE_EXHAUSTED)
        execution_failed = False
        error_msg = ""
        try:
            with handoff_span(
                tracer,
                agent_name="DICOMImageIngestionAgent",
                action_name="load_uncompressed_ct_pulmonary_series",
                execution_id=execution_id,
                workflow_name="wf_radiology_diagnostic_triage",
                handoff_to="RadiologyPathologyClassifierAgent",
                handoff_reason="Passing 34.2 MB uncompressed 16-bit DICOM series as raw base64 string",
                latency_type="HEAVY_VISION_PAYLOAD_IO",
                metadata={"media_size_mb": 34.2, "slices": 128, "modality": "CT_CHEST_ANGIOGRAM"},
            ) as span1:
                span1.set_attribute("media.size_mb", 34.2)
                span1.set_attribute("media_size_mb", 34.2)

                with tool_span(
                    tracer,
                    tool_name="extract_dicom_volume_raw_base64",
                    arguments={"series_uid": "1.2.840.113619.2.55.3.8819", "encode_format": "RAW_BASE64_UNCOMPRESSED"},
                    latency_type="HEAVY_VISION_PAYLOAD_IO",
                    execution_id=execution_id,
                ) as set_tool:
                    # Heavy serialization & base64 encoding delay (3.8s)
                    await asyncio.sleep(3.8)
                    exc = ResourceExhaustedError(
                        "ResourceExhaustedError: gRPC StatusCode.RESOURCE_EXHAUSTED. Serialized payload size 34.2MB exceeds "
                        "maximum ingress gRPC message length (16.0MB) on agent handoff transport channel."
                    )
                    set_tool({"status": "FAILED", "error": str(exc), "megabytes": 34.2})
                    record_span_error(span1, exc)
                    span1.set_attribute("status", "error")
                    span1.set_attribute("status_code", "Error")
                    raise exc
        except ResourceExhaustedError as exc:
            execution_failed = True
            error_msg = str(exc)
            record_span_error(root_span, exc)
            root_span.set_attribute("gen_ai.activity.status", "FAILED")
            root_span.set_attribute("status_code", "Error")

        duration_ms = (time.perf_counter() - start_time) * 1000
        root_span.set_attribute("gen_ai.latency.duration_ms", duration_ms)

        return ScenarioResult(
            scenario_name="Heavy Vision Payload I/O",
            latency_type="HEAVY_VISION_PAYLOAD_IO",
            execution_id=execution_id,
            trace_id=trace_id,
            total_duration_ms=duration_ms,
            status="FAILED" if execution_failed else "SUCCESS",
            step_count=1,
            summary="DICOMImageIngestionAgent attempted 34.2 MB uncompressed payload transfer, rejected by gRPC transport (ResourceExhaustedError).",
            details={"error": error_msg, "media_size_mb": 34.2, "transfer_duration_s": 3.8, "grpc_limit_mb": 16.0},
        )


# ===========================================================================
# SCENARIO 12: CLEAN OPTIMAL BASELINE (Healthy Reference Workflow)
# ===========================================================================
async def run_clean_baseline_scenario(client: Mistral, tracer: Tracer) -> ScenarioResult:
    """
    Scenario 12: High-Throughput E-Commerce Automated Order Fulfillment Reference Pipeline.
    Clean, right-sized 3-agent handoff chain executing in < 380ms with 100% memoized lookups,
    bounded token generation, and zero latency defects.
    """
    execution_id = f"lat-baseline-{uuid.uuid4().hex[:8]}"
    start_time = time.perf_counter()

    with tracer.start_as_current_span("trace_clean_optimal_baseline") as root_span:
        root_span.set_attribute("gen_ai.workflow.name", "wf_optimized_order_dispatch")
        root_span.set_attribute("gen_ai.workflow.execution_id", execution_id)
        root_span.set_attribute("gen_ai.latency.scenario", "clean_optimal_baseline")
        root_span.set_attribute("gen_ai.latency.type", "CLEAN_OPTIMAL_BASELINE")
        trace_id = format(root_span.get_span_context().trace_id, "032x")

        # Step 1: Order Validation
        with handoff_span(
            tracer,
            agent_name="OrderIntakeAgent",
            action_name="validate_order_payload",
            execution_id=execution_id,
            workflow_name="wf_optimized_order_dispatch",
            handoff_to="InventoryAllocationAgent",
            handoff_reason="Order schema and payment token pre-authorized; allocating warehouse stock",
            latency_type="CLEAN_OPTIMAL_BASELINE",
        ) as span1:
            await asyncio.sleep(0.08)
            span1.set_token_usage(input_tokens=280, output_tokens=30)
            span1.set_attribute("agent.order_status", "VALID")

        # Step 2: In-Memory Cached Inventory Allocation
        with handoff_span(
            tracer,
            agent_name="InventoryAllocationAgent",
            action_name="reserve_warehouse_stock",
            execution_id=execution_id,
            workflow_name="wf_optimized_order_dispatch",
            handoff_from="OrderIntakeAgent",
            handoff_to="ShippingLabelDispatcherAgent",
            handoff_reason="Warehouse bin stock reserved via Redis in-memory lookup",
            latency_type="CLEAN_OPTIMAL_BASELINE",
        ) as span2:
            with tool_span(tracer, "redis_atomic_stock_decrement", {"sku": "ELEC-FAST-01", "qty": 1}, execution_id=execution_id) as set_res:
                await asyncio.sleep(0.09)
                set_res({"reserved": True, "remaining_stock": 482, "cache_hit": True})

            span2.set_token_usage(input_tokens=310, output_tokens=35)
            span2.set_attribute("agent.allocation_status", "RESERVED")

        # Step 3: Fast Shipping Dispatch
        with handoff_span(
            tracer,
            agent_name="ShippingLabelDispatcherAgent",
            action_name="generate_carrier_consignment",
            execution_id=execution_id,
            workflow_name="wf_optimized_order_dispatch",
            handoff_from="InventoryAllocationAgent",
            handoff_reason="FedEx Express label generated; order ready for packing",
            latency_type="CLEAN_OPTIMAL_BASELINE",
        ) as span3:
            await asyncio.sleep(0.11)
            span3.set_token_usage(input_tokens=340, output_tokens=40)
            span3.set_attribute("agent.tracking_id", "TRK-EXP-99214")

        duration_ms = (time.perf_counter() - start_time) * 1000
        root_span.set_attribute("gen_ai.latency.duration_ms", duration_ms)

        return ScenarioResult(
            scenario_name="Clean Optimal Baseline",
            latency_type="CLEAN_OPTIMAL_BASELINE",
            execution_id=execution_id,
            trace_id=trace_id,
            total_duration_ms=duration_ms,
            status="SUCCESS",
            step_count=3,
            summary="Crisp, efficient 3-agent pipeline completed in < 380ms with 100% cached lookups and zero latency defects.",
            details={"wasted_latency_ms": 0, "efficiency_score": 100.0, "status": "OPTIMAL"},
        )


# ===========================================================================
# SCENARIO 13: DISTRIBUTED LOCK TIMEOUT & TWO-PHASE COMMIT DEADLOCK
# ===========================================================================
async def run_distributed_lock_timeout_scenario(client: Mistral, tracer: Tracer) -> ScenarioResult:
    """
    Scenario 13: Global Institutional FX Multi-Leg Settlement Deadlock.
    TokyoDeskClearingAgent and LondonDeskClearingAgent contend for cross-currency ledger locks.
    A distributed deadlock causes a 4.2s lock wait stall before throwing LockWaitTimeoutException
    and executing compensating rollback transactions.
    """
    execution_id = f"lat-deadlock-{uuid.uuid4().hex[:8]}"
    start_time = time.perf_counter()

    with tracer.start_as_current_span("trace_distributed_lock_timeout") as root_span:
        root_span.set_attribute("gen_ai.workflow.name", "wf_cross_border_fx_settlement")
        root_span.set_attribute("gen_ai.workflow.execution_id", execution_id)
        root_span.set_attribute("gen_ai.latency.scenario", "distributed_lock_timeout")
        root_span.set_attribute("gen_ai.latency.type", "DISTRIBUTED_LOCK_TIMEOUT")
        trace_id = format(root_span.get_span_context().trace_id, "032x")

        # Step 1: Trade Capture & Ticket Validation
        with handoff_span(
            tracer,
            agent_name="FXTradeCaptureAgent",
            action_name="validate_trade_ticket",
            execution_id=execution_id,
            workflow_name="wf_cross_border_fx_settlement",
            handoff_to="TokyoDeskClearingAgent",
            handoff_reason="Validated $25M USD/JPY swap ticket, initiating two-phase settlement locks",
            latency_type="DISTRIBUTED_LOCK_TIMEOUT",
        ) as span1:
            await asyncio.sleep(0.18)
            span1.set_token_usage(input_tokens=780, output_tokens=65)

        # Step 2: Tokyo Book Leg Lock
        with handoff_span(
            tracer,
            agent_name="TokyoDeskClearingAgent",
            action_name="acquire_jpy_nostro_hold",
            execution_id=execution_id,
            workflow_name="wf_cross_border_fx_settlement",
            handoff_from="FXTradeCaptureAgent",
            handoff_to="LondonDeskClearingAgent",
            handoff_reason="JPY 3.75B held in escrow, awaiting USD counterparty lock",
            latency_type="DISTRIBUTED_LOCK_TIMEOUT",
        ) as span2:
            await asyncio.sleep(0.22)
            span2.set_token_usage(input_tokens=650, output_tokens=55)
            span2.set_attribute("agent.escrow_hold", "JPY_3.75B_LOCKED")

        # Step 3: London Book Leg Lock (Severe Deadlock & Timeout)
        execution_failed = False
        error_msg = ""
        try:
            with handoff_span(
                tracer,
                agent_name="LondonDeskClearingAgent",
                action_name="acquire_usd_vostro_hold",
                execution_id=execution_id,
                workflow_name="wf_cross_border_fx_settlement",
                handoff_from="TokyoDeskClearingAgent",
                handoff_to="CompensatingReversalAgent",
                handoff_reason="Distributed deadlock occurred; lock wait timeout exceeded, initiating rollback",
                latency_type="DISTRIBUTED_LOCK_TIMEOUT",
                metadata={"lock_resource": "ledger:settlement:usd_jpy", "lock_timeout_ms": 4000},
            ) as span3:
                span3.set_attribute("status", "error")
                span3.set_attribute("status_code", "Error")

                with tool_span(
                    tracer,
                    tool_name="acquire_distributed_consul_lock",
                    arguments={"resource": "ledger:settlement:usd_jpy", "timeout_ms": 4000},
                    latency_type="DISTRIBUTED_LOCK_TIMEOUT",
                    execution_id=execution_id,
                ) as set_tool:
                    # 4.2s lock wait stall ending in timeout exception
                    await asyncio.sleep(4.2)
                    exc = TwoPhaseCommitDeadlockException(
                        "TwoPhaseCommitDeadlockException: Cross-region 2PC lock deadlock between Tokyo book ('JPY_NOSTRO') "
                        "and London book ('USD_VOSTRO'). Distributed lock acquisition timed out after 4200ms on resource 'ledger:settlement:usd_jpy'."
                    )
                    set_tool({"status": "FAILED", "error": str(exc)})
                    span3.set_token_usage(input_tokens=890, output_tokens=40)
                    record_span_error(span3, exc)
                    raise exc
        except TwoPhaseCommitDeadlockException as exc:
            execution_failed = True
            error_msg = str(exc)
            record_span_error(root_span, exc)
            root_span.set_attribute("gen_ai.activity.status", "FAILED")
            root_span.set_attribute("status_code", "Error")

        # Step 4: Compensating Rollback Transaction
        with handoff_span(
            tracer,
            agent_name="CompensatingReversalAgent",
            action_name="execute_two_phase_rollback",
            execution_id=execution_id,
            workflow_name="wf_cross_border_fx_settlement",
            handoff_from="LondonDeskClearingAgent",
            handoff_reason="Atomic compensation executed: released JPY escrow hold and aborted settlement",
            latency_type="DISTRIBUTED_LOCK_TIMEOUT",
        ) as span4:
            with tool_span(tracer, "release_escrow_hold", {"escrow_id": "ESC-JPY-9912"}, execution_id=execution_id) as set_roll:
                await asyncio.sleep(0.28)
                set_roll({"status": "ROLLED_BACK", "escrow_freed": True})

            span4.set_token_usage(input_tokens=740, output_tokens=70)
            span4.set_attribute("agent.compensation_status", "ROLLED_BACK")

        duration_ms = (time.perf_counter() - start_time) * 1000
        root_span.set_attribute("gen_ai.latency.duration_ms", duration_ms)
        root_span.set_attribute("status_code", "Error")

        return ScenarioResult(
            scenario_name="Distributed Lock Timeout",
            latency_type="DISTRIBUTED_LOCK_TIMEOUT",
            execution_id=execution_id,
            trace_id=trace_id,
            total_duration_ms=duration_ms,
            status="FAILED",
            step_count=4,
            summary="LondonDeskClearingAgent encountered a 4.2s distributed lock timeout deadlock on USD/JPY settlement ledger, forcing transaction rollback.",
            details={"lock_resource": "ledger:settlement:usd_jpy", "lock_wait_s": 4.2, "status": "ROLLED_BACK"},
        )


# ===========================================================================
# SCENARIO 14: EXPONENTIAL CONTEXT EXPLOSION (Zero-Day Cybersecurity Triage)
# ===========================================================================
async def run_context_explosion_scenario(client: Mistral, tracer: Tracer) -> ScenarioResult:
    """
    Scenario 14: Zero-Day Cybersecurity Incident Response & Uncompacted Memory Explosion.
    Four specialized incident agents naively append raw x86 assembly, core dumps, and Ghidra decompilation
    into the conversation history without sliding-window pruning. Context expands to 46,000+ tokens,
    causing 3.2s TTFT and 4.4s LLM generation latency drag.
    """
    execution_id = f"lat-explosion-{uuid.uuid4().hex[:8]}"
    start_time = time.perf_counter()

    with tracer.start_as_current_span("trace_exponential_context_explosion") as root_span:
        root_span.set_attribute("gen_ai.workflow.name", "wf_zero_day_vulnerability_triage")
        root_span.set_attribute("gen_ai.workflow.execution_id", execution_id)
        root_span.set_attribute("gen_ai.latency.scenario", "exponential_context_explosion")
        root_span.set_attribute("gen_ai.latency.type", "EXPONENTIAL_CONTEXT_EXPLOSION")
        trace_id = format(root_span.get_span_context().trace_id, "032x")

        # Step 1: Initial Threat Telemetry (2,800 tokens)
        with handoff_span(
            tracer,
            agent_name="ThreatTelemetryIngestAgent",
            action_name="extract_core_dump_memory",
            execution_id=execution_id,
            workflow_name="wf_zero_day_vulnerability_triage",
            handoff_to="BinaryDecompilationAgent",
            handoff_reason="Initial memory register crash state dumped (2,800 tokens)",
            latency_type="EXPONENTIAL_CONTEXT_EXPLOSION",
        ) as span1:
            await asyncio.sleep(0.2)
            span1.set_token_usage(input_tokens=2800, output_tokens=90)
            span1.set_attribute("prompt_tokens", 2800)

        # Step 2: Binary Decompilation (Appends 9,500 tokens of raw assembly)
        with handoff_span(
            tracer,
            agent_name="BinaryDecompilationAgent",
            action_name="decompile_assembly_subroutines",
            execution_id=execution_id,
            workflow_name="wf_zero_day_vulnerability_triage",
            handoff_from="ThreatTelemetryIngestAgent",
            handoff_to="ExploitPocSynthesizerAgent",
            handoff_reason="Appended 9,500 tokens of raw x86 assembly to dialogue history without pruning",
            latency_type="EXPONENTIAL_CONTEXT_EXPLOSION",
        ) as span2:
            await asyncio.sleep(0.35)
            span2.set_token_usage(input_tokens=12300, output_tokens=220)
            span2.set_attribute("prompt_tokens", 12300)

        # Step 3: Exploit PoC Synthesizer (Appends 16,000 tokens of Ghidra graphs)
        with handoff_span(
            tracer,
            agent_name="ExploitPocSynthesizerAgent",
            action_name="craft_buffer_overflow_payload",
            execution_id=execution_id,
            workflow_name="wf_zero_day_vulnerability_triage",
            handoff_from="BinaryDecompilationAgent",
            handoff_to="SecurityPatchAdvisoryAgent",
            handoff_reason="Context inflated to 28,500 tokens; forwarding full uncompacted trace to patch synthesizer",
            latency_type="EXPONENTIAL_CONTEXT_EXPLOSION",
        ) as span3:
            await asyncio.sleep(0.4)
            span3.set_token_usage(input_tokens=28500, output_tokens=310)
            span3.set_attribute("prompt_tokens", 28500)

        # Step 4: Final Patch Advisory Agent (Catastrophic Context Explosion: 46,200 tokens OOM)
        execution_failed = False
        error_msg = ""
        try:
            with handoff_span(
                tracer,
                agent_name="SecurityPatchAdvisoryAgent",
                action_name="generate_cve_remediation_bulletin",
                execution_id=execution_id,
                workflow_name="wf_zero_day_vulnerability_triage",
                handoff_from="ExploitPocSynthesizerAgent",
                handoff_reason="Catastrophic prompt inflation: 46,200 tokens processed across 4 uncompacted turns",
                latency_type="EXPONENTIAL_CONTEXT_EXPLOSION",
                metadata={"prompt_tokens": 46200, "ttft_ms": 3200},
            ) as span4:
                span4.set_attribute("prompt_tokens", 46200)
                span4.set_attribute("ttft_ms", 3200)
                span4.set_attribute("gen_ai.latency.ttft_ms", 3200)

                # Massive TTFT drag culminating in worker container OOM kill
                await asyncio.sleep(3.8)
                exc = MemoryAllocationError(
                    "MemoryAllocationError: Worker container OOMKilled (ExitCode 137). Context memory buffer of 46,200 tokens "
                    "exceeded 32,768 token resident memory ceiling on worker pod 'sec-triage-worker-7b9d'."
                )
                span4.set_token_usage(input_tokens=46200, output_tokens=0)
                record_span_error(span4, exc)
                span4.set_attribute("status", "error")
                span4.set_attribute("status_code", "Error")
                raise exc
        except MemoryAllocationError as exc:
            execution_failed = True
            error_msg = str(exc)
            record_span_error(root_span, exc)
            root_span.set_attribute("gen_ai.activity.status", "FAILED")
            root_span.set_attribute("status_code", "Error")

        duration_ms = (time.perf_counter() - start_time) * 1000
        root_span.set_attribute("gen_ai.latency.duration_ms", duration_ms)

        return ScenarioResult(
            scenario_name="Exponential Context Explosion",
            latency_type="EXPONENTIAL_CONTEXT_EXPLOSION",
            execution_id=execution_id,
            trace_id=trace_id,
            total_duration_ms=duration_ms,
            status="FAILED" if execution_failed else "SUCCESS",
            step_count=4,
            summary="Prompt context exploded to 46,200 uncompacted tokens, crashing worker pod with MemoryAllocationError (OOMKilled).",
            details={"error": error_msg, "initial_tokens": 2800, "final_prompt_tokens": 46200, "memory_ceiling_tokens": 32768},
        )
