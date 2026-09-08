# Agent Handoff Latency Error Test Matrix

This matrix documents the 7 agent handoff scenarios implemented in [`latencyTests/`](./). Each scenario generates pure OpenTelemetry traces with standardized GenAI attributes (`gen_ai.agent.*`, `gen_ai.latency.*`, `gen_ai.tool.*`) and uploads them directly to the Mistral Observability endpoint [`https://api.mistral.ai/telemetry/v1/traces`](https://docs.mistral.ai/api/endpoint/beta/observability/traces) without using Temporal or Mistral workflow engine primitives.

---

## 1. Scenario Evaluation Matrix

| # | Scenario | Latency Defect Type | Root Cause Simulated | Span & Tracing Signature | Latency Judge / Evaluator Expected Verdict | Remediation Recommendation |
|---|---|---|---|---|---|---|
| **1** | **Cascading Handoff Delay** | `CASCADING_HANDOFF_DELAY` | Upstream ingestion agent stalls on heavy 48-page OCR/chunking (3.2s), starving downstream entity extraction and synthesis agents. | Trace duration > 3.5s; >85% of total execution time concentrated in `DocumentIngestionAgent` span. | `LATENCY_BOTTLENECK_DETECTED` (Upstream Agent Bottleneck) | Enable streaming chunk handoffs, parallel page parsing, or background OCR preprocessing. |
| **2** | **Ping-Pong Loop Thrash** | `LOOP_THRASH_HANDOFF` | Circular clarification ambiguity between `TriageAgent` and `PolicyValidationAgent` bouncing 4 iterations back-and-forth before resolving. | High span count (8+ spans); cyclical `gen_ai.agent.handoff.iteration` increments; high cumulative latency. | `LOOP_THRASH_DETECTED` / `EXCESSIVE_AGENT_HANDOFFS` | Enforce explicit handoff state contracts, add maximum round-trip thresholds, or inject an arbitrator agent. |
| **3** | **Blocking Tool / I/O Stall** | `BLOCKING_TOOL_STALL` | `LegacyDatabaseAgent` executes a slow synchronous tool call `query_legacy_sap_erp` with a 3.6s I/O network hang. | Disproportionate `tool_query_legacy_sap_erp` span duration relative to LLM generation time. | `SLOW_TOOL_EXECUTION` / `IO_BLOCKING_LATENCY` | Implement caching layer, asynchronous non-blocking tool calls, or read replicas. |
| **4** | **Token Bloat & High TTFT** | `TOKEN_BLOAT_LATENCY` | `DocumentResearchAgent` passes 35,400 unpruned tokens across handoff to `ExecutiveSummaryAgent`, causing high TTFT (2.4s) and generation drag. | Elevated `llm.prompt_tokens` (35.4k); elevated `llm.time_to_first_token_ms` (2400ms); delayed LLM span. | `TOKEN_BLOAT_LATENCY` / `EXCESSIVE_CONTEXT_WINDOW` | Apply context pruning/summarization filter before agent handoff, or use chunked map-reduce synthesis. |
| **5** | **Fan-out Straggler Bottleneck** | `FANOUT_STRAGGLER_DELAY` | `UnderwritingCoordinator` fans out to 3 parallel agents: Identity (0.2s) & Fraud (0.3s) finish quickly, but Credit Bureau straggles (3.9s), gating aggregation. | Parallel child spans under parent where total wall-clock duration is strictly bounded by the slowest straggler span. | `ASYNC_FANOUT_STRAGGLER` / `CONCURRENCY_IMBALANCE` | Set aggressive per-agent SLA timeouts with fallback partial aggregations, or use hedging requests. |
| **6** | **Retry Storm & Degraded Fallback** | `RETRY_STORM_BACKOFF` | `PaymentSettlementAgent` attempts 3 consecutive timeout retries with exponential backoffs (0.8s, 1.6s) before falling back to `DegradedSettlementAgent`. | Sequence of failed `ach_payment_gateway` child spans with escalating backoff delays ending in `DEGRADED_SUCCESS`. | `RETRY_STORM_LATENCY` / `CASCADING_RETRY_EXHAUSTION` | Configure circuit breakers, jittered exponential backoffs, and immediate asynchronous queue offloading. |
| **7** | **Model Oversizing LLM Delay** | `MODEL_OVERSIZING_LLM_DELAY` | Frontier `mistral-large-latest` invoked for trivial 2-word boolean classification task, causing 3.8s LLM generation delay. | Single LLM span taking ~4s for < 10 output tokens; `task_complexity: LOW_SIMPLE_CLASSIFICATION`. | `MODEL_OVERSIZING_DETECTED` (Model Compute Overkill) | **Model Switch:** Downgrade model to `mistral-small-latest` or `ministral-8b-latest` (saves ~75% latency and cost). |
| **8** | **Unconstrained Generation Drag** | `UNCONSTRAINED_GEN_DRAG` | Prompt lacks `max_tokens` limit or structural bounds, causing LLM to emit 1,400+ verbose tokens (4.5s delay) for simple FAQ. | `llm.completion_tokens: 1420`; long generation duration relative to input prompt size. | `RUNAWAY_OUTPUT_GENERATION` (Unconstrained Completion) | **Formatting Guardrails:** Set `max_tokens: 150`, add strict JSON schema or bullet output instructions. |
| **9** | **Sequential I/O Waterfall** | `SEQUENTIAL_IO_WATERFALL` | `ProfileEnrichmentAgent` executes 4 independent network/DB tool calls in a synchronous sequence (4 x 0.85s = 3.4s). | 4 sequential non-overlapping `execute_tool` child spans under single agent step. | `SEQUENTIAL_IO_BOTTLENECK` (Un-batched Network I/O) | **Concurrency / Batching:** Parallelize independent calls with `asyncio.gather()` or use a batch API endpoint (reduces I/O from 3.4s to 0.9s). |
| **10** | **Uncached Repeated I/O Lookup** | `UNCACHED_REPEATED_IO` | `InvoiceCalculationAgent` and `TaxComplianceAgent` make duplicate remote API calls for identical static FX rate data (2 x 1.6s). | Multiple duplicate tool spans with identical arguments (`pair: EUR_USD`) across handoff chain. | `UNCACHED_IDEMPOTENT_IO` (Missing Cache Layer) | **Caching Layer:** Implement Redis / in-memory TTL caching decorator on static/slow-changing lookup tools. |
| **11** | **Heavy Vision Payload I/O** | `HEAVY_VISION_PAYLOAD_IO` | `HeavyVisionAgent` transmits raw uncompressed 25.4MB TIFF image over base64, causing 4.2s transfer and rasterization latency. | Disproportionate base64 payload transfer duration; `media.size_mb: 25.4`. | `UNCOMPRESSED_VISION_PAYLOAD` (Heavy Multi-modal I/O) | **Preprocessing / Compression:** Downscale resolution to max 1024px, convert to WebP, or OCR-extract text prior to LLM ingestion. |
| **12** | **Clean Optimal Baseline** | `CLEAN_OPTIMAL_BASELINE` | Fast, crisp 2-agent handoff (`ExpressIntakeAgent` $\rightarrow$ `ExpressDispatchAgent`) completing in < 400ms with zero errors or stalls. | Low total latency (< 400ms); zero error spans; clean 1:1 handoff chain. | `OPTIMAL_PERFORMANCE` (Healthy Baseline) | Control reference — no remediation needed. |

---

## 2. Telemetry Architecture

```
                                  OTel Root Span
               (gen_ai.latency.scenario, execution_id, trace_id)
                                        │
             ┌──────────────────────────┼──────────────────────────┐
             ▼                          ▼                          ▼
   ┌───────────────────┐      ┌───────────────────┐      ┌───────────────────┐
   │ Agent Handoff 1   │      │ Agent Handoff 2   │      │ Agent Handoff 3   │
   │ gen_ai.agent.name │ ───► │ gen_ai.agent.name │ ───► │ gen_ai.agent.name │
   │ handoff.to        │      │ handoff.from/to   │      │ handoff.from      │
   └─────────┬─────────┘      └─────────┬─────────┘      └───────────────────┘
             │                          │
             ▼                          ▼
   ┌───────────────────┐      ┌───────────────────┐
   │ Child Tool Span   │      │ LLM Ingestion     │
   │ gen_ai.tool.name  │      │ prompt/completion │
   └───────────────────┘      └───────────────────┘
```

---

## 3. How to Execute

From `trial-workflow/latencyTests`:

```bash
# Run all 7 scenarios and upload traces
python run_latency_suite.py

# Or run individual scenarios
python run_latency_suite.py --scenario cascade
python run_latency_suite.py --scenario loop
python run_latency_suite.py --scenario tool
python run_latency_suite.py --scenario bloat
python run_latency_suite.py --scenario fanout
python run_latency_suite.py --scenario retry
python run_latency_suite.py --scenario baseline
```

Recorded trace records will be automatically saved to [`LATEST_LATENCY_TRACE_IDS.json`](./LATEST_LATENCY_TRACE_IDS.json).
