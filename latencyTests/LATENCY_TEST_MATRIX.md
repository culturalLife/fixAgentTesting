# Multi-Agent Workflow Latency Benchmark Matrix (Enterprise Grade v2)

This matrix documents the 14 realistic multi-agent workflows built in `latencyTests` to benchmark, evaluate, and calibrate the **Mistral Observability Latency Evaluators**, **Cluster Workers (v2)**, and the **Auto-Fix Codestral Remediation Engine**.

Every scenario is authored as a realistic production workflow with authentic enterprise domain logic (customs clearance, maritime freight claims, treasury cash pooling, M&A antitrust due diligence, syndicated real estate lending, SWIFT wire transfers, cybersecurity incident response, radiology tele-consultation, and defense security clearances). Defect signatures arise **organically from distributed engineering bottlenecks, unpruned contexts, concurrency imbalances, and lock contentions**, triggering authentic enterprise-grade system exceptions.

---

## 1. Master Scenario & Defect Evaluation Matrix

| # | Scenario Name | CLI Key | Workflow Identifier | Target Latency Defect | Enterprise Failure Class | Root Defect Pattern & Architecture | Remediation Recommendation |
|---|---|---|---|---|---|---|---|
| **1** | **Cascading Handoff Delay** | `cascade` | `wf_cross_border_customs_clearance` | `CASCADING_HANDOFF_DELAY` | `DeadlineExceededError` | Upstream `CustomsDocumentParserAgent` suffers 3.8s OCR rasterization choke, breaching the 4200ms ingress SLA. | Stream parsed chunks asynchronously and parallelize rasterization with a worker pool. |
| **2** | **Ping-Pong Loop Thrash** | `loop` | `wf_freight_claim_arbitration` | `LOOP_THRASH_HANDOFF` | `MaxHandoffHopsExceededError` | Marine cargo reefer container damage claim repeatedly bounces over 8 handoff cycles, tripping recursion limit. | Enforce explicit handoff contracts with hard 2-turn round-trip thresholds and arbitration escalation. |
| **3** | **Blocking Tool Stall** | `tool` | `wf_treasury_liquidity_sweep` | `BLOCKING_TOOL_STALL` | `DatabaseLockTimeoutError` | `CoreBankingLedgerAgent` encounters row-level master ledger lock contention and HikariCP pool exhaustion (4.2s stall). | Implement non-blocking asynchronous database commits, connection pooling, and optimistic concurrency. |
| **4** | **Token Bloat & High TTFT** | `bloat` | `wf_ma_due_diligence_briefing` | `TOKEN_BLOAT_LATENCY` | `ContextWindowExceededError` | `DocumentCorpusCollectorAgent` passes unpruned 38,400-token corpus across handoff, breaching 16k cluster allocation limit. | Apply pre-handoff extractive summarization, vector chunking, or map-reduce synthesis (< 2k tokens). |
| **5** | **Fan-out Straggler Bottleneck** | `fanout` | `wf_commercial_loan_underwriting` | `FANOUT_STRAGGLER_DELAY` | `AsyncFanoutTimeoutError` | Parallel fan-out barrier collapses when Commercial Valuation straggles (3.5s vs peer 280ms), exceeding 3000ms SLA. | Configure aggressive per-agent SLA timeouts (1.5s) with cached spatial regression fallbacks. |
| **6** | **Retry Storm & Degraded Fallback** | `retry` | `wf_payment_gateway_routing` | `RETRY_STORM_BACKOFF` | `CircuitBreakerOpenException` | `PaymentProcessingAgent` encounters 504 Timeout, 429 Rate Limit, and 503 Unavailable; circuit breaker trips to DLQ. | Configure circuit breakers, jittered exponential backoffs, and immediate asynchronous queue offloading. |
| **7** | **Model Oversizing LLM Delay** | `oversize` | `wf_omnichannel_ticket_routing` | `MODEL_OVERSIZING_LLM_DELAY` | `InferenceQueueTimeoutError` | Frontier `mistral-large-latest` invoked for 1-token classification; GPU KV-cache allocation queue times out (> 3000ms). | **Model Downgrade:** Switch to `mistral-small-latest` or `ministral-8b-latest` (saves 75%+ latency and cost). |
| **8** | **Unconstrained Generation Drag** | `unconstrained` | `wf_regulatory_incident_disclosure` | `UNCONSTRAINED_GEN_DRAG` | `GatewayReadTimeoutError` | Missing `max_tokens` limits emits 1,460 verbose tokens over 4.2s, breaching Envoy reverse proxy 504 read timeout. | Set `max_tokens: 150`, add strict JSON schema or bulleted structure guidelines. |
| **9** | **Sequential I/O Waterfall** | `waterfall` | `wf_employee_clearance_pipeline` | `SEQUENTIAL_IO_WATERFALL` | `CumulativeTransactionBudgetExceeded` | Synchronous loop of 5 independent remote checks breaches 3000ms global transaction budget at check 4 (3309ms). | Replace sequential iteration with `asyncio.gather()` to execute checks concurrently in ~850ms. |
| **10** | **Uncached Repeated I/O Lookup** | `uncached` | `wf_multinational_tax_invoicing` | `UNCACHED_REPEATED_IO` | `UpstreamRateLimitExceeded` | 3 distinct agents make identical remote queries for ECB FX rates; 3rd unmemoized query triggers HTTP 429 rate limit. | Wrap lookup tool in Redis / `@lru_cache(ttl=3600)` decorator for idempotent queries. |
| **11** | **Heavy Vision Payload I/O** | `vision_io` | `wf_radiology_diagnostic_triage` | `HEAVY_VISION_PAYLOAD_IO` | `ResourceExhaustedError` | Passing 34.2 MB uncompressed DICOM scan over handoff exceeds the 16MB gRPC max message length. | Downsample image to 1024px WebP / crop clinical region of interest (ROI) prior to LLM transfer. |
| **12** | **Clean Optimal Baseline** | `baseline` | `wf_optimized_order_dispatch` | `CLEAN_OPTIMAL_BASELINE` | *None (Healthy Reference)* | Crisp 3-agent e-commerce order fulfillment executing in < 380ms with 100% cached lookups and zero wasted tokens. | Control reference — no remediation needed (`status="SUCCESS"`). |
| **13** | **Distributed Lock Timeout** | `deadlock` | `wf_cross_border_fx_settlement` | `DISTRIBUTED_LOCK_TIMEOUT` | `TwoPhaseCommitDeadlockException` | Tokyo and London clearing desks deadlock on cross-currency settlement ledger locks, triggering rollback. | Implement lock ordering hierarchy, reduce transaction scope, or use optimistic escrow reservations. |
| **14** | **Exponential Context Explosion** | `explosion` | `wf_zero_day_vulnerability_triage` | `EXPONENTIAL_CONTEXT_EXPLOSION` | `MemoryAllocationError` | 4 zero-day incident agents accumulate 46,200 tokens without compaction, triggering worker container OOM kill (Exit 137). | Enforce sliding-window compaction or summarize decompilation traces before agent handoffs. |

---

## 2. Latest Benchmark Execution Results (Enterprise Outages & Trace IDs)

All 14 scenarios were executed and traces were synchronously uploaded to Mistral Observability (`https://api.f418c72b5f65.dc.mistral.ai/telemetry/v1/traces`):

| # | Scenario Name | Status | Duration | Enterprise Exception | OTel Trace ID | Deep-Dive Trace Waterfall |
|---|---|---|---|---|---|---|
| **1** | Cascading Handoff Delay | **FAILED** | 9.11s | `DeadlineExceededError` | `f6e862ad76c17977cde3c3f9ddbf481a` | [View Waterfall](http://localhost:5173/traces/f6e862ad76c17977cde3c3f9ddbf481a) |
| **2** | Ping-Pong Loop Thrash | **FAILED** | 18.90s | `MaxHandoffHopsExceededError` | `950fc1d7d0e6812509b10aa1ad7cb087` | [View Waterfall](http://localhost:5173/traces/950fc1d7d0e6812509b10aa1ad7cb087) |
| **3** | Blocking Tool Stall | **FAILED** | 6.69s | `DatabaseLockTimeoutError` | `005f691e0331656473277a838e27950c` | [View Waterfall](http://localhost:5173/traces/005f691e0331656473277a838e27950c) |
| **4** | Token Bloat & High TTFT Latency | **FAILED** | 3.17s | `ContextWindowExceededError` | `3aeeeb3ce26b8d6c3fd54eb858e43442` | [View Waterfall](http://localhost:5173/traces/3aeeeb3ce26b8d6c3fd54eb858e43442) |
| **5** | Fanout Straggler Delay | **FAILED** | 3.66s | `AsyncFanoutTimeoutError` | `6007e7419e36baab18ee1ac74440f8bf` | [View Waterfall](http://localhost:5173/traces/6007e7419e36baab18ee1ac74440f8bf) |
| **6** | Retry Storm Backoff | **FAILED** | 4.77s | `CircuitBreakerOpenException` | `9512b8f379e58e83027fa11754809781` | [View Waterfall](http://localhost:5173/traces/9512b8f379e58e83027fa11754809781) |
| **7** | Model Oversizing LLM Delay | **FAILED** | 3.80s | `InferenceQueueTimeoutError` | `dc39d04eeaf4cd3fee7d81f30256ef1c` | [View Waterfall](http://localhost:5173/traces/dc39d04eeaf4cd3fee7d81f30256ef1c) |
| **8** | Unconstrained Generation Drag | **FAILED** | 4.42s | `GatewayReadTimeoutError` | `ff1d48f379d835ceabfc9ad9f910a7df` | [View Waterfall](http://localhost:5173/traces/ff1d48f379d835ceabfc9ad9f910a7df) |
| **9** | Sequential I/O Waterfall | **FAILED** | 3.53s | `CumulativeTransactionBudgetExceeded` | `68cfcddc09a02d8a8a97ca37f279802c` | [View Waterfall](http://localhost:5173/traces/68cfcddc09a02d8a8a97ca37f279802c) |
| **10** | Uncached Repeated I/O Lookup | **FAILED** | 4.23s | `UpstreamRateLimitExceeded` | `146144a825717792105ad2906ffdce45` | [View Waterfall](http://localhost:5173/traces/146144a825717792105ad2906ffdce45) |
| **11** | Heavy Vision Payload I/O | **FAILED** | 3.81s | `ResourceExhaustedError` | `83fc029e34d9c895c33dcf5564dbb5f2` | [View Waterfall](http://localhost:5173/traces/83fc029e34d9c895c33dcf5564dbb5f2) |
| **12** | Clean Optimal Baseline | **SUCCESS** | 0.30s | *Healthy Baseline* | `5fd74bcb85e9f13a8217e2b83efba5ca` | [View Waterfall](http://localhost:5173/traces/5fd74bcb85e9f13a8217e2b83efba5ca) |
| **13** | Distributed Lock Timeout | **FAILED** | 4.92s | `TwoPhaseCommitDeadlockException` | `165f63255bb8eea1a6a2d89edc265d37` | [View Waterfall](http://localhost:5173/traces/165f63255bb8eea1a6a2d89edc265d37) |
| **14** | Exponential Context Explosion | **FAILED** | 4.78s | `MemoryAllocationError` | `d0bb4ab9a26ecf4b81a62bc9df8687bc` | [View Waterfall](http://localhost:5173/traces/d0bb4ab9a26ecf4b81a62bc9df8687bc) |

---

## 3. Multi-Agent Telemetry Architecture

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

## 4. Running the Test Suite

Execute the benchmark suite using `python`:

```bash
# Run all 14 scenarios and synchronously upload traces to Mistral Observability:
python latencyTests/run_latency_suite.py

# Run specific scenarios:
python latencyTests/run_latency_suite.py --scenario cascade
python latencyTests/run_latency_suite.py --scenario loop
python latencyTests/run_latency_suite.py --scenario tool
python latencyTests/run_latency_suite.py --scenario bloat
python latencyTests/run_latency_suite.py --scenario fanout
python latencyTests/run_latency_suite.py --scenario retry
python latencyTests/run_latency_suite.py --scenario oversize
python latencyTests/run_latency_suite.py --scenario unconstrained
python latencyTests/run_latency_suite.py --scenario waterfall
python latencyTests/run_latency_suite.py --scenario uncached
python latencyTests/run_latency_suite.py --scenario vision_io
python latencyTests/run_latency_suite.py --scenario baseline
python latencyTests/run_latency_suite.py --scenario deadlock
python latencyTests/run_latency_suite.py --scenario explosion

# Resume from a specific scenario index onwards (e.g., from #6 onwards):
python latencyTests/run_latency_suite.py --from-scenario 6
```
