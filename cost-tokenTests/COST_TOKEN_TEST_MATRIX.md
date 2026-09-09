# Cost & Token-Behavior Multi-Agent Workflow Benchmark Matrix

This matrix documents the 9 realistic multi-agent workflows built in `cost-tokenTests` to evaluate the **Cost Agent** and **Token Efficiency Engine** detectors. 

Every scenario is authored as a realistic production workflow with authentic enterprise domain logic (accounts payable, mortgage underwriting, cross-border banking, commercial leasing, customer support triage, VIP travel concierge, supply chain inventory, and investment portfolio reporting). System defects arise **organically from common engineering oversights and operational edge cases**, without artificial error flags or test mocks.

---

## 1. Master Scenario & Detector Mapping

| # | Scenario Name | CLI Key | Workflow Identifier | Target Detector | Severity | Root Defect Pattern |
|---|---|---|---|---|---|---|
| **1** | **Vendor Invoice Reconciliation** | `retries` | `wf_vendor_invoice_reconciliation` | `RUNAWAY_RETRIES` (D3) | CRITICAL / HIGH | Model wraps JSON extraction in markdown fences; `json.loads` raises `JSONDecodeError`, triggering 3 activity retries. |
| **2** | **Dual Compliance Screening** | `redundant` | `wf_dual_compliance_screening` | `REDUNDANT_CALLS` (D2) | HIGH / MEDIUM | OFAC, AML, and Audit microservices independently execute identical statutory queries without memoization. |
| **3** | **Mortgage Underwriting & Settlement** | `failed` | `wf_mortgage_settlement_pipeline` | `FAILED_EXECUTIONS` (D5) | HIGH | Credit, income, and valuation succeed; wire settlement crashes on missing `wire_routing_number` schema key, wasting all upstream tokens. |
| **4** | **Commercial Lease Due Diligence** | `expensive` | `wf_commercial_lease_analysis` | `EXPENSIVE_PROMPT` (D4) | HIGH | Prompt builder injects 8,500+ tokens of unpruned commercial lease boilerplate across 5 queries without prefix caching ($p50 > 5,700$). |
| **5** | **Customer Support Ticket Triage** | `oversize` | `wf_ticket_classification_routing` | `OVER_PROVISIONED_MODEL` (D1) | LOW (Observation) | Global configuration defaulted to `mistral-large-latest` for single-turn, 1-token enum classification (`BILLING`) with >2,300 prompt tokens. |
| **6** | **VIP Travel Concierge Dialogue** | `leak` | `wf_concierge_support_dialogue` | `CONTEXT_GROWTH_LEAK` (D6) | HIGH | Conversation memory naively appends all turns across 5 agent handoffs without compaction (116 $\rightarrow$ 380 $\rightarrow$ 639 $\rightarrow$ 889 $\rightarrow$ 1,133 tokens). |
| **7** | **Retail Supply Chain Inventory Audit** | `amplification` | `wf_inventory_catalog_sync` | `TOOL_CALL_AMPLIFICATION` (D7) | HIGH / MEDIUM | Tool returns 20 KB uncompressed warehouse JSON dump; raw output is dumped into prompt, dominating $60\%$ of prompt tokens. |
| **8** | **Quantitative Portfolio Reporting** | `collapse` | `wf_portfolio_strategy_report` | `CACHE_COLLAPSE` (D8) | HIGH | Calls 1 & 2 warm up prefix cache (2,800 tokens); Call 3 prepends dynamic audit timestamps at byte 0, dropping cache read to 0 tokens. |
| **9** | **High-Efficiency Claims Baseline** | `baseline` | `wf_optimized_claims_processing` | `CLEAN_OPTIMAL_BASELINE` | N/A (Optimal) | Right-sized `mistral-small-latest`, static system prefix placed first, projected tool results, sliding-window memory, 0 wasted tokens. |

---

## 2. Latest Calibrated Benchmark Run Traces & UI Links

| # | Scenario Name | Target Anomaly | Execution ID | OTel Trace ID | Duration | Status | Deep Dive Links |
|---|---|---|---|---|---|---|---|
| **1** | Vendor Invoice Reconciliation | `RUNAWAY_RETRIES` | `exec-inv-rec-10cc8a13` | `a33448b001833ea65d4c8847372a4410` | 10.80s | SUCCESS | [Cost](http://localhost:5173/cost?trace_id=a33448b001833ea65d4c8847372a4410) · [Tokens](http://localhost:5173/tokens?trace_id=a33448b001833ea65d4c8847372a4410) |
| **2** | Dual Compliance Screening | `REDUNDANT_CALLS` | `exec-wire-comp-a7115bf9` | `6237c7e16820c26c16e310c437bab94b` | 4.98s | SUCCESS | [Cost](http://localhost:5173/cost?trace_id=6237c7e16820c26c16e310c437bab94b) · [Tokens](http://localhost:5173/tokens?trace_id=6237c7e16820c26c16e310c437bab94b) |
| **3** | Mortgage Underwriting Pipeline | `FAILED_EXECUTIONS` | `exec-mort-orig-64f4822c` | `e223a13f02a6ddf628a58adaf5bdca03` | 6.47s | FAILED | [Cost](http://localhost:5173/cost?trace_id=e223a13f02a6ddf628a58adaf5bdca03) · [Tokens](http://localhost:5173/tokens?trace_id=e223a13f02a6ddf628a58adaf5bdca03) |
| **4** | Commercial Lease Due Diligence | `EXPENSIVE_PROMPT` | `exec-lease-audit-238e52a6` | `40f066518e6eecba982d9230c0f250e3` | 10.03s | SUCCESS | [Cost](http://localhost:5173/cost?trace_id=40f066518e6eecba982d9230c0f250e3) · [Tokens](http://localhost:5173/tokens?trace_id=40f066518e6eecba982d9230c0f250e3) |
| **5** | Customer Support Ticket Triage | `OVER_PROVISIONED_MODEL` | `exec-triage-ovr-26ae12a1` | `42beb84460ddb29f655408f2452d3837` | 1.03s | SUCCESS | [Cost](http://localhost:5173/cost?trace_id=42beb84460ddb29f655408f2452d3837) · [Tokens](http://localhost:5173/tokens?trace_id=42beb84460ddb29f655408f2452d3837) |
| **6** | VIP Travel Concierge Dialogue | `CONTEXT_GROWTH_LEAK` | `exec-concierge-c8bea934` | `48ec543e2bf9bedfa0236d5629808c2a` | 9.75s | SUCCESS | [Cost](http://localhost:5173/cost?trace_id=48ec543e2bf9bedfa0236d5629808c2a) · [Tokens](http://localhost:5173/tokens?trace_id=48ec543e2bf9bedfa0236d5629808c2a) |
| **7** | Retail Supply Chain Inventory | `TOOL_CALL_AMPLIFICATION` | `exec-wh-audit-433ad43e` | `8e1356a6055a59c1b1c48a0a07ae6725` | 2.24s | SUCCESS | [Cost](http://localhost:5173/cost?trace_id=8e1356a6055a59c1b1c48a0a07ae6725) · [Tokens](http://localhost:5173/tokens?trace_id=8e1356a6055a59c1b1c48a0a07ae6725) |
| **8** | Portfolio Strategy Reporting | `CACHE_COLLAPSE` | `exec-portf-rep-5e34292e` | `0034fa394cd11aa1069ec01dcc208a64` | 7.20s | SUCCESS | [Cost](http://localhost:5173/cost?trace_id=0034fa394cd11aa1069ec01dcc208a64) · [Tokens](http://localhost:5173/tokens?trace_id=0034fa394cd11aa1069ec01dcc208a64) |
| **9** | Warranty Claims Baseline | `CLEAN_OPTIMAL_BASELINE` | `exec-opt-claim-d726eccd` | `358b18b442edd4262b526a7f04534d41` | 2.87s | SUCCESS | [Cost](http://localhost:5173/cost?trace_id=358b18b442edd4262b526a7f04534d41) · [Tokens](http://localhost:5173/tokens?trace_id=358b18b442edd4262b526a7f04534d41) |

---

## 3. In-Depth Scenario Profiles

### Scenario 1: Vendor Invoice Reconciliation
- **Business Domain**: Accounts Payable & Supply Chain Invoicing
- **Workflow ID**: `wf_vendor_invoice_reconciliation`
- **Execution ID**: `exec-inv-rec-10cc8a13`
- **OTel Trace ID**: `a33448b001833ea65d4c8847372a4410`
- **Agents Involved**: `InvoiceIngestionAgent` $\rightarrow$ `VendorValidationAgent` $\rightarrow$ `GeneralLedgerPostingAgent`
- **Organic Defect Mechanism**:
  The junior backend engineer implemented `InvoiceIngestionAgent.extract_invoice_line_items` by prompting the model for a JSON array of line items and immediately executing `json.loads(extraction_response)`. Because structured output schema constraints were omitted, the model naturally encapsulated its response within markdown fences (```` ```json [...] ``` ````). `json.loads` fails with `json.decoder.JSONDecodeError`. The workflow engine catches the failure and retries the activity:
  - Attempt 1: `wf.activity.attempt = 1` $\rightarrow$ `JSONDecodeError` recorded on child span
  - Attempt 2: `wf.activity.attempt = 2` $\rightarrow$ `JSONDecodeError` recorded on child span
  - Attempt 3: `wf.activity.attempt = 3` $\rightarrow$ Fallback regex extracts inner JSON array, allowing downstream posting to succeed.
- **Detector Flagged**: `RUNAWAY_RETRIES` (`detect_runaway_retries`) & `FAILED_EXECUTIONS` (`detect_failed_executions`)
- **Telemetry Evidence**: Span attribute `wf.activity.attempt > 1` and error status subtree. Wasted tokens equal the sum of tokens burned in attempts 1 and 2 (410 + 483 = 893 tokens).
- **Deep Dive UI**:
  - Cost: [http://localhost:5173/cost?trace_id=a33448b001833ea65d4c8847372a4410](http://localhost:5173/cost?trace_id=a33448b001833ea65d4c8847372a4410)
  - Tokens: [http://localhost:5173/tokens?trace_id=a33448b001833ea65d4c8847372a4410](http://localhost:5173/tokens?trace_id=a33448b001833ea65d4c8847372a4410)

---

### Scenario 2: Dual Compliance Screening
- **Business Domain**: High-Value International Wire Transfers ($750,000 to Singapore)
- **Workflow ID**: `wf_dual_compliance_screening`
- **Execution ID**: `exec-wire-comp-a7115bf9`
- **OTel Trace ID**: `6237c7e16820c26c16e310c437bab94b`
- **Agents Involved**: `OFACSanctionsAgent` $\rightarrow$ `RegulatoryComplianceAgent` $\rightarrow$ `AuditJournalAgent`
- **Organic Defect Mechanism**:
  Two independent development teams built separate microservices for international banking: Team A created `OFACSanctionsAgent` and Team B created `RegulatoryComplianceAgent`. Both microservices were wired into the transaction processing pipeline. Neither team implemented an in-memory memoization cache. Both agents execute identical queries with verbatim parameters against the same transaction data. A third agent, `AuditJournalAgent`, repeats the identical check during sign-off.
- **Detector Flagged**: `REDUNDANT_CALLS` (`detect_redundant_calls`)
- **Telemetry Evidence**: All three calls share the exact same `canonical_prompt_hash` within the same `execution_id`. Calls 2 and 3 are identified as 100% redundant waste (704 tokens wasted).
- **Deep Dive UI**:
  - Cost: [http://localhost:5173/cost?trace_id=6237c7e16820c26c16e310c437bab94b](http://localhost:5173/cost?trace_id=6237c7e16820c26c16e310c437bab94b)
  - Tokens: [http://localhost:5173/tokens?trace_id=6237c7e16820c26c16e310c437bab94b](http://localhost:5173/tokens?trace_id=6237c7e16820c26c16e310c437bab94b)

---

### Scenario 3: Mortgage Underwriting & Settlement Pipeline
- **Business Domain**: Residential Mortgage Origination ($420,000 Conventional Loan)
- **Workflow ID**: `wf_mortgage_settlement_pipeline`
- **Execution ID**: `exec-mort-orig-64f4822c`
- **OTel Trace ID**: `e223a13f02a6ddf628a58adaf5bdca03`
- **Agents Involved**: `CreditRiskAssessmentAgent` $\rightarrow$ `IncomeVerificationAgent` $\rightarrow$ `PropertyValuationAgent` $\rightarrow$ `WireDisbursementSettlementAgent`
- **Organic Defect Mechanism**:
  A schema refactoring renamed the borrower's bank routing field from `wire_routing_number` to `routing_number` in the intake payload. The first three underwriting agents process the borrower successfully, consuming upstream prompt and completion tokens. At step 4, `WireDisbursementSettlementAgent` attempts `payload["wire_routing_number"]`, throwing an unhandled `KeyError`. The entire workflow terminates with `status_code = Error` and `gen_ai.activity.status = FAILED`.
- **Detector Flagged**: `FAILED_EXECUTIONS` (`detect_failed_executions`)
- **Telemetry Evidence**: Root span marked as `FAILED`. 100% of tokens consumed by credit risk, income verification, and property valuation (1,188 tokens) yielded zero completed loan origination.
- **Deep Dive UI**:
  - Cost: [http://localhost:5173/cost?trace_id=e223a13f02a6ddf628a58adaf5bdca03](http://localhost:5173/cost?trace_id=e223a13f02a6ddf628a58adaf5bdca03)
  - Tokens: [http://localhost:5173/tokens?trace_id=e223a13f02a6ddf628a58adaf5bdca03](http://localhost:5173/tokens?trace_id=e223a13f02a6ddf628a58adaf5bdca03)

---

### Scenario 4: Commercial Lease Due Diligence Audit
- **Business Domain**: Commercial Real Estate Acquisition Due Diligence
- **Workflow ID**: `wf_commercial_lease_analysis`
- **Execution ID**: `exec-lease-audit-238e52a6`
- **OTel Trace ID**: `40f066518e6eecba982d9230c0f250e3`
- **Agents Involved**: `ClauseExtractionAgent` $\rightarrow$ `RiskSynthesizerAgent`
- **Organic Defect Mechanism**:
  Instead of utilizing retrieval-augmented generation (RAG) or semantic paragraph chunking, the prompt builder naively dumped 5 entire commercial master lease agreements (over 8,500 tokens of dense legal boilerplate containing full notary exhibits, boiler maintenance specifications, and indemnity clauses) into every clause evaluation turn across 5 separate covenants. Across queries, the median input tokens ($p50$) exceeds 5,739 tokens ($p90 = 5,747$).
- **Detector Flagged**: `EXPENSIVE_PROMPT` (`detect_expensive_prompts`)
- **Telemetry Evidence**: Cohort size $\ge 5$ spans, $p50 > 5,700$ tokens (context stuffing), user role dominates $>90\%$ of input tokens, zero cache reads on $>3,000$ token prompts.
- **Deep Dive UI**:
  - Cost: [http://localhost:5173/cost?trace_id=40f066518e6eecba982d9230c0f250e3](http://localhost:5173/cost?trace_id=40f066518e6eecba982d9230c0f250e3)
  - Tokens: [http://localhost:5173/tokens?trace_id=40f066518e6eecba982d9230c0f250e3](http://localhost:5173/tokens?trace_id=40f066518e6eecba982d9230c0f250e3)

---

### Scenario 5: Customer Support Ticket Triage
- **Business Domain**: Enterprise Helpdesk Ticket Routing
- **Workflow ID**: `wf_ticket_classification_routing`
- **Execution ID**: `exec-triage-ovr-26ae12a1`
- **OTel Trace ID**: `42beb84460ddb29f655408f2452d3837`
- **Agents Involved**: `InboundTicketClassifierAgent` $\rightarrow$ `QueueDispatcherAgent`
- **Organic Defect Mechanism**:
  The customer support infrastructure template defaulted to `mistral-large-latest`. `InboundTicketClassifierAgent` takes a single customer sentence (*"Why did my credit card get charged twice for renewal invoice INV-8819?"*) and prompts `mistral-large-latest` to classify it into `BILLING`, `TECHNICAL_BUG`, or `ACCOUNT_SECURITY`. The prompt includes a 2,300+ token corporate policy preamble, and the model outputs exactly 1 token: `BILLING`.
- **Detector Flagged**: `OVER_PROVISIONED_MODEL` (`detect_over_provisioned_models`)
- **Telemetry Evidence**: Flagged as Observation: `classify_ticket_category on mistral-large-latest (3,109 tokens)`. Matches `TOKEN_ASYMMETRY` (input > 2,000, output < 50, zero tools) and `SINGLE_TURN_NO_TOOLS`. Observability engine recommends downgrading to `mistral-small-latest`.
- **Deep Dive UI**:
  - Cost: [http://localhost:5173/cost?trace_id=42beb84460ddb29f655408f2452d3837](http://localhost:5173/cost?trace_id=42beb84460ddb29f655408f2452d3837)
  - Tokens: [http://localhost:5173/tokens?trace_id=42beb84460ddb29f655408f2452d3837](http://localhost:5173/tokens?trace_id=42beb84460ddb29f655408f2452d3837)

---

### Scenario 6: VIP Travel Concierge Multi-Turn Dialogue
- **Business Domain**: Luxury Hospitality & Itinerary Management
- **Workflow ID**: `wf_concierge_support_dialogue`
- **Execution ID**: `exec-concierge-c8bea934`
- **OTel Trace ID**: `48ec543e2bf9bedfa0236d5629808c2a`
- **Agents Involved**: `TravelConciergeAgent` $\rightarrow$ `DiningReservationAgent` $\rightarrow$ `BillingSpecialistAgent` $\rightarrow$ `ItineraryCoordinatorAgent` $\rightarrow$ `SummaryFinalizerAgent`
- **Organic Defect Mechanism**:
  The dialogue state manager naively appends every conversational exchange into an uncompacted history buffer (`conversation_messages.extend([user_entry, assistant_entry])`). As handoffs occur across specialized agents, the full unpruned message array is forwarded verbatim. Turn tokens grow monotonically:
  - Turn 1: 116 tokens
  - Turn 2: 380 tokens
  - Turn 3: 639 tokens
  - Turn 4: 889 tokens
  - Turn 5: 1,133 tokens
- **Detector Flagged**: `CONTEXT_GROWTH_LEAK` (`detect_context_growth_leak`) & `EXPENSIVE_PROMPT` (D4)
- **Telemetry Evidence**: Monotonic turn-over-turn token inflation without sliding-window pruning or compaction across 5 turns.
- **Deep Dive UI**:
  - Cost: [http://localhost:5173/cost?trace_id=48ec543e2bf9bedfa0236d5629808c2a](http://localhost:5173/cost?trace_id=48ec543e2bf9bedfa0236d5629808c2a)
  - Tokens: [http://localhost:5173/tokens?trace_id=48ec543e2bf9bedfa0236d5629808c2a](http://localhost:5173/tokens?trace_id=48ec543e2bf9bedfa0236d5629808c2a)

---

### Scenario 7: Retail Supply Chain Inventory Audit
- **Business Domain**: Omnichannel Retail Fulfillment & Stockout Prevention
- **Workflow ID**: `wf_inventory_catalog_sync`
- **Execution ID**: `exec-wh-audit-433ad43e`
- **OTel Trace ID**: `8e1356a6055a59c1b1c48a0a07ae6725`
- **Agents Involved**: `InventoryAuditorAgent` $\rightarrow$ `StockReconcilerAgent`
- **Organic Defect Mechanism**:
  `InventoryAuditorAgent` executes `query_warehouse_database` tool. The database query returns a raw, un-projected JSON dump of 60 warehouse bin records—including RFID chip serials, coordinate vectors, sensor temperatures, and batch barcodes (over 20 KB / 10,000+ characters). The agent prompt builder embeds this raw dump directly into the prompt: `f"Examine this inventory snapshot: {raw_dump}. Is SKU-ELEC-4412 in stock?"`.
- **Detector Flagged**: `TOOL_CALL_AMPLIFICATION` (`detect_tool_call_amplification`)
- **Telemetry Evidence**: Flagged as `Tool Call Amplification in query_warehouse_database (60.0% of prompt, 9,975 tokens)`. Tool return payload dominates $>40\%$ of downstream prompt tokens.
- **Deep Dive UI**:
  - Cost: [http://localhost:5173/cost?trace_id=8e1356a6055a59c1b1c48a0a07ae6725](http://localhost:5173/cost?trace_id=8e1356a6055a59c1b1c48a0a07ae6725)
  - Tokens: [http://localhost:5173/tokens?trace_id=8e1356a6055a59c1b1c48a0a07ae6725](http://localhost:5173/tokens?trace_id=8e1356a6055a59c1b1c48a0a07ae6725)

---

### Scenario 8: Quantitative Portfolio Strategy Reporting
- **Business Domain**: Institutional Asset Management & SEC Compliance Reporting
- **Workflow ID**: `wf_portfolio_strategy_report`
- **Execution ID**: `exec-portf-rep-5e34292e`
- **OTel Trace ID**: `0034fa394cd11aa1069ec01dcc208a64`
- **Agents Involved**: `PortfolioAnalyticsAgent` $\rightarrow$ `RegulatoryComplianceReviewAgent`
- **Organic Defect Mechanism**:
  The compliance review relies on a 2,800-token static SEC statutory disclosure prompt. Calls 1 and 2 place the static text at character 0, establishing a warm prefix cache (`gen_ai.usage.cache_read_input_tokens: 2800`). On Call 3, the logging middleware prepends dynamic tracking metadata (`f"TRACE_EVENT | Host: srv-09 | Nonce: {uuid4()} | Time: {time()} \n\n"`) at byte 0. This dynamic prefix at the start of the prompt invalidates Mistral's prefix cache key, dropping cache read to 0 tokens.
- **Detector Flagged**: `CACHE_COLLAPSE` (`detect_cache_collapse`)
- **Telemetry Evidence**: Mid-trace prompt cache invalidation where a warm prefix cache of 2,800 tokens collapses to 0 read tokens on Call 3.
- **Deep Dive UI**:
  - Cost: [http://localhost:5173/cost?trace_id=0034fa394cd11aa1069ec01dcc208a64](http://localhost:5173/cost?trace_id=0034fa394cd11aa1069ec01dcc208a64)
  - Tokens: [http://localhost:5173/tokens?trace_id=0034fa394cd11aa1069ec01dcc208a64](http://localhost:5173/tokens?trace_id=0034fa394cd11aa1069ec01dcc208a64)

---

### Scenario 9: High-Efficiency Warranty Claims Baseline
- **Business Domain**: Consumer Electronics Warranty Claims
- **Workflow ID**: `wf_optimized_claims_processing`
- **Execution ID**: `exec-opt-claim-d726eccd`
- **OTel Trace ID**: `358b18b442edd4262b526a7f04534d41`
- **Agents Involved**: `ClaimsIntakeAgent` $\rightarrow$ `WarrantyRuleEngineAgent` $\rightarrow$ `ApprovalNotificationAgent`
- **Architecture Highlights**:
  - Cost-efficient model: `mistral-small-latest`
  - Static system instructions placed first to maximize prefix cache hits
  - Database queries are projected to only 3 essential fields before LLM injection
  - Safe dictionary parsing with fallbacks preventing unhandled exceptions
  - Zero wasted tokens, serving as the benchmark reference baseline.
- **Detector Flagged**: None (`CLEAN_OPTIMAL_BASELINE`)
- **Telemetry Evidence**: 100% efficiency score, 0 wasted tokens, all spans completed successfully in 2.87s.
- **Deep Dive UI**:
  - Cost: [http://localhost:5173/cost?trace_id=358b18b442edd4262b526a7f04534d41](http://localhost:5173/cost?trace_id=358b18b442edd4262b526a7f04534d41)
  - Tokens: [http://localhost:5173/tokens?trace_id=358b18b442edd4262b526a7f04534d41](http://localhost:5173/tokens?trace_id=358b18b442edd4262b526a7f04534d41)

---

## 4. Running the Test Suite

Execute the suite directly using the CLI runner:

```bash
# Run all 9 scenarios using uv (recommended):
uv run cost-tokenTests/run_cost_suite.py

# Or using python directly:
python cost-tokenTests/run_cost_suite.py

# Run a specific scenario:
uv run cost-tokenTests/run_cost_suite.py --scenario retries
uv run cost-tokenTests/run_cost_suite.py --scenario redundant
uv run cost-tokenTests/run_cost_suite.py --scenario failed
uv run cost-tokenTests/run_cost_suite.py --scenario expensive
uv run cost-tokenTests/run_cost_suite.py --scenario oversize
uv run cost-tokenTests/run_cost_suite.py --scenario leak
uv run cost-tokenTests/run_cost_suite.py --scenario amplification
uv run cost-tokenTests/run_cost_suite.py --scenario collapse
uv run cost-tokenTests/run_cost_suite.py --scenario baseline

# Resume from scenario #4 onwards:
uv run cost-tokenTests/run_cost_suite.py --from-scenario 4
```

After execution, all trace IDs are persisted to `cost-tokenTests/LATEST_COST_TRACE_IDS.json` and accessible in the Observability Web UI:
- **Cost Engine Deep Dive**: `http://localhost:5173/cost?trace_id=<TRACE_ID>`
- **Token Efficiency Deep Dive**: `http://localhost:5173/tokens?trace_id=<TRACE_ID>`
