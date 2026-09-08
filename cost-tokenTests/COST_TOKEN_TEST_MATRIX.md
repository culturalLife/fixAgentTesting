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
| **4** | **Commercial Lease Due Diligence** | `expensive` | `wf_commercial_lease_analysis` | `EXPENSIVE_PROMPT` (D4) | HIGH | Prompt builder injects 9,000+ tokens of unpruned commercial lease boilerplate across queries without prefix caching ($p50 > 8,500$). |
| **5** | **Customer Support Ticket Triage** | `oversize` | `wf_ticket_classification_routing` | `OVER_PROVISIONED_MODEL` (D1) | LOW (Observation) | Global configuration defaulted to `mistral-large-latest` for single-turn, 1-token enum classification (`BILLING`). |
| **6** | **VIP Travel Concierge Dialogue** | `leak` | `wf_concierge_support_dialogue` | `CONTEXT_GROWTH_LEAK` (D6) | HIGH | Conversation memory naively appends all turns across agent handoffs without compaction (400 $\rightarrow$ 1,200 $\rightarrow$ 2,500 $\rightarrow$ 4,500 $\rightarrow$ 7,200 tokens). |
| **7** | **Retail Supply Chain Inventory Audit** | `amplification` | `wf_inventory_catalog_sync` | `TOOL_CALL_AMPLIFICATION` (D7) | HIGH / MEDIUM | Tool returns 18 KB uncompressed warehouse JSON dump; raw output is dumped into prompt, dominating $>70\%$ of prompt tokens. |
| **8** | **Quantitative Portfolio Reporting** | `collapse` | `wf_portfolio_strategy_report` | `CACHE_COLLAPSE` (D8) | HIGH | Calls 1 & 2 warm up prefix cache on 3,000-token SEC rules; Call 3 prepends dynamic audit timestamps at byte 0, dropping cache hit rate to 0%. |
| **9** | **High-Efficiency Claims Baseline** | `baseline` | `wf_optimized_claims_processing` | `CLEAN_OPTIMAL_BASELINE` | N/A (Optimal) | Right-sized `mistral-small-latest`, static system prefix placed first, projected tool results, sliding-window memory, 0 wasted tokens. |

---

## 2. In-Depth Scenario Profiles

### Scenario 1: Vendor Invoice Reconciliation
- **Business Domain**: Accounts Payable & Supply Chain Invoicing
- **Workflow ID**: `wf_vendor_invoice_reconciliation`
- **Agents Involved**: `InvoiceIngestionAgent` $\rightarrow$ `VendorValidationAgent` $\rightarrow$ `GeneralLedgerPostingAgent`
- **Organic Defect Mechanism**:
  The junior backend engineer implemented `InvoiceIngestionAgent.extract_invoice_line_items` by prompting the model for a JSON array of line items and immediately executing `json.loads(extraction_response)`. Because structured output schema constraints were omitted, the model naturally encapsulated its response within markdown fences (```` ```json [...] ``` ````). `json.loads` fails with `json.decoder.JSONDecodeError`. The workflow engine catches the failure and retries the activity:
  - Attempt 1: `wf.activity.attempt = 1` $\rightarrow$ `JSONDecodeError` recorded on child span
  - Attempt 2: `wf.activity.attempt = 2` $\rightarrow$ `JSONDecodeError` recorded on child span
  - Attempt 3: `wf.activity.attempt = 3` $\rightarrow$ Fallback regex extracts inner JSON array, allowing downstream posting to succeed.
- **Detector Flagged**: `RUNAWAY_RETRIES` (`detect_runaway_retries`)
- **Telemetry Evidence**: Span attribute `wf.activity.attempt > 1` and error status subtree. Wasted tokens equal the sum of tokens burned in attempts 1 and 2.

---

### Scenario 2: Dual Compliance Screening
- **Business Domain**: High-Value International Wire Transfers ($750,000 to Singapore)
- **Workflow ID**: `wf_dual_compliance_screening`
- **Agents Involved**: `OFACSanctionsAgent` $\rightarrow$ `RegulatoryComplianceAgent` $\rightarrow$ `AuditJournalAgent`
- **Organic Defect Mechanism**:
  Two independent development teams built separate microservices for international banking: Team A created `OFACSanctionsAgent` and Team B created `RegulatoryComplianceAgent`. Both microservices were wired into the transaction processing pipeline. Neither team implemented an in-memory memoization cache. Both agents execute identical queries with verbatim parameters against the same transaction data. A third agent, `AuditJournalAgent`, repeats the identical check during sign-off.
- **Detector Flagged**: `REDUNDANT_CALLS` (`detect_redundant_calls`)
- **Telemetry Evidence**: All three calls share the exact same `canonical_prompt_hash` (normalizing dynamic timestamps and UUIDs) within the same `execution_id`. Calls 2 and 3 are identified as 100% redundant waste.

---

### Scenario 3: Mortgage Underwriting & Settlement Pipeline
- **Business Domain**: Residential Mortgage Origination ($420,000 Conventional Loan)
- **Workflow ID**: `wf_mortgage_settlement_pipeline`
- **Agents Involved**: `CreditRiskAssessmentAgent` $\rightarrow$ `IncomeVerificationAgent` $\rightarrow$ `PropertyValuationAgent` $\rightarrow$ `WireDisbursementSettlementAgent`
- **Organic Defect Mechanism**:
  A schema refactoring renamed the borrower's bank routing field from `wire_routing_number` to `routing_number` in the intake payload. The first three underwriting agents process the borrower successfully, consuming upstream prompt and completion tokens. At step 4, `WireDisbursementSettlementAgent` attempts `payload["wire_routing_number"]`, throwing an unhandled `KeyError`. The entire workflow terminates with `status_code = Error` and `gen_ai.activity.status = FAILED`.
- **Detector Flagged**: `FAILED_EXECUTIONS` (`detect_failed_executions`)
- **Telemetry Evidence**: Root span marked as `FAILED`. 100% of tokens consumed by credit risk, income verification, and property valuation yielded zero completed loan origination.

---

### Scenario 4: Commercial Lease Due Diligence Audit
- **Business Domain**: Commercial Real Estate Acquisition Due Diligence
- **Workflow ID**: `wf_commercial_lease_analysis`
- **Agents Involved**: `ClauseExtractionAgent` $\rightarrow$ `RiskSynthesizerAgent`
- **Organic Defect Mechanism**:
  Instead of utilizing retrieval-augmented generation (RAG) or semantic paragraph chunking, the prompt builder naively dumped 5 entire commercial master lease agreements (over 9,000 tokens of dense legal boilerplate containing full notary exhibits, boiler maintenance specifications, and indemnity clauses) into every clause evaluation turn. Across queries, the median input tokens ($p50$) exceeds 8,500 tokens, and large spans show 0 prefix cache read tokens.
- **Detector Flagged**: `EXPENSIVE_PROMPT` (`detect_expensive_prompts`)
- **Telemetry Evidence**: $p50 > 8,000$ tokens (context stuffing), user role dominates $>90\%$ of input tokens, zero cache reads on $>3,000$ token prompts.

---

### Scenario 5: Customer Support Ticket Triage
- **Business Domain**: Enterprise Helpdesk Ticket Routing
- **Workflow ID**: `wf_ticket_classification_routing`
- **Agents Involved**: `InboundTicketClassifierAgent` $\rightarrow$ `QueueDispatcherAgent`
- **Organic Defect Mechanism**:
  The customer support infrastructure template defaulted to `mistral-large-latest`. `InboundTicketClassifierAgent` takes a single customer sentence (*"Why did my credit card get charged twice for renewal invoice INV-8819?"*) and prompts `mistral-large-latest` to classify it into `BILLING`, `TECHNICAL_BUG`, or `ACCOUNT_SECURITY`. The prompt includes a 2,200-token corporate policy preamble, and the model outputs exactly 1 token: `BILLING`.
- **Detector Flagged**: `OVER_PROVISIONED_MODEL` (`detect_over_provisioned_models`)
- **Telemetry Evidence**: Matches `LLM_TASK_CLASSIFIER` (simple classification), `TOKEN_ASYMMETRY` (input > 2,000, output < 50, zero tools), and `SINGLE_TURN_NO_TOOLS`. Observability engine recommends downgrading to `mistral-small-latest`.

---

### Scenario 6: VIP Travel Concierge Multi-Turn Dialogue
- **Business Domain**: Luxury Hospitality & Itinerary Management
- **Workflow ID**: `wf_concierge_support_dialogue`
- **Agents Involved**: `TravelConciergeAgent` $\rightarrow$ `DiningReservationAgent` $\rightarrow$ `BillingSpecialistAgent` $\rightarrow$ `ItineraryCoordinatorAgent`
- **Organic Defect Mechanism**:
  The dialogue state manager naively appends every conversational exchange into an uncompacted history buffer (`conversation_messages.extend([user_entry, assistant_entry])`). As handoffs occur across specialized agents, the full unpruned message array is forwarded verbatim. Turn tokens grow monotonically:
  - Turn 1: 420 tokens
  - Turn 2: 1,180 tokens
  - Turn 3: 2,410 tokens
  - Turn 4: 4,350 tokens
- **Detector Flagged**: `CONTEXT_GROWTH_LEAK` (`detect_context_growth_leak`) & `EXPENSIVE_PROMPT` (D4)
- **Telemetry Evidence**: Monotonic turn-over-turn token inflation without sliding-window pruning or compaction.

---

### Scenario 7: Retail Supply Chain Inventory Audit
- **Business Domain**: Omnichannel Retail Fulfillment & Stockout Prevention
- **Workflow ID**: `wf_inventory_catalog_sync`
- **Agents Involved**: `InventoryAuditorAgent` $\rightarrow$ `StockReconcilerAgent`
- **Organic Defect Mechanism**:
  `InventoryAuditorAgent` executes `query_warehouse_database` tool. The database query returns a raw, un-projected JSON dump of 60 warehouse bin records—including RFID chip serials, coordinate vectors, sensor temperatures, and batch barcodes (over 18 KB / 3,800 tokens). The agent prompt builder embeds this raw dump directly into the prompt: `f"Examine this inventory snapshot: {raw_dump}. Is SKU-ELEC-4412 in stock?"`. The tool output comprises $>70\%$ of the entire prompt token count.
- **Detector Flagged**: `TOOL_CALL_AMPLIFICATION` (`detect_tool_call_amplification`)
- **Telemetry Evidence**: Tool execution return payload dominates the subsequent LLM prompt input tokens ($\ge 40\%$).

---

### Scenario 8: Quantitative Portfolio Strategy Reporting
- **Business Domain**: Institutional Asset Management & SEC Compliance Reporting
- **Workflow ID**: `wf_portfolio_strategy_report`
- **Agents Involved**: `PortfolioAnalyticsAgent` $\rightarrow$ `RegulatoryComplianceReviewAgent`
- **Organic Defect Mechanism**:
  The compliance review relies on a 3,000-token static SEC statutory disclosure prompt. Calls 1 and 2 place the static text at character 0, establishing a warm prefix cache. On Call 3, the logging middleware prepends dynamic tracking metadata (`f"TRACE_EVENT | Host: srv-09 | Nonce: {uuid4()} | Time: {time()} \n\n"`) at byte 0. This dynamic prefix at the start of the prompt invalidates Mistral's prefix cache key, causing cache read tokens to drop from warm to 0.
- **Detector Flagged**: `CACHE_COLLAPSE` (`detect_cache_collapse`)
- **Telemetry Evidence**: Mid-trace prompt cache invalidation where a previously warm prefix cache drops to 0 read tokens on a subsequent call.

---

### Scenario 9: High-Efficiency Warranty Claims Baseline
- **Business Domain**: Consumer Electronics Warranty Claims
- **Workflow ID**: `wf_optimized_claims_processing`
- **Agents Involved**: `ClaimsIntakeAgent` $\rightarrow$ `WarrantyRuleEngineAgent` $\rightarrow$ `ApprovalNotificationAgent`
- **Architecture Highlights**:
  - Cost-efficient model: `mistral-small-latest`
  - Static system instructions placed first to maximize prefix cache hits
  - Database queries are projected to only 3 essential fields before LLM injection
  - Safe dictionary parsing with fallbacks preventing unhandled exceptions
  - Zero wasted tokens, serving as the benchmark reference baseline.
- **Detector Flagged**: None (`CLEAN_OPTIMAL_BASELINE`)
- **Telemetry Evidence**: 100% efficiency score, 0 wasted tokens, all spans completed successfully.

---

## 3. Running the Test Suite

Execute the suite directly using the CLI runner:

```bash
# Run all 9 scenarios and upload traces to Mistral Observability
python cost-tokenTests/run_cost_suite.py

# Run a specific scenario
python cost-tokenTests/run_cost_suite.py --scenario retries
python cost-tokenTests/run_cost_suite.py --scenario redundant
python cost-tokenTests/run_cost_suite.py --scenario failed
python cost-tokenTests/run_cost_suite.py --scenario expensive
python cost-tokenTests/run_cost_suite.py --scenario oversize
python cost-tokenTests/run_cost_suite.py --scenario leak
python cost-tokenTests/run_cost_suite.py --scenario amplification
python cost-tokenTests/run_cost_suite.py --scenario collapse
python cost-tokenTests/run_cost_suite.py --scenario baseline

# Resume from scenario #4 onwards
python cost-tokenTests/run_cost_suite.py --from-scenario 4
```

After execution, all trace IDs are persisted to `cost-tokenTests/LATEST_COST_TRACE_IDS.json` and accessible in the Observability Web UI:
- **Cost Engine Deep Dive**: `http://localhost:5173/cost?trace_id=<TRACE_ID>`
- **Token Efficiency Deep Dive**: `http://localhost:5173/tokens?trace_id=<TRACE_ID>`
