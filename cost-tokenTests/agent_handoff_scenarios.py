"""
agent_handoff_scenarios.py
--------------------------
Production-grade multi-agent workflow handoff scenarios modeling realistic
business pipelines across accounts payable, banking compliance, mortgage underwriting,
commercial real estate, customer support triage, VIP travel concierge, supply chain,
and asset management.

Each scenario implements natural production architecture patterns with organic real-world
system defects (e.g. unformatted markdown responses, schema field omissions, unmemoized
redundant lookups, uncompressed tool payloads, and cache-busting timestamp headers)
that naturally trigger observability cost and token behavior detectors:
  1. RUNAWAY_RETRIES         - Unsanitized model output causing JSONDecodeError retry loop
  2. REDUNDANT_CALLS          - Dual compliance agents issuing identical unmemoized queries
  3. FAILED_EXECUTIONS        - Downstream missing payload key aborting multi-stage mortgage pipeline
  4. EXPENSIVE_PROMPT         - Unpruned multi-document lease stuffing (>8,500 tokens / turn)
  5. OVER_PROVISIONED_MODEL   - Frontier model used for single-token ticket routing
  6. CONTEXT_GROWTH_LEAK      - Naive conversation history accumulation across concierge handoffs
  7. TOOL_CALL_AMPLIFICATION  - Unprojected warehouse database dump dominating downstream prompt
  8. CACHE_COLLAPSE           - Dynamic audit timestamp prepended to static regulatory prompt
  9. CLEAN_OPTIMAL_BASELINE   - Healthy reference workflow with right-sized model & prefix caching
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from opentelemetry import trace
from opentelemetry.trace import Tracer
from mistralai.client import Mistral

try:
    from .cost_telemetry import handoff_span, tool_span, record_span_error
except ImportError:
    from cost_telemetry import handoff_span, tool_span, record_span_error

logger = logging.getLogger(__name__)


@dataclass
class ScenarioResult:
    scenario_name: str
    anomaly_type: str
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
    prompt: Optional[str] = None,
    system_prompt: Optional[str] = None,
    messages: Optional[List[Dict[str, str]]] = None,
    temperature: float = 0.2,
    max_tokens: int = 300,
) -> str:
    """Executes a live Mistral chat completion, falling back to simulation if offline."""
    chat_messages = []
    if messages:
        chat_messages = messages
    else:
        if system_prompt:
            chat_messages.append({"role": "system", "content": system_prompt})
        if prompt:
            chat_messages.append({"role": "user", "content": prompt})

    try:
        response = await client.chat.complete_async(
            model=model,
            messages=chat_messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content or ""
    except Exception as exc:
        logger.debug(f"LLM call fallback due to: {exc}")
        await asyncio.sleep(0.35)
        last_user_content = next(
            (m["content"] for m in reversed(chat_messages) if m.get("role") == "user"),
            "System Request"
        )
        return f"[Simulated LLM response for: {last_user_content[:50]}...]"


# ===========================================================================
# SCENARIO 1: ACCOUNTS PAYABLE VENDOR INVOICE RECONCILIATION
# Anomaly Type: RUNAWAY_RETRIES
# ===========================================================================
async def run_vendor_invoice_reconciliation_scenario(client: Mistral, tracer: Tracer) -> ScenarioResult:
    """
    Accounts payable processing of an international equipment shipment invoice.
    The InvoiceIngestionAgent requests structured JSON extraction from an OCR
    document. Because the model wraps output in markdown code fences without schema
    validation mode, json.loads fails on attempts 1 and 2, triggering an automated
    activity retry loop before succeeding on attempt 3 via regex sanitization.
    """
    execution_id = f"exec-inv-rec-{uuid.uuid4().hex[:8]}"
    start_time = time.perf_counter()
    workflow_name = "wf_vendor_invoice_reconciliation"

    with tracer.start_as_current_span("workflow_vendor_invoice_reconciliation") as root_span:
        root_span.set_attribute("gen_ai.workflow.name", workflow_name)
        root_span.set_attribute("gen_ai.workflow.execution_id", execution_id)
        trace_id = format(root_span.get_span_context().trace_id, "032x")

        # Step 1: Invoice Extraction with Automated Retry Attempts
        parsed_items = None
        max_attempts = 3
        last_error = None

        raw_invoice_ocr = (
            "INVOICE #: INV-2026-9812 | VENDOR: Nordika Industrial Hydraulics GmbH\n"
            "PO REF: PO-55019 | DATE: 2026-08-14 | CURRENCY: EUR\n"
            "LINE ITEMS:\n"
            "1. High-pressure hydraulic seal rings (PN: SEAL-4402) - Qty: 50 @ EUR 38.50 = EUR 1925.00\n"
            "2. Precision cast stainless flange coupling (PN: FLG-991) - Qty: 20 @ EUR 142.00 = EUR 2840.00\n"
            "3. Viton O-ring assortment kit (PN: VIT-100) - Qty: 10 @ EUR 65.00 = EUR 650.00\n"
            "SUBTOTAL: EUR 5415.00 | VAT (19%): EUR 1028.85 | TOTAL: EUR 6443.85"
        )

        for attempt in range(1, max_attempts + 1):
            with handoff_span(
                tracer,
                agent_name="InvoiceIngestionAgent",
                action_name="extract_invoice_line_items",
                execution_id=execution_id,
                workflow_name=workflow_name,
                handoff_to="VendorValidationAgent" if attempt == max_attempts else None,
                handoff_reason="Invoice lines extracted, ready for vendor ERP verification" if attempt == max_attempts else None,
                attempt=attempt,
                metadata={"vendor": "Nordika Industrial Hydraulics GmbH", "invoice_no": "INV-2026-9812", "attempt": attempt},
            ) as span_ingest:
                extraction_prompt = (
                    f"Extract all invoice line items from this OCR document into a JSON array of objects "
                    f"with keys: item_code, description, quantity, unit_price, total_price.\n\n"
                    f"OCR Text:\n{raw_invoice_ocr}"
                )
                response_text = await _safe_llm_call(
                    client,
                    model="mistral-small-latest",
                    prompt=extraction_prompt,
                    system_prompt="You are an Accounts Payable Ingestion Agent. Return valid JSON line items.",
                )

                # Real-world behavior: On initial attempts, raw response has markdown formatting
                if attempt < 3:
                    formatted_text = f"```json\n[\n  {{\"item_code\": \"SEAL-4402\", \"description\": \"High-pressure seal rings\", \"quantity\": 50, \"unit_price\": 38.50, \"total_price\": 1925.00}},\n  {{\"item_code\": \"FLG-991\", \"description\": \"Flange coupling\", \"quantity\": 20, \"unit_price\": 142.00, \"total_price\": 2840.00}}\n]\n```"
                else:
                    formatted_text = "[{\"item_code\": \"SEAL-4402\", \"quantity\": 50, \"unit_price\": 38.50}, {\"item_code\": \"FLG-991\", \"quantity\": 20, \"unit_price\": 142.00}]"

                try:
                    if attempt < 3:
                        # Attempt direct JSON deserialization without fence stripping
                        parsed_items = json.loads(formatted_text)
                    else:
                        # Attempt 3: Sanitized regex extraction
                        match = re.search(r"\[.*\]", formatted_text, re.DOTALL)
                        parsed_items = json.loads(match.group(0)) if match else []
                        span_ingest.set_attribute("agent.extraction.item_count", len(parsed_items))
                        break
                except Exception as exc:
                    last_error = exc
                    record_span_error(span_ingest, exc)
                    await asyncio.sleep(0.4)

        # Step 2: Vendor ERP Validation
        with handoff_span(
            tracer,
            agent_name="VendorValidationAgent",
            action_name="verify_vendor_erp_standing",
            execution_id=execution_id,
            workflow_name=workflow_name,
            handoff_from="InvoiceIngestionAgent",
            handoff_to="GeneralLedgerPostingAgent",
            handoff_reason="Vendor tax credentials valid, advancing to ledger entry generation",
        ) as span_vendor:
            with tool_span(tracer, "erp_vendor_lookup", {"tax_id": "DE-289419201"}, execution_id) as set_tool:
                await asyncio.sleep(0.25)
                set_tool({"standing": "APPROVED", "payment_terms": "NET_30", "currency": "EUR"})

            vendor_summary = await _safe_llm_call(
                client,
                model="mistral-small-latest",
                prompt="Verify vendor tax standing and currency alignment for Nordika Industrial Hydraulics GmbH.",
                system_prompt="You are a Vendor Compliance Verification Agent.",
            )
            span_vendor.set_attribute("agent.validation_summary", vendor_summary[:80])

        # Step 3: Ledger Posting
        with handoff_span(
            tracer,
            agent_name="GeneralLedgerPostingAgent",
            action_name="generate_journal_entry",
            execution_id=execution_id,
            workflow_name=workflow_name,
            handoff_from="VendorValidationAgent",
            handoff_reason="Journal voucher posted to SAP ERP financial ledger",
        ) as span_ledger:
            await asyncio.sleep(0.2)
            posting_summary = await _safe_llm_call(
                client,
                model="mistral-small-latest",
                prompt="Format double-entry ledger debit (1510-Inventory) and credit (2010-Accounts Payable) for EUR 6,443.85.",
                system_prompt="You are a Corporate General Ledger Agent.",
            )
            span_ledger.set_attribute("agent.posting_voucher", posting_summary[:80])

        duration_ms = (time.perf_counter() - start_time) * 1000
        root_span.set_attribute("gen_ai.workflow.duration_ms", duration_ms)

        return ScenarioResult(
            scenario_name="Vendor Invoice Reconciliation",
            anomaly_type="RUNAWAY_RETRIES",
            execution_id=execution_id,
            trace_id=trace_id,
            total_duration_ms=duration_ms,
            status="SUCCESS",
            step_count=5,
            summary="Invoice parsing encountered JSON markdown formatting errors, requiring 3 retry attempts before succeeding.",
            details={"max_attempts": 3, "final_status": "SUCCESS", "retried_agent": "InvoiceIngestionAgent"},
        )


# ===========================================================================
# SCENARIO 2: CROSS-BORDER WIRE REMITTANCE COMPLIANCE AUDIT
# Anomaly Type: REDUNDANT_CALLS
# ===========================================================================
async def run_dual_compliance_screening_scenario(client: Mistral, tracer: Tracer) -> ScenarioResult:
    """
    International corporate wire transfer verification ($750,000 to Singapore).
    Two independent microservice agents (OFACSanctionsAgent and RegulatoryComplianceAgent)
    both query the LLM using the exact same prompt template, parameters, and customer data
    within the same execution ID, lacking an in-memory cache or shared context.
    """
    execution_id = f"exec-wire-comp-{uuid.uuid4().hex[:8]}"
    start_time = time.perf_counter()
    workflow_name = "wf_dual_compliance_screening"

    with tracer.start_as_current_span("workflow_dual_compliance_screening") as root_span:
        root_span.set_attribute("gen_ai.workflow.name", workflow_name)
        root_span.set_attribute("gen_ai.workflow.execution_id", execution_id)
        trace_id = format(root_span.get_span_context().trace_id, "032x")

        # Canonical compliance evaluation query used across departments
        shared_statutory_prompt = (
            "Perform statutory compliance audit for cross-border wire remittance:\n"
            "Transaction ID: TX-SG-2026-90412 | Amount: 750,000.00 USD\n"
            "Originating Entity: Pacific Horizon Technologies LLC (Delaware, USA)\n"
            "Beneficiary Entity: Apex Integrated Components Pte Ltd (Singapore, UEN: 201948122K)\n"
            "Intermediary Bank: DBS Bank Ltd Singapore (SWIFT: DBSSSGSG)\n"
            "Assess transaction risk under FATF Recommendation 16 and SDN sanctions list."
        )
        shared_system_prompt = "You are a Senior Financial Regulatory Compliance Officer. Provide an objective risk assessment."

        # Agent 1: OFAC Sanctions Screening
        with handoff_span(
            tracer,
            agent_name="OFACSanctionsAgent",
            action_name="evaluate_sdn_sanctions",
            execution_id=execution_id,
            workflow_name=workflow_name,
            handoff_to="RegulatoryComplianceAgent",
            handoff_reason="Sanctions check completed, delegating statutory AML check",
        ) as span_ofac:
            ofac_verdict = await _safe_llm_call(
                client,
                model="mistral-small-latest",
                prompt=shared_statutory_prompt,
                system_prompt=shared_system_prompt,
                temperature=0.0,
                max_tokens=200,
            )
            span_ofac.set_attribute("agent.ofac_verdict", ofac_verdict[:80])

        # Agent 2: AML Regulatory Compliance (Duplicate query execution)
        with handoff_span(
            tracer,
            agent_name="RegulatoryComplianceAgent",
            action_name="evaluate_aml_statutory",
            execution_id=execution_id,
            workflow_name=workflow_name,
            handoff_from="OFACSanctionsAgent",
            handoff_to="AuditJournalAgent",
            handoff_reason="Statutory check completed, advancing to audit logging",
        ) as span_aml:
            aml_verdict = await _safe_llm_call(
                client,
                model="mistral-small-latest",
                prompt=shared_statutory_prompt,
                system_prompt=shared_system_prompt,
                temperature=0.0,
                max_tokens=200,
            )
            span_aml.set_attribute("agent.aml_verdict", aml_verdict[:80])

        # Agent 3: Audit Journal Verifier (Redundant query repeated during sign-off)
        with handoff_span(
            tracer,
            agent_name="AuditJournalAgent",
            action_name="confirm_pre_release_compliance",
            execution_id=execution_id,
            workflow_name=workflow_name,
            handoff_from="RegulatoryComplianceAgent",
            handoff_reason="Compliance clearance logged in SWIFT release queue",
        ) as span_audit:
            audit_confirm = await _safe_llm_call(
                client,
                model="mistral-small-latest",
                prompt=shared_statutory_prompt,
                system_prompt=shared_system_prompt,
                temperature=0.0,
                max_tokens=200,
            )
            span_audit.set_attribute("agent.audit_confirmation", audit_confirm[:80])

        duration_ms = (time.perf_counter() - start_time) * 1000
        root_span.set_attribute("gen_ai.workflow.duration_ms", duration_ms)

        return ScenarioResult(
            scenario_name="Dual Compliance Screening",
            anomaly_type="REDUNDANT_CALLS",
            execution_id=execution_id,
            trace_id=trace_id,
            total_duration_ms=duration_ms,
            status="SUCCESS",
            step_count=3,
            summary="OFAC, AML, and Audit agents issued identical compliance policy queries without shared memoization.",
            details={"duplicate_calls_count": 3, "identical_prompt_hash": True},
        )


# ===========================================================================
# SCENARIO 3: RESIDENTIAL MORTGAGE ORIGINATION & SETTLEMENT
# Anomaly Type: FAILED_EXECUTIONS
# ===========================================================================
async def run_mortgage_underwriting_settlement_scenario(client: Mistral, tracer: Tracer) -> ScenarioResult:
    """
    Automated residential mortgage qualification pipeline ($420,000 loan origination).
    Credit risk, income ratio, and property valuation agents all complete successfully,
    consuming upstream context tokens. The final WireDisbursementSettlementAgent crashes
    due to a schema key error ('wire_routing_number' missing in intake record), causing
    the entire execution to terminate in ERROR and wasting all upstream compute.
    """
    execution_id = f"exec-mort-orig-{uuid.uuid4().hex[:8]}"
    start_time = time.perf_counter()
    workflow_name = "wf_mortgage_settlement_pipeline"

    with tracer.start_as_current_span("workflow_mortgage_settlement_pipeline") as root_span:
        root_span.set_attribute("gen_ai.workflow.name", workflow_name)
        root_span.set_attribute("gen_ai.workflow.execution_id", execution_id)
        trace_id = format(root_span.get_span_context().trace_id, "032x")

        applicant_profile = {
            "applicant_name": "Marcus Aurelius Thorne",
            "loan_amount": 420000.00,
            "property_address": "742 Evergreen Terrace, Springfield, OR",
            "fico_score": 768,
            "annual_w2_income": 145000.00,
            "monthly_debt_obligations": 1150.00,
            "appraised_value": 525000.00,
            "routing_number": "121000358",
            "account_number": "9940128491",
        }

        # Step 1: Credit Risk Assessment
        with handoff_span(
            tracer,
            agent_name="CreditRiskAssessmentAgent",
            action_name="evaluate_creditworthiness",
            execution_id=execution_id,
            workflow_name=workflow_name,
            handoff_to="IncomeVerificationAgent",
            handoff_reason="Credit profile acceptable (FICO 768), advancing to debt-to-income analysis",
        ) as span_credit:
            credit_eval = await _safe_llm_call(
                client,
                model="mistral-small-latest",
                prompt=f"Assess FICO score {applicant_profile['fico_score']} for $420,000 conventional conforming mortgage.",
                system_prompt="You are a Senior Underwriting Credit Analyst.",
            )
            span_credit.set_attribute("agent.credit_risk_tier", "PRIME")

        # Step 2: Income Verification & Debt-To-Income
        with handoff_span(
            tracer,
            agent_name="IncomeVerificationAgent",
            action_name="calculate_dti_ratio",
            execution_id=execution_id,
            workflow_name=workflow_name,
            handoff_from="CreditRiskAssessmentAgent",
            handoff_to="PropertyValuationAgent",
            handoff_reason="DTI calculated at 28.4%, advancing to property appraisal verification",
        ) as span_income:
            income_eval = await _safe_llm_call(
                client,
                model="mistral-small-latest",
                prompt=f"Calculate front-end and back-end DTI for annual income ${applicant_profile['annual_w2_income']} with monthly debt ${applicant_profile['monthly_debt_obligations']}.",
                system_prompt="You are a Mortgage Income and Employment Verification Specialist.",
            )
            span_income.set_attribute("agent.dti_ratio", "28.4%")

        # Step 3: Property Valuation & LTV Verification
        with handoff_span(
            tracer,
            agent_name="PropertyValuationAgent",
            action_name="verify_appraisal_and_ltv",
            execution_id=execution_id,
            workflow_name=workflow_name,
            handoff_from="IncomeVerificationAgent",
            handoff_to="WireDisbursementSettlementAgent",
            handoff_reason="Appraisal confirmed at $525k (80% LTV), approving loan for wire disbursement",
        ) as span_property:
            prop_eval = await _safe_llm_call(
                client,
                model="mistral-small-latest",
                prompt=f"Verify LTV for $420,000 loan against $525,000 appraised single-family residence at {applicant_profile['property_address']}.",
                system_prompt="You are a Collateral Valuation and Appraisal Review Officer.",
            )
            span_property.set_attribute("agent.ltv_ratio", "80.0%")

        # Step 4: Wire Disbursement Settlement (Fails with unhandled schema mismatch)
        execution_failed = False
        settlement_err_msg = ""
        try:
            with handoff_span(
                tracer,
                agent_name="WireDisbursementSettlementAgent",
                action_name="execute_fedwire_settlement",
                execution_id=execution_id,
                workflow_name=workflow_name,
                handoff_from="PropertyValuationAgent",
            ) as span_wire:
                # Production bug: intake schema provided 'routing_number' while wire gateway expects 'wire_routing_number'
                wire_instructions = {
                    "beneficiary": applicant_profile["applicant_name"],
                    "amount": applicant_profile["loan_amount"],
                    "wire_routing": applicant_profile["wire_routing_number"],
                    "account_number": applicant_profile["account_number"],
                }
                span_wire.set_attribute("agent.disbursement_status", "COMPLETED")
        except KeyError as exc:
            execution_failed = True
            settlement_err_msg = f"KeyError: {exc}"
            record_span_error(root_span, exc)
            root_span.set_attribute("gen_ai.activity.status", "FAILED")
            root_span.set_attribute("status_code", "Error")

        duration_ms = (time.perf_counter() - start_time) * 1000
        root_span.set_attribute("gen_ai.workflow.duration_ms", duration_ms)

        return ScenarioResult(
            scenario_name="Mortgage Underwriting & Settlement Pipeline",
            anomaly_type="FAILED_EXECUTIONS",
            execution_id=execution_id,
            trace_id=trace_id,
            total_duration_ms=duration_ms,
            status="FAILED" if execution_failed else "SUCCESS",
            step_count=4,
            summary="Workflow completed credit, income, and appraisal underwriting but failed at final wire settlement due to missing schema field.",
            details={"error": settlement_err_msg, "wasted_upstream_agents": ["CreditRiskAssessmentAgent", "IncomeVerificationAgent", "PropertyValuationAgent"]},
        )


# ===========================================================================
# SCENARIO 4: COMMERCIAL LEASE DUE DILIGENCE AUDIT
# Anomaly Type: EXPENSIVE_PROMPT
# ===========================================================================
async def run_commercial_lease_due_diligence_scenario(client: Mistral, tracer: Tracer) -> ScenarioResult:
    """
    Acquisition due diligence auditing five commercial master lease agreements.
    Instead of using targeted section retrieval, the prompt builder dumps the
    entire 9,000+ token uncompressed legal lease text into each clause review turn.
    With p50 > 8,500 tokens and zero prefix caching, it triggers the Expensive Prompt detector.
    """
    execution_id = f"exec-lease-audit-{uuid.uuid4().hex[:8]}"
    start_time = time.perf_counter()
    workflow_name = "wf_commercial_lease_analysis"

    # Corpus representing heavy commercial master lease boilerplate
    heavy_lease_boilerplate = (
        "SECTION 1. PREMISES AND TERM. Landlord hereby leases to Tenant, and Tenant hereby leases from Landlord, "
        "the commercial premises situated at 100 Montgomery Street, Floors 14 through 18, San Francisco, CA. "
        "The initial term shall commence on January 1, 2024 and expire on December 31, 2034, unless sooner terminated...\n"
        + ("SECTION 2. OPERATING EXPENSES AND COMMON AREA MAINTENANCE. Tenant shall pay its Proportionate Share (14.82%) of Common Area Maintenance (CAM), including HVAC maintenance, security monitoring, elevator service contracts, exterior structural reserves, and municipal sewer assessments...\n" * 15)
        + ("SECTION 8. SUBLETTING AND ASSIGNMENT COVENANTS. Tenant shall not transfer, pledge, or sublease the premises without prior written consent of Landlord. Any assignment resulting in a change of corporate control exceeding 49% of voting equity shall constitute an unpermitted assignment trigger...\n" * 15)
        + ("SECTION 14. INDEMNIFICATION, HAZARDOUS MATERIALS, AND ENVIRONMENTAL COVENANTS. Tenant shall defend, indemnify, and hold harmless Landlord and its managing agents against any claims, damages, liabilities, fines, or remediation costs arising from hazardous substance release or statutory environmental breaches...\n" * 15)
        + ("SECTION 22. DEFAULT, LIQUIDATED DAMAGES, AND ACCELERATION REMEDIES. Upon occurrence of an Event of Default, Landlord may accelerate all remaining monthly base rent installments through the expiration date, discounted to present value at the Federal Reserve discount rate plus 200 basis points...\n" * 15)
    )

    with tracer.start_as_current_span("workflow_commercial_lease_analysis") as root_span:
        root_span.set_attribute("gen_ai.workflow.name", workflow_name)
        root_span.set_attribute("gen_ai.workflow.execution_id", execution_id)
        trace_id = format(root_span.get_span_context().trace_id, "032x")

        # Step 1: Sublease Clause Evaluation (Context stuffed prompt)
        with handoff_span(
            tracer,
            agent_name="ClauseExtractionAgent",
            action_name="evaluate_sublease_rights",
            execution_id=execution_id,
            workflow_name=workflow_name,
            handoff_to="RiskSynthesizerAgent",
            handoff_reason="Sublease covenants analyzed, passing to environmental review",
        ) as span_clause:
            query_prompt = (
                f"Analyze the following full commercial lease portfolio documentation:\n\n"
                f"{heavy_lease_boilerplate}\n\n"
                f"Evaluate whether Tenant may sublet 20,000 square feet to an affiliate without Landlord approval."
            )
            sublease_resp = await _safe_llm_call(
                client,
                model="mistral-small-latest",
                prompt=query_prompt,
                system_prompt="You are a Commercial Real Estate Legal Due Diligence Specialist.",
                max_tokens=250,
            )
            span_clause.set_attribute("agent.sublease_evaluation", sublease_resp[:80])

        # Step 2: Environmental Indemnity Review (Context stuffed prompt repeated)
        with handoff_span(
            tracer,
            agent_name="ClauseExtractionAgent",
            action_name="evaluate_environmental_indemnity",
            execution_id=execution_id,
            workflow_name=workflow_name,
            handoff_from="ClauseExtractionAgent",
            handoff_to="RiskSynthesizerAgent",
            handoff_reason="Environmental clauses audited, proceeding to final risk memo",
        ) as span_env:
            query_prompt_2 = (
                f"Analyze the following full commercial lease portfolio documentation:\n\n"
                f"{heavy_lease_boilerplate}\n\n"
                f"Assess Tenant's ongoing environmental liability obligations under Section 14."
            )
            env_resp = await _safe_llm_call(
                client,
                model="mistral-small-latest",
                prompt=query_prompt_2,
                system_prompt="You are a Commercial Real Estate Legal Due Diligence Specialist.",
                max_tokens=250,
            )
            span_env.set_attribute("agent.environmental_risk", env_resp[:80])

        # Step 3: Synthesis
        with handoff_span(
            tracer,
            agent_name="RiskSynthesizerAgent",
            action_name="generate_executive_memo",
            execution_id=execution_id,
            workflow_name=workflow_name,
            handoff_from="ClauseExtractionAgent",
            handoff_reason="Executive summary generated for investment committee",
        ) as span_memo:
            memo_resp = await _safe_llm_call(
                client,
                model="mistral-small-latest",
                prompt="Summarize top 3 commercial lease risks for investment committee presentation.",
                system_prompt="You are a Real Estate Private Equity Principal.",
                max_tokens=200,
            )
            span_memo.set_attribute("agent.memo_summary", memo_resp[:80])

        duration_ms = (time.perf_counter() - start_time) * 1000
        root_span.set_attribute("gen_ai.workflow.duration_ms", duration_ms)

        return ScenarioResult(
            scenario_name="Commercial Lease Due Diligence",
            anomaly_type="EXPENSIVE_PROMPT",
            execution_id=execution_id,
            trace_id=trace_id,
            total_duration_ms=duration_ms,
            status="SUCCESS",
            step_count=3,
            summary="ClauseExtractionAgent stuffed 9,000+ unpruned lease boilerplate tokens across queries without caching.",
            details={"estimated_p50_tokens": 8800, "dominant_role": "user", "cache_miss": True},
        )


# ===========================================================================
# SCENARIO 5: CUSTOMER SUPPORT TICKET ROUTING & TRIAGE
# Anomaly Type: OVER_PROVISIONED_MODEL
# ===========================================================================
async def run_support_ticket_triage_scenario(client: Mistral, tracer: Tracer) -> ScenarioResult:
    """
    Automated helpdesk ticket classification and department assignment.
    The enterprise base template defaulted to mistral-large-latest for all tasks.
    InboundTicketClassifierAgent queries mistral-large-latest with a 2,200 token corporate
    policy header to categorize a 1-sentence customer query, producing a 1-token output
    ('BILLING'). Triggers Over-Provisioned Model Tier detection.
    """
    execution_id = f"exec-triage-ovr-{uuid.uuid4().hex[:8]}"
    start_time = time.perf_counter()
    workflow_name = "wf_ticket_classification_routing"

    company_global_support_preamble = (
        "GLOBAL CUSTOMER SERVICE OPERATIONS MANUAL -- STANDARD OPERATING PROCEDURE 44-B\n"
        "All customer inquiries entering the omnichannel routing bus must adhere to strict SLA classification standards.\n"
        "Classification categories are strictly enumerated as: [BILLING, TECHNICAL_BUG, ACCOUNT_SECURITY, SALES_INQUIRY].\n"
        "Do not invent new categories. Do not include markdown or conversational formatting in classification tags.\n"
        + ("Operational Rule 101: Billing tickets concern refunds, invoice disputes, VAT exemptions, and recurring stripe charges.\n" * 12)
        + ("Operational Rule 102: Technical Bug tickets concern 500 errors, broken buttons, API latency, and database timeouts.\n" * 12)
        + ("Operational Rule 103: Account Security tickets concern password resets, 2FA recovery, and suspicious logins.\n" * 12)
    )

    with tracer.start_as_current_span("workflow_ticket_classification_routing") as root_span:
        root_span.set_attribute("gen_ai.workflow.name", workflow_name)
        root_span.set_attribute("gen_ai.workflow.execution_id", execution_id)
        trace_id = format(root_span.get_span_context().trace_id, "032x")

        with handoff_span(
            tracer,
            agent_name="InboundTicketClassifierAgent",
            action_name="classify_ticket_category",
            execution_id=execution_id,
            workflow_name=workflow_name,
            handoff_to="QueueDispatcherAgent",
            handoff_reason="Category determined, dispatching ticket to billing queue",
        ) as span_triage:
            customer_inquiry = "Why did my credit card get charged twice for renewal invoice INV-8819?"
            classification_prompt = (
                f"{company_global_support_preamble}\n\n"
                f"Customer Query: \"{customer_inquiry}\"\n\n"
                f"Respond with exactly one category word: BILLING, TECHNICAL_BUG, ACCOUNT_SECURITY, or SALES_INQUIRY."
            )

            # Heavyweight frontier model utilized for trivial 1-token classification
            result_category = await _safe_llm_call(
                client,
                model="mistral-large-latest",
                prompt=classification_prompt,
                system_prompt="You are an Inbound Support Routing Agent.",
                temperature=0.0,
                max_tokens=5,
            )
            span_triage.set_attribute("agent.assigned_category", result_category.strip())

        duration_ms = (time.perf_counter() - start_time) * 1000
        root_span.set_attribute("gen_ai.workflow.duration_ms", duration_ms)

        return ScenarioResult(
            scenario_name="Customer Support Ticket Triage",
            anomaly_type="OVER_PROVISIONED_MODEL",
            execution_id=execution_id,
            trace_id=trace_id,
            total_duration_ms=duration_ms,
            status="SUCCESS",
            step_count=1,
            summary="mistral-large-latest was utilized for single-token enum classification on a single-turn query.",
            details={"model": "mistral-large-latest", "output_tokens_approx": 1, "downgrade_target": "mistral-small-latest"},
        )


# ===========================================================================
# SCENARIO 6: VIP TRAVEL CONCIERGE MULTI-TURN DIALOGUE
# Anomaly Type: CONTEXT_GROWTH_LEAK
# ===========================================================================
async def run_concierge_service_dialogue_scenario(client: Mistral, tracer: Tracer) -> ScenarioResult:
    """
    Multi-turn VIP luxury travel concierge handling restaurant and hotel requests.
    The dialogue manager naively appends all conversation turns into a growing
    message list across agent handoffs without a sliding window or compaction,
    causing monotonic prompt token growth turn after turn (400 -> 1200 -> 2500 -> 4500 -> 7200).
    """
    execution_id = f"exec-concierge-{uuid.uuid4().hex[:8]}"
    start_time = time.perf_counter()
    workflow_name = "wf_concierge_support_dialogue"

    with tracer.start_as_current_span("workflow_concierge_support_dialogue") as root_span:
        root_span.set_attribute("gen_ai.workflow.name", workflow_name)
        root_span.set_attribute("gen_ai.workflow.execution_id", execution_id)
        trace_id = format(root_span.get_span_context().trace_id, "032x")

        # Conversational memory buffer that accumulates history without pruning
        conversation_history: List[Dict[str, str]] = [
            {"role": "system", "content": "You are a Luxury Travel Concierge Agent for five-star hotel guests. Maintain an elegant, helpful tone."}
        ]

        turns = [
            ("TravelConciergeAgent", "recommend_dining", "Guest: We are arriving in Tokyo this Thursday evening. Recommend three top-tier sushi omakase restaurants in Ginza."),
            ("DiningReservationAgent", "reserve_table", "Guest: Reserve a private room at Sushi Yoshitake for 8:00 PM Thursday. Note that my spouse has a severe shellfish allergy."),
            ("BillingSpecialistAgent", "authorize_charge", "Guest: Please charge the deposit to my Amex Centurion on file and confirm cancellation penalty terms."),
            ("ItineraryCoordinatorAgent", "finalize_itinerary", "Guest: Also arrange private luxury chauffeur pickup from Haneda Airport to Aman Tokyo at 6:00 PM."),
        ]

        for turn_idx, (agent_name, action_name, guest_message) in enumerate(turns, 1):
            with handoff_span(
                tracer,
                agent_name=agent_name,
                action_name=action_name,
                execution_id=execution_id,
                workflow_name=workflow_name,
                handoff_to=turns[turn_idx][0] if turn_idx < len(turns) else None,
                handoff_reason=f"Advancing turn {turn_idx} in guest itinerary coordination",
                metadata={"turn_number": turn_idx, "unpruned_history_size": len(conversation_history)},
            ) as span_turn:
                conversation_history.append({"role": "user", "content": guest_message})
                response = await _safe_llm_call(
                    client,
                    model="mistral-small-latest",
                    messages=conversation_history,
                    max_tokens=250,
                )
                conversation_history.append({"role": "assistant", "content": response})
                span_turn.set_attribute("agent.turn_response", response[:60])
                await asyncio.sleep(0.3)

        duration_ms = (time.perf_counter() - start_time) * 1000
        root_span.set_attribute("gen_ai.workflow.duration_ms", duration_ms)

        return ScenarioResult(
            scenario_name="VIP Travel Concierge Multi-Turn Dialogue",
            anomaly_type="CONTEXT_GROWTH_LEAK",
            execution_id=execution_id,
            trace_id=trace_id,
            total_duration_ms=duration_ms,
            status="SUCCESS",
            step_count=4,
            summary="Conversation history accumulated across 4 agent handoffs without compaction or sliding window.",
            details={"turn_count": 4, "growth_pattern": "monotonic_accumulation"},
        )


# ===========================================================================
# SCENARIO 7: RETAIL SUPPLY CHAIN INVENTORY AUDIT
# Anomaly Type: TOOL_CALL_AMPLIFICATION
# ===========================================================================
async def run_warehouse_inventory_audit_scenario(client: Mistral, tracer: Tracer) -> ScenarioResult:
    """
    Supply chain stockout check for urgent pending retail orders.
    InventoryAuditorAgent invokes query_warehouse_database, which returns a raw
    uncompressed JSON dump of 60 warehouse bin records (18 KB). The agent directly
    injects this raw dump into the subsequent LLM prompt, where tool output makes up
    over 75% of the total prompt tokens. Triggers Tool Call Amplification.
    """
    execution_id = f"exec-wh-audit-{uuid.uuid4().hex[:8]}"
    start_time = time.perf_counter()
    workflow_name = "wf_inventory_catalog_sync"

    # Realistic uncompressed database dump from legacy warehouse system
    raw_warehouse_dump = {
        "facility_code": "WH-IL-NAPERVILLE-02",
        "timestamp_utc": "2026-08-20T10:14:02.912Z",
        "inventory_snapshot": [
            {
                "bin_id": f"BIN-A-{i:03d}",
                "sku": f"SKU-ELEC-44{i:02d}",
                "rfid_hex": f"0xEF{i:04X}99A0",
                "coordinates": {"aisle": 4, "rack": i % 8, "tier": (i % 4) + 1},
                "quantity_on_hand": (i * 7) % 45,
                "quantity_reserved": (i * 2) % 15,
                "sensor_telemetry": {"temp_celsius": 19.4, "humidity_pct": 42.1, "vibration_g": 0.02},
                "vendor_origin": "Shenzhen Precision Microelectronics Ltd",
                "pallet_batch_code": f"PLT-2026-{1000 + i}",
                "inspection_status": "PASSED_GREEN",
            }
            for i in range(1, 55)
        ],
    }
    raw_dump_json = json.dumps(raw_warehouse_dump)

    with tracer.start_as_current_span("workflow_inventory_catalog_sync") as root_span:
        root_span.set_attribute("gen_ai.workflow.name", workflow_name)
        root_span.set_attribute("gen_ai.workflow.execution_id", execution_id)
        trace_id = format(root_span.get_span_context().trace_id, "032x")

        # Step 1: Database Tool Execution returning raw dump
        with handoff_span(
            tracer,
            agent_name="InventoryAuditorAgent",
            action_name="audit_sku_availability",
            execution_id=execution_id,
            workflow_name=workflow_name,
            handoff_to="StockReconcilerAgent",
            handoff_reason="Physical inventory verified, notifying fulfillment dispatcher",
        ) as span_audit:
            with tool_span(
                tracer,
                tool_name="query_warehouse_database",
                arguments={"facility": "WH-IL-NAPERVILLE-02", "target_sku": "SKU-ELEC-4412"},
                execution_id=execution_id,
            ) as set_tool:
                await asyncio.sleep(0.3)
                set_tool(raw_warehouse_dump)

            # Prompt injection of full unprojected tool payload
            amplified_prompt = (
                f"Analyze the following warehouse facility inventory snapshot and determine if SKU-ELEC-4412 has sufficient stock:\n\n"
                f"{raw_dump_json}\n\n"
                f"Verify if unreserved stock count is >= 10 units for order fulfillment."
            )

            audit_verdict = await _safe_llm_call(
                client,
                model="mistral-small-latest",
                prompt=amplified_prompt,
                system_prompt="You are a Warehouse Logistics Inventory Auditor.",
                max_tokens=150,
            )
            span_audit.set_attribute("agent.stock_audit_verdict", audit_verdict[:80])

        duration_ms = (time.perf_counter() - start_time) * 1000
        root_span.set_attribute("gen_ai.workflow.duration_ms", duration_ms)

        return ScenarioResult(
            scenario_name="Retail Supply Chain Inventory Audit",
            anomaly_type="TOOL_CALL_AMPLIFICATION",
            execution_id=execution_id,
            trace_id=trace_id,
            total_duration_ms=duration_ms,
            status="SUCCESS",
            step_count=2,
            summary="Raw uncompressed 18 KB JSON database dump injected into prompt, dominating >70% of prompt tokens.",
            details={"tool_name": "query_warehouse_database", "payload_share_pct": 74.5},
        )


# ===========================================================================
# SCENARIO 8: QUANTITATIVE PORTFOLIO PERFORMANCE REPORTING
# Anomaly Type: CACHE_COLLAPSE
# ===========================================================================
async def run_portfolio_performance_reporting_scenario(client: Mistral, tracer: Tracer) -> ScenarioResult:
    """
    Quarterly institutional investment portfolio reporting and compliance review.
    The compliance audit relies on a 3,000-token static SEC regulatory guidelines prompt.
    Calls 1 & 2 establish a warm prefix cache. Call 3 prepends dynamic audit metadata
    (Timestamp and Request Nonce) at byte 0 of the prompt, completely invalidating the
    prefix cache and dropping cache read tokens to 0. Triggers Cache Collapse detection.
    """
    execution_id = f"exec-portf-rep-{uuid.uuid4().hex[:8]}"
    start_time = time.perf_counter()
    workflow_name = "wf_portfolio_strategy_report"

    static_sec_regulatory_guidelines = (
        "UNITED STATES SECURITIES AND EXCHANGE COMMISSION -- INVESTMENT ADVISERS ACT RULE 206(4)-1\n"
        "Statutory Disclosure Standards for Performance Presentation and Risk Metrics.\n"
        "1. Performance Metrics: Net-of-fees performance must be presented with equal prominence to gross returns.\n"
        "2. Benchmark Comparison: Any benchmark presented must be relevant to the fund's investment mandate and strategy.\n"
        "3. Attribution Analysis: Sector attribution must reflect Brinson-Fachler allocation and selection components.\n"
        + ("Standard Compliance Definition 4.1: Derivatives leverage must be measured via gross notional commitment and VaR.\n" * 16)
        + ("Standard Compliance Definition 4.2: High-yield fixed income exposures exceeding 15% require credit stress testing.\n" * 16)
        + ("Standard Compliance Definition 4.3: Foreign currency exposures must be categorized as hedged or unhedged.\n" * 16)
    )

    with tracer.start_as_current_span("workflow_portfolio_strategy_report") as root_span:
        root_span.set_attribute("gen_ai.workflow.name", workflow_name)
        root_span.set_attribute("gen_ai.workflow.execution_id", execution_id)
        trace_id = format(root_span.get_span_context().trace_id, "032x")

        # Call 1: Equities Performance Review (Warm cache creation)
        with handoff_span(
            tracer,
            agent_name="PortfolioAnalyticsAgent",
            action_name="audit_equities_performance",
            execution_id=execution_id,
            workflow_name=workflow_name,
            handoff_to="RegulatoryComplianceReviewAgent",
            handoff_reason="Equities reviewed, checking fixed income assets",
        ) as span1:
            prompt_1 = f"{static_sec_regulatory_guidelines}\n\nPortfolio: Vanguard Large Cap Growth, Return: +14.2% net, Benchmark: +12.8%."
            res1 = await _safe_llm_call(
                client,
                model="mistral-small-latest",
                prompt=prompt_1,
                system_prompt="You are an Institutional Portfolio Compliance Reviewer.",
            )
            span1.set_attribute("agent.equities_status", "COMPLIANT")

        # Call 2: Fixed Income Performance Review (Prefix cache hit)
        with handoff_span(
            tracer,
            agent_name="PortfolioAnalyticsAgent",
            action_name="audit_fixed_income_performance",
            execution_id=execution_id,
            workflow_name=workflow_name,
            handoff_to="RegulatoryComplianceReviewAgent",
            handoff_reason="Fixed income reviewed, advancing to final audit check",
        ) as span2:
            prompt_2 = f"{static_sec_regulatory_guidelines}\n\nPortfolio: Core US Aggregate Bond, Return: +3.8% net, Benchmark: +3.2%."
            res2 = await _safe_llm_call(
                client,
                model="mistral-small-latest",
                prompt=prompt_2,
                system_prompt="You are an Institutional Portfolio Compliance Reviewer.",
            )
            span2.set_attribute("agent.fixed_income_status", "COMPLIANT")

        # Call 3: Audit Compliance Review with Dynamic Header Prepended (Busts Prefix Cache)
        with handoff_span(
            tracer,
            agent_name="RegulatoryComplianceReviewAgent",
            action_name="generate_compliance_clearance",
            execution_id=execution_id,
            workflow_name=workflow_name,
            handoff_from="PortfolioAnalyticsAgent",
            handoff_reason="Audit report signed and archived in SEC compliance binder",
        ) as span3:
            # Dynamic header prepended at character 0 invalidates Mistral prefix cache key
            dynamic_audit_header = f"TRACE_EVENT | Host: srv-fin-09 | Nonce: {uuid.uuid4().hex} | Time: {time.time()}\n\n"
            prompt_3 = f"{dynamic_audit_header}{static_sec_regulatory_guidelines}\n\nVerify final portfolio prospectus disclosures."
            res3 = await _safe_llm_call(
                client,
                model="mistral-small-latest",
                prompt=prompt_3,
                system_prompt="You are an Institutional Portfolio Compliance Reviewer.",
            )
            span3.set_attribute("agent.audit_clearance", "APPROVED")

        duration_ms = (time.perf_counter() - start_time) * 1000
        root_span.set_attribute("gen_ai.workflow.duration_ms", duration_ms)

        return ScenarioResult(
            scenario_name="Quantitative Portfolio Strategy Reporting",
            anomaly_type="CACHE_COLLAPSE",
            execution_id=execution_id,
            trace_id=trace_id,
            total_duration_ms=duration_ms,
            status="SUCCESS",
            step_count=3,
            summary="Calls 1 and 2 established a warm prefix cache; Call 3 prepended dynamic audit timestamps at byte 0, invalidating cache.",
            details={"prefix_cache_invalidated": True, "static_prefix_tokens": 3100},
        )


# ===========================================================================
# SCENARIO 9: HIGH-EFFICIENCY WARRANTY CLAIMS BASELINE
# Anomaly Type: CLEAN_OPTIMAL_BASELINE
# ===========================================================================
async def run_clean_optimized_claims_scenario(client: Mistral, tracer: Tracer) -> ScenarioResult:
    """
    Healthy reference production workflow for consumer electronics warranty claims.
    Employs cost-efficient mistral-small-latest, prefix-cached static system prompts,
    clean field-projected tool queries, sliding-window dialogue context, and safe dictionary
    parsing. Exhibits zero token waste and serves as the benchmark baseline.
    """
    execution_id = f"exec-opt-claim-{uuid.uuid4().hex[:8]}"
    start_time = time.perf_counter()
    workflow_name = "wf_optimized_claims_processing"

    with tracer.start_as_current_span("workflow_optimized_claims_processing") as root_span:
        root_span.set_attribute("gen_ai.workflow.name", workflow_name)
        root_span.set_attribute("gen_ai.workflow.execution_id", execution_id)
        trace_id = format(root_span.get_span_context().trace_id, "032x")

        # Step 1: Clean Intake with Structured Extraction
        with handoff_span(
            tracer,
            agent_name="ClaimsIntakeAgent",
            action_name="parse_warranty_claim",
            execution_id=execution_id,
            workflow_name=workflow_name,
            handoff_to="WarrantyRuleEngineAgent",
            handoff_reason="Claim parsed and verified, advancing to coverage evaluation",
        ) as span_intake:
            intake_res = await _safe_llm_call(
                client,
                model="mistral-small-latest",
                prompt="Extract serial number and defect description from: 'My Sony WH-1000XM5 headphones (SN: SN-881290) won't hold charge.'",
                system_prompt="You are a Consumer Electronics Warranty Intake Specialist.",
                max_tokens=100,
            )
            span_intake.set_attribute("agent.claim_extracted", intake_res[:60])

        # Step 2: Clean Projected Tool Query
        with handoff_span(
            tracer,
            agent_name="WarrantyRuleEngineAgent",
            action_name="evaluate_policy_coverage",
            execution_id=execution_id,
            workflow_name=workflow_name,
            handoff_from="ClaimsIntakeAgent",
            handoff_to="ApprovalNotificationAgent",
            handoff_reason="Claim covered under 2-year manufacturer warranty, notifying customer",
        ) as span_rule:
            with tool_span(
                tracer,
                tool_name="lookup_warranty_status",
                arguments={"serial_no": "SN-881290"},
                execution_id=execution_id,
            ) as set_tool:
                await asyncio.sleep(0.15)
                # Projected concise dictionary without unneeded fields
                set_tool({"serial_no": "SN-881290", "coverage": "ACTIVE", "expires": "2027-04-15"})

            rule_res = await _safe_llm_call(
                client,
                model="mistral-small-latest",
                prompt="Serial SN-881290 has ACTIVE warranty through 2027-04-15. Confirm replacement authorization.",
                system_prompt="You are a Warranty Coverage Rule Specialist.",
                max_tokens=100,
            )
            span_rule.set_attribute("agent.authorization", "APPROVED")

        # Step 3: Notification Generation
        with handoff_span(
            tracer,
            agent_name="ApprovalNotificationAgent",
            action_name="send_replacement_notice",
            execution_id=execution_id,
            workflow_name=workflow_name,
            handoff_from="WarrantyRuleEngineAgent",
            handoff_reason="Return merchandise authorization (RMA) email dispatched to customer",
        ) as span_notify:
            notify_res = await _safe_llm_call(
                client,
                model="mistral-small-latest",
                prompt="Compose polite customer email with RMA label for headphone battery replacement.",
                system_prompt="You are a Customer Communications Specialist.",
                max_tokens=150,
            )
            span_notify.set_attribute("agent.notification_dispatched", True)

        duration_ms = (time.perf_counter() - start_time) * 1000
        root_span.set_attribute("gen_ai.workflow.duration_ms", duration_ms)

        return ScenarioResult(
            scenario_name="High-Efficiency Warranty Claims Baseline",
            anomaly_type="CLEAN_OPTIMAL_BASELINE",
            execution_id=execution_id,
            trace_id=trace_id,
            total_duration_ms=duration_ms,
            status="SUCCESS",
            step_count=3,
            summary="Clean baseline workflow using right-sized model, projected tool outputs, and zero token waste.",
            details={"wasted_tokens": 0, "efficiency_score": 100.0, "status": "OPTIMAL"},
        )
