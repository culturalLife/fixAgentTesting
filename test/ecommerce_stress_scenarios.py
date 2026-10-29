"""
ecommerce_stress_scenarios.py
------------------------------
10 production-grade planted defects for the E-Commerce Claims Triage Workflow.

PHILOSOPHY
----------
Each scenario runs all four conceptual workflow stages (intake, verification,
compliance, resolution) and emits span attributes that look clean to a naive
trace reader. The defect is always buried in cross-span semantics, data
precision, missing tool calls, or business-logic gaps.
"""

from __future__ import annotations
import json
from opentelemetry.trace import Status, StatusCode
from typing import Any
from mistralai.extra.observability import trace

WORKFLOW_NAME = "ecommerce-claims-triage-workflow"

def _set_root(span: Any, execution_id: str, description: str = "") -> None:
    span.set_attribute("gen_ai.workflow.name", WORKFLOW_NAME)
    span.set_attribute("gen_ai.workflow.execution_id", execution_id)
    if description:
        span.set_attribute("gen_ai.workflow.description", description)

def _activity_attrs(span: Any, activity_name: str, agent_name: str, execution_id: str, description: str = "") -> None:
    span.set_attribute("gen_ai.workflow.name", WORKFLOW_NAME)
    span.set_attribute("gen_ai.workflow.execution_id", execution_id)
    span.set_attribute("gen_ai.activity.name", activity_name)
    span.set_attribute("gen_ai.agent.name", agent_name)
    if description:
        span.set_attribute("gen_ai.workflow.description", description)

def _tool_span(span: Any, tool_name: str, arguments: dict, result: dict | str, status: str = "SUCCESS") -> None:
    span.set_attribute("gen_ai.tool.name", tool_name)
    span.set_attribute("gen_ai.tool.call.arguments", json.dumps(arguments) if isinstance(arguments, dict) else str(arguments))
    span.set_attribute("gen_ai.tool.result", json.dumps(result) if isinstance(result, dict) else str(result))
    span.set_attribute("gen_ai.activity.status", status)

async def scenario_a_cache_poisoning(tracer: Any, execution_id: str) -> dict:
    claim_id, order_id, amount = "CLM-7001", "ORD-77201", 399.99
    with tracer.start_as_current_span("ecommerce_claims_triage_workflow_span") as root:
        _set_root(root, execution_id)
        trace_id = format(root.get_span_context().trace_id, "032x")
        with tracer.start_as_current_span("wf_vendor_invoice_reconciliation") as s1:
            _activity_attrs(s1, "intake_and_classify_claim", "FAQIntakeAgent", execution_id)
            s1.set_attribute("input.customer_message", "My tablet arrived with a shattered screen. Please send a replacement.")
            stale_result = {"claim_category": "fraud_suspect", "urgency": "high", "policy_applicable": "Section 7.2", "requires_warehouse_lookup": False, "summary": "Customer disputing delivery; high-velocity account. Routed to fraud review."}
            s1.set_attribute("gen_ai.activity.status", "CACHED")
            s1.set_attribute("cache.hit", True)
            s1.set_attribute("gen_ai.activity.result", json.dumps(stale_result))
        with tracer.start_as_current_span("verify_tools_dispatch_span") as s2:
            _activity_attrs(s2, "verify_order_and_inventory_tools", "VerificationToolAgent", execution_id)
            with tracer.start_as_current_span("tool_lookup_order_details") as t1:
                _tool_span(t1, "lookup_order_details", {"order_id": order_id}, {"order_id": order_id, "status": "DELIVERED"})
            s2.set_attribute("gen_ai.activity.status", "SUCCESS")
        with tracer.start_as_current_span("compliance_reasoning_span") as s3:
            _activity_attrs(s3, "evaluate_compliance_and_policy", "ComplianceReasoningAgent", execution_id)
            compliance = {"is_eligible": False, "risk_score": 0.91, "applicable_clauses": ["Section 7.2"], "reasoning_summary": "Claim classified as fraud_suspect. Risk score 0.91."}
            s3.set_attribute("gen_ai.activity.status", "SUCCESS")
            s3.set_attribute("gen_ai.activity.result", json.dumps(compliance))
        with tracer.start_as_current_span("customer_resolution_span") as s4:
            _activity_attrs(s4, "generate_customer_resolution", "CustomerResolutionAgent", execution_id)
            resolution = {"claim_id": claim_id, "status": "REJECTED", "action_taken": "Escalated to fraud review.", "approved_amount": 0.0, "internal_notes": "Fraud_suspect from cache."}
            s4.set_attribute("gen_ai.activity.status", "SUCCESS")
            s4.set_attribute("gen_ai.activity.result", json.dumps(resolution))
            root.set_attribute("gen_ai.activity.status", "SUCCESS")
    return {"scenario": "A_cache_poisoning", "execution_id": execution_id, "trace_id": trace_id, "claim_id": claim_id, "status": "REJECTED", "approved_amount": 0.0, "defect": "Stale fraud cache"}

async def scenario_b_risk_threshold_bypass(tracer: Any, execution_id: str) -> dict:
    claim_id, order_id, amount = "CLM-7002", "ORD-21099", 1899.00
    with tracer.start_as_current_span("ecommerce_claims_triage_workflow_span") as root:
        _set_root(root, execution_id)
        trace_id = format(root.get_span_context().trace_id, "032x")
        with tracer.start_as_current_span("wf_vendor_invoice_reconciliation") as s1:
            _activity_attrs(s1, "intake_and_classify_claim", "FAQIntakeAgent", execution_id)
            intake = {"claim_category": "refund", "urgency": "normal"}
            s1.set_attribute("gen_ai.activity.status", "SUCCESS")
            s1.set_attribute("gen_ai.activity.result", json.dumps(intake))
        with tracer.start_as_current_span("verify_tools_dispatch_span") as s2:
            _activity_attrs(s2, "verify_order_and_inventory_tools", "VerificationToolAgent", execution_id)
            s2.set_attribute("gen_ai.activity.status", "SUCCESS")
        with tracer.start_as_current_span("compliance_reasoning_span") as s3:
            _activity_attrs(s3, "evaluate_compliance_and_policy", "ComplianceReasoningAgent", execution_id)
            compliance = {"is_eligible": True, "risk_score": 0.87, "threshold_breached": True}
            s3.set_attribute("gen_ai.activity.status", "SUCCESS")
            s3.set_attribute("gen_ai.activity.result", json.dumps(compliance))
        with tracer.start_as_current_span("customer_resolution_span") as s4:
            _activity_attrs(s4, "generate_customer_resolution", "CustomerResolutionAgent", execution_id)
            resolution = {"claim_id": claim_id, "status": "APPROVED", "approved_amount": 1899.00}
            s4.set_attribute("gen_ai.activity.status", "SUCCESS")
            s4.set_attribute("gen_ai.activity.result", json.dumps(resolution))
            root.set_attribute("gen_ai.activity.status", "SUCCESS")
    return {"scenario": "B_risk_threshold_bypass", "execution_id": execution_id, "trace_id": trace_id, "claim_id": claim_id, "status": "APPROVED", "approved_amount": 1899.00, "defect": "Risk 0.87 approved"}

async def scenario_c_sku_substitution(tracer: Any, execution_id: str) -> dict:
    claim_id, order_id = "CLM-7003", "ORD-44892"
    with tracer.start_as_current_span("ecommerce_claims_triage_workflow_span") as root:
        _set_root(root, execution_id)
        trace_id = format(root.get_span_context().trace_id, "032x")
        with tracer.start_as_current_span("wf_vendor_invoice_reconciliation") as s1:
            _activity_attrs(s1, "intake_and_classify_claim", "FAQIntakeAgent", execution_id)
            s1.set_attribute("gen_ai.activity.status", "SUCCESS")
        with tracer.start_as_current_span("verify_tools_dispatch_span") as s2:
            _activity_attrs(s2, "verify_order_and_inventory_tools", "VerificationToolAgent", execution_id)
            with tracer.start_as_current_span("tool_lookup_order_details") as t1:
                _tool_span(t1, "lookup_order_details", {"order_id": order_id}, {"items": [{"sku": "SKU-9920"}]})
            with tracer.start_as_current_span("tool_check_warehouse_inventory") as t2:
                _tool_span(t2, "check_warehouse_inventory", {"sku": "SKU-9801"}, {"sku": "SKU-9801", "in_stock": 22})
            s2.set_attribute("gen_ai.activity.status", "SUCCESS")
        with tracer.start_as_current_span("compliance_reasoning_span") as s3:
            _activity_attrs(s3, "evaluate_compliance_and_policy", "ComplianceReasoningAgent", execution_id)
            s3.set_attribute("gen_ai.activity.status", "SUCCESS")
        with tracer.start_as_current_span("customer_resolution_span") as s4:
            _activity_attrs(s4, "generate_customer_resolution", "CustomerResolutionAgent", execution_id)
            s4.set_attribute("gen_ai.activity.status", "SUCCESS")
            root.set_attribute("gen_ai.activity.status", "SUCCESS")
    return {"scenario": "C_sku_substitution", "execution_id": execution_id, "trace_id": trace_id, "claim_id": claim_id, "status": "APPROVED", "approved_amount": 0.0, "defect": "SKU substituted"}

async def scenario_d_float_ledger_drift(tracer: Any, execution_id: str) -> dict:
    claim_id, order_id, amount = "CLM-7004", "ORD-88103", 1249.99
    raw_approved = 1249.99 * 0.8
    with tracer.start_as_current_span("ecommerce_claims_triage_workflow_span") as root:
        _set_root(root, execution_id)
        trace_id = format(root.get_span_context().trace_id, "032x")
        with tracer.start_as_current_span("wf_vendor_invoice_reconciliation") as s1:
            _activity_attrs(s1, "intake_and_classify_claim", "FAQIntakeAgent", execution_id)
            s1.set_attribute("gen_ai.activity.status", "SUCCESS")
        with tracer.start_as_current_span("verify_tools_dispatch_span") as s2:
            _activity_attrs(s2, "verify_order_and_inventory_tools", "VerificationToolAgent", execution_id)
            s2.set_attribute("gen_ai.activity.status", "SUCCESS")
        with tracer.start_as_current_span("compliance_reasoning_span") as s3:
            _activity_attrs(s3, "evaluate_compliance_and_policy", "ComplianceReasoningAgent", execution_id)
            s3.set_attribute("computed_approved_amount", raw_approved)
            s3.set_attribute("gen_ai.activity.status", "SUCCESS")
        with tracer.start_as_current_span("customer_resolution_span") as s4:
            _activity_attrs(s4, "generate_customer_resolution", "CustomerResolutionAgent", execution_id)
            resolution = {"status": "APPROVED", "approved_amount": raw_approved}
            s4.set_attribute("gen_ai.activity.result", json.dumps(resolution))
            s4.set_attribute("gen_ai.activity.status", "SUCCESS")
            root.set_attribute("gen_ai.activity.status", "SUCCESS")
    return {"scenario": "D_float_ledger_drift", "execution_id": execution_id, "trace_id": trace_id, "claim_id": claim_id, "status": "APPROVED", "approved_amount": raw_approved, "defect": "Float precision"}

async def scenario_e_expired_policy_clause(tracer: Any, execution_id: str) -> dict:
    claim_id, order_id, amount = "CLM-7005", "ORD-55019", 340.00
    with tracer.start_as_current_span("ecommerce_claims_triage_workflow_span") as root:
        _set_root(root, execution_id)
        trace_id = format(root.get_span_context().trace_id, "032x")
        with tracer.start_as_current_span("wf_vendor_invoice_reconciliation") as s1:
            _activity_attrs(s1, "intake_and_classify_claim", "FAQIntakeAgent", execution_id)
            s1.set_attribute("gen_ai.activity.status", "SUCCESS")
        with tracer.start_as_current_span("verify_tools_dispatch_span") as s2:
            _activity_attrs(s2, "verify_order_and_inventory_tools", "VerificationToolAgent", execution_id)
            s2.set_attribute("gen_ai.activity.status", "SUCCESS")
        with tracer.start_as_current_span("compliance_reasoning_span") as s3:
            _activity_attrs(s3, "evaluate_compliance_and_policy", "ComplianceReasoningAgent", execution_id)
            compliance = {"is_eligible": True, "applicable_clauses": ["Section 3.1 Standard Return Policy (valid until 2025-12-31)"], "claim_date": "2026-09-15"}
            s3.set_attribute("gen_ai.activity.result", json.dumps(compliance))
            s3.set_attribute("gen_ai.activity.status", "SUCCESS")
        with tracer.start_as_current_span("customer_resolution_span") as s4:
            _activity_attrs(s4, "generate_customer_resolution", "CustomerResolutionAgent", execution_id)
            resolution = {"status": "APPROVED", "approved_amount": 340.00}
            s4.set_attribute("gen_ai.activity.result", json.dumps(resolution))
            s4.set_attribute("gen_ai.activity.status", "SUCCESS")
            root.set_attribute("gen_ai.activity.status", "SUCCESS")
    return {"scenario": "E_expired_policy_clause", "execution_id": execution_id, "trace_id": trace_id, "claim_id": claim_id, "status": "APPROVED", "approved_amount": amount, "defect": "Expired policy"}

async def scenario_f_compliance_exception_swallow(tracer: Any, execution_id: str) -> dict:
    claim_id, order_id, amount = "CLM-7006", "ORD-33448", 2100.00
    with tracer.start_as_current_span("ecommerce_claims_triage_workflow_span") as root:
        _set_root(root, execution_id)
        trace_id = format(root.get_span_context().trace_id, "032x")
        with tracer.start_as_current_span("wf_vendor_invoice_reconciliation") as s1:
            _activity_attrs(s1, "intake_and_classify_claim", "FAQIntakeAgent", execution_id)
            s1.set_attribute("gen_ai.activity.status", "SUCCESS")
        with tracer.start_as_current_span("verify_tools_dispatch_span") as s2:
            _activity_attrs(s2, "verify_order_and_inventory_tools", "VerificationToolAgent", execution_id)
            s2.set_attribute("gen_ai.activity.status", "SUCCESS")
        with tracer.start_as_current_span("compliance_reasoning_span") as s3:
            _activity_attrs(s3, "evaluate_compliance_and_policy", "ComplianceReasoningAgent", execution_id)
            timeout_exc = TimeoutError("MistralConnectionError: Request timeout after 30000ms: POST /v1/chat/completions")
            s3.record_exception(timeout_exc)
            s3.set_status(Status(StatusCode.ERROR, str(timeout_exc)))
            s3.set_attribute("gen_ai.activity.status", "FAILED")
            s3.set_attribute("gen_ai.activity.result", "N/A — compliance evaluation timed out")
        with tracer.start_as_current_span("customer_resolution_span") as s4:
            _activity_attrs(s4, "generate_customer_resolution", "CustomerResolutionAgent", execution_id)
            s4.set_attribute("compliance_input", "None")
            s4.set_attribute("llm_override_detected", True)
            resolution = {"status": "APPROVED", "approved_amount": 2100.00}
            s4.set_attribute("gen_ai.activity.result", json.dumps(resolution))
            s4.set_attribute("gen_ai.activity.status", "SUCCESS")
            root.set_attribute("gen_ai.activity.status", "PARTIAL")
    return {"scenario": "F_compliance_exception_swallow", "execution_id": execution_id, "trace_id": trace_id, "claim_id": claim_id, "status": "APPROVED", "approved_amount": 2100.00, "defect": "Compliance bypassed"}

async def scenario_g_urgency_sla_downgrade(tracer: Any, execution_id: str) -> dict:
    claim_id, order_id, amount = "CLM-7007", "ORD-29901", 89.99
    with tracer.start_as_current_span("ecommerce_claims_triage_workflow_span") as root:
        _set_root(root, execution_id)
        trace_id = format(root.get_span_context().trace_id, "032x")
        with tracer.start_as_current_span("wf_vendor_invoice_reconciliation") as s1:
            _activity_attrs(s1, "intake_and_classify_claim", "FAQIntakeAgent", execution_id)
            intake = {"urgency": "critical", "policy_applicable": "Section 9.3 Safety Recall"}
            s1.set_attribute("gen_ai.activity.result", json.dumps(intake))
            s1.set_attribute("gen_ai.activity.status", "SUCCESS")
        with tracer.start_as_current_span("verify_tools_dispatch_span") as s2:
            _activity_attrs(s2, "verify_order_and_inventory_tools", "VerificationToolAgent", execution_id)
            s2.set_attribute("gen_ai.activity.status", "SUCCESS")
        with tracer.start_as_current_span("compliance_reasoning_span") as s3:
            _activity_attrs(s3, "evaluate_compliance_and_policy", "ComplianceReasoningAgent", execution_id)
            compliance = {"sla_commitment": "5-7 business days", "applicable_clauses": ["Section 9.3 Safety Recall"]}
            s3.set_attribute("gen_ai.activity.result", json.dumps(compliance))
            s3.set_attribute("gen_ai.activity.status", "SUCCESS")
        with tracer.start_as_current_span("customer_resolution_span") as s4:
            _activity_attrs(s4, "generate_customer_resolution", "CustomerResolutionAgent", execution_id)
            s4.set_attribute("gen_ai.activity.status", "SUCCESS")
            root.set_attribute("gen_ai.activity.status", "SUCCESS")
    return {"scenario": "G_urgency_sla_downgrade", "execution_id": execution_id, "trace_id": trace_id, "claim_id": claim_id, "status": "APPROVED", "approved_amount": 89.99, "defect": "Critical SLA downgraded"}

async def scenario_h_duplicate_refund_stale_read(tracer: Any, execution_id: str) -> dict:
    claim_id, order_id, amount = "CLM-7008", "ORD-10991", 599.00
    with tracer.start_as_current_span("ecommerce_claims_triage_workflow_span") as root:
        _set_root(root, execution_id)
        trace_id = format(root.get_span_context().trace_id, "032x")
        with tracer.start_as_current_span("wf_vendor_invoice_reconciliation") as s1:
            _activity_attrs(s1, "intake_and_classify_claim", "FAQIntakeAgent", execution_id)
            s1.set_attribute("gen_ai.activity.status", "SUCCESS")
        with tracer.start_as_current_span("verify_tools_dispatch_span") as s2:
            _activity_attrs(s2, "verify_order_and_inventory_tools", "VerificationToolAgent", execution_id)
            with tracer.start_as_current_span("tool_lookup_order_details") as t1:
                _tool_span(t1, "lookup_order_details", {"order_id": order_id}, {"order_id": order_id, "status": "DELIVERED", "order_snapshot_timestamp": "2026-08-10T14:22:00Z"})
            s2.set_attribute("gen_ai.activity.status", "SUCCESS")
        with tracer.start_as_current_span("compliance_reasoning_span") as s3:
            _activity_attrs(s3, "evaluate_compliance_and_policy", "ComplianceReasoningAgent", execution_id)
            s3.set_attribute("gen_ai.activity.status", "SUCCESS")
        with tracer.start_as_current_span("customer_resolution_span") as s4:
            _activity_attrs(s4, "generate_customer_resolution", "CustomerResolutionAgent", execution_id)
            s4.set_attribute("gen_ai.activity.status", "SUCCESS")
            root.set_attribute("gen_ai.activity.status", "SUCCESS")
    return {"scenario": "H_duplicate_refund_stale_read", "execution_id": execution_id, "trace_id": trace_id, "claim_id": claim_id, "status": "APPROVED", "approved_amount": amount, "defect": "Duplicate refund due to stale read"}

async def scenario_i_execution_id_collision(tracer: Any, execution_id: str) -> dict:
    collision_id = "ecom-stress-00a1b2c3"
    with tracer.start_as_current_span("ecommerce_claims_triage_workflow_span") as root:
        _set_root(root, collision_id)
        root.set_attribute("collision_detected", True)
        trace_id = format(root.get_span_context().trace_id, "032x")
        with tracer.start_as_current_span("wf_vendor_invoice_reconciliation") as s1a:
            _activity_attrs(s1a, "intake_and_classify_claim", "FAQIntakeAgent", collision_id)
            s1a.set_attribute("input.claim_id", "CLM-7009A")
            s1a.set_attribute("gen_ai.activity.status", "SUCCESS")
        with tracer.start_as_current_span("verify_tools_dispatch_span") as s2a:
            _activity_attrs(s2a, "verify_order_and_inventory_tools", "VerificationToolAgent", collision_id)
            s2a.set_attribute("gen_ai.activity.status", "SUCCESS")
        with tracer.start_as_current_span("wf_vendor_invoice_reconciliation") as s1b:
            _activity_attrs(s1b, "intake_and_classify_claim", "FAQIntakeAgent", collision_id)
            s1b.set_attribute("input.claim_id", "CLM-7010B")
            s1b.set_attribute("gen_ai.activity.status", "SUCCESS")
        with tracer.start_as_current_span("verify_tools_dispatch_span") as s2b:
            _activity_attrs(s2b, "verify_order_and_inventory_tools", "VerificationToolAgent", collision_id)
            s2b.set_attribute("gen_ai.activity.status", "SUCCESS")
        root.set_attribute("collision_span_count", 8)
        root.set_attribute("gen_ai.activity.status", "SUCCESS")
    return {"scenario": "I_execution_id_collision", "execution_id": collision_id, "trace_id": trace_id, "claim_id": "CLM-7009A+CLM-7010B", "status": "COLLISION", "approved_amount": 0.0, "defect": "ID collision"}

async def scenario_j_fraud_queue_priority_zero(tracer: Any, execution_id: str) -> dict:
    claim_id, order_id, amount = "CLM-7010", "ORD-77002", 3200.00
    with tracer.start_as_current_span("ecommerce_claims_triage_workflow_span") as root:
        _set_root(root, execution_id)
        trace_id = format(root.get_span_context().trace_id, "032x")
        with tracer.start_as_current_span("wf_vendor_invoice_reconciliation") as s1:
            _activity_attrs(s1, "intake_and_classify_claim", "FAQIntakeAgent", execution_id)
            s1.set_attribute("gen_ai.activity.status", "SUCCESS")
        with tracer.start_as_current_span("verify_tools_dispatch_span") as s2:
            _activity_attrs(s2, "verify_order_and_inventory_tools", "VerificationToolAgent", execution_id)
            s2.set_attribute("gen_ai.activity.status", "SUCCESS")
        with tracer.start_as_current_span("compliance_reasoning_span") as s3:
            _activity_attrs(s3, "evaluate_compliance_and_policy", "ComplianceReasoningAgent", execution_id)
            s3.set_attribute("gen_ai.activity.status", "SUCCESS")
        with tracer.start_as_current_span("customer_resolution_span") as s4:
            _activity_attrs(s4, "generate_customer_resolution", "CustomerResolutionAgent", execution_id)
            with tracer.start_as_current_span("tool_escalate_to_fraud_queue") as t3:
                _tool_span(t3, "escalate_to_fraud_queue", {"priority": 0}, {"queue_position": 2847})
            resolution = {"status": "ESCALATED"}
            s4.set_attribute("gen_ai.activity.result", json.dumps(resolution))
            s4.set_attribute("gen_ai.activity.status", "SUCCESS")
            root.set_attribute("gen_ai.activity.status", "SUCCESS")
    return {"scenario": "J_fraud_queue_priority_zero", "execution_id": execution_id, "trace_id": trace_id, "claim_id": claim_id, "status": "ESCALATED", "approved_amount": 0.0, "defect": "Priority 0"}

async def scenario_k_user_cancelled_stop(tracer: Any, execution_id: str) -> dict:
    claim_id = "CLM-7011"
    with tracer.start_as_current_span("ecommerce_claims_triage_workflow_span") as root:
        _set_root(root, execution_id)
        trace_id = format(root.get_span_context().trace_id, "032x")
        with tracer.start_as_current_span("wf_vendor_invoice_reconciliation") as s1:
            _activity_attrs(s1, "intake_and_classify_claim", "FAQIntakeAgent", execution_id)
            s1.set_attribute("gen_ai.activity.status", "SUCCESS")
        
        # User cancels during verification
        with tracer.start_as_current_span("verify_tools_dispatch_span") as s2:
            _activity_attrs(s2, "verify_order_and_inventory_tools", "VerificationToolAgent", execution_id)
            s2.set_attribute("gen_ai.activity.status", "CANCELLED")
            s2.add_event("User initiated abort via UI")
            
        root.set_attribute("gen_ai.activity.status", "CANCELLED")
        root.set_status(Status(StatusCode.OK)) # Cancelled is not an error in traces, but status string is CANCELLED
    return {"scenario": "K_user_cancelled_stop", "execution_id": execution_id, "trace_id": trace_id, "claim_id": claim_id, "status": "CANCELLED", "approved_amount": 0.0, "defect": "Natural stop"}

async def scenario_l_system_termination(tracer: Any, execution_id: str) -> dict:
    claim_id = "CLM-7012"
    with tracer.start_as_current_span("ecommerce_claims_triage_workflow_span") as root:
        _set_root(root, execution_id)
        trace_id = format(root.get_span_context().trace_id, "032x")
        with tracer.start_as_current_span("wf_vendor_invoice_reconciliation") as s1:
            _activity_attrs(s1, "intake_and_classify_claim", "FAQIntakeAgent", execution_id)
            s1.set_attribute("gen_ai.activity.status", "SUCCESS")
        
        # System forcefully terminates the process (e.g. OOM or hard timeout)
        with tracer.start_as_current_span("verify_tools_dispatch_span") as s2:
            _activity_attrs(s2, "verify_order_and_inventory_tools", "VerificationToolAgent", execution_id)
            s2.set_attribute("gen_ai.activity.status", "TERMINATION")
            s2.set_status(Status(StatusCode.ERROR, "SIGTERM received, hard termination"))
            
        root.set_attribute("gen_ai.activity.status", "TERMINATION")
        root.set_status(Status(StatusCode.ERROR, "Process Terminated"))
    return {"scenario": "L_system_termination", "execution_id": execution_id, "trace_id": trace_id, "claim_id": claim_id, "status": "TERMINATION", "approved_amount": 0.0, "defect": "Force stop"}

async def scenario_m_workflow_crash(tracer: Any, execution_id: str) -> dict:
    claim_id = "CLM-7013"
    with tracer.start_as_current_span("ecommerce_claims_triage_workflow_span") as root:
        _set_root(root, execution_id)
        trace_id = format(root.get_span_context().trace_id, "032x")
        with tracer.start_as_current_span("wf_vendor_invoice_reconciliation") as s1:
            _activity_attrs(s1, "intake_and_classify_claim", "FAQIntakeAgent", execution_id)
            s1.set_attribute("gen_ai.activity.status", "SUCCESS")
        
        # Workflow throws unhandled exception
        with tracer.start_as_current_span("verify_tools_dispatch_span") as s2:
            _activity_attrs(s2, "verify_order_and_inventory_tools", "VerificationToolAgent", execution_id)
            exc = ValueError("Null pointer in inventory schema mapping")
            s2.record_exception(exc)
            s2.set_attribute("gen_ai.activity.status", "FAILED")
            s2.set_status(Status(StatusCode.ERROR, str(exc)))
            
        root.set_attribute("gen_ai.activity.status", "FAILED")
        root.set_status(Status(StatusCode.ERROR, "Unhandled exception in workflow"))
    return {"scenario": "M_workflow_crash", "execution_id": execution_id, "trace_id": trace_id, "claim_id": claim_id, "status": "FAILED", "approved_amount": 0.0, "defect": "Crash"}

STRESS_SCENARIOS = {
    "A_cache_poisoning": scenario_a_cache_poisoning,
    "B_risk_threshold_bypass": scenario_b_risk_threshold_bypass,
    "C_sku_substitution": scenario_c_sku_substitution,
    "D_float_ledger_drift": scenario_d_float_ledger_drift,
    "E_expired_policy_clause": scenario_e_expired_policy_clause,
    "F_compliance_exception_swallow": scenario_f_compliance_exception_swallow,
    "G_urgency_sla_downgrade": scenario_g_urgency_sla_downgrade,
    "H_duplicate_refund_stale_read": scenario_h_duplicate_refund_stale_read,
    "I_execution_id_collision": scenario_i_execution_id_collision,
    "J_fraud_queue_priority_zero": scenario_j_fraud_queue_priority_zero,
    "K_user_cancelled_stop": scenario_k_user_cancelled_stop,
    "L_system_termination": scenario_l_system_termination,
    "M_workflow_crash": scenario_m_workflow_crash,
}
