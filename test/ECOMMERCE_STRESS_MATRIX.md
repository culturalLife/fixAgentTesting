# E-Commerce Claims Stress Test Matrix

This document defines 10 highly sophisticated, production-grade stress test anomalies for the E-Commerce Claims Triage Workflow. These tests verify the robustness of Mistral Observability Judge and telemetry pipelines. They are intentionally designed to mimic subtle, complex errors found in high-scale enterprise microservice environments, bypassing simple pass/fail validations.

| Scenario ID | Name | Subsystem / Defect Type | Expected Trace Signature | Judge Evaluation Challenge |
| :--- | :--- | :--- | :--- | :--- |
| **A** | `cache_poisoning` | State/Cache | Intake returns CACHED containing stale fraud verdict; rest of workflow evaluates as fraud despite tools showing regular delivery. | The judge must detect a semantic conflict between the intake cache output (`fraud_suspect`) and the actual `lookup_order_details` output (`DELIVERED`), recognizing that stale data poisoned the decision. |
| **B** | `risk_threshold_bypass` | Logic/Compliance | Risk score is `0.87` (breached) but Customer Resolution agent approves anyway. | The trace appears successful with status `SUCCESS`. The judge must flag the violation of business rules by cross-referencing `risk_score` against `status: APPROVED`. |
| **C** | `sku_substitution` | Data/Verification | Tool lookup shows `SKU-9920` ordered, but inventory check uses `SKU-9801` (bait-and-switch). | The judge needs to trace arguments across sequential tool calls to notice that the agent hallucinated or substituted the SKU identifier midway through verification. |
| **D** | `float_ledger_drift` | Data/Precision | Approved amount `1249.99 * 0.8` is computed with floating-point drift leading to imprecise financial resolution. | The judge must evaluate numerical invariants and flag the precision issue on the approved amount, rather than just accepting a generic `SUCCESS` status. |
| **E** | `expired_policy_clause` | Logic/Compliance | Agent evaluates a policy valid until `2025-12-31` on a claim dated `2026-09-15`. | The judge must read the text strings in the applicable_clauses array and understand temporal semantics relative to the claim date. |
| **F** | `compliance_exception_swallow` | Control Flow | Compliance step raises `MistralConnectionError` (TimeoutError), but workflow continues and overrides it. | The judge must detect that the critical compliance span failed (`StatusCode.ERROR`), but the root span and final resolution masked it as `PARTIAL` or `SUCCESS` due to a dangerous try-catch swallow. |
| **G** | `urgency_sla_downgrade` | SLA/Priority | Intake identifies claim as `critical` (Safety Recall), but compliance assigns a `5-7 business days` SLA. | The judge must correlate `urgency: critical` across spans with the actual `sla_commitment` strings to identify a downgraded service level. |
| **H** | `duplicate_refund_stale_read` | Concurrency | Order lookup reads state from `2026-08-10`, missing recent refund events, leading to double refund approval. | The judge must flag the `order_snapshot_timestamp` being heavily skewed from the current time, recognizing a stale read concurrency vulnerability. |
| **I** | `execution_id_collision` | State/Concurrency | Two independent claims (`CLM-7009A` and `CLM-7010B`) are incorrectly tagged with the exact same `execution_id` `ecom-stress-00a1b2c3`. | The judge must aggregate the spans and realize there are 8 spans instead of 4, with conflicting inputs under a single trace execution context. |
| **J** | `fraud_queue_priority_zero` | Escalation/Routing | Customer Resolution routes to fraud queue with `priority: 0`, essentially blackholing it. | The judge must flag the specific `arguments: {"priority": 0}` in the `escalate_to_fraud_queue` tool call as an anomalous routing error, despite the tool succeeding. |
| **K** | `user_cancelled_stop` | State/Cancellation | User cancels workflow midway. The root span emits `gen_ai.activity.status = CANCELLED` and `StatusCode.OK`. | The judge must recognize this is a natural, graceful user cancellation, not a failure or unhandled crash. |
| **L** | `system_termination` | Control Flow | System force-kills process (e.g. timeout). Root span emits `gen_ai.activity.status = TERMINATION` and `StatusCode.ERROR`. | The judge must detect that the execution was interrupted by an external system force (hard kill), rather than an internal logic failure. |
| **M** | `workflow_crash` | Logic/Exception | Workflow throws unhandled `ValueError`. Root span emits `gen_ai.activity.status = FAILED` and `StatusCode.ERROR`. | The judge must extract the specific exception (`ValueError`) from the failed span and categorize this as a true internal crash. |

## How to run
```powershell
# Set API keys
$env:MISTRAL_API_KEY="your-mistral-api-key"

# Run the suite
python c:\Users\Himanshu\Downloads\Document-Ai-branch-vm2\backend-v2\trial-workflow\test\run_ecommerce_stress_suite.py
```
This generates `LATEST_ECOMMERCE_STRESS_TRACE_IDS.json`, which can then be passed to a judge to determine whether it detects these complex planted bugs.
