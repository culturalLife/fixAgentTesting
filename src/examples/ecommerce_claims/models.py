from __future__ import annotations
from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class ClaimType(str, Enum):
    REFUND = "refund"
    REPLACEMENT = "replacement"
    INSPECTION = "inspection"
    FRAUD_SUSPECT = "fraud_suspect"


class UrgencyLevel(str, Enum):
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"


class CustomerClaimInput(BaseModel):
    claim_id: str = Field(description="Unique claim identifier, e.g., CLM-9021")
    customer_id: str = Field(description="Customer ID, e.g., CUST-4412")
    order_id: str = Field(description="Original purchase order ID")
    claim_type: str = Field(description="Claim type or reason: damage, refund, missing")
    claim_amount: float = Field(description="Total requested reimbursement in USD")
    customer_message: str = Field(description="Customer query or description")
    attachments: List[str] = Field(default_factory=list, description="URLs or paths of submitted receipts/photos")


class IntakeClassification(BaseModel):
    claim_category: ClaimType
    urgency: UrgencyLevel
    policy_applicable: str
    requires_warehouse_lookup: bool
    summary: str


class ToolExecutionResult(BaseModel):
    tool_name: str
    arguments: Dict[str, Any]
    status: str
    output: Dict[str, Any]


class ComplianceReasoningResult(BaseModel):
    is_eligible: bool
    risk_score: float = Field(description="Risk assessment score between 0.0 and 1.0")
    applicable_clauses: List[str]
    reasoning_summary: str


class ResolutionReport(BaseModel):
    claim_id: str
    status: str
    action_taken: str
    approved_amount: float
    customer_facing_response: str
    internal_notes: str
