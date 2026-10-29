from __future__ import annotations
from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

class CancellationReason(str, Enum):
    PERSONAL = "personal"
    WEATHER = "weather"
    FLIGHT_DELAY = "flight_delay"
    OTHER = "other"

class HotelCancellationInput(BaseModel):
    request_id: str = Field(description="Unique request identifier")
    booking_id: str = Field(description="Booking ID, e.g., BK-10293")
    customer_id: str = Field(description="Customer ID, e.g., CUST-999")
    reason: str = Field(description="Reason for cancellation")
    customer_message: str = Field(description="Customer message")

class CancellationClassification(BaseModel):
    category: CancellationReason
    is_eligible_for_refund: bool
    penalty_applicable: bool

class ToolExecutionResult(BaseModel):
    tool_name: str
    arguments: Dict[str, Any]
    status: str
    output: Dict[str, Any]
    error: Optional[str] = None

class CancellationReport(BaseModel):
    request_id: str
    status: str
    refund_amount: float
    message: str
