import sys
import json
import asyncio
from typing import List

from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, 'src')

from examples.ecommerce_claims.activities import evaluate_compliance_and_policy
from examples.ecommerce_claims.models import (
    CustomerClaimInput,
    IntakeClassification,
    ToolExecutionResult,
    ComplianceReasoningResult,
    ClaimType,
    UrgencyLevel,
)

async def test_evaluate_compliance_and_policy():
    # Test input set 1: Trace failure scenario
    claim1 = CustomerClaimInput(
        claim_id="CLM-9021",
        customer_id="CUST-4412",
        order_id="ORD-12345",
        claim_type="refund",
        claim_amount=199.99,
        customer_message="Product arrived damaged and doesn't work",
        attachments=[
            "https://example.com/photo1.jpg"
        ]
    )
    
    classification1 = IntakeClassification(
        claim_category=ClaimType.REFUND,
        urgency=UrgencyLevel.HIGH,
        policy_applicable="Standard Return Policy 30-Day",
        requires_warehouse_lookup=True,
        summary="Customer requests refund for damaged product"
    )
    
    tools1 = [
        ToolExecutionResult(
            tool_name="lookup_order_details",
            arguments={"order_id": "ORD-12345", "customer_id": "CUST-4412"},
            status="SUCCESS",
            output={
                "order_id": "ORD-12345",
                "order_date": "2026-07-28",
                "status": "DELIVERED",
                "items": [
                    {
                        "sku": "SKU-9920",
                        "name": "Wireless Headphones",
                        "price": 199.99
                    }
                ],
                "delivery_confirmed": True
            }
        )
    ]
    
    result1 = await evaluate_compliance_and_policy(claim1, classification1, tools1)
    print("Test 1 Result:", result1.model_dump_json(indent=2))
    
    # Test input set 2: Normal case with string reasoning_summary
    claim2 = CustomerClaimInput(
        claim_id="CLM-9022",
        customer_id="CUST-4413",
        order_id="ORD-12346",
        claim_type="replacement",
        claim_amount=149.99,
        customer_message="Item missing from delivery",
        attachments=[]
    )
    
    classification2 = IntakeClassification(
        claim_category=ClaimType.REPLACEMENT,
        urgency=UrgencyLevel.NORMAL,
        policy_applicable="Standard Return Policy 30-Day",
        requires_warehouse_lookup=True,
        summary="Customer requests replacement for missing item"
    )
    
    tools2 = [
        ToolExecutionResult(
            tool_name="lookup_order_details",
            arguments={"order_id": "ORD-12346", "customer_id": "CUST-4413"},
            status="SUCCESS",
            output={
                "order_id": "ORD-12346",
                "order_date": "2026-07-29",
                "status": "DELIVERED",
                "items": [
                    {
                        "sku": "SKU-9921",
                        "name": "Smart Watch",
                        "price": 149.99
                    }
                ],
                "delivery_confirmed": True
            }
        )
    ]
    
    result2 = await evaluate_compliance_and_policy(claim2, classification2, tools2)
    print("Test 2 Result:", result2.model_dump_json(indent=2))
    
    # Test input set 3: Edge case with empty reasoning_summary
    claim3 = CustomerClaimInput(
        claim_id="CLM-9023",
        customer_id="CUST-4414",
        order_id="ORD-12347",
        claim_type="inspection",
        claim_amount=0,
        customer_message="",
        attachments=[]
    )
    
    classification3 = IntakeClassification(
        claim_category=ClaimType.INSPECTION,
        urgency=UrgencyLevel.LOW,
        policy_applicable="Standard Return Policy 30-Day",
        requires_warehouse_lookup=False,
        summary="Empty claim for inspection"
    )
    
    tools3: List[ToolExecutionResult] = []
    
    result3 = await evaluate_compliance_and_policy(claim3, classification3, tools3)
    print("Test 3 Result:", result3.model_dump_json(indent=2))

if __name__ == "__main__":
    asyncio.run(test_evaluate_compliance_and_policy())