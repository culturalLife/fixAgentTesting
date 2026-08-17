import asyncio
import json
import os
from dotenv import load_dotenv

load_dotenv(override=True)

from mistralai.extra.workflows import WorkflowEncodingConfig, configure_workflow_encoding
from mistralai.workflows.client import get_mistral_client


async def main():
    api_key = os.environ.get("MISTRAL_API_KEY", "")
    client = get_mistral_client(
        api_key=api_key,
        server_url=os.environ.get("SERVER_URL", "https://api.mistral.ai"),
    )
    await configure_workflow_encoding(WorkflowEncodingConfig(), client=client)

    print("Submitting workflow execution 'order-processing-pipeline'...")
    input_data = {
        "order_id": "ORD-99482",
        "customer_id": "CUST-88301",
        "items": [
            {
                "item_id": "ITEM-101",
                "sku": "MED-SUPPLY-A",
                "quantity": 2,
                "unit_price": 45.00,
                "is_tax_exempt": True,
                "category_code": "MED"
            },
            {
                "item_id": "ITEM-102",
                "sku": "GIFT-CARD-B",
                "quantity": 1,
                "unit_price": 50.00,
                "is_tax_exempt": True,
                "category_code": "GFT"
            }
        ],
        "promo_code": "SUMMER_SALE_20",
        "shipping_country": "US"
    }

    try:
        result = await client.workflows.execute_workflow_and_wait_async(
            workflow_identifier="order-processing-pipeline",
            input=input_data,
            deployment_name=os.environ.get("DEPLOYMENT_NAME", "default"),
        )
        print("\n=================== WORKFLOW RESULT ===================")
        print(json.dumps(result, indent=2))
    except Exception as exc:
        print("\n=================== WORKFLOW CRASHED AS EXPECTED ===================")
        print(f"Error Type: {type(exc).__name__}")
        print(f"Details: {exc}")


if __name__ == "__main__":
    asyncio.run(main())
