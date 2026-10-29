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

    print("Submitting workflow execution 'hotel-cancellation-workflow'...")
    input_data = {
        "request_id": "REQ-101",
        "booking_id": "BK-10293",
        "customer_id": "CUST-999",
        "reason": "Flight got delayed and missed connection.",
        "customer_message": "My flight was delayed by 10 hours so I cannot make it to the hotel today. I need to cancel and get a refund."
    }

    try:
        result = await client.workflows.execute_workflow_and_wait_async(
            workflow_identifier="hotel-cancellation-workflow",
            input=input_data,
            deployment_name=os.environ.get("DEPLOYMENT_NAME", "default"),
        )
        print("\n=================== WORKFLOW RESULT ===================")
        print(json.dumps(result, indent=2))
    except Exception as exc:
        print("\n=================== WORKFLOW ENCOUNTERED AN ERROR ===================")
        print(f"Error Type: {type(exc).__name__}")
        print(f"Details: {exc}")


if __name__ == "__main__":
    asyncio.run(main())
