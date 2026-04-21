import boto3
import os
from dotenv import load_dotenv

load_dotenv()

print("Region:", os.getenv("AWS_DEFAULT_REGION"))

client = boto3.client(
    service_name="bedrock",
    region_name=os.getenv("AWS_DEFAULT_REGION")
)

response = client.list_foundation_models()

print("✅ Bedrock is accessible")
print("Available models count:", len(response.get("modelSummaries", [])))


