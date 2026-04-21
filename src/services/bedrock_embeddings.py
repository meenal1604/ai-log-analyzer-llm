import json
import os
import boto3
from dotenv import load_dotenv

# Load .env variables
load_dotenv()


class BedrockEmbeddingService:
    """
    AWS Bedrock – Titan Text Embeddings v2
    Produces 1536-dimension embeddings
    """

    def __init__(self):
        self.region = os.getenv("AWS_DEFAULT_REGION", "us-east-1")

        # Correct Titan Embeddings model ID
        self.model_id = "amazon.titan-embed-text-v2:0"

        # Bedrock Runtime client
        self.client = boto3.client(
            service_name="bedrock-runtime",
            region_name=self.region
        )

    def embed_text(self, text: str):
        """
        Convert text to embedding vector
        """

        payload = {
            "inputText": text
        }

        response = self.client.invoke_model(
            modelId=self.model_id,
            body=json.dumps(payload).encode("utf-8"),
            accept="application/json",
            contentType="application/json"
        )

        response_body = json.loads(response["body"].read())

        # Titan v2 response format
        return response_body["embedding"]

