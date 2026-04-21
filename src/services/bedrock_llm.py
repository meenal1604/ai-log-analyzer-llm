import boto3
import json
import os

class BedrockLLM:
    def __init__(self):
        self.client = boto3.client(
            "bedrock-runtime",
            region_name=os.getenv("AWS_DEFAULT_REGION")
        )
        self.model_id = "meta.llama3-8b-instruct-v1:0"

    def generate(self, prompt: str) -> str:
        body = {
            "prompt": prompt,
            "max_gen_len": 512,
            "temperature": 0.3,
            "top_p": 0.9
        }

        response = self.client.invoke_model(
            modelId=self.model_id,
            body=json.dumps(body),
            contentType="application/json",
            accept="application/json"
        )

        result = json.loads(response["body"].read())
        return result.get("generation", "")
