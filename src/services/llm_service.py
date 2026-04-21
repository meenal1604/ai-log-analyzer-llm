import streamlit as st
from openai import OpenAI


class LLMService:
    def __init__(self):
        # ✅ client is created ONCE here
        self.client = OpenAI(
            api_key=st.secrets["OPENAI_API_KEY"]
        )

    def generate_rca(self, query: str, logs: str, kb_context: str) -> str:
        prompt = f"""
You are an AI Production Support Engineer.

Analyze the following logs and knowledge base context
and generate a clear Root Cause Analysis.

User Query:
{query}

Relevant Logs:
{logs}

Knowledge Base Context:
{kb_context}

Provide:
1. Root Cause
2. Impact
3. Recommended Fix
4. Prevention
"""

        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0.3
        )

        return response.choices[0].message.content
