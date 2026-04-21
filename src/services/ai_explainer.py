from src.services.bedrock_llm import BedrockLLM

class AIExplainer:
    def __init__(self):
        self.llm = BedrockLLM()

    def explain(self, logs, rca_summary, kb_fixes):
        context = f"""
Logs Summary:
{logs}

RCA Summary:
{rca_summary}

KB Fixes:
{kb_fixes}
"""

        prompt = f"""
You are an SRE expert.
Explain the root cause in simple terms and suggest fixes.

Context:
{context}

Answer:
"""

        return self.llm.generate(prompt)
