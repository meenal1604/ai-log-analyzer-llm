import subprocess

class LocalLLM:
    def __init__(self, model="mistral"):
        self.model = model

    def generate(self, prompt: str) -> str:
        try:
            result = subprocess.run(
                ["ollama", "run", self.model],
                input=prompt,
                capture_output=True,
                text=True,
                encoding="utf-8",
                timeout=60
            )
            return result.stdout.strip()
        except Exception as e:
            return f"LLM Error: {str(e)}"
