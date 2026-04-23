import requests
from typing import Optional

from .base_llm import BaseLLM


class OllamaLLM(BaseLLM):

    def __init__(self, model: str = "llama3"):
        super().__init__(model)

    def generate(
        self,
        prompt: str,
        system_prompt: str = "",
        temperature: float = 0.1,
        max_tokens: Optional[int] = None,
    ) -> str:

        payload = {
            "model": self.model,
            "prompt": f"{system_prompt}\n\n{prompt}",
            "stream": False,
            "options": {
                "temperature": temperature
            },
        }

        if max_tokens:
            payload["options"]["num_predict"] = max_tokens

        response = requests.post(
            "http://localhost:11434/api/generate",
            json=payload,
            timeout=60,
        )

        response.raise_for_status()

        return response.json().get("response", "")