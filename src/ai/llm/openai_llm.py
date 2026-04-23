# src/ai/llm/openai_llm.py

from openai import OpenAI
from typing import Optional

from src.core.config import Constants
from src.ai.llm.base_llm import BaseLLM


class OpenAILLM(BaseLLM):

    def __init__(self, model: Optional[str] = None):
        if not Constants.OPENAI_API_KEY:
            raise RuntimeError("OPENAI_API_KEY não configurada.")

        super().__init__(model or Constants.OPENAI_CHAT_MODEL)

        self.client = OpenAI(api_key=Constants.OPENAI_API_KEY)

    def generate(
        self,
        prompt: str,
        system_prompt: str = "",
        temperature: float = 0.1,
        max_tokens: Optional[int] = None,
    ) -> str:

        completion = self.client.chat.completions.create(
            model=self.model,
            temperature=temperature,
            max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
        )

        return completion.choices[0].message.content or ""