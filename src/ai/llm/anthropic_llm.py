# src/ai/llm/anthropic_llm.py

import os
from typing import Optional
from anthropic import AsyncAnthropic, Anthropic

from src.ai.llm.base_llm import BaseLLM


class AnthropicLLM(BaseLLM):

    def __init__(self, model: str = "claude-haiku-4-5"):
        super().__init__(model)
        api_key = os.getenv("ANTHROPIC_API_KEY")

        self.client = Anthropic(api_key=api_key)
        self.async_client = AsyncAnthropic(api_key=api_key)

    # ✅ versão sync
    def generate(
        self,
        prompt: str,
        system_prompt: str = "",
        temperature: float = 0.1,
        max_tokens: Optional[int] = 100,
    ) -> str:

        response = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[{
                "role": "user",
                "content": f"{system_prompt}\n\n{prompt}"
            }]
        )

        return response.content[0].text.strip()

    # 🚀 versão async
    async def agenerate(
        self,
        prompt: str,
        system_prompt: str = "",
        temperature: float = 0.1,
        max_tokens: Optional[int] = 100,
    ) -> str:

        response = await self.async_client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[{
                "role": "user",
                "content": f"{system_prompt}\n\n{prompt}"
            }]
        )

        return response.content[0].text.strip()