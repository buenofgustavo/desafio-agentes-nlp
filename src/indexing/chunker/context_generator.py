# src/ingestion/context_generator.py
import asyncio
import os
from anthropic import AsyncAnthropic, RateLimitError, InternalServerError, APIConnectionError
from typing import List, Tuple
from dataclasses import dataclass
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from src.utils.logger import LoggingService

logger = LoggingService.setup_logger(__name__)

@dataclass
class ContextRequest:
    parent_text: str
    child_text: str
    index: int

class ContextGenerator:
    # 1. MELHORIA: O Prompt agora é otimizado para o padrão Anthropic
    SYSTEM_PROMPT = """\
    Você é um assistente jurídico focado no setor elétrico brasileiro (ANEEL).
    Sua tarefa é ler um trecho de um documento maior e um pequeno fragmento (chunk) retirado dele.
    Escreva de 1 a 2 frases curtas (máximo 35 palavras) em Português que situe o fragmento.
    Exemplo: "Este trecho define os critérios de cálculo da tarifa de distribuição (TUSD) no contexto da Resolução 414."
    Não resuma o fragmento. Responda APENAS com a frase, sem introduções."""

    USER_PROMPT_TEMPLATE = """\
    <documento>
    {parent_text}
    </documento>

    <fragmento>
    {child_text}
    </fragmento>"""

    CONCURRENCY = 20 
    MODEL = "claude-3-5-haiku-latest"
    MAX_TOKENS = 80

    def __init__(self, api_key: str = None):
        self.client = AsyncAnthropic(api_key=api_key or os.getenv("ANTHROPIC_API_KEY"))

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        retry=retry_if_exception_type((RateLimitError, InternalServerError, APIConnectionError)),
        before_sleep=lambda retry_state: logger.warning(f"Rate limit atingido. Tentando novamente (Tentativa {retry_state.attempt_number})...")
    )
    async def _call_anthropic(self, req: ContextRequest) -> str:
        message = await self.client.messages.create(
            model=self.MODEL,
            max_tokens=self.MAX_TOKENS,
            system=self.SYSTEM_PROMPT,
            messages=[{
                "role": "user",
                "content": self.USER_PROMPT_TEMPLATE.format(
                    parent_text=req.parent_text,
                    child_text=req.child_text,
                )
            }]
        )
        return message.content[0].text.strip()

    async def _generate_one(self, req: ContextRequest, semaphore: asyncio.Semaphore) -> Tuple[int, str]:
        async with semaphore:
            try:
                context = await self._call_anthropic(req)
                return req.index, context
            except Exception as e:
                logger.error(f"Falha definitiva ao gerar contexto para o chunk {req.index}: {e}")
                return req.index, ""

    async def generate_contexts_async(self, requests: List[ContextRequest]) -> List[str]:
        semaphore = asyncio.Semaphore(self.CONCURRENCY)
        tasks = [self._generate_one(req, semaphore) for req in requests]
        results = await asyncio.gather(*tasks)

        ordered = [""] * len(requests)
        for index, context in results:
            ordered[index] = context

        return ordered

    def generate_contexts(self, requests: List[ContextRequest]) -> List[str]:
        return asyncio.run(self.generate_contexts_async(requests))