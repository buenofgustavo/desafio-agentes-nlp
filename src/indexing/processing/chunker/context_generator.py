# src/ingestion/context_generator.py
import asyncio
import os
from anthropic import AsyncAnthropic
from typing import List, Tuple
from dataclasses import dataclass


@dataclass
class ContextRequest:
    parent_text: str
    child_text: str
    index: int


class ContextGenerator:
    CONTEXT_PROMPT = """\
        You will receive a document excerpt and a specific chunk from that excerpt.
        Write a single short sentence (max 25 words) in Portuguese that describes
        what the chunk is about and where it fits within the document.
        Do NOT summarize the chunk — just situate it.
        Respond with the sentence only, no preamble.
        
        <document>
        {parent_text}
        </document>
        
        <chunk>
        {child_text}
        </chunk>"""

    CONCURRENCY = 20
    MODEL = "claude-haiku-4-5-20251001"
    MAX_TOKENS = 80

    def __init__(self, api_key: str = None):
        self.client = AsyncAnthropic(api_key=api_key or os.getenv("ANTHROPIC_API_KEY"))

    async def _generate_one(self, req: ContextRequest, semaphore: asyncio.Semaphore) -> Tuple[int, str]:
        async with semaphore:
            try:
                message = await self.client.messages.create(
                    model=self.MODEL,
                    max_tokens=self.MAX_TOKENS,
                    messages=[{
                        "role": "user",
                        "content": self.CONTEXT_PROMPT.format(
                            parent_text=req.parent_text[:2000],
                            child_text=req.child_text,
                        )
                    }]
                )
                return req.index, message.content[0].text.strip()
            except Exception as e:
                print(f"  ⚠ Context generation failed for chunk {req.index}: {e}")
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
