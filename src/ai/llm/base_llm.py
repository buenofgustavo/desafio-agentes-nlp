from abc import ABC, abstractmethod
from typing import Optional


class BaseLLM(ABC):
    """
    Interface base para qualquer LLM.
    """

    def __init__(self, model: Optional[str] = None):
        self.model = model

    # obrigatório (sync)
    @abstractmethod
    def generate(
        self,
        prompt: str,
        system_prompt: str = "",
        temperature: float = 0.1,
        max_tokens: Optional[int] = None,
    ) -> str:
        pass

    # opcional (async)
    async def agenerate(
        self,
        prompt: str,
        system_prompt: str = "",
        temperature: float = 0.1,
        max_tokens: Optional[int] = None,
    ) -> str:
        raise NotImplementedError(f"{self.__class__.__name__} não implementa agenerate()")