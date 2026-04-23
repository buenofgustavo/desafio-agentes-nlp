# src/ai/llm/factory.py

from typing import Optional, Dict, Tuple

from src.ai.llm.base_llm import BaseLLM
from src.ai.llm.openai_llm import OpenAILLM
from src.ai.llm.anthropic_llm import AnthropicLLM
from src.ai.llm.ollama_llm import OllamaLLM

from src.utils.logger import LoggingService

logger = LoggingService.setup_logger(__name__)


# cache: (provider, model) -> instância
_instances: Dict[Tuple[str, str], BaseLLM] = {}


def get_llm(provider, model) -> BaseLLM:

    """
    Retorna uma instância de LLM com cache por provider + model.
    """

    if provider is None or provider == "":
        raise ValueError("provider é obrigatório")

    if model is None or model == "":
        raise ValueError("model é obrigatório")

    key = (provider, model)
    logger.info(f'Carregando LLM: {key}')
    
    if key not in _instances:

        if provider == "openai":
            _instances[key] = OpenAILLM(model=model)

        elif provider == "anthropic":
            _instances[key] = AnthropicLLM(model=model)

        elif provider == "ollama":
            _instances[key] = OllamaLLM(model=model)
            
        else:
            raise ValueError(f"Provider não suportado: {provider}")

    return _instances[key]