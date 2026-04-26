"""Módulo de expansão de query: HyDE + reformulação de query.

Extraído em uma classe independente para que possa ser testado
de forma independente do agente LangGraph.
"""
from __future__ import annotations

import json
import re

from src.ai.llm.base_llm import BaseLLM
from src.agent.prompts import HYDE_PROMPT, QUERY_REFORMULATION_PROMPT
from src.core.config import Constants
from src.utils.logger import LoggingService

logger = LoggingService.setup_logger(__name__)


def _parse_json_safely(raw: str) -> object:
    """Remove blocos de markdown e analisa o JSON, retornando None em caso de falha."""
    cleaned = re.sub(r"```(?:json)?\s*", "", raw).strip().rstrip("`")
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        logger.debug("Resposta bruta do LLM que falhou na análise JSON: %s", raw)
        return None


class QueryExpander:
    """Gera documentos hipotéticos HyDE e reformulações de query.

    Args:
        llm_client: Qualquer implementação de ``BaseLLM`` (ex: ``AnthropicLLM``).
    """

    def __init__(self, llm_client: BaseLLM) -> None:
        self._llm = llm_client

    def generate_hyde_document(self, query: str) -> str:
        """Gera um trecho de documento hipotético para a query fornecida.

        Args:
            query: A pergunta original do usuário.

        Returns:
            Um trecho hipotético de 2-3 parágrafos, ou ``""`` em caso de falha.
        """
        prompt = HYDE_PROMPT.format(query=query)
        try:
            hyde_doc = self._llm.generate(
                prompt=prompt,
                temperature=0.1,
                max_tokens=Constants.LLM_MAX_TOKENS,
            )
            logger.debug(
                "Documento HyDE gerado (%d caracteres)", len(hyde_doc)
            )
            return hyde_doc.strip()
        except Exception:
            logger.error("Falha ao gerar documento HyDE", exc_info=True)
            raise

    def generate_reformulations(self, query: str, n: int) -> list[str]:
        """Gera *n* reformulações da query.

        Args:
            query: A pergunta original do usuário.
            n: Número de reformulações a serem produzidas.

        Returns:
            Uma lista de queries reformuladas. Retorna ``[query]``
            se o LLM retornar uma saída impossível de analisar.
        """
        prompt = QUERY_REFORMULATION_PROMPT.format(query=query, n=n)
        try:
            raw = self._llm.generate(
                prompt=prompt,
                temperature=0.3,
                max_tokens=Constants.LLM_MAX_TOKENS,
            )
        except Exception:
            logger.error("Falha ao gerar reformulações", exc_info=True)
            raise

        parsed = _parse_json_safely(raw)
        if isinstance(parsed, list) and all(isinstance(s, str) for s in parsed):
            logger.debug("Reformulações geradas: %d", len(parsed))
            return parsed[:n]

        logger.warning(
            "Resposta de reformulação inválida — usando query original. "
            "Bruto: %s",
            raw[:200],
        )
        return [query]

    def expand(self, query: str) -> tuple[str, list[str]]:
        """Executa o pipeline completo de expansão.

        Returns:
            Uma tupla ``(hyde_document, reformulations)``.
            Se ``HYDE_ENABLED`` for ``False``, retorna ``("", [query])``.
        """
        if not Constants.HYDE_ENABLED:
            logger.info("HyDE desabilitado — retornando query original")
            return "", [query]

        hyde_doc = self.generate_hyde_document(query)
        reformulations = self.generate_reformulations(
            query, Constants.QUERY_REFORMULATIONS
        )
        return hyde_doc, reformulations
