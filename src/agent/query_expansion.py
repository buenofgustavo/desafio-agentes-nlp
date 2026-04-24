"""Query expansion module: HyDE + query reformulation.

Extracted into a standalone class so it can be tested independently
of the LangGraph agent.
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
    """Strip markdown fences and parse JSON, returning None on failure."""
    cleaned = re.sub(r"```(?:json)?\s*", "", raw).strip().rstrip("`")
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        logger.debug("Raw LLM response that failed JSON parse: %s", raw)
        return None


class QueryExpander:
    """Generates HyDE hypothetical documents and query reformulations.

    Args:
        llm_client: Any ``BaseLLM`` implementation (e.g. ``AnthropicLLM``).
    """

    def __init__(self, llm_client: BaseLLM) -> None:
        self._llm = llm_client

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate_hyde_document(self, query: str) -> str:
        """Generate a hypothetical document passage for the given query.

        Args:
            query: The user's original question.

        Returns:
            A 2-3 paragraph hypothetical passage, or ``""`` on failure.
        """
        prompt = HYDE_PROMPT.format(query=query)
        try:
            hyde_doc = self._llm.generate(
                prompt=prompt,
                temperature=0.1,
                max_tokens=Constants.LLM_MAX_TOKENS,
            )
            logger.debug(
                "HyDE document generated (%d chars)", len(hyde_doc)
            )
            return hyde_doc.strip()
        except Exception:
            logger.error("Falha ao gerar documento HyDE", exc_info=True)
            raise

    def generate_reformulations(self, query: str, n: int) -> list[str]:
        """Generate *n* reformulations of the query.

        Args:
            query: The user's original question.
            n: Number of reformulations to produce.

        Returns:
            A list of reformulated queries. Falls back to ``[query]``
            if the LLM returns unparseable output.
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
            "Raw: %s",
            raw[:200],
        )
        return [query]

    def expand(self, query: str) -> tuple[str, list[str]]:
        """Run the full expansion pipeline.

        Returns:
            A tuple ``(hyde_document, reformulations)``.
            If ``HYDE_ENABLED`` is ``False``, returns ``("", [query])``.
        """
        if not Constants.HYDE_ENABLED:
            logger.info("HyDE desabilitado — retornando query original")
            return "", [query]

        hyde_doc = self.generate_hyde_document(query)
        reformulations = self.generate_reformulations(
            query, Constants.QUERY_REFORMULATIONS
        )
        return hyde_doc, reformulations
