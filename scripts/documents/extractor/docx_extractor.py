"""Extração de texto de arquivos Microsoft Word (DOCX)."""
from typing import Any, Iterator, Union
from docx import Document
from docx.table import Table
from docx.text.paragraph import Paragraph
from docx.oxml.ns import qn
from pathlib import Path

from src.utils.logger import LoggingService

logger = LoggingService.setup_logger(__name__)


class DocxExtractor:
    """Classe responsável por extrair texto de arquivos DOCX.
    
    Percorre o corpo do documento na ordem original, intercalando
    parágrafos e tabelas para preservar o contexto posicional.
    """

    @staticmethod
    def _iter_block_items(document: Any) -> Iterator[Union[Paragraph, Table]]:
        """Itera sobre os elementos do corpo do documento na ordem em que aparecem.
        
        Yields objetos Paragraph e Table na sequência correta do documento.
        """
        body = document.element.body
        for child in body.iterchildren():
            if child.tag == qn('w:p'):
                yield Paragraph(child, body)
            elif child.tag == qn('w:tbl'):
                yield Table(child, body)

    @classmethod
    def extract_text_from_docx(cls, file_path: Path) -> str:
        """Extrai texto de um arquivo DOCX preservando a ordem de parágrafos e tabelas."""
        try:
            document = Document(str(file_path))

            logger.debug(f"Começando extração de documento DOCX: {file_path.name}")

            document_rows = []
            paragraph_count = 0
            table_rows_count = 0

            for block in cls._iter_block_items(document):
                if isinstance(block, Paragraph):
                    text = block.text.strip()
                    if text:
                        document_rows.append(text)
                        paragraph_count += 1
                elif isinstance(block, Table):
                    for row in block.rows:
                        cells = [cell.text.strip() for cell in row.cells if cell.text.strip()]
                        if cells:
                            document_rows.append(" | ".join(cells))
                            table_rows_count += 1

            full_text = "\n".join(document_rows)

            logger.debug(
                f"Extração finalizada: {file_path.name} | "
                f"parágrafos={paragraph_count}, linhas_tabelas={table_rows_count}, "
                f"total_linhas={len(document_rows)}"
            )

            if not full_text.strip():
                logger.warning(f"Nenhum texto extraído de: {file_path.name}")

            return full_text
        except Exception as e:
            logger.error(f"Erro ao extrair texto do DOCX {file_path.name}: {e}")
            return ""