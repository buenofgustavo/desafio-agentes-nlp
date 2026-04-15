"""Extração de texto de planilhas (XLSX, XLSM) utilizando pandas."""
import pandas as pd
from pathlib import Path
from typing import Literal

from src.utils.logger import LoggingService
from src.utils.file_utils import FileManager

logger = LoggingService.setup_logger(__name__)

class SpreadsheetExtractor:
    """Classe responsável por extrair texto de arquivos de planilha (XLSX, XLSM) utilizando pandas."""

    SPREADSHEET_ENGINE_MAP = {
        ".xlsx": "openpyxl",
        ".xlsm": "openpyxl",
    }
    
    @staticmethod
    def extract_text_from_spreadsheet(file_path: Path, file_extension: Literal[".xlsx", ".xlsm"]) -> str:
        """Extrai texto de um arquivo de planilha. Retorna o texto extraído ou uma string vazia em caso de falha."""
        if not FileManager.verify_file_exists(file_path):
            logger.warning(f"Arquivo de planilha não encontrado: {file_path}")
            return ""

        if not FileManager.verify_file_has_content(file_path):
            logger.warning(f"Arquivo vazio (0 bytes) ignorado: {file_path.name}")
            return ""

        engine = SpreadsheetExtractor.SPREADSHEET_ENGINE_MAP.get(file_extension)

        try:
            sheets = pd.read_excel(file_path, sheet_name=None, header=None, engine=engine)  # type: ignore
            logger.debug(f"Planilha lida com sucesso: {file_path} | ext: {file_extension}")
        except Exception as e:
            logger.error(f"Pandas falhou na leitura da planilha: {file_path}\n Erro: {e}")
            return ""

        lines = []
        for sheet_name, dataframe in sheets.items():
            dataframe = dataframe.fillna("").astype(str)

            for _, row in dataframe.iterrows():
                columns = [col.strip() for col in row if col.strip()]
                if columns:
                    lines.append(" | ".join(columns))

        if not lines:
            logger.warning(f"Nenhum texto extraído da planilha: {file_path}")
            return ""

        return "\n".join(lines)
        