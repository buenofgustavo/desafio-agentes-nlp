"""Manipulação e extração de arquivos compactados ZIP."""
import zipfile

from pathlib import Path
from src.utils.logger import LoggingService
from src.utils.file_utils import FileManager

logger = LoggingService.setup_logger(__name__)

class ZipExtractor:
    """Classe responsável por extrair arquivos ZIP."""

    @staticmethod
    def extract_zip_file(zip_path: Path, extract_to: Path) -> None:
        """Extrai um arquivo ZIP para um diretório especificado."""
        if not zip_path.exists() or not zip_path.is_file():
            logger.warning(f"Arquivo ZIP não encontrado: {zip_path}")
            return

        extract_to.mkdir(parents=True, exist_ok=True)
        
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            for file_info in zip_ref.infolist():
                extracted_path = extract_to / file_info.filename

                try:
                    zip_ref.extract(file_info, extract_to)
                    logger.debug(f"Arquivo extraído '{file_info.filename}' para '{extract_to}'")
                except zipfile.BadZipFile as e:
                    logger.warning(f"Erro de integridade (CRC/Corrompido) no arquivo '{file_info.filename}': {e}")
                    FileManager.remove_path(extracted_path)
                except Exception as e:
                    logger.error(f"Erro inesperado ao extrair '{file_info.filename}': {e}")

    