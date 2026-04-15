"""Manipulação e extração de arquivos compactados RAR."""
import rarfile

from pathlib import Path
from src.utils.logger import LoggingService
from src.utils.file_utils import FileManager

logger = LoggingService.setup_logger(__name__)

class RarExtractor:
    """Classe responsável por extrair arquivos RAR."""

    @staticmethod
    def extract_rar_file(rar_path: Path, extract_to: Path) -> None:
        """Extrai um arquivo RAR para um diretório especificado."""
        if not rar_path.exists() or not rar_path.is_file():
            logger.warning(f"Arquivo RAR não encontrado: {rar_path}")
            return

        extract_to.mkdir(parents=True, exist_ok=True)

        try:
            with rarfile.RarFile(rar_path, 'r') as rar_ref:
                for file_info in rar_ref.infolist():
                    extracted_path = extract_to / file_info.filename

                    try:
                        rar_ref.extract(file_info, extract_to)
                        logger.debug(f"Arquivo extraído '{file_info.filename}' para '{extract_to}'")
                    except rarfile.BadRarFile as e:
                        logger.warning(f"Erro de integridade (corrompido) no arquivo '{file_info.filename}': {e}")
                        FileManager.remove_path(extracted_path)
                    except Exception as e:
                        logger.error(f"Erro inesperado ao extrair '{file_info.filename}': {e}")

        except rarfile.Error as e:
            logger.error(f"Erro ao abrir arquivo RAR '{rar_path}': {e}")