"""Utilitários para manipulação de arquivos e diretórios."""

import filetype  # type: ignore
import json
import re
import shutil
from pathlib import Path
from typing import List, Dict, Any

from src.utils.logger import LoggingService

logger = LoggingService.setup_logger(__name__)

class FileManager:
    """Classe responsável pelo gerenciamento de arquivos do projeto."""

    @staticmethod
    def get_file_name(file_path: Path) -> str:
        """Retorna o nome do arquivo sem a extensão."""
        return file_path.stem

    @staticmethod
    def guess_file_extension(file_path: Path) -> str:
        """Tenta adivinhar a extensão de um arquivo sem extensão baseado no conteúdo."""
        try:
            kind = filetype.guess(file_path)
            if kind is not None:
                return f".{kind.extension}"
        except Exception as e:
            logger.warning(f"Erro ao tentar adivinhar extensão do arquivo {file_path.name}: {e}")
        return ""

    @staticmethod
    def get_file_extension(file_path: Path) -> str:
        """Retorna a extensão do arquivo em letras minúsculas sem espaços."""
        return file_path.suffix.lower().replace(" ", "")

    @staticmethod
    def verify_file_has_content(file_path: Path) -> bool:
        """Verifica se o arquivo existe e possui conteúdo (tamanho > 0)."""
        return file_path.stat().st_size > 0

    @staticmethod
    def verify_file_exists(file_path: Path) -> bool:
        """Verifica se o caminho existe e é um arquivo."""
        return file_path.exists() and file_path.is_file()

    @staticmethod
    def verify_folder_exists(folder_path: Path) -> bool:
        """Verifica se o caminho existe e é um diretório."""
        return folder_path.exists() and folder_path.is_dir()

    @staticmethod
    def remove_path(path: Path) -> None:
        """Remove um arquivo ou diretório de forma recursiva."""
        try:
            if path.exists():
                if path.is_file():
                    path.unlink()
                elif path.is_dir():
                    shutil.rmtree(path)
                logger.info(f"Caminho removido: {path}")
        except Exception as e:
            logger.error(f"Erro ao remover caminho {path}: {e}")