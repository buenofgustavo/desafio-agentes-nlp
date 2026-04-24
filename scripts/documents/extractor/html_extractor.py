"""Responsável por extrair texto de arquivos HTML, utilizando BeautifulSoup para parsing e limpeza do conteúdo."""

from pathlib import Path
from bs4 import BeautifulSoup

from src.utils.logger import LoggingService
from src.utils.file_utils import FileManager

logger = LoggingService.setup_logger(__name__)


class HtmlExtractor:
    """Classe responsável por extrair texto de arquivos HTML."""

    @classmethod
    def extract_text_from_html(cls, file_path: Path) -> str:
        """Extrai texto limpo de um arquivo HTML.
        
        Remove as tags <script> e <style> para evitar a inclusão de código no texto
        extraído e utiliza o BeautifulSoup para formatar e limpar o conteúdo.
        """
        if not FileManager.verify_file_exists(file_path):
            logger.warning(f"Arquivo HTML não encontrado: {file_path}")
            return ""

        if not FileManager.verify_file_has_content(file_path):
            logger.warning(f"Arquivo vazio (0 bytes) ignorado: {file_path.name}")
            return ""

        try:
            logger.debug(f"Começando extração de documento HTML: {file_path.name}")

            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                html_content = f.read()

            soup = BeautifulSoup(html_content, "html.parser")

            # Remove tags de script e style, pois não contêm texto útil
            for element in soup(["script", "style"]):
                element.decompose()

            # Extrai o texto, usando quebra de linha como separador
            text = soup.get_text(separator="\n")

            # Limpa espaços em branco excessivos e linhas vazias
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            clean_text = "\n".join(chunk for chunk in chunks if chunk)

            if not clean_text.strip():
                logger.warning(f"Nenhum texto extraído de: {file_path.name}")
                return ""

            logger.debug(f"Extração finalizada: {file_path.name} | tamanho={len(clean_text)}")
            return clean_text

        except Exception as e:
            logger.error(f"Erro ao extrair texto do HTML {file_path.name}: {e}")
            return ""