"""Responsável por baixar os arquivos PDFs a partir dos links presentes nos arquivos JSON e armazená-los na pasta designada."""
import logging
import random
import time
from curl_cffi import requests as cffi_requests
from urllib.parse import urlparse
from typing import List
from pathlib import Path
from src.core.models import AneelRecord
from src.core.config import Constants

class PdfDownloader:
    """Classe responsável pelo download dos arquivos PDF."""

    BROWSER_IMPERSONATIONS = ["chrome110", "chrome107", "chrome104", "chrome101"]

    HEADERS = {
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,application/pdf,*/*;q=0.8",
        "Accept-Language": "pt-BR,pt;q=0.9,en-US;q=0.8,en;q=0.7",
        "Accept-Encoding": "gzip, deflate, br",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1",
        "Sec-Fetch-Dest": "document",
        "Sec-Fetch-Mode": "navigate",
        "Sec-Fetch-Site": "same-origin",
        "Sec-Fetch-User": "?1",
    }

    @classmethod
    def _create_resilient_session(cls) -> cffi_requests.Session:
        """Cria sessão com impersonation de browser real (bypass de TLS fingerprinting)."""
        session = cffi_requests.Session(
            impersonate=random.choice(cls.BROWSER_IMPERSONATIONS),
        )
        session.headers.update(cls.HEADERS)
        return session

    @staticmethod
    def _verify_is_pdf(response: cffi_requests.Response) -> bool:
        """Verifica se a response contém um PDF como conteúdo (utiliza headers e content na verificação)"""
        content_type = response.headers.get("Content-Type", "").lower()        
        return "application/pdf" in content_type and response.content.startswith(b'%PDF-')
        
    @staticmethod
    def _base_url(url: str) -> str:
        """Extrai a origem (scheme + host) de qualquer URL."""
        parsed = urlparse(url)
        return f"{parsed.scheme}://{parsed.netloc}"

    @staticmethod
    def _parent_url(url: str) -> str:
        """Retorna a URL do diretório pai (ex: .../cedoc/file.pdf → .../cedoc/)."""
        parsed = urlparse(url)
        parent = str(Path(parsed.path).parent) + "/"
        return f"{parsed.scheme}://{parsed.netloc}{parent}"

    @classmethod
    def _warm_up_session(cls, session: cffi_requests.Session, target_url: str) -> None:
        """
        Simula navegação real: visita homepage → diretório pai → URL alvo.
        Isso garante que o servidor emita cookies de sessão válidos.
        """
        base = cls._base_url(target_url)
        parent = cls._parent_url(target_url)

        steps = [
            (base, {"Referer": "", "Sec-Fetch-Site": "none"}),
            (parent, {"Referer": base, "Sec-Fetch-Site": "same-origin"}),
        ]

        for url, extra_headers in steps:
            try:
                session.get(url=url, timeout=15, headers=extra_headers)
                time.sleep(random.uniform(0.5, 1.5))
            except Exception:
                pass  # Warm-up falhou parcialmente — tenta continuar mesmo assim

    @classmethod
    def download_pdfs(cls, records: List[AneelRecord], output_dir: Path | str = Constants.PDFS_DIR, skip_existing: bool = True) -> None:
        """Faz o download dos arquivos PDF para pasta de destino"""
        if isinstance(output_dir, str):
            output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        sessions: dict[str, cffi_requests.Session] = {}

        for record in records:
            for pdf in record.pdfs:
                target_path = output_dir / pdf.arquivo

                if skip_existing and target_path.exists():
                    continue
                
                host = cls._base_url(pdf.url)
                # Reutiliza sessão existente por host ou cria uma nova
                if host not in sessions:
                    sessions[host] = cls._create_resilient_session()
                    cls._warm_up_session(sessions[host], pdf.url)

                session = sessions[host]

                try:
                    delay = random.uniform(1.5, 3.5)
                    time.sleep(delay)

                    response = session.get(url=pdf.url, headers={"Referer": host}, timeout=15)
                    response.raise_for_status()

                    if not cls._verify_response_contains_pdf(response):
                        logging.warning(f"Arquivo {pdf.arquivo} não é um PDF válido!")
                        continue

                    target_path.write_bytes(response.content)
                    logging.info(f"PDF salvo: {pdf.arquivo}")

                except cffi_requests.exceptions.HTTPError as e:
                    logging.error(f"HTTP error no download do PDF {pdf.arquivo}: {e}")
                except cffi_requests.exceptions.RequestException as e:
                    logging.error(f"Network error no download do PDF {pdf.arquivo}: {e}")
                