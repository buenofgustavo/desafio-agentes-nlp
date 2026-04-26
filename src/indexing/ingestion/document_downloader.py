"""Responsável por baixar os arquivos a partir dos links presentes nos arquivos JSON e armazená-los na pasta designada."""
import re
import random
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from curl_cffi import requests as cf_requests
from urllib.parse import urlparse, urlunparse
from pathlib import Path
from typing import List, Dict, Tuple
from src.core.models import AneelRecord
from src.core.config import Constants
from src.utils.logger import LoggingService

logger = LoggingService.setup_logger(__name__)


class DocumentDownloader:
    """Baixa documentos legislativos com curl_cffi (Chrome TLS) e concorrência."""

    _EXCLUDED_EXTENSIONS = {".html"}

    _BASE_HEADERS = {
        "Accept-Language": "pt-BR,pt;q=0.9,en-US;q=0.8",
        "Accept-Encoding": "gzip, deflate, br",
        "Connection": "keep-alive",
    }

    _MAX_RETRIES = 3
    _MAX_WORKERS = 20
    _RETRYABLE_STATUS_CODES = {403, 429, 500, 502, 503, 504}

    def __init__(self) -> None:
        self._thread_local = threading.local()
        self._domain_locks: Dict[str, threading.Lock] = {}
        self._domain_last_request: Dict[str, float] = {}
        self._global_lock = threading.Lock()
        self._all_sessions: List[cf_requests.Session] = []

        # Estatísticas (thread-safe)
        self._stats_lock = threading.Lock()
        self._stats = {
            "success": 0,
            "skipped_existing": 0,
            "skipped_excluded": 0,
            "failed_404": 0,
            "failed_403": 0,
            "failed_timeout": 0,
            "failed_other": 0,
        }

    def __enter__(self) -> "DocumentDownloader":
        return self

    def __exit__(self, *_) -> None:
        self._cleanup_sessions()

    # ── Gerenciamento de Sessão (thread-local) ──────────────────────

    def _get_session(self) -> cf_requests.Session:
        """Retorna a sessão thread-local, criando-a se necessário.

        Cada thread do ThreadPoolExecutor recebe sua própria sessão
        curl_cffi para garantir thread-safety e manter cookies
        Cloudflare isolados por thread.
        """
        if not hasattr(self._thread_local, "session") or self._thread_local.session is None:
            session = cf_requests.Session()
            self._thread_local.session = session
            self._thread_local.warmed_up_hosts = set()
            with self._global_lock:
                self._all_sessions.append(session)
        return self._thread_local.session

    def _cleanup_sessions(self) -> None:
        """Fecha todas as sessões thread-local registradas."""
        for session in self._all_sessions:
            try:
                session.close()
            except Exception:
                pass
        self._all_sessions.clear()

    # ── Auxiliares de URL ──────────────────────────────────────────

    @staticmethod
    def _sanitize_url(url: str) -> str:
        """Limpa e normaliza a URL: strip, corrige double-scheme, força HTTPS."""
        url = url.strip()
        # Fix double-scheme: "https://  http://www2..." → "https://www2..."
        url = re.sub(r"^https?://\s+https?://", "https://", url)
        # Force HTTPS
        parsed = urlparse(url)
        if parsed.scheme == "http":
            parsed = parsed._replace(scheme="https")
        return urlunparse(parsed)

    @staticmethod
    def _get_extension(url: str) -> str:
        """Extrai a extensão do arquivo da URL."""
        path = urlparse(url).path.rstrip()
        filename = path.rsplit("/", 1)[-1] if "/" in path else path
        if "." in filename:
            return "." + filename.rsplit(".", 1)[-1].lower()
        return ""

    @staticmethod
    def _base_url(url: str) -> str:
        """Extrai a origem (scheme + host) de qualquer URL."""
        parsed = urlparse(url)
        return f"{parsed.scheme}://{parsed.netloc}"

    @staticmethod
    def _url_directory(url: str) -> str:
        """Retorna o diretório (path pai) da URL.

        Exemplo: https://www2.aneel.gov.br/cedoc/doc.pdf
              → https://www2.aneel.gov.br/cedoc/
        """
        parsed = urlparse(url)
        path = parsed.path
        dir_path = path.rsplit("/", 1)[0] + "/" if "/" in path else "/"
        return f"{parsed.scheme}://{parsed.netloc}{dir_path}"

    # ── Controle de Fluxo (Rate Limiting) ──────────────────────────

    def _get_domain_lock(self, domain: str) -> threading.Lock:
        """Retorna o lock de rate limiting para um domínio (thread-safe)."""
        with self._global_lock:
            if domain not in self._domain_locks:
                self._domain_locks[domain] = threading.Lock()
            return self._domain_locks[domain]

    def _rate_limit(self, url: str) -> None:
        """Aplica rate limiting por domínio para evitar sobrecarga no servidor."""
        domain = urlparse(url).netloc
        lock = self._get_domain_lock(domain)
        with lock:
            now = time.monotonic()
            last = self._domain_last_request.get(domain, 0)
            delay = random.uniform(0.2, 0.5)
            elapsed = now - last
            if elapsed < delay:
                time.sleep(delay - elapsed)
            self._domain_last_request[domain] = time.monotonic()

    # ── Pré-aquecimento (Warm-up) ──────────────────────────────────

    def _ensure_warmed_up(self, url: str) -> None:
        """Faz warm-up do domínio para obter cookies Cloudflare (thread-local).

        Visita o diretório do documento (ex: /cedoc/) para que o
        Cloudflare emita os cookies de sessão vinculados à fingerprint
        Chrome antes de baixar o documento real.
        """
        session = self._get_session()
        host = self._base_url(url)
        warmed = getattr(self._thread_local, "warmed_up_hosts", set())

        if host in warmed:
            return

        warmup_url = self._url_directory(url)
        try:
            session.get(
                warmup_url,
                impersonate="chrome",
                headers={
                    **self._BASE_HEADERS,
                    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                    "Referer": f"{host}/",
                },
                timeout=30,
            )
            logger.debug(f"Warm-up concluído: {warmup_url}")
        except Exception as e:
            logger.warning(f"Warm-up falhou para {warmup_url} (continuando): {e}")

        warmed.add(host)
        self._thread_local.warmed_up_hosts = warmed

    # ── Estatísticas ───────────────────────────────────────────────

    def _increment_stat(self, key: str) -> None:
        """Incrementa um contador de estatística (thread-safe)."""
        with self._stats_lock:
            self._stats[key] += 1

    # ── Lógica de Download ─────────────────────────────────────────

    def _fetch_bytes(self, url: str) -> bytes | None:
        """Baixa o conteúdo de uma URL com retry e exponential backoff.

        Estratégia de retry:
          - HTTP 404      → falha imediata (recurso inexistente)
          - HTTP 403/429/5xx → retry até _MAX_RETRIES vezes
          - Timeout / TLS → retry até _MAX_RETRIES vezes
        """
        session = self._get_session()
        referer = self._url_directory(url)
        ext = self._get_extension(url)
        accept = "application/pdf,*/*;q=0.8" if ext == ".pdf" else "*/*"

        for attempt in range(1, self._MAX_RETRIES + 1):
            try:
                self._rate_limit(url)
                response = session.get(
                    url,
                    impersonate="chrome",
                    headers={
                        **self._BASE_HEADERS,
                        "Accept": accept,
                        "Referer": referer,
                    },
                    timeout=300,
                )

                # ── HTTP 404: não existe, não retenta ──
                if response.status_code == 404:
                    logger.warning(f"HTTP 404 (não encontrado): {url}")
                    self._increment_stat("failed_404")
                    return None

                # ── Status retentável (403, 429, 5xx) ──
                if response.status_code in self._RETRYABLE_STATUS_CODES:
                    if attempt < self._MAX_RETRIES:
                        wait = 2 ** attempt + random.uniform(0, 1)
                        logger.info(
                            f"HTTP {response.status_code} na tentativa {attempt}/{self._MAX_RETRIES} "
                            f"para {url}. Retry em {wait:.1f}s"
                        )
                        time.sleep(wait)
                        continue

                    if response.status_code == 403:
                        self._increment_stat("failed_403")
                    else:
                        self._increment_stat("failed_other")
                    logger.error(
                        f"Falha HTTP {response.status_code} após {attempt} tentativa(s): {url}"
                    )
                    return None

                # ── Outros erros HTTP (4xx não cobertos) ──
                if response.status_code >= 400:
                    self._increment_stat("failed_other")
                    logger.error(f"HTTP {response.status_code}: {url}")
                    return None

                # ── Sucesso ──
                return response.content

            except cf_requests.RequestsError as e:
                error_str = str(e)
                is_timeout = "timed out" in error_str.lower() or "curl: (28)" in error_str
                is_tls = "curl: (35)" in error_str
                is_retryable = is_timeout or is_tls

                if is_retryable and attempt < self._MAX_RETRIES:
                    wait = 2 ** attempt + random.uniform(0, 1)
                    logger.info(
                        f"Erro na tentativa {attempt}/{self._MAX_RETRIES} para {url} "
                        f"({error_str[:80]}). Retry em {wait:.1f}s"
                    )
                    time.sleep(wait)
                    continue

                if is_timeout:
                    self._increment_stat("failed_timeout")
                else:
                    self._increment_stat("failed_other")

                logger.error(
                    f"Falha definitiva após {attempt} tentativa(s): {url} - {error_str[:120]}"
                )
                return None

        return None

    def _process_single_download(self, url: str, target_path: Path) -> bool:
        """Processa o download de um único documento. Retorna True se bem-sucedido."""
        self._ensure_warmed_up(url)
        content = self._fetch_bytes(url)

        if content:
            target_path.write_bytes(content)
            self._increment_stat("success")
            return True

        return False

    # ── API Pública ────────────────────────────────────────────────

    def download_documents(
        self,
        records: List[AneelRecord],
        output_dir: Path | str = Constants.DOCUMENTS_DIR,
        skip_existing: bool = True,
    ) -> None:
        """Faz o download dos documentos com concorrência (ThreadPoolExecutor).

        Fluxo:
          1. Constrói lista de tarefas (URL + caminho destino), pulando existentes
          2. Distribui entre _MAX_WORKERS threads
          3. Cada thread tem sua própria sessão curl_cffi com cookies Cloudflare
          4. Rate limiting por domínio evita sobrecarga no servidor
          5. Retry automático para erros transientes (403, timeout, TLS)
        """
        if isinstance(output_dir, str):
            output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Contém a tupla (url, caminho_destino)
        tasks: List[Tuple[str, Path]] = []
        for record in records:
            for pdf in record.pdfs:
                url = self._sanitize_url(pdf.url)
                ext = self._get_extension(url)

                if ext in self._EXCLUDED_EXTENSIONS:
                    self._increment_stat("skipped_excluded")
                    continue

                target_path = output_dir / pdf.arquivo

                if skip_existing and target_path.exists():
                    self._increment_stat("skipped_existing")
                    continue

                tasks.append((url, target_path))

        total = len(tasks)
        logger.info(f"Iniciando download de {total} documentos com {self._MAX_WORKERS} workers")

        if total == 0:
            logger.info("Nenhum documento pendente para download.")
            self._log_summary()
            return

        completed = 0
        completed_lock = threading.Lock()

        def _worker(task: Tuple[str, Path]):
            nonlocal completed
            url, target_path = task
            result = self._process_single_download(url, target_path)
            with completed_lock:
                completed += 1
                if completed % 200 == 0 or completed == total:
                    logger.info(
                        f"Progresso: {completed}/{total} ({100 * completed / total:.1f}%)"
                    )
                    self._log_summary()
            return result

        with ThreadPoolExecutor(max_workers=self._MAX_WORKERS) as executor:
            futures = [executor.submit(_worker, task) for task in tasks]
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    logger.error(f"Erro inesperado em worker: {e}")

        self._log_summary()

    def _log_summary(self) -> None:
        """Loga o resumo das estatísticas de download."""
        logger.info(
            f"Resumo do download: "
            f"Sucesso={self._stats['success']}, "
            f"Já existiam={self._stats['skipped_existing']}, "
            f"Extensão excluída={self._stats['skipped_excluded']}, "
            f"Falhas 404={self._stats['failed_404']}, "
            f"Falhas 403={self._stats['failed_403']}, "
            f"Timeouts={self._stats['failed_timeout']}, "
            f"Outros erros={self._stats['failed_other']}"
        )