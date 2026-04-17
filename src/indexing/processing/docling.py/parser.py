"""
Docling GCS Parser
------------------
Reads documents from a GCS bucket, converts them to Markdown using Docling,
and writes the results back to the same bucket under a different prefix.

Supported formats: PDF, HTML, HTM, XLSX, XLSM, ZIP (extracted), RAR (extracted)

Environment variables (set in /etc/docling.env on the VM):
  GCS_BUCKET       - bucket name (no gs:// prefix)
  INPUT_PREFIX     - source prefix, e.g. "documents/"
  OUTPUT_PREFIX    - destination prefix, e.g. "markdowns/"
  STATE_BLOB       - path inside bucket to store the progress log
  MAX_WORKERS      - parallel download/upload threads (default: 4)
  BATCH_SIZE       - files to process before flushing state (default: 10)
"""

GCS_BUCKET = "aneel-raw-data"      # bucket name (no gs:// prefix)
INPUT_PREFIX = "aneel-documents/"    # source prefix, e.g. "documents/"
OUTPUT_PREFIX = "docling-markdowns/"   # destination prefix, e.g. "markdowns/"
STATE_BLOB = "processing_state/processed.json"      # path inside bucket to store the progress log
MAX_WORKERS = 4      # parallel download/upload threads (default: 4)
BATCH_SIZE = 10      # files to process before flushing state (default: 10)

import os
import sys
import logging
import tempfile
import zipfile
import time
import json
import traceback
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

# ── GCS ──────────────────────────────────────────────────────────────────────
from google.cloud import storage

# ── Docling ───────────────────────────────────────────────────────────────────
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions, EasyOcrOptions
from docling.datamodel.base_models import InputFormat

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────
BUCKET_NAME   = os.environ["GCS_BUCKET"]
INPUT_PREFIX  = os.environ.get("INPUT_PREFIX", "documents/")
OUTPUT_PREFIX = os.environ.get("OUTPUT_PREFIX", "markdowns/")
STATE_BLOB    = os.environ.get("STATE_BLOB",   "processing_state/processed.json")
MAX_WORKERS   = int(os.environ.get("MAX_WORKERS", "4"))
BATCH_SIZE    = int(os.environ.get("BATCH_SIZE",  "10"))

SUPPORTED_EXT = {".pdf", ".html", ".htm", ".xlsx", ".xlsm"}
ARCHIVE_EXT   = {".zip", ".rar"}

# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("/var/log/docling_parser.log"),
    ],
)
log = logging.getLogger("docling-parser")


# ─────────────────────────────────────────────────────────────────────────────
# State management  (resume-safe)
# ─────────────────────────────────────────────────────────────────────────────

def load_state(bucket: storage.Bucket) -> dict:
    """Load the processing state from GCS (dict keyed by blob name)."""
    blob = bucket.blob(STATE_BLOB)
    if blob.exists():
        data = blob.download_as_text()
        return json.loads(data)
    return {}


def save_state(bucket: storage.Bucket, state: dict) -> None:
    blob = bucket.blob(STATE_BLOB)
    blob.upload_from_string(json.dumps(state, indent=2), content_type="application/json")
    log.info("State saved (%d entries).", len(state))


# ─────────────────────────────────────────────────────────────────────────────
# Docling converter  (one shared instance – GPU is initialised once)
# ─────────────────────────────────────────────────────────────────────────────

def build_converter() -> DocumentConverter:
    """
    Build a Docling converter optimised for GPU inference.
    EasyOCR is used for scanned PDFs; set use_gpu=False if running CPU-only.
    """
    pdf_opts = PdfPipelineOptions(
        do_ocr=True,
        do_table_structure=True,
        ocr_options=EasyOcrOptions(
            use_gpu=True,
            force_full_page_ocr=False
        ),
    )
    return DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pdf_opts),
        }
    )


# ─────────────────────────────────────────────────────────────────────────────
# File helpers
# ─────────────────────────────────────────────────────────────────────────────

def extract_zip(zip_path: Path, dest_dir: Path) -> list[Path]:
    """Extract a ZIP and return list of supported file paths inside."""
    files = []
    with zipfile.ZipFile(zip_path, "r") as zf:
        for name in zf.namelist():
            if Path(name).suffix.lower() in SUPPORTED_EXT:
                zf.extract(name, dest_dir)
                files.append(dest_dir / name)
    return files


def extract_rar(rar_path: Path, dest_dir: Path) -> list[Path]:
    """Extract a RAR archive. Requires 'unrar' or 'rarfile' library."""
    try:
        import rarfile
        files = []
        with rarfile.RarFile(str(rar_path)) as rf:
            for name in rf.namelist():
                if Path(name).suffix.lower() in SUPPORTED_EXT:
                    rf.extract(name, str(dest_dir))
                    files.append(dest_dir / name)
        return files
    except ImportError:
        log.error("rarfile not installed. Run: pip install rarfile && apt-get install unrar")
        return []


def convert_file(converter: DocumentConverter, file_path: Path) -> str | None:
    """Convert a single file to Markdown. Returns None on failure."""
    try:
        result = converter.convert(str(file_path))
        return result.document.export_to_markdown()
    except Exception:
        log.error("Conversion failed for %s:\n%s", file_path, traceback.format_exc())
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Per-blob processing
# ─────────────────────────────────────────────────────────────────────────────

def output_blob_name(input_blob_name: str, relative_name: str | None = None) -> str:
    """
    Map an input blob path to its output markdown path.

    documents/foo/bar.pdf  →  markdowns/foo/bar.md
    For files extracted from archives, `relative_name` is appended.
    """
    # Strip INPUT_PREFIX
    rel = input_blob_name
    if rel.startswith(INPUT_PREFIX):
        rel = rel[len(INPUT_PREFIX):]

    if relative_name:
        # e.g. archive.zip → archive/inner_file.md
        stem = Path(rel).stem
        rel   = str(Path(stem) / relative_name)

    # Replace extension with .md
    rel = str(Path(rel).with_suffix(".md"))
    return OUTPUT_PREFIX + rel


def process_blob(
    blob: storage.Blob,
    bucket: storage.Bucket,
    converter: DocumentConverter,
) -> dict:
    """
    Download one blob, convert to markdown(s), upload results.
    Returns a status dict for state tracking.
    """
    ext  = Path(blob.name).suffix.lower()
    stat = {"blob": blob.name, "status": "skipped", "outputs": [], "error": None}

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        local = tmp_path / Path(blob.name).name

        # ── Download ──────────────────────────────────────────────────────
        log.info(f"⬇  Downloading  {blob.name}  ({(blob.size or 0) / 1e6:.1f} MB)")
        blob.download_to_filename(str(local))

        # ── Determine files to convert ────────────────────────────────────
        if ext == ".zip":
            to_convert = [(f, output_blob_name(blob.name, f.name)) for f in extract_zip(local, tmp_path)]
        elif ext == ".rar":
            to_convert = [(f, output_blob_name(blob.name, f.name)) for f in extract_rar(local, tmp_path)]
        elif ext in SUPPORTED_EXT:
            to_convert = [(local, output_blob_name(blob.name))]
        else:
            log.warning(f"Unsupported extension '{ext}' for blob {blob.name} — skipping.")
            stat["status"] = "unsupported"
            return stat

        # ── Convert & upload ──────────────────────────────────────────────
        any_ok = False
        for file_path, out_blob_name in to_convert:
            log.info(f"   Converting  {file_path.name}")
            t0       = time.time()
            markdown = convert_file(converter, file_path)
            elapsed  = time.time() - t0

            if markdown is None:
                stat["error"] = f"Conversion failed for {file_path.name}"
                continue

            # Upload markdown
            out_blob = bucket.blob(out_blob_name)
            out_blob.upload_from_string(markdown.encode(), content_type="text/markdown")
            log.info(f"   ✔  Uploaded  {out_blob_name}  ({elapsed:.1f}s)")
            stat["outputs"].append(out_blob_name)
            any_ok = True

        stat["status"] = "ok" if any_ok else "error"
    return stat


# ─────────────────────────────────────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────────────────────────────────────

def main():
    log.info("=" * 60)
    log.info("Docling GCS Parser starting")
    log.info(f"  Bucket      : {BUCKET_NAME}")
    log.info(f"  Input prefix: {INPUT_PREFIX}")
    log.info(f"  Output prefix: {OUTPUT_PREFIX}")
    log.info(f"  Workers     : {MAX_WORKERS}")
    log.info("=" * 60)

    gcs_client = storage.Client()
    bucket = gcs_client.bucket(BUCKET_NAME)

    # Load previous state (for resume)
    state = load_state(bucket)
    log.info(f"Loaded state: {len(state)} already-processed blobs.")

    # List blobs
    all_blobs = list(bucket.list_blobs(prefix=INPUT_PREFIX))
    blobs_to_process = [
        b for b in all_blobs
        if b.name != INPUT_PREFIX                        # skip the prefix "folder" itself
        and Path(b.name).suffix.lower() in SUPPORTED_EXT | ARCHIVE_EXT
        and b.name not in state                          # skip already processed
    ]

    total = len(blobs_to_process)
    log.info(f"Blobs to process: {total}  (skipping {len(state)} already done)")

    if total == 0:
        log.info("Nothing to do. Exiting.")
        return

    # Build converter ONCE (loads GPU models)
    log.info("Initialising Docling converter (loading models)…")
    converter = build_converter()
    log.info("Converter ready.")

    # Process in batches
    done = 0
    errors = 0

    for i in range(0, total, BATCH_SIZE):
        batch = blobs_to_process[i : i + BATCH_SIZE]
        log.info(f"── Batch {i // BATCH_SIZE + 1}/{-(-total // BATCH_SIZE)} ──────────────────────")

        # Use threads only for I/O (download/upload); conversion is sequential on GPU
        # If you want full parallelism on CPU, increase MAX_WORKERS and remove the lock.
        for blob in batch:
            try:
                result = process_blob(blob, bucket, converter)
                state[blob.name] = result
                if result["status"] == "ok":
                    done += 1
                else:
                    errors += 1
            except Exception:
                log.error(f"Unexpected error on {blob.name}:\n{traceback.format_exc()}")
                state[blob.name] = {"blob": blob.name, "status": "error", "error": traceback.format_exc()}
                errors += 1

        # Flush state after each batch (crash-safe)
        save_state(bucket, state)
        log.info(f"Progress: {done + errors}/{total} done, {errors} errors")

    log.info("=" * 60)
    log.info(f"Finished!  ✔ {done} succeeded  ✘ {errors} errors")
    log.info("=" * 60)


if __name__ == "__main__":
    main()