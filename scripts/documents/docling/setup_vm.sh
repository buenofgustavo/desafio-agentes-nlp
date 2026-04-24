#!/usr/bin/env bash
# =============================================================================
#  setup_vm.sh  –  Run ONCE after SSH-ing into the GCP VM
#
#  Assumptions:
#    - VM was created with image: common-cu129-ubuntu-2204-nvidia-580
#    - NVIDIA drivers are already installed (nvidia-smi works)
#    - VM has a Service Account attached via GCP IAM (No JSON keys needed)
#
#  Usage:
#    1. Upload this script, parser.py, and docling.service to the VM:
#         gcloud compute scp parser.py setup_vm.sh docling.service \
#           docling-parser:~ --zone=us-central1-a
#    2. SSH in:
#         gcloud compute ssh docling-parser --zone=us-central1-a
#    3. Edit the variables below, then:
#         chmod +x setup_vm.sh && ./setup_vm.sh
# =============================================================================
set -euo pipefail

# ── Edit these before running ─────────────────────────────────────────────────
GCS_BUCKET="aneel-raw-data"      # bucket name (no gs:// prefix)
INPUT_PREFIX="aneel-documents/"    # source prefix, e.g. "documents/"
OUTPUT_PREFIX="docling-markdowns/"   # destination prefix
STATE_BLOB="processing_state/processed.json"
MAX_WORKERS="10"
BATCH_SIZE="100"
# ─────────────────────────────────────────────────────────────────────────────

log() { echo -e "\n\033[1;34m▶  $*\033[0m"; }

# ── 1. Verify GPU ─────────────────────────────────────────────────────────────
log "Verifying GPU"
nvidia-smi || { echo "ERROR: nvidia-smi failed."; exit 1; }

# ── 2. System dependencies ────────────────────────────────────────────────────
log "Installing system dependencies"
sudo apt-get update -qq
sudo apt-get install -y -qq \
    python3-venv python3-pip \
    libgl1 libglib2.0-0 \
    poppler-utils tesseract-ocr \
    unrar-free

# ── 3. Directory Ownership & Virtualenv ───────────────────────────────────────
log "Creating Python virtualenv at /opt/docling_env"
# Create directories and immediately grant ownership to your user (ireis)
sudo mkdir -p /opt/docling_env /opt/docling_parser
sudo chown -R $USER:$USER /opt/docling_env /opt/docling_parser

# Create the venv as your normal user (No sudo here!)
python3 -m venv /opt/docling_env

# Install uv for blazing-fast package management
/opt/docling_env/bin/pip install uv -q

# ── 4. PyTorch with cu124 wheels (compatible with driver 580 / CUDA 12.9+) ───
log "Installing PyTorch (cu124)"

# Activate the virtual environment first!
source /opt/docling_env/bin/activate

uv pip install \
    torch torchvision \
    --index-url https://download.pytorch.org/whl/cu124 -q

# ── 5. Docling + GCS + EasyOCR + ftfy ─────────────────────────────────────────
log "Installing Python dependencies"
uv pip install \
    "docling>=2.5.0" \
    "google-cloud-storage>=2.16.0" \
    "rarfile>=4.1" \
    "easyocr" \
    "ftfy" -q

# ── 6. Sanity check ───────────────────────────────────────────────────────────
log "Verifying PyTorch CUDA"
/opt/docling_env/bin/python3 -c \
    "import torch; assert torch.cuda.is_available(), 'CUDA not available!'; \
     print('OK –', torch.cuda.get_device_name(0))"

# ── 7. Deploy parser ──────────────────────────────────────────────────────────
log "Deploying parser"
cp parser.py /opt/docling_parser/parser.py

# ── 8. Environment config (No keys!) ──────────────────────────────────────────
log "Writing /etc/docling.env"
sudo tee /etc/docling.env > /dev/null <<ENV
GCS_BUCKET=${GCS_BUCKET}
INPUT_PREFIX=${INPUT_PREFIX}
OUTPUT_PREFIX=${OUTPUT_PREFIX}
STATE_BLOB=${STATE_BLOB}
MAX_WORKERS=${MAX_WORKERS}
BATCH_SIZE=${BATCH_SIZE}
ENV
sudo chmod 600 /etc/docling.env

# ── 9. Systemd service ────────────────────────────────────────────────────────
log "Installing systemd service"
sudo cp docling.service /etc/systemd/system/docling-parser.service
sudo systemctl daemon-reload
sudo systemctl enable docling-parser.service

log "Setup complete!"
echo ""
echo "  Start parser:  sudo systemctl start docling-parser"
echo "  Live logs:     sudo journalctl -u docling-parser -f"
echo "  Log file:      tail -f /var/log/docling_parser.log"
echo ""
echo "  Manual run (for testing):"
echo "    sudo env \$(cat /etc/docling.env | xargs) /opt/docling_env/bin/python3 /opt/docling_parser/parser.py"