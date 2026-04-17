#!/usr/bin/env bash
# =============================================================================
#  setup_vm.sh  –  Run ONCE after SSH-ing into the new GCP VM
#
#  What this does:
#    1. Installs NVIDIA drivers + CUDA toolkit
#    2. Installs Python 3.11 + virtualenv
#    3. Installs all Python dependencies (docling, google-cloud-storage, etc.)
#    4. Installs unrar (for .rar files)
#    5. Uploads your parser.py and writes /etc/docling.env
#    6. Installs and enables the systemd service
#
#  Usage:
#    chmod +x setup_vm.sh
#    ./setup_vm.sh
# =============================================================================
set -euo pipefail

# ── Edit these before running ─────────────────────────────────────────────────
GCS_BUCKET="aneel-raw-data"      # bucket name (no gs:// prefix)
INPUT_PREFIX="aneel-documents/"    # source prefix, e.g. "documents/"
OUTPUT_PREFIX="docling-markdowns/"   # destination prefix, e.g. 
STATE_BLOB="processing_state/processed.json"
MAX_WORKERS="4"
BATCH_SIZE="10"
# ─────────────────────────────────────────────────────────────────────────────

log() { echo -e "\n\033[1;34m▶  $*\033[0m"; }

# ── 1. System packages ────────────────────────────────────────────────────────
log "Updating system packages"
sudo apt-get update -qq
sudo apt-get install -y -qq \
    build-essential curl wget git unzip \
    python3.11 python3.11-venv python3.11-dev python3-pip \
    libgl1 libglib2.0-0 poppler-utils tesseract-ocr \
    unrar-free   # for .rar support

# ── 2. NVIDIA drivers + CUDA 12.x ─────────────────────────────────────────────
log "Installing NVIDIA drivers and CUDA toolkit"

# Remove old drivers if any
sudo apt-get remove -y --purge '^nvidia-.*' '^cuda-.*' 2>/dev/null || true

# Add CUDA repo
CUDA_KEYRING="cuda-keyring_1.1-1_all.deb"
wget -q "https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/${CUDA_KEYRING}"
sudo dpkg -i "${CUDA_KEYRING}"
sudo apt-get update -qq
sudo apt-get install -y -qq cuda-toolkit-12-4 nvidia-driver-550

log "NVIDIA driver installed. GPU info (may show after reboot):"
nvidia-smi 2>/dev/null || echo "(nvidia-smi not yet active – reboot may be needed)"

# ── 3. Python virtualenv ──────────────────────────────────────────────────────
log "Creating Python virtualenv at /opt/docling_env"
sudo python3.11 -m venv /opt/docling_env
sudo /opt/docling_env/bin/pip install --upgrade pip wheel setuptools

# ── 4. Install Python dependencies ───────────────────────────────────────────
log "Installing Python dependencies (this may take 5-10 minutes)"
# Install PyTorch with CUDA 12.1 support first (required by docling's EasyOCR)
sudo /opt/docling_env/bin/pip install \
    torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install the rest
sudo /opt/docling_env/bin/pip install \
    "docling>=2.5.0" \
    "google-cloud-storage>=2.16.0" \
    "rarfile>=4.1" \
    "tqdm>=4.66.0"

# ── 5. Create app directory and copy parser ───────────────────────────────────
log "Setting up app directory"
sudo mkdir -p /opt/docling_parser
sudo cp parser.py /opt/docling_parser/parser.py
sudo chown -R nobody:nogroup /opt/docling_parser 2>/dev/null || true

# ── 6. Write environment config ───────────────────────────────────────────────
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
log "Environment written to /etc/docling.env"

# ── 7. Install systemd service ────────────────────────────────────────────────
log "Installing systemd service"
sudo cp docling.service /etc/systemd/system/docling-parser.service
sudo systemctl daemon-reload
sudo systemctl enable docling-parser.service

log "Setup complete!"
echo ""
echo "  Next steps:"
echo "  1. Reboot if this is the first NVIDIA driver install:  sudo reboot"
echo "  2. After reboot, verify GPU:                           nvidia-smi"
echo "  3. Start the parser:                                   sudo systemctl start docling-parser"
echo "  4. Watch live logs:                                    sudo journalctl -u docling-parser -f"
echo "  5. Check log file:                                     tail -f /var/log/docling_parser.log"
echo ""
echo "  To run manually (without systemd):"
echo "    source /opt/docling_env/bin/activate"
echo "    env \$(cat /etc/docling.env | xargs) python3 /opt/docling_parser/parser.py"