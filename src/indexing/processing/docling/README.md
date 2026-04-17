# Docling GCS Parser – Setup Guide

## Files in this package

| File | Purpose |
|---|---|
| `parser.py` | Main parser – reads from GCS, converts docs, writes Markdowns back |
| `requirements.txt` | Python dependencies reference |
| `setup_vm.sh` | One-time VM initialisation script |
| `docling.service` | Systemd unit (process management) |

---

## Step 1 – Create the GCP VM (Spot instance)

```bash
gcloud compute instances create docling-parser \
  --project=YOUR_PROJECT \
  --zone=us-central1-a \
  --machine-type=n1-standard-8 \
  --accelerator=type=nvidia-tesla-t4,count=1 \
  --maintenance-policy=TERMINATE \
  --provisioning-model=SPOT \
  --instance-termination-action=STOP \
  --image-family=common-cu129-ubuntu-2204-nvidia-580 \
  --image-project=deeplearning-platform-release \
  --boot-disk-size=150GB \
  --boot-disk-type=pd-ssd \
  --metadata="install-nvidia-driver=True" \
  --scopes=https://www.googleapis.com/auth/cloud-platform
```

> **Spot saves ~70% cost.** The parser is resume-safe — if the VM is preempted,
> just restart it and it picks up from the GCS checkpoint automatically.
> `--instance-termination-action=STOP` preserves the disk (and cached models)
> on preemption instead of deleting the VM.

---

## Step 2 – Copy files to the VM

```bash
# From your local machine (in the directory with all 4 files)
gcloud compute scp parser.py setup_vm.sh docling.service key.json \
  docling-parser:~ --zone=us-central1-a
```

> `key.json` is your service account key. The account needs
> `roles/storage.objectCreator` (write) on the bucket. Public read is fine.

---

## Step 3 – SSH in and run setup

```bash
gcloud compute ssh docling-parser --zone=us-central1-a
```

Inside the VM:

```bash
# Edit GCS_BUCKET (and optionally prefixes) at the top of setup_vm.sh
nano setup_vm.sh

# Run setup (~10 minutes)
chmod +x setup_vm.sh && ./setup_vm.sh
```

No reboot needed — NVIDIA drivers are pre-installed in the image.

---

## Step 4 – Start the parser

```bash
sudo systemctl start docling-parser

# Watch live logs
sudo journalctl -u docling-parser -f

# Or the log file
tail -f /var/log/docling_parser.log
```

---

## Step 5 – Monitor progress

```bash
# Files written so far
gsutil ls gs://YOUR_BUCKET/markdowns/ | wc -l

# ok vs error counts from the checkpoint
gsutil cat gs://YOUR_BUCKET/processing_state/processed.json \
  | python3 -c "
import sys, json
s = json.load(sys.stdin)
ok  = sum(1 for v in s.values() if v['status'] == 'ok')
err = sum(1 for v in s.values() if v['status'] == 'error')
print(f'ok: {ok}  errors: {err}  total: {len(s)}')
"
```

---

## Step 6 – Shut down when done

```bash
# Stop the VM (disk preserved, no charges except disk storage)
gcloud compute instances stop docling-parser --zone=us-central1-a

# Or delete entirely
gcloud compute instances delete docling-parser --zone=us-central1-a
```

---

## Retry failed files

```bash
# Remove error entries from checkpoint so they get reprocessed next run
gsutil cat gs://YOUR_BUCKET/processing_state/processed.json \
  | python3 -c "
import sys, json
s = json.load(sys.stdin)
cleaned = {k: v for k, v in s.items() if v.get('status') == 'ok'}
print(json.dumps(cleaned, indent=2))
" | gsutil cp - gs://YOUR_BUCKET/processing_state/processed.json

sudo systemctl start docling-parser
```

---

## GCS bucket layout

```
gs://YOUR_BUCKET/
├── documents/                  ← input files (PDF, HTML, XLSX, ZIP, RAR…)
│   ├── folder1/report.pdf
│   └── folder2/archive.zip
│       └── (extracted: doc1.pdf, doc2.html)
├── markdowns/                  ← output (mirrors documents/ structure)
│   ├── folder1/report.md
│   └── folder2/archive/
│       ├── doc1.md
│       └── doc2.md
└── processing_state/
    └── processed.json          ← resume checkpoint (written by parser)
```

---

## Authentication

| Access | Method |
|---|---|
| **Read** (downloading source docs) | Public (`allUsers`) — no credentials needed |
| **Write** (uploading markdowns + checkpoint) | Service account key at `/opt/docling_parser/secrets/key.json` |

The key path is set via `GOOGLE_APPLICATION_CREDENTIALS` in `/etc/docling.env`
and picked up automatically by the GCS Python client.

---

## OCR behaviour

| File type | OCR used? |
|---|---|
| Native/digital PDF | ❌ No — text extracted directly |
| Scanned PDF | ✅ Yes — EasyOCR on GPU, page-by-page |
| HTML / HTM | ❌ Never |
| XLSX / XLSM | ❌ Never |

OCR fires **only on pages with no extractable text** (`force_full_page_ocr=False`).
This keeps native PDFs fast (~3–5s) while still handling scanned ones (~20–60s).

---

## Environment variables (`/etc/docling.env`)

| Variable | Default | Description |
|---|---|---|
| `GCS_BUCKET` | *(required)* | Bucket name without `gs://` |
| `INPUT_PREFIX` | `documents/` | Source prefix |
| `OUTPUT_PREFIX` | `markdowns/` | Destination prefix |
| `STATE_BLOB` | `processing_state/processed.json` | Checkpoint path |
| `MAX_WORKERS` | `4` | Download/upload threads |
| `BATCH_SIZE` | `10` | Files per batch before flushing state |
| `GOOGLE_APPLICATION_CREDENTIALS` | *(set by setup_vm.sh)* | Service account key path |

---

## Estimated processing time (T4 GPU)

| File type | Avg per file |
|---|---|
| PDF — native/digital | 3 – 5 s |
| PDF — scanned (OCR) | 20 – 60 s |
| HTML / HTM | 1 – 3 s |
| XLSX / XLSM | 2 – 5 s |

For ~26k PDFs: **2–6 days** on a single T4 (depending on how many are scanned).
To go faster, run multiple VMs processing different sub-folders in parallel.