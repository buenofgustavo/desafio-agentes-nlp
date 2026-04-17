# Docling GCS Parser – Setup Guide

## Files in this package

| File | Purpose |
|---|---|
| `parser.py` | Main parser – reads from GCS, converts docs, writes Markdowns back |
| `requirements.txt` | Python dependencies |
| `setup_vm.sh` | One-time VM initialisation script |
| `docling.service` | Systemd unit (auto-start / process management) |

---

## Step 1 – Create the GCP VM

> **Recommended spec:** `n1-standard-8` + NVIDIA T4  
> ~$0.73/hr on-demand, ~$0.22/hr spot (use spot if the job can restart)

```bash
# Replace YOUR_PROJECT and YOUR_ZONE as needed (cheapest T4 zones: us-central1-a, us-east1-c)
gcloud compute instances create docling-parser \
  --project=YOUR_PROJECT \
  --zone=us-central1-a \
  --machine-type=n1-standard-8 \
  --accelerator=type=nvidia-tesla-t4,count=1 \
  --maintenance-policy=TERMINATE \
  --provisioning-model=STANDARD \
  --image-family=ubuntu-2204-lts \
  --image-project=ubuntu-os-cloud \
  --boot-disk-size=150GB \
  --boot-disk-type=pd-ssd \
  --scopes=https://www.googleapis.com/auth/cloud-platform
```

> **Tip:** Replace `--provisioning-model=STANDARD` with `--provisioning-model=SPOT`
> to save ~70% on cost. The parser is resume-safe (it tracks state in GCS), so
> spot preemption is fine.

> **Boot disk:** 150 GB is recommended because Docling downloads ~3 GB of AI
> models on first run and PDFs are cached locally during processing.

---

## Step 2 – Copy files to the VM

```bash
# From your local machine
gcloud compute scp parser.py setup_vm.sh docling.service \
  docling-parser:~ --zone=us-central1-a
```

---

## Step 3 – SSH in and run setup

```bash
gcloud compute ssh docling-parser --zone=us-central1-a
```

Then inside the VM:

```bash
# 1. Edit GCS settings at the top of setup_vm.sh
nano setup_vm.sh        # set GCS_BUCKET, INPUT_PREFIX, OUTPUT_PREFIX

# 2. Run setup (takes ~10 minutes)
chmod +x setup_vm.sh
./setup_vm.sh

# 3. Reboot to activate NVIDIA drivers
sudo reboot
```

---

## Step 4 – Start the parser

After the VM comes back up:

```bash
gcloud compute ssh docling-parser --zone=us-central1-a

# Verify GPU
nvidia-smi

# Start the parser
sudo systemctl start docling-parser

# Watch live logs
sudo journalctl -u docling-parser -f

# Or tail the log file directly
tail -f /var/log/docling_parser.log
```

---

## Step 5 – Monitor and stop

```bash
# Check status
sudo systemctl status docling-parser

# Stop before shutting down VM
sudo systemctl stop docling-parser

# Check how many markdowns were written
gsutil ls gs://YOUR_BUCKET/markdowns/ | wc -l

# Check error count in state file
gsutil cat gs://YOUR_BUCKET/processing_state/processed.json \
  | python3 -c "import sys,json; s=json.load(sys.stdin); \
    print('ok:', sum(1 for v in s.values() if v['status']=='ok')); \
    print('errors:', sum(1 for v in s.values() if v['status']=='error'))"
```

---

## Step 6 – Shut down the VM when done (saves money!)

```bash
gcloud compute instances stop docling-parser --zone=us-central1-a
# or delete it entirely:
gcloud compute instances delete docling-parser --zone=us-central1-a
```

---

## Resume / retry failed files

The parser stores progress in `gs://YOUR_BUCKET/processing_state/processed.json`.
Any blob already in that file (with `status: ok`) is **skipped** on the next run.

To retry only errored files:

```bash
# Remove error entries from state so they get reprocessed
gsutil cat gs://YOUR_BUCKET/processing_state/processed.json \
  | python3 -c "
import sys, json
s = json.load(sys.stdin)
cleaned = {k: v for k, v in s.items() if v.get('status') == 'ok'}
print(json.dumps(cleaned, indent=2))
" | gsutil cp - gs://YOUR_BUCKET/processing_state/processed.json
```

---

## GCS bucket layout

```
gs://YOUR_BUCKET/
├── documents/                  ← your input files (PDF, HTML, XLSX, ZIP, RAR…)
│   ├── folder1/
│   │   └── report.pdf
│   └── folder2/
│       └── archive.zip
│           └── (extracted: doc1.pdf, doc2.html)
├── markdowns/                  ← output (mirrors documents/ structure)
│   ├── folder1/
│   │   └── report.md
│   └── folder2/
│       └── archive/
│           ├── doc1.md
│           └── doc2.md
└── processing_state/
    └── processed.json          ← resume checkpoint
```

---

## Environment variables (in /etc/docling.env)

| Variable | Default | Description |
|---|---|---|
| `GCS_BUCKET` | *(required)* | Bucket name without `gs://` |
| `INPUT_PREFIX` | `documents/` | Source prefix inside the bucket |
| `OUTPUT_PREFIX` | `markdowns/` | Destination prefix inside the bucket |
| `STATE_BLOB` | `processing_state/processed.json` | Resume checkpoint path |
| `MAX_WORKERS` | `4` | Download/upload threads |
| `BATCH_SIZE` | `10` | Files per batch before flushing state |

---

## Estimated processing time

| File type | Avg per file (T4 GPU) |
|---|---|
| PDF (text-based) | 5 – 15 s |
| PDF (scanned / OCR needed) | 20 – 60 s |
| HTML / HTM | 1 – 3 s |
| XLSX / XLSM | 2 – 5 s |

For your ~26 k PDFs, expect **roughly 2 – 6 days** of continuous processing on a
single T4. To go faster, create multiple VMs processing different `INPUT_PREFIX`
sub-folders, or use a larger GPU (L4 is ~2× faster than T4 for ~2× the cost).