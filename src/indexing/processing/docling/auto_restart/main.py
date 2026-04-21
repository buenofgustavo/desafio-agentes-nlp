"""
Cloud Function: auto_restart_vm
--------------------------------
Triggered by Cloud Scheduler every 5 minutes.
Checks if the Spot VM is in STOPPED or TERMINATED state and starts it.

Environment variables (set automatically by deploy_auto_restart.sh):
  PROJECT_ID   - GCP project ID
  ZONE         - VM zone, e.g. us-central1-a
  INSTANCE     - VM instance name, e.g. docling-parser
"""

import os
import logging
import functions_framework
from googleapiclient import discovery
from googleapiclient.errors import HttpError

PROJECT_ID = os.environ["PROJECT_ID"]
ZONE       = os.environ["ZONE"]
INSTANCE   = os.environ["INSTANCE"]

log = logging.getLogger("auto-restart")
logging.basicConfig(level=logging.INFO)


@functions_framework.http
def auto_restart_vm(request):
    """HTTP-triggered Cloud Function (called by Cloud Scheduler)."""
    try:
        compute = discovery.build("compute", "v1")

        # ── Get current VM status ─────────────────────────────────────────────
        result = compute.instances().get(
            project=PROJECT_ID,
            zone=ZONE,
            instance=INSTANCE,
        ).execute()

        status = result.get("status", "UNKNOWN")
        log.info("VM '%s' current status: %s", INSTANCE, status)

        # ── Start if stopped/terminated ───────────────────────────────────────
        if status in ("STOPPED", "TERMINATED"):
            log.info("VM is %s — sending start request…", status)
            op = compute.instances().start(
                project=PROJECT_ID,
                zone=ZONE,
                instance=INSTANCE,
            ).execute()
            log.info("Start operation launched: %s", op.get("name"))
            return (f"VM '{INSTANCE}' was {status} — start triggered.", 200)

        elif status == "RUNNING":
            log.info("VM is already RUNNING — nothing to do.")
            return (f"VM '{INSTANCE}' is already RUNNING.", 200)

        else:
            # STAGING, STOPPING, REPAIRING, etc. — transient state, do nothing
            log.info("VM is in transient state '%s' — skipping.", status)
            return (f"VM '{INSTANCE}' is in state {status} — skipped.", 200)

    except HttpError as e:
        log.error("Compute API error: %s", e)
        return (f"Compute API error: {e}", 500)

    except Exception as e:
        log.error("Unexpected error: %s", e)
        return (f"Unexpected error: {e}", 500)