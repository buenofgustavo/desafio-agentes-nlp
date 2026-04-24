#!/usr/bin/env bash
# =============================================================================
#  deploy_auto_restart.sh
#  Run this ONCE from your LOCAL machine (not the VM).
#
#  What it does:
#    1. Creates a dedicated service account for the function
#    2. Grants it permission to start/stop Compute Engine instances
#    3. Deploys a Cloud Function (2nd gen) that checks VM state
#    4. Creates a Cloud Scheduler job that calls the function every 5 minutes
#
#  Requirements:
#    - gcloud CLI authenticated (gcloud auth login)
#    - Billing enabled on the project
#    - APIs: Cloud Functions, Cloud Scheduler, Compute Engine
# =============================================================================
set -euo pipefail

# ── Edit these ────────────────────────────────────────────────────────────────
PROJECT_ID="desafio-agentes-nlp-ceia"
ZONE="us-central1-a"
INSTANCE="docling-parser"
REGION="us-central1"          # must match ZONE region
SCHEDULE="*/5 * * * *"        # every 5 minutes
# ─────────────────────────────────────────────────────────────────────────────

FUNCTION_NAME="auto-restart-docling-vm"
SA_NAME="docling-vm-restarter"
SA_EMAIL="${SA_NAME}@${PROJECT_ID}.iam.gserviceaccount.com"

log() { echo -e "\n\033[1;34m▶  $*\033[0m"; }

# ── 1. Enable required APIs ───────────────────────────────────────────────────
log "Enabling required APIs"
gcloud services enable \
    cloudfunctions.googleapis.com \
    cloudscheduler.googleapis.com \
    compute.googleapis.com \
    cloudbuild.googleapis.com \
    run.googleapis.com \
    --project="${PROJECT_ID}"

# ── 2. Create service account ─────────────────────────────────────────────────
log "Creating service account: ${SA_EMAIL}"
gcloud iam service-accounts create "${SA_NAME}" \
    --display-name="Docling VM Restarter" \
    --project="${PROJECT_ID}" 2>/dev/null || echo "(already exists, continuing)"

# ── 3. Grant Compute Instance Admin (start/stop VMs) ─────────────────────────
log "Granting roles/compute.instanceAdmin.v1 to service account"
gcloud projects add-iam-policy-binding "${PROJECT_ID}" \
    --member="serviceAccount:${SA_EMAIL}" \
    --role="roles/compute.instanceAdmin.v1" \
    --condition=None

# ── 4. Deploy Cloud Function ──────────────────────────────────────────────────
log "Deploying Cloud Function: ${FUNCTION_NAME}"
gcloud functions deploy "${FUNCTION_NAME}" \
    --gen2 \
    --region="${REGION}" \
    --project="${PROJECT_ID}" \
    --runtime=python311 \
    --entry-point=auto_restart_vm \
    --trigger-http \
    --no-allow-unauthenticated \
    --service-account="${SA_EMAIL}" \
    --set-env-vars="PROJECT_ID=${PROJECT_ID},ZONE=${ZONE},INSTANCE=${INSTANCE}" \
    --memory=256Mi \
    --timeout=60s \
    --source=./auto_restart/

FUNCTION_URL=$(gcloud functions describe "${FUNCTION_NAME}" \
    --gen2 --region="${REGION}" --project="${PROJECT_ID}" \
    --format="value(serviceConfig.uri)")
log "Function deployed at: ${FUNCTION_URL}"

# ── 5. Grant Scheduler permission to invoke the function ─────────────────────
log "Granting Cloud Scheduler permission to invoke function"
gcloud functions add-invoker-policy-binding "${FUNCTION_NAME}" \
    --gen2 \
    --region="${REGION}" \
    --project="${PROJECT_ID}" \
    --member="serviceAccount:${SA_EMAIL}"

# ── 6. Create Cloud Scheduler job ─────────────────────────────────────────────
log "Creating Cloud Scheduler job (every 5 minutes)"
gcloud scheduler jobs create http "restart-${INSTANCE}" \
    --location="${REGION}" \
    --project="${PROJECT_ID}" \
    --schedule="${SCHEDULE}" \
    --uri="${FUNCTION_URL}" \
    --http-method=GET \
    --oidc-service-account-email="${SA_EMAIL}" \
    --attempt-deadline=60s \
    --time-zone="UTC" 2>/dev/null || \
gcloud scheduler jobs update http "restart-${INSTANCE}" \
    --location="${REGION}" \
    --project="${PROJECT_ID}" \
    --schedule="${SCHEDULE}" \
    --uri="${FUNCTION_URL}" \
    --http-method=GET \
    --oidc-service-account-email="${SA_EMAIL}" \
    --attempt-deadline=60s \
    --time-zone="UTC"

log "Done!"
echo ""
echo "  The Scheduler will check the VM every 5 minutes."
echo "  If the VM is STOPPED (preempted), it will be restarted automatically."
echo ""
echo "  To test immediately:"
echo "    gcloud scheduler jobs run restart-${INSTANCE} --location=${REGION} --project=${PROJECT_ID}"
echo ""
echo "  To watch Scheduler logs:"
echo "    gcloud scheduler jobs describe restart-${INSTANCE} --location=${REGION} --project=${PROJECT_ID}"
echo ""
echo "  To watch Function logs:"
echo "    gcloud functions logs read ${FUNCTION_NAME} --gen2 --region=${REGION} --project=${PROJECT_ID} --limit=20"
echo ""
echo "  To tear everything down:"
echo "    gcloud scheduler jobs delete restart-${INSTANCE} --location=${REGION} --project=${PROJECT_ID}"
echo "    gcloud functions delete ${FUNCTION_NAME} --gen2 --region=${REGION} --project=${PROJECT_ID}"