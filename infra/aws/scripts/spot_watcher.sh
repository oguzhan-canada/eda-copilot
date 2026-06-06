#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# spot_watcher.sh — IMDSv2 Spot interruption monitor
#
# Polls EC2 instance metadata every 5s for the 2-minute interruption warning.
# On interruption: syncs critical artifacts to S3 and writes a marker file.
#
# Usage: bash spot_watcher.sh &   (run in background)
# ─────────────────────────────────────────────────────────────────────────────
set -uo pipefail

source /opt/ppa/env.sh 2>/dev/null || true

POLL_INTERVAL=5
MARKER_FILE="/opt/ppa/.spot_interrupted"

echo "[spot_watcher] Monitoring for Spot interruption (every ${POLL_INTERVAL}s)..."

while true; do
    # Get IMDSv2 token
    TOKEN=$(curl -s -X PUT "http://169.254.169.254/latest/api/token" \
      -H "X-aws-ec2-metadata-token-ttl-seconds: 21600" 2>/dev/null) || true

    # Check for interruption action
    HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" \
      -H "X-aws-ec2-metadata-token: $TOKEN" \
      "http://169.254.169.254/latest/meta-data/spot/instance-action" 2>/dev/null) || true

    if [ "$HTTP_CODE" = "200" ]; then
        echo "[spot_watcher] *** SPOT INTERRUPTION DETECTED ***"
        echo "$(date -Iseconds)" > "$MARKER_FILE"

        # Emergency sync to S3
        echo "[spot_watcher] Emergency sync to S3..."

        # Sync models (most expensive to recreate)
        aws s3 sync /opt/ppa/models/ "s3://${S3_BUCKET}/ppa-run/models/" \
          --quiet 2>/dev/null || true

        # Sync results
        aws s3 sync /opt/ppa/results/ "s3://${S3_BUCKET}/ppa-run/results/" \
          --quiet 2>/dev/null || true

        # Sync features
        aws s3 sync /opt/ppa/features/ "s3://${S3_BUCKET}/ppa-run/features/" \
          --quiet 2>/dev/null || true

        # Sync data (manifest, run status)
        aws s3 sync /opt/ppa/data/ "s3://${S3_BUCKET}/ppa-run/data/" \
          --exclude "raw/openabc_d/*" \
          --quiet 2>/dev/null || true

        echo "[spot_watcher] Emergency sync complete. Instance will terminate soon."
        exit 0
    fi

    sleep "$POLL_INTERVAL"
done
