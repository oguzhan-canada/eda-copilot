#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# upload_code.sh — Upload project code and orchestration scripts to S3
#
# Run this BEFORE launching Spot instances so bootstrap can pull the code.
# No Docker or ECR needed — instances install directly from S3 + pip + source.
#
# Prerequisites:
#   - AWS CLI configured
#   - Terraform already applied (S3 bucket exists)
#
# Usage:
#   bash aws/scripts/upload_code.sh
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Get S3 bucket from terraform output
S3_BUCKET=$(cd "$SCRIPT_DIR/../terraform" && terraform output -raw s3_bucket 2>/dev/null || echo "")
if [ -z "$S3_BUCKET" ]; then
    echo "ERROR: Could not determine S3 bucket. Run 'terraform apply' first."
    exit 1
fi

echo "═══════════════════════════════════════════════════════════════"
echo " Uploading project to S3"
echo " Bucket:  $S3_BUCKET"
echo " Project: $PROJECT_ROOT"
echo "═══════════════════════════════════════════════════════════════"

# ── Upload project code (excluding aws/, data/, models/, .git) ────────────────
echo ""
echo "▸ Uploading project code..."
aws s3 sync "$PROJECT_ROOT/" "s3://${S3_BUCKET}/code/" \
    --exclude ".git/*" \
    --exclude "aws/*" \
    --exclude "data/raw/*" \
    --exclude "models/*" \
    --exclude "results/*" \
    --exclude "__pycache__/*" \
    --exclude "*.pyc" \
    --exclude ".pytest_cache/*"
echo "  ✓ Code uploaded to s3://${S3_BUCKET}/code/"

# ── Upload orchestration scripts ──────────────────────────────────────────────
echo ""
echo "▸ Uploading orchestration scripts..."
aws s3 sync "$SCRIPT_DIR/" "s3://${S3_BUCKET}/scripts/" \
    --exclude "upload_code.sh"
echo "  ✓ Scripts uploaded to s3://${S3_BUCKET}/scripts/"

# ── Upload requirements.txt to root for bootstrap ────────────────────────────
echo ""
echo "▸ Uploading requirements.txt..."
aws s3 cp "$PROJECT_ROOT/requirements.txt" "s3://${S3_BUCKET}/code/requirements.txt"

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo " Upload complete!"
echo ""
echo " Next steps:"
echo "   1. cd aws/terraform"
echo "   2. terraform apply -var='launch_cpu=true'   # Stages 1-2"
echo "   3. SSH in: ssh -i ~/.ssh/ppa-key.pem ubuntu@<ip>"
echo "   4. Wait for bootstrap (~15 min), then:"
echo "      source /opt/ppa/env.sh"
echo "      bash /opt/ppa/scripts/run_stages_1_2.sh"
echo "═══════════════════════════════════════════════════════════════"
