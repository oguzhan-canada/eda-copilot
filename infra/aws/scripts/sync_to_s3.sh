#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# sync_to_s3.sh — Incremental sync of artifacts to S3
#
# Usage:
#   bash sync_to_s3.sh              # Sync everything
#   bash sync_to_s3.sh features     # Sync only features/
#   bash sync_to_s3.sh models       # Sync only models/
#   bash sync_to_s3.sh data         # Sync data/ (excluding raw OpenABC-D)
#   bash sync_to_s3.sh results      # Sync results/
#   bash sync_to_s3.sh checkpoint   # Sync models + results (mid-training save)
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

source /opt/ppa/env.sh 2>/dev/null || true

S3_PREFIX="s3://${S3_BUCKET}/ppa"
LOCAL_ROOT="/opt/ppa"
TARGET="${1:-all}"

sync_dir() {
    local src="$1" dst="$2"
    shift 2
    echo "  Syncing $src → $dst ..."
    aws s3 sync "$src" "$dst" --quiet "$@"
}

case "$TARGET" in
    features)
        sync_dir "$LOCAL_ROOT/features/" "$S3_PREFIX/features/"
        ;;
    models)
        sync_dir "$LOCAL_ROOT/models/" "$S3_PREFIX/models/"
        ;;
    data)
        sync_dir "$LOCAL_ROOT/data/" "$S3_PREFIX/data/" \
            --exclude "raw/openabc_d/*.tar.gz"
        ;;
    results)
        sync_dir "$LOCAL_ROOT/results/" "$S3_PREFIX/results/"
        ;;
    checkpoint)
        sync_dir "$LOCAL_ROOT/models/" "$S3_PREFIX/models/"
        sync_dir "$LOCAL_ROOT/results/" "$S3_PREFIX/results/"
        ;;
    all)
        sync_dir "$LOCAL_ROOT/features/" "$S3_PREFIX/features/"
        sync_dir "$LOCAL_ROOT/data/" "$S3_PREFIX/data/" \
            --exclude "raw/openabc_d/*.tar.gz"
        sync_dir "$LOCAL_ROOT/models/" "$S3_PREFIX/models/"
        sync_dir "$LOCAL_ROOT/results/" "$S3_PREFIX/results/"
        ;;
    *)
        echo "Usage: $0 {features|models|data|results|checkpoint|all}"
        exit 1
        ;;
esac

echo "  Sync complete: $TARGET → $S3_PREFIX"
