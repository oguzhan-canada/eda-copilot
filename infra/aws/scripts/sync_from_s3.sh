#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# sync_from_s3.sh — Pull artifacts from S3 to local workspace
#
# Usage:
#   bash sync_from_s3.sh              # Pull everything
#   bash sync_from_s3.sh features     # Pull only features/
#   bash sync_from_s3.sh models       # Pull only models/
#   bash sync_from_s3.sh data         # Pull data/ (manifest + metadata)
#   bash sync_from_s3.sh training     # Pull features + data (pre-training)
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

source /opt/ppa/env.sh 2>/dev/null || true

S3_PREFIX="s3://${S3_BUCKET}/ppa"
LOCAL_ROOT="/opt/ppa"
TARGET="${1:-all}"

sync_dir() {
    local src="$1" dst="$2"
    shift 2
    echo "  Pulling $src → $dst ..."
    mkdir -p "$dst"
    aws s3 sync "$src" "$dst" --quiet "$@"
}

case "$TARGET" in
    features)
        sync_dir "$S3_PREFIX/features/" "$LOCAL_ROOT/features/"
        ;;
    models)
        sync_dir "$S3_PREFIX/models/" "$LOCAL_ROOT/models/"
        ;;
    data)
        sync_dir "$S3_PREFIX/data/" "$LOCAL_ROOT/data/"
        ;;
    training)
        sync_dir "$S3_PREFIX/features/" "$LOCAL_ROOT/features/"
        sync_dir "$S3_PREFIX/data/" "$LOCAL_ROOT/data/"
        ;;
    all)
        sync_dir "$S3_PREFIX/features/" "$LOCAL_ROOT/features/"
        sync_dir "$S3_PREFIX/data/" "$LOCAL_ROOT/data/"
        sync_dir "$S3_PREFIX/models/" "$LOCAL_ROOT/models/"
        sync_dir "$S3_PREFIX/results/" "$LOCAL_ROOT/results/"
        ;;
    *)
        echo "Usage: $0 {features|models|data|training|all}"
        exit 1
        ;;
esac

echo "  Pull complete: $S3_PREFIX → $TARGET"
