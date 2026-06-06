#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# run_stages_1_2.sh — Orchestrate Stages 1-2 on CPU Spot instance
#
# Runs Python directly (no Docker). Handles:
#   1a. Download OpenABC-D (resumable)
#   1b. Download CircuitNet + DEF fix
#   1c. Run OpenROAD sweep (5 designs × 11 clocks, parallel)
#   2a. Extract graph features
#   2b. Extract spatial features
#   2c. Extract timing features
#   2d. Build unified manifest
#   → Incremental S3 sync after each major step
#
# Usage: bash run_stages_1_2.sh [--resume]
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source /opt/ppa/env.sh 2>/dev/null || true

# Start Spot interruption watcher in background
bash "$SCRIPT_DIR/spot_watcher.sh" &
WATCHER_PID=$!
trap "kill $WATCHER_PID 2>/dev/null || true" EXIT

RESUME="${1:-}"
LOG_DIR="/opt/ppa/results/logs"
CODE_DIR="/opt/ppa/code"
mkdir -p "$LOG_DIR"

log() { echo "[$(date '+%H:%M:%S')] $1"; }

# ═══════════════════════════════════════════════════════════════════════════════
# STAGE 1: DATA COLLECTION
# ═══════════════════════════════════════════════════════════════════════════════

log "═══ STAGE 1: Data Collection ═══"

# ── 1a. OpenABC-D ─────────────────────────────────────────────────────────────
log "Step 1a: Downloading OpenABC-D..."
python3 "$CODE_DIR/scripts/download_openabc.py" \
    --output "$PPA_DATA_ROOT/raw/openabc_d/" \
    2>&1 | tee "$LOG_DIR/download_openabc.log"

log "  Syncing OpenABC-D metadata to S3..."
bash "$SCRIPT_DIR/sync_to_s3.sh" data

# ── 1b. CircuitNet ────────────────────────────────────────────────────────────
log "Step 1b: Downloading CircuitNet..."
python3 "$CODE_DIR/scripts/download_circuitnet.py" \
    --output "$PPA_DATA_ROOT/raw/circuitnet/" \
    2>&1 | tee "$LOG_DIR/download_circuitnet.log"

log "  Syncing CircuitNet data to S3..."
bash "$SCRIPT_DIR/sync_to_s3.sh" data

# ── 1c. OpenROAD Sweep ────────────────────────────────────────────────────────
log "Step 1c: Running OpenROAD sweep (5 designs × 11 clocks)..."
python3 "$CODE_DIR/scripts/run_openroad.py" \
    --output "$PPA_DATA_ROOT/raw/openroad_runs/" \
    --max-jobs 4 \
    2>&1 | tee "$LOG_DIR/run_openroad.log"

log "  Syncing OpenROAD results to S3..."
bash "$SCRIPT_DIR/sync_to_s3.sh" data

# ═══════════════════════════════════════════════════════════════════════════════
# STAGE 2: FEATURE EXTRACTION
# ═══════════════════════════════════════════════════════════════════════════════

log "═══ STAGE 2: Feature Extraction ═══"

# ── 2a. Graph features ────────────────────────────────────────────────────────
log "Step 2a: Extracting graph features..."
python3 "$CODE_DIR/scripts/extract_graph.py" \
    --input "$PPA_DATA_ROOT/raw/openroad_runs/" \
    --output /opt/ppa/features/graph/ \
    2>&1 | tee "$LOG_DIR/extract_graph.log"

# ── 2b. Spatial features ─────────────────────────────────────────────────────
log "Step 2b: Extracting spatial features..."
python3 "$CODE_DIR/scripts/extract_spatial.py" \
    --input "$PPA_DATA_ROOT/raw/openroad_runs/" \
    --output /opt/ppa/features/spatial/ \
    2>&1 | tee "$LOG_DIR/extract_spatial.log"

# ── 2c. Timing features ──────────────────────────────────────────────────────
log "Step 2c: Extracting timing features..."
python3 "$CODE_DIR/scripts/extract_timing.py" \
    --input "$PPA_DATA_ROOT/raw/openroad_runs/" \
    --output /opt/ppa/features/timing_stats.csv \
    2>&1 | tee "$LOG_DIR/extract_timing.log"

# ── 2d. Build manifest ───────────────────────────────────────────────────────
log "Step 2d: Building unified manifest..."
python3 "$CODE_DIR/scripts/build_manifest.py" \
    --openabc "$PPA_DATA_ROOT/raw/openabc_d/openabc_manifest_partial.csv" \
    --circuitnet "$PPA_DATA_ROOT/raw/circuitnet/circuitnet_manifest_partial.csv" \
    --openroad "$PPA_DATA_ROOT/raw/openroad_runs/" \
    --timing /opt/ppa/features/timing_stats.csv \
    --spatial /opt/ppa/features/spatial/spatial_stats.csv \
    --output "$PPA_DATA_ROOT/manifest_real.csv" \
    2>&1 | tee "$LOG_DIR/build_manifest.log"

# ── Final sync ────────────────────────────────────────────────────────────────
log "Final sync: all artifacts to S3..."
bash "$SCRIPT_DIR/sync_to_s3.sh" all

# ── Validate manifest paths are relative ──────────────────────────────────────
log "Validating manifest path portability..."
python3 -c "
import pandas as pd, sys
df = pd.read_csv('$PPA_DATA_ROOT/manifest_real.csv')
abs_paths = [c for c in df.columns if 'path' in c.lower()
             for v in df[c].dropna() if str(v).startswith('/')]
if abs_paths:
    print(f'WARNING: {len(abs_paths)} absolute paths found in manifest')
    sys.exit(1)
print('OK: All manifest paths are relative')
"

log ""
log "═══════════════════════════════════════════════════════════════"
log " STAGES 1-2 COMPLETE"
log " Artifacts synced to s3://${S3_BUCKET}/ppa-run/"
log " Next: Launch GPU instance and run run_stages_3_5.sh"
log "═══════════════════════════════════════════════════════════════"
