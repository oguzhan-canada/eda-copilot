#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# run_stages_3_5.sh — Orchestrate Stages 3-5 on GPU Spot instance
#
# Runs Python directly (no Docker). Handles:
#   3a. Train GBDT (leave-one-design-out)
#   3b. Train GAT with criticality (5 folds) → checkpoint sync per fold
#   3c. Train GAT without criticality (ablation)
#   4a. Select best checkpoint
#   4b. Bayesian Optimization (50 trials)
#   4c. RL PPO training (500k steps)
#   5.  Evaluation + final report
#   → S3 checkpoint after each step
#
# Usage: bash run_stages_3_5.sh
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source /opt/ppa/env.sh 2>/dev/null || true

# Start Spot interruption watcher
bash "$SCRIPT_DIR/spot_watcher.sh" &
WATCHER_PID=$!
trap "kill $WATCHER_PID 2>/dev/null || true" EXIT

LOG_DIR="/opt/ppa/results/logs"
CODE_DIR="/opt/ppa/code"
mkdir -p "$LOG_DIR"

log() { echo "[$(date '+%H:%M:%S')] $1"; }

# ── Pull training data from S3 ───────────────────────────────────────────────
log "Pulling training data from S3..."
bash "$SCRIPT_DIR/sync_from_s3.sh" training

# Verify manifest exists
if [ ! -f "$PPA_DATA_ROOT/manifest_real.csv" ]; then
    echo "ERROR: manifest_real.csv not found. Run Stages 1-2 first."
    exit 1
fi

# ═══════════════════════════════════════════════════════════════════════════════
# STAGE 3: MODEL TRAINING
# ═══════════════════════════════════════════════════════════════════════════════

log "═══ STAGE 3: Model Training ═══"

# ── 3a. GBDT baseline ────────────────────────────────────────────────────────
log "Step 3a: Training GBDT (leave-one-design-out)..."
python3 "$CODE_DIR/scripts/train_gbdt.py" \
    --manifest "$PPA_DATA_ROOT/manifest_real.csv" \
    --output /opt/ppa/models/gbdt/ \
    --seed 42 \
    2>&1 | tee "$LOG_DIR/train_gbdt.log"

log "  Syncing GBDT models to S3..."
bash "$SCRIPT_DIR/sync_to_s3.sh" models

# ── 3b. GAT with criticality (5-fold) ────────────────────────────────────────
log "Step 3b: Training GAT + criticality (5 folds)..."
python3 "$CODE_DIR/scripts/train_gat.py" \
    --manifest "$PPA_DATA_ROOT/manifest_real.csv" \
    --output /opt/ppa/models/real/ \
    --all-folds \
    --seed 42 \
    2>&1 | tee "$LOG_DIR/train_gat_crit.log"

log "  Syncing GAT models to S3 (checkpoint)..."
bash "$SCRIPT_DIR/sync_to_s3.sh" models

# ── 3c. GAT ablation — no criticality ────────────────────────────────────────
log "Step 3c: Training GAT ablation (no criticality)..."
python3 "$CODE_DIR/scripts/train_gat.py" \
    --manifest "$PPA_DATA_ROOT/manifest_real.csv" \
    --output /opt/ppa/models/real/ \
    --all-folds \
    --no-criticality \
    --seed 42 \
    2>&1 | tee "$LOG_DIR/train_gat_nocrit.log"

log "  Syncing all models to S3..."
bash "$SCRIPT_DIR/sync_to_s3.sh" models

# ═══════════════════════════════════════════════════════════════════════════════
# STAGE 4: OPTIMIZATION
# ═══════════════════════════════════════════════════════════════════════════════

log "═══ STAGE 4: Optimization ═══"

# ── 4a. Select best checkpoint ────────────────────────────────────────────────
log "Step 4a: Selecting best GAT checkpoint..."
BEST_CKPT=$(python3 -c "
import json, torch
d = json.load(open('/opt/ppa/models/real/best_checkpoints.json'))
best = min(d.values(), key=lambda p: torch.load(p, weights_only=False, map_location='cpu')['val_mae'])
print(best)
")
log "  Best checkpoint: $BEST_CKPT"

# ── 4b. Bayesian Optimization ────────────────────────────────────────────────
log "Step 4b: Running BO (50 trials, 10 EDA verifications)..."
python3 "$CODE_DIR/optimize/run_bo.py" \
    --model "$BEST_CKPT" \
    --manifest "$PPA_DATA_ROOT/manifest_real.csv" \
    --trials 50 \
    --eda-budget 10 \
    --results /opt/ppa/results/bo_real/ \
    --seed 42 \
    2>&1 | tee "$LOG_DIR/run_bo.log"

log "  Syncing BO results to S3..."
bash "$SCRIPT_DIR/sync_to_s3.sh" results

# ── 4c. RL PPO ───────────────────────────────────────────────────────────────
log "Step 4c: Running RL PPO (500k steps)..."
python3 "$CODE_DIR/optimize/run_rl.py" \
    --model "$BEST_CKPT" \
    --manifest "$PPA_DATA_ROOT/manifest_real.csv" \
    --steps 500000 \
    --eval-freq 10000 \
    --results /opt/ppa/results/rl_real/ \
    --seed 42 \
    2>&1 | tee "$LOG_DIR/run_rl.log"

log "  Syncing RL results to S3..."
bash "$SCRIPT_DIR/sync_to_s3.sh" results

# ═══════════════════════════════════════════════════════════════════════════════
# STAGE 5: EVALUATION
# ═══════════════════════════════════════════════════════════════════════════════

log "═══ STAGE 5: Evaluation ═══"

log "Step 5: Running evaluation..."
python3 "$CODE_DIR/eval/eval_models.py" \
    --manifest "$PPA_DATA_ROOT/manifest_real.csv" \
    --gbdt-model /opt/ppa/models/gbdt/ \
    --gat-dir /opt/ppa/models/real/ \
    --results /opt/ppa/results/ \
    2>&1 | tee "$LOG_DIR/eval_models.log"

# ── Final sync ────────────────────────────────────────────────────────────────
log "Final sync: all results to S3..."
bash "$SCRIPT_DIR/sync_to_s3.sh" all

log ""
log "═══════════════════════════════════════════════════════════════"
log " STAGES 3-5 COMPLETE"
log " All results synced to s3://${S3_BUCKET}/ppa-run/"
log ""
log " Key outputs:"
log "   models/gbdt/          — GBDT checkpoints"
log "   models/real/           — GAT checkpoints (5 folds × 2 configs)"
log "   results/bo_real/       — BO: summary, Pareto front, convergence plot"
log "   results/rl_real/       — RL: summary, reward curve, eval log"
log "   results/               — Evaluation report + plots"
log "═══════════════════════════════════════════════════════════════"
