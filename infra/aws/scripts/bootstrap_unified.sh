#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# bootstrap_unified.sh — Phase 0 unified instance (ORFS v3.0 + ML stack)
#
# Target: c6i.4xlarge Spot, Ubuntu 22.04 (Deep Learning AMI)
# Installs: ORFS v3.0 + conda env (ppa-ext) + V2PYG + AutoTuner deps
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail
exec > /var/log/bootstrap_unified.log 2>&1

ORFS_TAG="v3.0"
ORFS_SHA="181e9133776117ea1b9f74dbacbfdaadff8c331b"
S3_BUCKET="${S3_BUCKET:-ppa-framework-30ee10a0}"
AWS_REGION="${AWS_REGION:-us-east-1}"

echo "══════════════════════════════════════════════════════════════"
echo " PPA Framework Extension — Unified Instance Bootstrap"
echo " ORFS: ${ORFS_TAG} (${ORFS_SHA:0:8})"
echo " Region: ${AWS_REGION}"
echo " S3: ${S3_BUCKET}"
echo "══════════════════════════════════════════════════════════════"

# ── System packages ──────────────────────────────────────────────────────────
apt-get update -y
apt-get install -y git cmake build-essential tcl-dev libreadline-dev \
    wget curl unzip parallel htop tmux swig libboost-all-dev libeigen3-dev \
    liblemon-dev zlib1g-dev libbz2-dev

# ── Miniconda (if not already present) ───────────────────────────────────────
if ! command -v conda &>/dev/null; then
    echo "▸ Installing Miniconda..."
    wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh
    bash /tmp/miniconda.sh -b -p /opt/miniconda3
    eval "$(/opt/miniconda3/bin/conda shell.bash hook)"
    conda init bash
    rm /tmp/miniconda.sh
else
    eval "$(conda shell.bash hook)"
fi

# ── OpenROAD-flow-scripts at pinned tag ──────────────────────────────────────
echo "▸ Installing ORFS ${ORFS_TAG}..."
cd /opt
if [ ! -d "OpenROAD-flow-scripts" ]; then
    git clone https://github.com/The-OpenROAD-Project/OpenROAD-flow-scripts.git
fi
cd OpenROAD-flow-scripts
git fetch --tags
git checkout "${ORFS_TAG}"

# Verify checkout
ACTUAL_SHA=$(git rev-parse HEAD)
echo "  Checked out: ${ACTUAL_SHA}"
if [ "${ACTUAL_SHA}" != "${ORFS_SHA}" ]; then
    echo "  WARNING: SHA mismatch. Expected ${ORFS_SHA}, got ${ACTUAL_SHA}."
    echo "  Tag may have been retagged. Proceeding with tag checkout."
fi

# Build OpenROAD + Yosys
sudo ./setup.sh
./build_openroad.sh --local 2>&1 | tail -30

export PATH="/opt/OpenROAD-flow-scripts/tools/install/OpenROAD/bin:$PATH"
openroad -version && echo "✓ OpenROAD installed" || { echo "✗ OpenROAD build FAILED"; exit 1; }

# Verify ASAP7 designs exist
ASAP7_DIR="/opt/OpenROAD-flow-scripts/flow/designs/asap7"
for design in aes ibex jpeg riscv32i gcd ethmac uart; do
    if [ -d "${ASAP7_DIR}/${design}" ]; then
        echo "  ✓ ${design}"
    else
        echo "  ⚠ ${design} not found in ASAP7 designs"
    fi
done

# ── Conda environment ────────────────────────────────────────────────────────
echo "▸ Creating conda env ppa-ext..."
cd /opt/ppa
conda env create -f environment.yml || conda env update -f environment.yml
conda activate ppa-ext

# Install PyTorch CPU + PyG ecosystem
pip install torch==2.5.0+cpu --extra-index-url https://download.pytorch.org/whl/cpu
pip install torch-scatter==2.1.2+pt25cpu torch-sparse==0.6.18+pt25cpu \
    --find-links https://data.pyg.org/whl/torch-2.5.0+cpu.html
pip install torch-geometric==2.7.0

# Install remaining deps
pip install -r requirements_extension.txt \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    --find-links https://data.pyg.org/whl/torch-2.5.0+cpu.html

# ── V2PYG (Verilog-to-PyG) ──────────────────────────────────────────────────
echo "▸ Installing V2PYG..."
cd /opt
if [ ! -d "Verilog-to-PyG" ]; then
    git clone https://github.com/Yu-Maryland/Verilog-to-PyG.git
fi
cd Verilog-to-PyG
pip install -e . 2>/dev/null || echo "  V2PYG: no setup.py, will use as submodule"

# ── AutoTuner deps (Ray already in requirements) ─────────────────────────────
echo "▸ Verifying AutoTuner..."
python -c "import ray; print(f'  ✓ Ray {ray.__version__}')"
python -c "import hyperopt; print(f'  ✓ HyperOpt {hyperopt.__version__}')"

# ── Validation ───────────────────────────────────────────────────────────────
echo ""
echo "▸ Running import validation..."
python -c "
import torch; print(f'  torch {torch.__version__}')
import torch_geometric; print(f'  PyG {torch_geometric.__version__}')
import torch_scatter; print(f'  scatter OK')
import torch_sparse; print(f'  sparse OK')
import botorch; print(f'  BoTorch {botorch.__version__}')
import xgboost; print(f'  XGBoost {xgboost.__version__}')
import stable_baselines3; print(f'  SB3 {stable_baselines3.__version__}')
import gymnasium; print(f'  Gymnasium {gymnasium.__version__}')
print('  ✓ All imports passed')
"

# ── Create workspace ─────────────────────────────────────────────────────────
mkdir -p /opt/ppa/{data/raw,features/graph,results,models,scripts,code}
chown -R ubuntu:ubuntu /opt/ppa

# ── Pull project code from S3 ────────────────────────────────────────────────
echo "▸ Syncing project from S3..."
aws s3 sync "s3://${S3_BUCKET}/ppa/" /opt/ppa/ \
    --exclude "data/raw/openabc_d/*" \
    --exclude "data/raw/openroad_runs/*" \
    || echo "  S3 sync partial (some paths may not exist yet)"

# ── Write environment config ─────────────────────────────────────────────────
cat > /opt/ppa/env.sh <<'ENVEOF'
export AWS_REGION="${AWS_REGION:-us-east-1}"
export S3_BUCKET="${S3_BUCKET:-ppa-framework-30ee10a0}"
export PPA_DATA_ROOT=/opt/ppa/data
export WORK_DIR=/opt/ppa
export PYTHONPATH=/opt/ppa/code
export PATH="/opt/OpenROAD-flow-scripts/tools/install/OpenROAD/bin:$PATH"
export OPENROAD_BIN=/opt/OpenROAD-flow-scripts/tools/install/OpenROAD/bin/openroad
export YOSYS_BIN=/opt/OpenROAD-flow-scripts/tools/install/yosys/bin/yosys
export PDK_ROOT=/opt/OpenROAD-flow-scripts/flow/platforms
export V2PYG_ROOT=/opt/Verilog-to-PyG
eval "$(conda shell.bash hook)"
conda activate ppa-ext
ENVEOF

echo ""
echo "══════════════════════════════════════════════════════════════"
echo " Bootstrap complete. SSH in and run:"
echo "   source /opt/ppa/env.sh"
echo "   cd /opt/ppa && pytest tests/ -v"
echo "══════════════════════════════════════════════════════════════"
