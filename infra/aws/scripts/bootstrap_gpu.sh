#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# bootstrap_gpu.sh — Userdata script for GPU Spot instance (Stages 3-5)
#
# Deep Learning AMI (Ubuntu 22.04) already has NVIDIA drivers, CUDA, PyTorch.
# Installs OpenROAD (for Stage 4 EDA verification) + project Python deps.
# No Docker needed.
# Template variables filled by Terraform: aws_region, s3_bucket
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail
exec > /var/log/bootstrap.log 2>&1

echo "══════════════════════════════════════════════════════════════"
echo " PPA Framework — GPU Instance Bootstrap (Direct Install)"
echo " Region:  ${aws_region}"
echo " S3:      ${s3_bucket}"
echo "══════════════════════════════════════════════════════════════"

# ── Verify GPU ────────────────────────────────────────────────────────────────
echo "▸ Checking NVIDIA GPU..."
nvidia-smi || { echo "ERROR: nvidia-smi failed — no GPU driver"; exit 1; }

# ── System packages ──────────────────────────────────────────────────────────
apt-get update -y
apt-get install -y git cmake build-essential tcl-dev libreadline-dev \
    wget curl unzip parallel htop tmux

# ── OpenROAD from source (needed for Stage 4 EDA verification) ───────────────
echo "▸ Installing OpenROAD-flow-scripts..."
cd /opt
git clone --depth 1 https://github.com/The-OpenROAD-Project/OpenROAD-flow-scripts.git
cd OpenROAD-flow-scripts
sudo ./setup.sh
./build_openroad.sh --local 2>&1 | tail -30

export PATH="/opt/OpenROAD-flow-scripts/tools/install/OpenROAD/bin:$PATH"
openroad -version && echo "✓ OpenROAD installed" || { echo "✗ OpenROAD FAILED"; exit 1; }

# ── Python deps (GPU PyTorch already on DLAMI, add project-specific) ─────────
echo "▸ Installing additional Python packages..."
pip install --upgrade pip
pip install pandas numpy pyyaml networkx scipy scikit-learn xgboost matplotlib
pip install torch-scatter torch-sparse torch-geometric \
    -f https://data.pyg.org/whl/torch-2.1.0+cu118.html
pip install botorch==0.9.5 gpytorch==1.11.0 stable-baselines3==2.0.0 tensorboard==2.17.1

# Verify PyTorch GPU
python3 -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'; print(f'✓ PyTorch {torch.__version__} with CUDA {torch.version.cuda}')"

# ── Create workspace ─────────────────────────────────────────────────────────
mkdir -p /opt/ppa/{data,features,models,results/logs,scripts,code}
chown -R ubuntu:ubuntu /opt/ppa

# ── Pull project code + scripts from S3 ──────────────────────────────────────
echo "▸ Pulling project code from S3..."
aws s3 sync s3://${s3_bucket}/code/ /opt/ppa/code/ || true
aws s3 sync s3://${s3_bucket}/scripts/ /opt/ppa/scripts/ || true
chmod +x /opt/ppa/scripts/*.sh 2>/dev/null || true

# ── Write environment config ─────────────────────────────────────────────────
cat > /opt/ppa/env.sh <<ENVEOF
export AWS_REGION="${aws_region}"
export S3_BUCKET="${s3_bucket}"
export PPA_DATA_ROOT=/opt/ppa/data
export WORK_DIR=/opt/ppa
export PYTHONPATH=/opt/ppa/code
export PATH="/opt/OpenROAD-flow-scripts/tools/install/OpenROAD/bin:\$PATH"
export OPENROAD_BIN=/opt/OpenROAD-flow-scripts/tools/install/OpenROAD/bin/openroad
export YOSYS_BIN=/opt/OpenROAD-flow-scripts/tools/install/yosys/bin/yosys
export PDK_ROOT=/opt/OpenROAD-flow-scripts/flow/platforms
ENVEOF

echo ""
echo "══════════════════════════════════════════════════════════════"
echo " Bootstrap complete (~15 min). SSH in and run:"
echo "   source /opt/ppa/env.sh"
echo "   bash /opt/ppa/scripts/run_stages_3_5.sh"
echo "══════════════════════════════════════════════════════════════"
