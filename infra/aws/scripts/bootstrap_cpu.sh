#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# bootstrap_cpu.sh — Userdata script for CPU Spot instance (Stages 1-2)
#
# Deep Learning AMI (Ubuntu 22.04) already has Python 3.10, conda, pip.
# Installs OpenROAD from source + project Python deps. No Docker needed.
# Template variables filled by Terraform: aws_region, s3_bucket
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail
exec > /var/log/bootstrap.log 2>&1

echo "══════════════════════════════════════════════════════════════"
echo " PPA Framework — CPU Instance Bootstrap (Direct Install)"
echo " Region:  ${aws_region}"
echo " S3:      ${s3_bucket}"
echo "══════════════════════════════════════════════════════════════"

# ── System packages ──────────────────────────────────────────────────────────
apt-get update -y
apt-get install -y git cmake build-essential tcl-dev libreadline-dev \
    wget curl unzip parallel htop tmux

# ── OpenROAD from source ─────────────────────────────────────────────────────
echo "▸ Installing OpenROAD-flow-scripts..."
cd /opt
git clone --depth 1 https://github.com/The-OpenROAD-Project/OpenROAD-flow-scripts.git
cd OpenROAD-flow-scripts
sudo ./setup.sh
./build_openroad.sh --local 2>&1 | tail -30

export PATH="/opt/OpenROAD-flow-scripts/tools/install/OpenROAD/bin:$PATH"
openroad -version && echo "✓ OpenROAD installed" || { echo "✗ OpenROAD FAILED"; exit 1; }

# ── Python deps (CPU-only PyTorch) ───────────────────────────────────────────
echo "▸ Installing Python packages..."
pip install --upgrade pip
pip install pandas numpy pyyaml networkx scipy scikit-learn xgboost matplotlib
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install torch-scatter torch-sparse torch-geometric \
    -f https://data.pyg.org/whl/torch-2.1.0+cpu.html

# ── Create workspace ─────────────────────────────────────────────────────────
mkdir -p /opt/ppa/{data/raw/openabc_d,data/raw/circuitnet,data/raw/openroad_runs}
mkdir -p /opt/ppa/{features,results/logs,scripts,code}
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
echo "   bash /opt/ppa/scripts/run_stages_1_2.sh"
echo "══════════════════════════════════════════════════════════════"
