#!/bin/bash
# GPU instance setup script for QLoRA training
# Run on a fresh g5.2xlarge (Ubuntu 22.04, Deep Learning AMI)
#
# Usage:
#   chmod +x scripts/setup_gpu.sh && ./scripts/setup_gpu.sh

set -e

echo "=== GPU Instance Setup for EDA LoRA Training ==="

# Verify GPU
nvidia-smi
echo ""

# Clone or sync project data
if [ ! -d "llm-eda-kg" ]; then
    echo "Syncing project from S3..."
    aws s3 sync s3://eda-kg-e6c0f9f2/project/ llm-eda-kg/ --exclude "*.pyc" --exclude "__pycache__/*"
fi

cd llm-eda-kg

# Install dependencies
pip install -r requirements-train.txt --quiet
echo "Dependencies installed"

# Verify torch + CUDA
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA {torch.cuda.get_device_name(0)}')"

# Sync training data from S3
echo "Syncing training data..."
mkdir -p data/train
aws s3 sync s3://eda-kg-e6c0f9f2/train/ data/train/

echo ""
echo "Setup complete. Run training with:"
echo "  python -m pipeline.train.train_lora \\"
echo "    --train-data data/train/train.jsonl \\"
echo "    --val-data data/train/val.jsonl \\"
echo "    --output-dir results/train/lora-eda \\"
echo "    --s3-bucket eda-kg-e6c0f9f2 \\"
echo "    --epochs 3"
