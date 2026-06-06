#!/bin/bash
set -e

# LoRA baseline evaluation — runs on g5.xlarge spot instance
# Self-terminates after completion

BUCKET="eda-kg-e6c0f9f2"
REGION="us-east-1"

echo "=== Starting LoRA baseline eval $(date) ==="

# Wait for GPU driver
timeout 120 bash -c 'until nvidia-smi &>/dev/null; do sleep 5; done' || true
nvidia-smi || echo "WARNING: No GPU detected"

# Install deps
pip install -q torch transformers peft accelerate bitsandbytes sentencepiece protobuf 2>&1 | tail -5

# Download benchmark and eval script from S3
mkdir -p /tmp/lora-eval
aws s3 cp s3://$BUCKET/lora-eda/final/ /tmp/lora-eval/adapter/ --recursive --region $REGION
aws s3 cp s3://$BUCKET/edabench/edabench_v1.jsonl /tmp/lora-eval/edabench_v1.jsonl --region $REGION 2>/dev/null || true
aws s3 cp s3://$BUCKET/eval/lora_baseline_eval.py /tmp/lora-eval/lora_baseline_eval.py --region $REGION 2>/dev/null || true

# If benchmark not on S3 yet, fail
if [ ! -f /tmp/lora-eval/edabench_v1.jsonl ]; then
    echo "ERROR: edabench_v1.jsonl not found on S3"
    # Self-terminate
    INSTANCE_ID=$(curl -s http://169.254.169.254/latest/meta-data/instance-id)
    aws ec2 terminate-instances --instance-ids $INSTANCE_ID --region $REGION
    exit 1
fi

cd /tmp/lora-eval

# Run evaluation
echo "=== Starting inference $(date) ==="
python lora_baseline_eval.py \
    --benchmark edabench_v1.jsonl \
    --adapter-path ./adapter/ \
    --output baseline_lora_only.json \
    --max-new-tokens 512

echo "=== Inference complete $(date) ==="

# Upload results
aws s3 cp baseline_lora_only.json s3://$BUCKET/eval/baseline_lora_only.json --region $REGION
echo "Results uploaded to s3://$BUCKET/eval/baseline_lora_only.json"

# Self-terminate
INSTANCE_ID=$(curl -s http://169.254.169.254/latest/meta-data/instance-id)
echo "Self-terminating instance $INSTANCE_ID"
aws ec2 terminate-instances --instance-ids $INSTANCE_ID --region $REGION
