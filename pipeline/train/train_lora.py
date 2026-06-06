"""
QLoRA fine-tuning of Mistral-7B-Instruct on EDA instruction dataset.

Uses HuggingFace PEFT + trl SFTTrainer with 4-bit NF4 quantization.

Usage (on GPU instance):
    pip install -r requirements-train.txt
    python -m pipeline.train.train_lora \
        --train-data data/train/train.jsonl \
        --val-data data/train/val.jsonl \
        --output-dir results/train/lora-eda \
        --epochs 3

Resume from checkpoint:
    python -m pipeline.train.train_lora \
        --resume-from-checkpoint results/train/lora-eda/checkpoint-500

Requirements:
    torch>=2.1
    transformers>=4.40
    peft>=0.10
    trl>=0.8
    bitsandbytes>=0.43
    datasets
    accelerate
    wandb (optional)
"""

import argparse
import json
import os
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from trl import SFTTrainer, SFTConfig


BASE_MODEL = "mistralai/Mistral-7B-Instruct-v0.3"
MAX_SEQ_LENGTH = 1024  # Cap sequence length for memory


def load_dataset_jsonl(path: str) -> Dataset:
    """Load JSONL instruction data into HF Dataset."""
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))
    return Dataset.from_list(records)


def format_prompt(record: dict) -> str:
    """Format instruction/input/output into Mistral chat template."""
    return (
        f"<s>[INST] {record['instruction']}\n\n{record['input']} [/INST]"
        f"{record['output']}</s>"
    )


def main():
    parser = argparse.ArgumentParser(description="QLoRA fine-tuning for EDA copilot")
    parser.add_argument("--train-data", default="data/train/train.jsonl")
    parser.add_argument("--val-data", default="data/train/val.jsonl")
    parser.add_argument("--output-dir", default="results/train/lora-eda")
    parser.add_argument("--base-model", default=BASE_MODEL)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.1)
    parser.add_argument("--max-seq-length", type=int, default=MAX_SEQ_LENGTH)
    parser.add_argument("--eval-steps", type=int, default=100)
    parser.add_argument("--save-steps", type=int, default=500)
    parser.add_argument("--resume-from-checkpoint", type=str, default=None)
    parser.add_argument("--s3-bucket", type=str, default=None,
                        help="S3 bucket for checkpoint sync")
    args = parser.parse_args()

    print(f"{'='*60}")
    print(f"QLoRA Training — EDA Copilot")
    print(f"{'='*60}")
    print(f"Base model:    {args.base_model}")
    print(f"Train data:    {args.train_data}")
    print(f"Val data:      {args.val_data}")
    print(f"Output dir:    {args.output_dir}")
    print(f"Epochs:        {args.epochs}")
    print(f"Batch size:    {args.batch_size} (effective: {args.batch_size * args.grad_accum})")
    print(f"LoRA r={args.lora_r}, alpha={args.lora_alpha}, dropout={args.lora_dropout}")
    print(f"GPU:           {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU (WILL FAIL)'}")
    print(f"{'='*60}\n")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for QLoRA training")

    # Load datasets
    print("Loading datasets...")
    train_ds = load_dataset_jsonl(args.train_data)
    val_ds = load_dataset_jsonl(args.val_data)
    print(f"  Train: {len(train_ds)} records")
    print(f"  Val:   {len(val_ds)} records")

    # Add formatted text column
    train_ds = train_ds.map(
        lambda x: {"text": format_prompt(x)},
        desc="Formatting train prompts",
    )
    val_ds = val_ds.map(
        lambda x: {"text": format_prompt(x)},
        desc="Formatting val prompts",
    )

    # 4-bit quantization config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )

    # Load model
    print(f"\nLoading {args.base_model} with 4-bit quantization...")
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16,
    )
    model = prepare_model_for_kbit_training(model)

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # LoRA config
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        bias="none",
    )

    model = get_peft_model(model, lora_config)
    trainable, total = model.get_nb_trainable_parameters()
    print(f"Trainable parameters: {trainable:,} / {total:,} "
          f"({trainable / total * 100:.2f}%)")

    # Training arguments
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    training_args = SFTConfig(
        max_length=args.max_seq_length,
        output_dir=str(output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_steps=50,
        weight_decay=0.01,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        fp16=False,
        bf16=False,
        optim="adamw_torch",
        max_grad_norm=0.3,
        report_to="none",
        dataloader_num_workers=2,
        gradient_checkpointing=True,
    )

    # Trainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        args=training_args,
        processing_class=tokenizer,
    )

    # Train
    print("\nStarting training...")
    if args.resume_from_checkpoint:
        print(f"Resuming from: {args.resume_from_checkpoint}")

    train_result = trainer.train(
        resume_from_checkpoint=args.resume_from_checkpoint
    )

    # Save final model
    print("\nSaving final model...")
    trainer.save_model(str(output_dir / "final"))
    tokenizer.save_pretrained(str(output_dir / "final"))

    # Save training metrics
    metrics = train_result.metrics
    metrics["train_samples"] = len(train_ds)
    metrics["val_samples"] = len(val_ds)

    with open(output_dir / "training_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Training metrics saved to {output_dir / 'training_metrics.json'}")

    # Evaluate
    print("\nRunning final evaluation...")
    eval_metrics = trainer.evaluate()
    with open(output_dir / "eval_metrics.json", "w") as f:
        json.dump(eval_metrics, f, indent=2)
    print(f"Eval loss: {eval_metrics.get('eval_loss', 'N/A')}")

    # Sync to S3 if configured
    if args.s3_bucket:
        s3_path = f"s3://{args.s3_bucket}/lora-eda/"
        print(f"\nSyncing to {s3_path}...")
        os.system(f"aws s3 sync {output_dir / 'final'} {s3_path}final/")
        os.system(f"aws s3 cp {output_dir / 'training_metrics.json'} {s3_path}")
        os.system(f"aws s3 cp {output_dir / 'eval_metrics.json'} {s3_path}")
        print("S3 sync complete")

    print(f"\n{'='*60}")
    print("Training complete!")
    print(f"Model saved: {output_dir / 'final'}")
    print(f"Eval loss:   {eval_metrics.get('eval_loss', 'N/A')}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
