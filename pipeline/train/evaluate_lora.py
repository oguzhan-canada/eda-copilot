"""
Evaluate LoRA fine-tuned model against base model on validation set.

Compares perplexity/loss of the fine-tuned LoRA model vs the base
Mistral-7B-Instruct to verify training improved EDA task performance.

Usage:
    python -m pipeline.train.evaluate_lora \
        --lora-path results/train/lora-eda/final \
        --eval-data data/train/val.jsonl \
        --output results/train/eval_results.json
"""

import argparse
import json
import time
from pathlib import Path

import torch
from datasets import Dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from tqdm import tqdm


BASE_MODEL = "mistralai/Mistral-7B-Instruct-v0.3"


def load_eval_data(path: str) -> list[dict]:
    """Load JSONL eval records."""
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))
    return records


def format_prompt(record: dict) -> str:
    """Mistral chat template."""
    return (
        f"<s>[INST] {record['instruction']}\n\n{record['input']} [/INST]"
        f"{record['output']}</s>"
    )


def compute_perplexity(model, tokenizer, texts: list[str], batch_size: int = 4) -> dict:
    """Compute average perplexity over texts."""
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    for i in tqdm(range(0, len(texts), batch_size), desc="Evaluating"):
        batch = texts[i : i + batch_size]
        encodings = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=1024,
        ).to(model.device)

        with torch.no_grad():
            outputs = model(**encodings, labels=encodings["input_ids"])

        # Accumulate loss weighted by token count
        n_tokens = encodings["attention_mask"].sum().item()
        total_loss += outputs.loss.item() * n_tokens
        total_tokens += n_tokens

    avg_loss = total_loss / total_tokens
    perplexity = torch.exp(torch.tensor(avg_loss)).item()

    return {
        "avg_loss": avg_loss,
        "perplexity": perplexity,
        "total_tokens": total_tokens,
        "num_examples": len(texts),
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate LoRA vs base model")
    parser.add_argument("--lora-path", required=True, help="Path to LoRA adapter")
    parser.add_argument("--eval-data", default="data/train/val.jsonl")
    parser.add_argument("--output", default="results/train/eval_results.json")
    parser.add_argument("--base-model", default=BASE_MODEL)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Limit eval samples (for quick testing)")
    args = parser.parse_args()

    print(f"{'='*60}")
    print("LoRA Evaluation — EDA Copilot")
    print(f"{'='*60}")

    # Load eval data
    records = load_eval_data(args.eval_data)
    if args.max_samples:
        records = records[: args.max_samples]
    texts = [format_prompt(r) for r in records]
    print(f"Eval samples: {len(texts)}")

    # 4-bit quantization config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # --- Evaluate BASE model ---
    print(f"\nLoading base model: {args.base_model}")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16,
    )

    t0 = time.time()
    base_results = compute_perplexity(base_model, tokenizer, texts, args.batch_size)
    base_time = time.time() - t0
    print(f"Base model — Loss: {base_results['avg_loss']:.4f}, "
          f"PPL: {base_results['perplexity']:.2f} ({base_time:.1f}s)")

    del base_model
    torch.cuda.empty_cache()

    # --- Evaluate LoRA model ---
    print(f"\nLoading LoRA model: {args.lora_path}")
    lora_base = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16,
    )
    lora_model = PeftModel.from_pretrained(lora_base, args.lora_path)

    t0 = time.time()
    lora_results = compute_perplexity(lora_model, tokenizer, texts, args.batch_size)
    lora_time = time.time() - t0
    print(f"LoRA model — Loss: {lora_results['avg_loss']:.4f}, "
          f"PPL: {lora_results['perplexity']:.2f} ({lora_time:.1f}s)")

    # --- Compare ---
    loss_improvement = base_results["avg_loss"] - lora_results["avg_loss"]
    ppl_improvement = base_results["perplexity"] - lora_results["perplexity"]
    improved = lora_results["avg_loss"] < base_results["avg_loss"]

    print(f"\n{'='*60}")
    print(f"Loss improvement: {loss_improvement:+.4f} ({'PASS' if improved else 'FAIL'})")
    print(f"PPL improvement:  {ppl_improvement:+.2f}")
    print(f"{'='*60}")

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    results = {
        "base_model": args.base_model,
        "lora_path": args.lora_path,
        "eval_data": args.eval_data,
        "num_samples": len(texts),
        "base": base_results,
        "lora": lora_results,
        "comparison": {
            "loss_improvement": loss_improvement,
            "perplexity_improvement": ppl_improvement,
            "lora_is_better": improved,
        },
    }

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
