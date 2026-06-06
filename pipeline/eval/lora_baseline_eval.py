#!/usr/bin/env python3
"""
LoRA-only baseline evaluation for EDABench.

Runs the fine-tuned Mistral-7B model (without any retrieval) on all
EDABench items and saves results in the same format as evaluate_system.py.

Usage (on GPU instance):
    python lora_baseline_eval.py \
        --benchmark edabench_v1.jsonl \
        --adapter-path ./lora-eda-final/ \
        --output baseline_lora_only.json
"""
import argparse
import json
import time
from pathlib import Path
from statistics import mean


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--adapter-path", required=True)
    parser.add_argument("--output", default="baseline_lora_only.json")
    parser.add_argument("--base-model", default="mistralai/Mistral-7B-Instruct-v0.3")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    args = parser.parse_args()

    # Load benchmark
    items = []
    with open(args.benchmark, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                items.append(json.loads(line))
    print(f"Loaded {len(items)} benchmark items")

    # Load model
    print(f"Loading base model: {args.base_model}")
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import PeftModel

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )

    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        quantization_config=bnb_config,
        device_map="auto",
    )
    print(f"Loading LoRA adapter: {args.adapter_path}")
    model = PeftModel.from_pretrained(base_model, args.adapter_path)
    model.eval()
    print("Model loaded")

    # Run inference
    results = []
    latencies = []

    for i, item in enumerate(items):
        query = item["query"]
        print(f"  [{i+1}/{len(items)}] {item['id']}: {query[:60]}...", end="", flush=True)

        prompt = f"[INST] You are an EDA engineering assistant. Answer the following question accurately and concisely.\n\n{query} [/INST]"

        t0 = time.perf_counter()
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                temperature=0.1,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
            )

        response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        latency = time.perf_counter() - t0

        result = {
            "id": item["id"],
            "query": query,
            "task_category": item["task_category"],
            "difficulty": item.get("difficulty", "medium"),
            "retrieval_precision": None,
            "source_recall": None,
            "category_match": True,
            "latency_s": round(latency, 3),
            "graph_facts_count": 0,
            "chunks_count": 0,
            "chunk_sources": [],
            "system_answer": response.strip(),
            "judge_scores": None,
        }
        results.append(result)
        latencies.append(latency)
        print(f" ({latency:.1f}s, {len(response)} chars)", flush=True)

    # Save results
    output = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "benchmark_file": args.benchmark,
        "model": args.base_model,
        "adapter": args.adapter_path,
        "num_items": len(items),
        "mean_latency_s": round(mean(latencies), 3),
        "results": results,
    }
    Path(args.output).write_text(json.dumps(output, indent=2))
    print(f"\nResults written to {args.output}")
    print(f"Mean latency: {mean(latencies):.1f}s")


if __name__ == "__main__":
    main()
