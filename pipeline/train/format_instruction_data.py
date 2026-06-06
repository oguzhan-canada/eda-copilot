"""
Format synthetic QA pairs into instruction-tuning dataset for QLoRA.

Reads qa_train.jsonl + qa_newcats_train.jsonl, merges, applies
class-weighted sampling to balance minority categories, then
splits 80/10/10 into train/val/test.

Output format per record:
{
    "instruction": "You are an EDA expert...",
    "input": "<question>",
    "output": "<answer>",
    "task_category": "error_diagnosis",
    "source": "synthetic"
}

Usage:
    python -m pipeline.train.format_instruction_data
    python -m pipeline.train.format_instruction_data --min-category-pct 0.15
"""

import argparse
import json
import os
import random
from collections import Counter
from pathlib import Path


SYSTEM_INSTRUCTION = (
    "You are an EDA (Electronic Design Automation) expert assistant. "
    "Answer the user's question about ASIC design, synthesis, place-and-route, "
    "timing analysis, DRC, or EDA tool usage. Be specific, cite relevant tools "
    "or standards, and provide actionable guidance."
)

# Map non-standard categories to canonical ones
CATEGORY_MAP = {
    "error_diagnosis": "error_diagnosis",
    "rtl_qa": "rtl_qa",
    "constraint_generation": "constraint_generation",
    "cross_tool_knowledge": "cross_tool_knowledge",
    "optimization_advisory": "cross_tool_knowledge",  # Merge into cross_tool
    "general_eda": "cross_tool_knowledge",
    "drc_rule_lookup": "drc_rule_lookup",
}


def load_qa_pairs(data_dir: str) -> list[dict]:
    """Load and merge all QA JSONL files."""
    pairs = []
    for filename in ["qa_train.jsonl", "qa_newcats_train.jsonl"]:
        path = Path(data_dir) / filename
        if not path.exists():
            print(f"  Skipping {path} (not found)")
            continue
        with open(path, encoding="utf-8") as f:
            for line in f:
                record = json.loads(line)
                pairs.append(record)
        print(f"  Loaded {path.name}: {len(pairs)} cumulative")
    return pairs


def format_record(qa: dict) -> dict:
    """Convert a QA pair to instruction format."""
    raw_cat = qa.get("task_category", "error_diagnosis")
    category = CATEGORY_MAP.get(raw_cat, raw_cat)

    return {
        "instruction": SYSTEM_INSTRUCTION,
        "input": qa["question"].strip(),
        "output": qa["answer"].strip(),
        "task_category": category,
        "source": "synthetic",
        "judge_score": qa.get("judge_score"),
        "source_file": qa.get("source_file", ""),
    }


def oversample_minorities(
    records: list[dict],
    min_pct: float = 0.15,
    seed: int = 42,
) -> list[dict]:
    """Oversample minority categories to reach min_pct of total.

    Only oversamples categories below the threshold — does not
    downsample majority categories.
    """
    rng = random.Random(seed)

    by_cat = {}
    for r in records:
        cat = r["task_category"]
        by_cat.setdefault(cat, []).append(r)

    total = len(records)
    min_count = int(total * min_pct)
    augmented = list(records)  # Start with all originals

    for cat, items in by_cat.items():
        if len(items) < min_count:
            deficit = min_count - len(items)
            oversampled = rng.choices(items, k=deficit)
            augmented.extend(oversampled)
            print(f"  Oversampled {cat}: {len(items)} -> {len(items) + deficit} "
                  f"(+{deficit})")

    rng.shuffle(augmented)
    return augmented


def stratified_split(
    records: list[dict],
    train_pct: float = 0.8,
    val_pct: float = 0.1,
    seed: int = 42,
) -> tuple[list, list, list]:
    """Stratified 80/10/10 split preserving category distribution."""
    rng = random.Random(seed)

    by_cat = {}
    for r in records:
        cat = r["task_category"]
        by_cat.setdefault(cat, []).append(r)

    train, val, test = [], [], []

    for cat, items in by_cat.items():
        rng.shuffle(items)
        n = len(items)
        n_train = int(n * train_pct)
        n_val = int(n * val_pct)

        train.extend(items[:n_train])
        val.extend(items[n_train:n_train + n_val])
        test.extend(items[n_train + n_val:])

    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)

    return train, val, test


def main():
    parser = argparse.ArgumentParser(
        description="Format QA pairs into instruction-tuning dataset"
    )
    parser.add_argument(
        "--data-dir", default="data/synthetic",
        help="Directory containing qa_train.jsonl files",
    )
    parser.add_argument(
        "--output-dir", default="data/train",
        help="Output directory for train/val/test splits",
    )
    parser.add_argument(
        "--min-category-pct", type=float, default=0.15,
        help="Minimum percentage for each category (oversampling threshold)",
    )
    parser.add_argument(
        "--min-judge-score", type=float, default=0.6,
        help="Filter out QA pairs below this judge score",
    )
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    print("Loading QA pairs...")
    raw_pairs = load_qa_pairs(args.data_dir)
    print(f"Total raw pairs: {len(raw_pairs)}")

    # Filter by quality
    filtered = [
        p for p in raw_pairs
        if p.get("judge_score", 1.0) >= args.min_judge_score
    ]
    print(f"After quality filter (score >= {args.min_judge_score}): {len(filtered)}")

    # Format to instruction tuning
    print("\nFormatting records...")
    formatted = [format_record(p) for p in filtered]

    # Show category distribution before oversampling
    cats = Counter(r["task_category"] for r in formatted)
    print(f"\nCategory distribution (before oversampling):")
    for cat, n in cats.most_common():
        print(f"  {cat:30s} {n:6,} ({n / len(formatted) * 100:.1f}%)")

    # Oversample minorities
    print(f"\nOversampling minorities to >= {args.min_category_pct * 100:.0f}%...")
    balanced = oversample_minorities(
        formatted, min_pct=args.min_category_pct, seed=args.seed
    )

    # Show post-oversampling distribution
    cats_post = Counter(r["task_category"] for r in balanced)
    print(f"\nCategory distribution (after oversampling):")
    for cat, n in cats_post.most_common():
        print(f"  {cat:30s} {n:6,} ({n / len(balanced) * 100:.1f}%)")

    # Stratified split
    print(f"\nSplitting 80/10/10 (stratified)...")
    train, val, test = stratified_split(balanced, seed=args.seed)
    print(f"  Train: {len(train)}")
    print(f"  Val:   {len(val)}")
    print(f"  Test:  {len(test)}")

    # Verify split category distribution
    for name, split in [("Train", train), ("Val", val), ("Test", test)]:
        cats_split = Counter(r["task_category"] for r in split)
        min_cat_pct = min(n / len(split) for n in cats_split.values()) * 100
        print(f"  {name} min category: {min_cat_pct:.1f}%")

    # Write output
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for name, data in [("train", train), ("val", val), ("test", test)]:
        path = out_dir / f"{name}.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            for record in data:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(f"  Written: {path} ({len(data)} records)")

    # Write metadata
    metadata = {
        "total_raw": len(raw_pairs),
        "quality_filtered": len(filtered),
        "min_judge_score": args.min_judge_score,
        "oversampled_total": len(balanced),
        "min_category_pct": args.min_category_pct,
        "splits": {
            "train": len(train),
            "val": len(val),
            "test": len(test),
        },
        "categories": dict(cats_post.most_common()),
        "seed": args.seed,
    }
    meta_path = out_dir / "dataset_metadata.json"
    json.dump(metadata, open(meta_path, "w"), indent=2)
    print(f"  Metadata: {meta_path}")


if __name__ == "__main__":
    main()
