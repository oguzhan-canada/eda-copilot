"""
Component 1.2 — Contamination check against EDABench holdout seeds.

MinHash scan of qa_train.jsonl against EDABench seed Q&A pairs.
Near-duplicates are quarantined (not discarded) for traceability.

Usage:
    python -m pipeline.synth_qa.contamination_check \
        --input data/synthetic/qa_train.jsonl \
        --holdout data/edabench/seeds \
        --output results/contamination/qa_check.json \
        --quarantine data/synthetic/qa_contaminated.jsonl
"""
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import yaml
from datasketch import MinHash, MinHashLSH


NUM_PERM = 128
DEFAULT_THRESHOLD = 0.70  # Looser than corpus dedup — catch any overlap


def text_to_minhash(text: str, num_perm: int = NUM_PERM) -> MinHash:
    """Compute MinHash from text using word-level 5-grams."""
    m = MinHash(num_perm=num_perm)
    words = text.lower().split()
    for i in range(max(1, len(words) - 4)):
        shingle = " ".join(words[i:i + 5])
        m.update(shingle.encode("utf-8"))
    return m


def load_seeds(holdout_dir: Path) -> list[dict]:
    """Load all seed YAML files from the holdout directory."""
    seeds = []
    for f in holdout_dir.rglob("*.yaml"):
        with open(f, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
        if isinstance(data, list):
            seeds.extend(data)
        elif isinstance(data, dict) and "id" in data:
            seeds.append(data)
    return seeds


def main():
    parser = argparse.ArgumentParser(description="Contamination check vs EDABench seeds")
    parser.add_argument("--input", required=True, help="qa_train.jsonl path")
    parser.add_argument("--holdout", required=True, help="EDABench seeds directory")
    parser.add_argument("--output", default="results/contamination/qa_check.json")
    parser.add_argument("--quarantine", default="data/synthetic/qa_contaminated.jsonl")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    args = parser.parse_args()

    input_path = Path(args.input)
    holdout_dir = Path(args.holdout)
    output_path = Path(args.output)
    quarantine_path = Path(args.quarantine)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    quarantine_path.parent.mkdir(parents=True, exist_ok=True)

    # Load holdout seeds
    seeds = load_seeds(holdout_dir)
    print(f"Loaded {len(seeds)} holdout seed Q&A pairs")

    # Build LSH index from seeds
    lsh = MinHashLSH(threshold=args.threshold, num_perm=NUM_PERM)
    seed_hashes = {}

    for seed in seeds:
        seed_id = seed.get("id", "unknown")
        seed_text = f"{seed.get('question', '')} {seed.get('gold_answer', '')}"
        mh = text_to_minhash(seed_text)
        seed_hashes[seed_id] = mh
        try:
            lsh.insert(f"seed_{seed_id}", mh)
        except ValueError:
            pass

    print(f"Built LSH index with {len(seed_hashes)} seeds (threshold={args.threshold})")

    # Scan training Q&A
    qa_records = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                qa_records.append(json.loads(line))

    print(f"Scanning {len(qa_records)} training Q&A pairs...")

    contaminated = []
    clean = []

    for qa in qa_records:
        qa_text = f"{qa.get('question', '')} {qa.get('answer', '')}"
        qa_mh = text_to_minhash(qa_text)

        matches = lsh.query(qa_mh)
        seed_matches = [m for m in matches if m.startswith("seed_")]

        if seed_matches:
            # Compute exact Jaccard with each matched seed
            best_match = None
            best_sim = 0.0
            for match_key in seed_matches:
                seed_id = match_key.replace("seed_", "")
                if seed_id in seed_hashes:
                    sim = qa_mh.jaccard(seed_hashes[seed_id])
                    if sim > best_sim:
                        best_sim = sim
                        best_match = seed_id

            qa["contamination"] = {
                "matched_seed": best_match,
                "jaccard_similarity": round(best_sim, 4),
                "threshold": args.threshold,
            }
            contaminated.append(qa)
        else:
            clean.append(qa)

    # Write quarantined records
    if contaminated:
        with open(quarantine_path, "w", encoding="utf-8") as f:
            for qa in contaminated:
                f.write(json.dumps(qa, ensure_ascii=False) + "\n")

    # Overwrite input with clean records only
    with open(input_path, "w", encoding="utf-8") as f:
        for qa in clean:
            f.write(json.dumps(qa, ensure_ascii=False) + "\n")

    # Write report
    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "threshold": args.threshold,
        "num_perm": NUM_PERM,
        "holdout_seeds": len(seeds),
        "total_qa_scanned": len(qa_records),
        "contaminated": len(contaminated),
        "clean": len(clean),
        "contamination_rate_pct": round(len(contaminated) / max(len(qa_records), 1) * 100, 4),
        "quarantined_to": str(quarantine_path),
        "contaminated_details": [
            {
                "case_id": qa.get("case_id"),
                "matched_seed": qa["contamination"]["matched_seed"],
                "jaccard_similarity": qa["contamination"]["jaccard_similarity"],
            }
            for qa in contaminated
        ],
    }

    output_path.write_text(json.dumps(report, indent=2))

    # Summary
    print(f"\n{'='*60}")
    print(f"CONTAMINATION CHECK SUMMARY")
    print(f"{'='*60}")
    print(f"Total scanned:      {len(qa_records):,}")
    print(f"Clean:              {len(clean):,}")
    print(f"Contaminated:       {len(contaminated):,}")
    print(f"Contamination rate: {report['contamination_rate_pct']:.2f}%")
    print(f"Target:             0 near-duplicates")

    if contaminated:
        print(f"\nQuarantined records:")
        for qa in contaminated:
            c = qa["contamination"]
            print(f"  {qa.get('case_id'):30s} -> seed {c['matched_seed']} "
                  f"(sim={c['jaccard_similarity']:.3f})")

    print(f"\nClean output:       {input_path}")
    print(f"Quarantine:         {quarantine_path}")
    print(f"Report:             {output_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
