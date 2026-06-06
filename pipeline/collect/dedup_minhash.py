"""
Component 1.1 (cont.) — MinHash deduplication of raw corpus.

Reads all text files from the raw corpus directory, computes MinHash
signatures, identifies near-duplicates using LSH, and writes
deduplicated copies to the staging directory.

Usage:
    python -m pipeline.collect.dedup_minhash \
        --input C:/eda-kg-data/corpus/raw_docs \
        --output C:/eda-kg-data/corpus/staging/dedup \
        --manifest data/manifests/artifacts_manifest.parquet \
        --threshold 0.90
"""
import argparse
import hashlib
import json
import shutil
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from datasketch import MinHash, MinHashLSH
from tqdm import tqdm


TEXT_EXTENSIONS = {
    ".v", ".sv", ".vhd", ".vhdl", ".tcl", ".py", ".md", ".txt", ".rst",
    ".lib", ".lef", ".def", ".sdc", ".json", ".yaml", ".yml",
    ".log", ".rpt", ".cfg", ".mk", ".sh", ".c", ".cc", ".cpp", ".h",
    ".html", ".xml", ".csv",
}

# Files smaller than this are likely empty or trivial — skip
MIN_FILE_BYTES = 100

# Files larger than this are likely generated/binary-ish — skip for perf
MAX_FILE_BYTES = 512 * 1024  # 512 KB

# Number of permutations for MinHash (higher = more accurate, slower)
NUM_PERM = 128


def compute_minhash(text: str, num_perm: int = NUM_PERM) -> MinHash:
    """Compute MinHash signature from text using word-level shingles."""
    m = MinHash(num_perm=num_perm)
    # Cap text to first 50K chars for performance
    text = text[:50000].lower()
    words = text.split()
    for i in range(len(words) - 4):
        shingle = " ".join(words[i:i + 5])
        m.update(shingle.encode("utf-8"))
    return m


def read_text_file(path: Path) -> str:
    """Read a text file, returning empty string on failure."""
    try:
        return path.read_text(encoding="utf-8", errors="ignore")
    except (PermissionError, OSError):
        return ""


def main():
    parser = argparse.ArgumentParser(description="MinHash deduplication of raw corpus")
    parser.add_argument("--input", required=True, help="Raw corpus directory")
    parser.add_argument("--output", required=True, help="Deduplicated output directory")
    parser.add_argument("--manifest", default="data/manifests/artifacts_manifest.parquet")
    parser.add_argument("--threshold", type=float, default=0.90,
                        help="Jaccard similarity threshold for near-duplicate detection")
    parser.add_argument("--report", default="results/reports/dedup_report.json")
    args = parser.parse_args()

    input_root = Path(args.input)
    output_root = Path(args.output)
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    output_root.mkdir(parents=True, exist_ok=True)

    # Discover all text files
    print("Scanning for text files...")
    all_files = []
    skipped_large = 0
    for f in input_root.rglob("*"):
        if f.is_file() and f.suffix.lower() in TEXT_EXTENSIONS and ".git" not in f.parts:
            sz = f.stat().st_size
            if sz < MIN_FILE_BYTES:
                continue
            if sz > MAX_FILE_BYTES:
                skipped_large += 1
                continue
            all_files.append(f)
    print(f"Skipped {skipped_large} files > 512 KB (likely generated)")

    print(f"Found {len(all_files):,} text files to process")

    # Phase 1: Compute MinHash signatures
    print("Computing MinHash signatures...")
    lsh = MinHashLSH(threshold=args.threshold, num_perm=NUM_PERM)
    signatures = {}
    content_hashes = {}
    skipped = 0

    for f in tqdm(all_files, desc="Hashing"):
        text = read_text_file(f)
        if len(text) < MIN_FILE_BYTES:
            skipped += 1
            continue

        # Exact duplicate check via SHA-256
        content_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
        content_hashes[str(f)] = content_hash

        mh = compute_minhash(text)
        key = str(f)
        signatures[key] = mh

        try:
            lsh.insert(key, mh)
        except ValueError:
            # Duplicate key — already inserted (exact same path)
            pass

    print(f"Computed {len(signatures):,} signatures ({skipped} files skipped as too small)")

    # Phase 2: Find near-duplicate clusters
    print("Finding near-duplicate clusters...")
    duplicates = set()
    exact_dupes = set()
    clusters = defaultdict(list)

    # First: exact duplicates by content hash
    hash_to_files = defaultdict(list)
    for fpath, chash in content_hashes.items():
        hash_to_files[chash].append(fpath)

    for chash, fpaths in hash_to_files.items():
        if len(fpaths) > 1:
            # Keep the first, mark rest as exact duplicates
            for dup in fpaths[1:]:
                exact_dupes.add(dup)

    # Second: near-duplicates via LSH
    for key, mh in tqdm(signatures.items(), desc="LSH query"):
        if key in exact_dupes:
            continue
        result = lsh.query(mh)
        if len(result) > 1:
            cluster_key = min(result)  # canonical = alphabetically first
            clusters[cluster_key].extend(result)
            for r in result:
                if r != cluster_key:
                    duplicates.add(r)

    near_dupes = duplicates - exact_dupes
    all_dupes = exact_dupes | duplicates

    print(f"Exact duplicates: {len(exact_dupes):,}")
    print(f"Near-duplicates:  {len(near_dupes):,}")
    print(f"Total duplicates: {len(all_dupes):,}")

    # Phase 3: Copy unique files to output
    print("Copying unique files to staging...")
    copied = 0
    for f in tqdm(all_files, desc="Copying"):
        if str(f) in all_dupes:
            continue
        rel_path = f.relative_to(input_root)
        dest = output_root / rel_path
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(f, dest)
        copied += 1

    # Phase 4: Write dedup report
    total_input = len(all_files)
    dedup_rate = len(all_dupes) / total_input * 100 if total_input > 0 else 0

    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "input_dir": str(input_root),
        "output_dir": str(output_root),
        "threshold": args.threshold,
        "num_perm": NUM_PERM,
        "total_input_files": total_input,
        "exact_duplicates": len(exact_dupes),
        "near_duplicates": len(near_dupes),
        "total_removed": len(all_dupes),
        "files_after_dedup": copied,
        "dedup_rate_percent": round(dedup_rate, 2),
        "skipped_too_small": skipped,
        "target_dedup_rate": "<5%",
    }

    report_path.write_text(json.dumps(report, indent=2))
    print(f"\nDedup report written to {report_path}")

    # Phase 5: Update manifest with dedup_status
    manifest_path = Path(args.manifest)
    if manifest_path.exists():
        df = pd.read_parquet(manifest_path)
        # Count dupes per source
        source_dedup = {}
        for dup_path in all_dupes:
            # Extract source_id from path (first subdir under input_root)
            try:
                rel = Path(dup_path).relative_to(input_root)
                source_id = rel.parts[0]
                source_dedup[source_id] = source_dedup.get(source_id, 0) + 1
            except (ValueError, IndexError):
                pass

        df["dedup_removed"] = df["source_id"].map(lambda x: source_dedup.get(x, 0))
        df["dedup_status"] = "complete"
        df.to_parquet(manifest_path, index=False)
        print(f"Manifest updated with dedup_status: {manifest_path}")

    # Summary
    print(f"\n{'='*60}")
    print(f"DEDUP SUMMARY")
    print(f"{'='*60}")
    print(f"Input files:        {total_input:,}")
    print(f"Exact duplicates:   {len(exact_dupes):,}")
    print(f"Near-duplicates:    {len(near_dupes):,}")
    print(f"Total removed:      {len(all_dupes):,}")
    print(f"Files after dedup:  {copied:,}")
    print(f"Dedup rate:         {dedup_rate:.2f}%")
    print(f"Target rate:        <5%")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
