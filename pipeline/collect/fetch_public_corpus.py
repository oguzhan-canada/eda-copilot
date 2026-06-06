"""
Component 1.1 — Public corpus collection.
Reads configs/corpus_sources.yaml, clones repos, builds manifest.

Usage:
    python -m pipeline.collect.fetch_public_corpus \
        --config configs/corpus_sources.yaml \
        --output data/corpus/raw_docs
"""
import argparse
import hashlib
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yaml
from tqdm import tqdm


def sha256_dir(path: Path) -> str:
    """Stable checksum of all files in a directory."""
    h = hashlib.sha256()
    for f in sorted(path.rglob("*")):
        if f.is_file():
            try:
                h.update(f.read_bytes())
            except (PermissionError, OSError):
                pass
    return h.hexdigest()[:16]


def clone_or_update(url: str, dest: Path) -> bool:
    """Clone if missing, pull if exists. Returns True on success."""
    if (dest / ".git").exists():
        result = subprocess.run(
            ["git", "-C", str(dest), "pull", "--quiet"],
            capture_output=True, text=True
        )
        return result.returncode == 0
    dest.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        ["git", "clone", "--depth=1", "--quiet", url, str(dest)],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        print(f"  ERROR cloning {url}: {result.stderr.strip()}", file=sys.stderr)
        return False
    return True


def count_tokens_approx(path: Path, extensions: list) -> int:
    """Rough token count: chars / 4 for text files."""
    total_chars = 0
    for ext in extensions:
        for f in path.rglob(f"*{ext}"):
            try:
                total_chars += len(f.read_text(errors="ignore"))
            except (PermissionError, OSError):
                pass
    return total_chars // 4


def main():
    parser = argparse.ArgumentParser(description="Fetch public EDA corpus sources")
    parser.add_argument("--config", default="configs/corpus_sources.yaml")
    parser.add_argument("--output", default="data/corpus/raw_docs")
    parser.add_argument("--manifest", default="data/manifests/artifacts_manifest.parquet")
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    sources = cfg["sources"]
    output_root = Path(args.output)
    manifest_path = Path(args.manifest)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    records = []
    text_exts = [".v", ".vhd", ".sv", ".tcl", ".py", ".md", ".txt", ".rst",
                 ".lib", ".lef", ".def", ".sdc", ".json", ".yaml", ".yml",
                 ".log", ".rpt", ".cfg", ".mk", ".sh"]

    for src in tqdm(sources, desc="Fetching corpus"):
        dest = output_root / src["id"]
        print(f"\n[{src['id']}] Fetching from {src['url']}")

        ok = clone_or_update(src["url"], dest)
        if not ok:
            print(f"  SKIP {src['id']} — clone failed")
            records.append({
                "source_id":    src["id"],
                "tier":         src["tier"],
                "source_type":  src.get("type", "github_repo"),
                "url":          src["url"],
                "license":      src["license"],
                "tool_name":    src.get("tool_name", ""),
                "tool_version": src.get("tool_version", ""),
                "version_tag":  src.get("version_tag", ""),
                "local_path":   str(dest),
                "checksum":     "",
                "fetch_date":   datetime.now(timezone.utc).isoformat(),
                "file_count":   0,
                "token_count_approx": 0,
                "status":       "failed",
            })
            continue

        checksum = sha256_dir(dest)
        token_count = count_tokens_approx(dest, text_exts)
        file_count = sum(1 for _ in dest.rglob("*") if _.is_file())

        records.append({
            "source_id":    src["id"],
            "tier":         src["tier"],
            "source_type":  src.get("type", "github_repo"),
            "url":          src["url"],
            "license":      src["license"],
            "tool_name":    src.get("tool_name", ""),
            "tool_version": src.get("tool_version", ""),
            "version_tag":  src.get("version_tag", ""),
            "local_path":   str(dest),
            "checksum":     checksum,
            "fetch_date":   datetime.now(timezone.utc).isoformat(),
            "file_count":   file_count,
            "token_count_approx": token_count,
            "status":       "ok",
        })
        print(f"  OK  {file_count:,} files  ~{token_count:,} tokens  checksum={checksum}")

    df = pd.DataFrame(records)
    df.to_parquet(manifest_path, index=False)

    jsonl_path = manifest_path.with_suffix(".jsonl")
    df.to_json(jsonl_path, orient="records", lines=True)

    print(f"\nManifest written: {manifest_path}")
    print(f"Sources fetched:  {len([r for r in records if r['status'] == 'ok'])}/{len(sources)}")
    print(df[["source_id", "tier", "file_count", "token_count_approx"]].to_string(index=False))


if __name__ == "__main__":
    main()
