"""
Component 1.1 (cont.) — Tier 3 forum mining from GitHub Issues.

Extracts Q&A-style data from OpenROAD, Yosys, and OpenLane GitHub Issues.
Targets error discussions, tool usage questions, and debugging threads.

Usage:
    python -m pipeline.collect.mine_forums \
        --output C:/eda-kg-data/corpus/raw_docs/forums \
        --manifest data/manifests/artifacts_manifest.parquet
"""
import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from tqdm import tqdm


# Repos to mine — highest-value EDA issue trackers
REPOS = [
    {
        "owner": "The-OpenROAD-Project",
        "repo": "OpenROAD",
        "tool_name": "OpenROAD",
        "tool_version": "26Q1",
        "license": "BSD-3-Clause",
        "tier": 3,
    },
    {
        "owner": "The-OpenROAD-Project",
        "repo": "OpenROAD-flow-scripts",
        "tool_name": "ORFS",
        "tool_version": "26Q1",
        "license": "BSD-3-Clause",
        "tier": 3,
    },
    {
        "owner": "YosysHQ",
        "repo": "yosys",
        "tool_name": "Yosys",
        "tool_version": "0.36",
        "license": "ISC",
        "tier": 3,
    },
    {
        "owner": "The-OpenROAD-Project",
        "repo": "OpenLane",
        "tool_name": "OpenLane",
        "tool_version": "2.x",
        "license": "Apache-2.0",
        "tier": 3,
    },
    {
        "owner": "The-OpenROAD-Project",
        "repo": "OpenSTA",
        "tool_name": "OpenSTA",
        "tool_version": "2.6",
        "license": "GPL-3.0",
        "tier": 3,
    },
]

# Labels that indicate error/debug discussion (case-insensitive match)
ERROR_LABELS = {"bug", "error", "crash", "timing", "drc", "violation", "fix", "issue"}

# Max issues per repo (to stay within API limits)
MAX_ISSUES_PER_REPO = 500


def fetch_issues_gh(owner: str, repo: str, max_issues: int = MAX_ISSUES_PER_REPO) -> list:
    """Fetch closed issues with comments using gh CLI."""
    cmd = [
        "gh", "issue", "list",
        "--repo", f"{owner}/{repo}",
        "--state", "closed",
        "--limit", str(max_issues),
        "--json", "number,title,body,labels,comments,createdAt,closedAt,url",
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120,
                                encoding="utf-8", errors="replace")
        if result.returncode != 0:
            print(f"  Warning: gh CLI error for {owner}/{repo}: {result.stderr.strip()[:200]}")
            return []
        return json.loads(result.stdout) if result.stdout and result.stdout.strip() else []
    except (subprocess.TimeoutExpired, json.JSONDecodeError) as e:
        print(f"  Warning: {e}")
        return []


def extract_qa_from_issue(issue: dict, tool_name: str) -> dict | None:
    """Extract a Q&A pair from a GitHub issue if it has useful content."""
    title = issue.get("title", "")
    body = issue.get("body", "") or ""
    comments = issue.get("comments", [])
    labels = [l.get("name", "") for l in issue.get("labels", [])]

    # Skip issues with no body or no comments (no answer)
    if not body.strip() or not comments:
        return None

    # Skip very short issues (likely not diagnostic)
    if len(body) < 50:
        return None

    # Build the answer from all comments concatenated
    answer_parts = []
    for c in comments:
        c_body = c.get("body", "") or ""
        if len(c_body) > 20:
            answer_parts.append(c_body)

    if not answer_parts:
        return None

    answer = "\n---\n".join(answer_parts[:5])  # Cap at 5 comments

    # Extract error-related tags
    error_tags = [l for l in labels if l.lower() in ERROR_LABELS]

    return {
        "source": "github_issues",
        "tool_name": tool_name,
        "issue_number": issue.get("number"),
        "url": issue.get("url", ""),
        "question_title": title,
        "question_body": body[:5000],  # Cap body length
        "answer": answer[:10000],  # Cap answer length
        "labels": labels,
        "error_tags": error_tags,
        "created_at": issue.get("createdAt", ""),
        "closed_at": issue.get("closedAt", ""),
        "comment_count": len(comments),
    }


def main():
    parser = argparse.ArgumentParser(description="Mine Tier 3 Q&A from GitHub Issues")
    parser.add_argument("--output", default="C:/eda-kg-data/corpus/raw_docs/forums")
    parser.add_argument("--manifest", default="data/manifests/artifacts_manifest.parquet")
    parser.add_argument("--max-per-repo", type=int, default=MAX_ISSUES_PER_REPO)
    args = parser.parse_args()

    output_root = Path(args.output)
    output_root.mkdir(parents=True, exist_ok=True)

    all_qa = []
    manifest_records = []

    for repo_info in tqdm(REPOS, desc="Mining repos"):
        owner = repo_info["owner"]
        repo = repo_info["repo"]
        tool_name = repo_info["tool_name"]

        print(f"\n[{owner}/{repo}] Fetching issues...")
        issues = fetch_issues_gh(owner, repo, args.max_per_repo)
        print(f"  Fetched {len(issues)} closed issues")

        qa_pairs = []
        for issue in issues:
            qa = extract_qa_from_issue(issue, tool_name)
            if qa:
                qa_pairs.append(qa)

        # Write to JSONL
        repo_output = output_root / f"{owner}_{repo}_issues.jsonl"
        with open(repo_output, "w", encoding="utf-8") as f:
            for qa in qa_pairs:
                f.write(json.dumps(qa, ensure_ascii=False) + "\n")

        all_qa.extend(qa_pairs)

        manifest_records.append({
            "source_id": f"github_issues_{owner}_{repo}",
            "tier": repo_info["tier"],
            "source_type": "github_issues",
            "url": f"https://github.com/{owner}/{repo}/issues",
            "license": repo_info["license"],
            "tool_name": tool_name,
            "tool_version": repo_info["tool_version"],
            "version_tag": f"{tool_name}_issues",
            "local_path": str(repo_output),
            "checksum": "",
            "fetch_date": datetime.now(timezone.utc).isoformat(),
            "file_count": 1,
            "token_count_approx": sum(len(qa.get("question_body", "")) + len(qa.get("answer", "")) for qa in qa_pairs) // 4,
            "status": "ok",
            "qa_pairs": len(qa_pairs),
        })

        print(f"  Extracted {len(qa_pairs)} Q&A pairs -> {repo_output.name}")

    # Update manifest
    manifest_path = Path(args.manifest)
    if manifest_path.exists():
        df_existing = pd.read_parquet(manifest_path)
        df_new = pd.DataFrame(manifest_records)
        # Drop qa_pairs (not in original schema)
        if "qa_pairs" in df_new.columns:
            df_new = df_new.drop(columns=["qa_pairs"])
        # Align columns: add missing cols as None
        for col in df_existing.columns:
            if col not in df_new.columns:
                df_new[col] = None
        # Keep only existing columns, in order
        df_new = df_new[df_existing.columns]
        df = pd.concat([df_existing, df_new], ignore_index=True)
        # Convert all object columns to string to avoid Arrow mixed-type errors
        for col in df.columns:
            if df[col].dtype == object:
                df[col] = df[col].astype(str).replace("None", "").replace("nan", "")
        df.to_parquet(manifest_path, index=False)
        print(f"\nManifest updated: {manifest_path} ({len(df)} rows)")

    # Summary
    print(f"\n{'='*60}")
    print(f"FORUM MINING SUMMARY")
    print(f"{'='*60}")
    print(f"Repos mined:        {len(REPOS)}")
    print(f"Total Q&A pairs:    {len(all_qa):,}")
    error_qa = sum(1 for qa in all_qa if qa.get("error_tags"))
    print(f"Error-tagged Q&A:   {error_qa:,}")
    print(f"Target:             5,000+")
    print(f"{'='*60}")

    for rec in manifest_records:
        print(f"  {rec['source_id']:45s} {rec.get('qa_pairs', 0):>5} Q&A")


if __name__ == "__main__":
    main()
