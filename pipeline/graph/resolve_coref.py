"""
Coreference Resolution for EDA Knowledge Graph Triples.

Normalizes entity references across documents so that the same real-world
entity (design, module, version, rule) maps to a single canonical node ID
in Neo4j. Without this, triple extraction produces disconnected islands.

Minimum viable coref (Phase 1):
  - Normalize Design.name against configs/base.yaml canonical list
  - Normalize Version.id against known ORFS version strings
  - Lowercase + strip whitespace on Module.name and Rule.id
  - Deduplicate identical triples from different source files
  - Flag cross-document matches as confidence=0.7

Usage:
    python -m pipeline.graph.resolve_coref \
        --input data/triples/triples_raw.jsonl \
        --output data/triples/triples_resolved.jsonl \
        --config configs/base.yaml
"""

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import yaml


# Canonical entity registries (populated from config)
CANONICAL_DESIGNS = set()
CANONICAL_VERSIONS = set()
DESIGN_ALIASES = {}  # lowercase alias -> canonical name
VERSION_ALIASES = {}  # alias -> canonical version id


def load_canonical_entities(config_path: Path):
    """Load canonical entity names from base.yaml."""
    global CANONICAL_DESIGNS, CANONICAL_VERSIONS, DESIGN_ALIASES, VERSION_ALIASES

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Designs from config
    designs = config.get("designs", [])
    for item in designs:
        if isinstance(item, dict):
            name = item.get("name", "")
        else:
            name = str(item)
        if name:
            CANONICAL_DESIGNS.add(name)
            DESIGN_ALIASES[name.lower()] = name
            DESIGN_ALIASES[name.lower().replace("_", "")] = name
            DESIGN_ALIASES[name.lower().replace("-", "_")] = name

    # Known ORFS versions
    orfs_versions = config.get("orfs_versions", [])
    for item in orfs_versions:
        if isinstance(item, dict):
            ver_key = item.get("name", "")
            tag = item.get("version_tag", ver_key)
        else:
            ver_key = str(item)
            tag = ver_key
        CANONICAL_VERSIONS.add(tag)
        VERSION_ALIASES[tag.lower()] = tag
        VERSION_ALIASES[ver_key.lower()] = tag
        VERSION_ALIASES[tag.lower().replace("_", ".")] = tag
        VERSION_ALIASES[tag.lower().replace(".", "_")] = tag

    # Hardcoded known versions
    known_versions = {
        "orfs_v3_0": "ORFS_v3.0",
        "orfs_v3.0": "ORFS_v3.0",
        "orfs v3.0": "ORFS_v3.0",
        "orfs_26q1": "ORFS_26Q1",
        "orfs 26q1": "ORFS_26Q1",
        "v3.0": "ORFS_v3.0",
        "26q1": "ORFS_26Q1",
    }
    VERSION_ALIASES.update(known_versions)


def normalize_id(entity_id: str, label: str) -> str:
    """Normalize an entity ID based on its label type."""
    cleaned = entity_id.strip()

    if label == "Design":
        # Try to match against canonical designs
        lookup = cleaned.lower().replace("design_", "").replace("_", "")
        for alias, canonical in DESIGN_ALIASES.items():
            if alias in lookup or lookup in alias:
                return f"design_{canonical}"
        return f"design_{cleaned.lower()}"

    elif label == "Version":
        # Match against known version strings
        lookup = cleaned.lower().replace("tool_", "").replace("version_", "")
        for alias, canonical in VERSION_ALIASES.items():
            if alias == lookup or alias in lookup:
                return f"version_{canonical.lower().replace('.', '_')}"
        return f"version_{cleaned.lower()}"

    elif label == "Module":
        # Lowercase, strip path prefixes and common suffixes
        mod = cleaned.lower()
        mod = re.sub(r"^(module_|cell_|pin_|net_|port_)", "", mod)
        mod = re.sub(r"\s+", "_", mod)
        return f"module_{mod}"

    elif label == "Rule":
        # Lowercase, normalize separators
        rule = cleaned.lower()
        rule = re.sub(r"^(rule_|clk_|sdc_)", "", rule)
        rule = re.sub(r"[\s-]+", "_", rule)
        return f"rule_{rule}"

    elif label == "Violation":
        return f"violation_{cleaned.lower()}"

    elif label == "Fix":
        return f"fix_{cleaned.lower()}"

    elif label == "PDK":
        pdk = cleaned.lower().replace("pdk_", "")
        return f"pdk_{pdk}"

    elif label == "ToolRun":
        return f"toolrun_{cleaned.lower()}"

    elif label == "TimingPath":
        return f"tp_{cleaned.lower()}"

    elif label == "Report":
        return f"report_{cleaned.lower()}"

    else:
        return cleaned.lower()


def resolve_triple(triple: dict) -> dict:
    """Apply coreference resolution to a single triple."""
    resolved = triple.copy()

    # Normalize subject
    resolved["subject_id"] = normalize_id(
        triple["subject_id"], triple["subject_label"]
    )

    # Normalize object
    resolved["object_id"] = normalize_id(
        triple["object_id"], triple["object_label"]
    )

    # Cross-document matches get lower confidence
    if resolved["subject_id"] != triple["subject_id"] or \
       resolved["object_id"] != triple["object_id"]:
        resolved["confidence"] = min(triple.get("confidence", 1.0), 0.7)
        resolved["coref_applied"] = True
    else:
        resolved["coref_applied"] = False

    return resolved


def deduplicate_triples(triples: list) -> list:
    """Remove exact duplicate triples (same subject, predicate, object).
    
    Keeps the highest-confidence version of each unique triple.
    """
    seen = {}  # (subj_id, pred, obj_id) -> best triple

    for t in triples:
        key = (t["subject_id"], t["predicate"], t["object_id"])
        if key not in seen or t.get("confidence", 0) > seen[key].get("confidence", 0):
            seen[key] = t

    return list(seen.values())


def compute_stats(triples: list) -> dict:
    """Compute resolution statistics."""
    stats = {
        "total_input": len(triples),
        "coref_applied": sum(1 for t in triples if t.get("coref_applied")),
        "labels": Counter(),
        "predicates": Counter(),
        "confidence_buckets": {"high": 0, "medium": 0, "low": 0},
    }

    for t in triples:
        stats["labels"][t["subject_label"]] += 1
        stats["labels"][t["object_label"]] += 1
        stats["predicates"][t["predicate"]] += 1

        conf = t.get("confidence", 1.0)
        if conf >= 0.9:
            stats["confidence_buckets"]["high"] += 1
        elif conf >= 0.7:
            stats["confidence_buckets"]["medium"] += 1
        else:
            stats["confidence_buckets"]["low"] += 1

    return stats


def main():
    parser = argparse.ArgumentParser(description="Resolve coreferences in extracted triples")
    parser.add_argument("--input", required=True, help="Raw triples JSONL")
    parser.add_argument("--output", default="data/triples/triples_resolved.jsonl",
                        help="Resolved triples output")
    parser.add_argument("--config", default="configs/base.yaml",
                        help="Base config with canonical entity lists")
    parser.add_argument("--stats", default="results/graph/coref_stats.json",
                        help="Resolution statistics output")
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        print(f"ERROR: Config not found: {config_path}")
        sys.exit(1)

    # Load canonical entities
    load_canonical_entities(config_path)
    print(f"Loaded {len(CANONICAL_DESIGNS)} canonical designs, "
          f"{len(CANONICAL_VERSIONS)} canonical versions", flush=True)

    # Load raw triples
    input_path = Path(args.input)
    raw_triples = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                raw_triples.append(json.loads(line))

    print(f"Loaded {len(raw_triples):,} raw triples", flush=True)

    # Phase 1: Normalize entity IDs
    print("Phase 1: Normalizing entity IDs...", flush=True)
    resolved = [resolve_triple(t) for t in raw_triples]

    # Phase 2: Deduplicate
    print("Phase 2: Deduplicating...", flush=True)
    deduped = deduplicate_triples(resolved)

    # Write output
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for t in deduped:
            f.write(json.dumps(t, ensure_ascii=False) + "\n")

    # Compute and save stats
    stats = compute_stats(deduped)
    stats["total_after_dedup"] = len(deduped)
    stats["duplicates_removed"] = len(resolved) - len(deduped)
    stats["dedup_rate"] = f"{(1 - len(deduped)/len(resolved))*100:.1f}%"

    stats_path = Path(args.stats)
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    # Convert Counter to dict for JSON serialization
    stats["labels"] = dict(stats["labels"])
    stats["predicates"] = dict(stats["predicates"])
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    # Summary
    print(f"\n{'='*60}", flush=True)
    print(f"COREFERENCE RESOLUTION SUMMARY", flush=True)
    print(f"{'='*60}", flush=True)
    print(f"Input triples:      {len(raw_triples):,}", flush=True)
    print(f"Coref applied:      {stats['coref_applied']:,}", flush=True)
    print(f"After dedup:        {len(deduped):,}", flush=True)
    print(f"Duplicates removed: {stats['duplicates_removed']:,} ({stats['dedup_rate']})", flush=True)
    print(f"Confidence:         high={stats['confidence_buckets']['high']}, "
          f"med={stats['confidence_buckets']['medium']}, "
          f"low={stats['confidence_buckets']['low']}", flush=True)
    print(f"Output:             {output_path}", flush=True)
    print(f"Stats:              {stats_path}", flush=True)
    print(f"{'='*60}", flush=True)


if __name__ == "__main__":
    main()
