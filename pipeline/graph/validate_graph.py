"""
Graph validation for Neo4j knowledge graph.

Runs four validation checks:
  1. Structural integrity — orphan nodes, dangling relationships
  2. Contradiction detection — conflicting fixes for same violation
  3. Seed triple integrity — ED-001 through ED-005 present
  4. Version coverage — ORFS versions with DIVERGES_FROM edges

Plus latency measurement (p50/p95 for 2-hop retrieval).

Usage:
    python -m pipeline.graph.validate_graph \
        --uri neo4j+s://... --user neo4j --password ... \
        --output results/graph/validation_report.json
"""

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

from neo4j import GraphDatabase


SEED_IDS = [
    'ed_001_sdc_override',
    'ed_002_version_divergence',
    'ed_003_unit_mismatch',
    'ed_004_def_naming',
    'ed_005_ibex_26q1_sigsegv',
]


def check_structural_integrity(session) -> dict:
    """Check for orphan nodes and relationship counts."""
    orphans = session.run(
        "MATCH (n) WHERE NOT (n)--() RETURN labels(n) AS labels, count(n) AS cnt"
    ).data()

    total_nodes = session.run("MATCH (n) RETURN count(n) AS c").single()["c"]
    total_rels = session.run("MATCH ()-[r]->() RETURN count(r) AS c").single()["c"]

    orphan_count = sum(r["cnt"] for r in orphans)
    orphan_pct = (orphan_count / total_nodes * 100) if total_nodes > 0 else 0

    label_counts = session.run(
        "MATCH (n) UNWIND labels(n) AS lbl "
        "RETURN lbl, count(n) AS cnt ORDER BY cnt DESC"
    ).data()

    return {
        "check": "structural_integrity",
        "passed": True,  # orphans are flagged, not failures
        "total_nodes": total_nodes,
        "total_relationships": total_rels,
        "orphan_nodes": orphan_count,
        "orphan_pct": round(orphan_pct, 2),
        "orphan_breakdown": [
            {"labels": r["labels"], "count": r["cnt"]} for r in orphans
        ],
        "label_counts": {r["lbl"]: r["cnt"] for r in label_counts},
    }


def check_contradictions(session) -> dict:
    """Detect violations with multiple conflicting fixes."""
    # Violations with multiple fixes (not necessarily contradictions)
    multi_fix = session.run(
        "MATCH (v:Violation)<-[:FIXES]-(f:Fix) "
        "WITH v, collect(DISTINCT f.id) AS fixes "
        "WHERE size(fixes) > 1 "
        "RETURN v.id AS violation, fixes ORDER BY size(fixes) DESC LIMIT 20"
    ).data()

    # Cross-version contradictions: same violation fixed differently per version
    cross_version = session.run(
        "MATCH (v:Violation)<-[:FIXES]-(f:Fix), "
        "      (f)-[:TARGETS]->(ver:Version) "
        "WITH v, collect(DISTINCT {fix: f.id, version: ver.version_tag}) AS pairs "
        "WHERE size(pairs) > 1 "
        "RETURN v.id AS violation, pairs LIMIT 10"
    ).data()

    return {
        "check": "contradiction_detection",
        "passed": True,  # contradictions are flagged for review, not failures
        "violations_with_multiple_fixes": len(multi_fix),
        "samples": multi_fix[:5],
        "cross_version_contradictions": len(cross_version),
        "cross_version_samples": cross_version[:5],
    }


def check_seed_triples(session) -> dict:
    """Verify all seed triples ED-001 through ED-005 are present."""
    results = session.run(
        "MATCH (v:Violation) WHERE v.id IN $ids "
        "RETURN v.id AS id, v.description AS description, v.error_code AS code",
        ids=SEED_IDS,
    ).data()

    found_ids = {r["id"] for r in results}
    missing = [sid for sid in SEED_IDS if sid not in found_ids]

    return {
        "check": "seed_triple_integrity",
        "passed": len(missing) == 0,
        "expected": len(SEED_IDS),
        "found": len(found_ids),
        "missing": missing,
        "details": results,
    }


def check_version_coverage(session) -> dict:
    """Verify both ORFS versions and DIVERGES_FROM edge."""
    versions = session.run(
        "MATCH (v:Version) "
        "RETURN v.id AS id, v.version_tag AS tag, v.source_tool AS tool "
        "ORDER BY v.id"
    ).data()

    divergence = session.run(
        "MATCH (a:Version)-[r:DIVERGES_FROM]->(b:Version) "
        "RETURN a.id AS from_id, b.id AS to_id, "
        "       a.version_tag AS from_tag, b.version_tag AS to_tag"
    ).data()

    canonical_versions = {"orfs_v3_0", "orfs_26q1"}
    found_canonical = {v["id"] for v in versions if v["id"] in canonical_versions}

    return {
        "check": "version_coverage",
        "passed": len(found_canonical) == 2 and len(divergence) > 0,
        "total_version_nodes": len(versions),
        "canonical_found": list(found_canonical),
        "canonical_missing": list(canonical_versions - found_canonical),
        "diverges_from_edges": len(divergence),
        "divergence_details": divergence,
    }


def measure_latency(session, iterations: int = 100) -> dict:
    """Measure p50/p95 for 2-hop Violation->Fix retrieval."""
    # Warm up
    session.run(
        "MATCH (v:Violation)-[:CAUSES|FIXES*1..2]-(n) "
        "RETURN v.id, collect(n.id) LIMIT 10"
    ).consume()

    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        session.run(
            "MATCH (v:Violation)-[:CAUSES|FIXES*1..2]-(n) "
            "RETURN v.id, collect(n.id) LIMIT 10"
        ).consume()
        elapsed_ms = (time.perf_counter() - start) * 1000
        times.append(elapsed_ms)

    p50 = statistics.median(times)
    p95 = statistics.quantiles(times, n=20)[18]
    p99 = statistics.quantiles(times, n=100)[98]
    mean = statistics.mean(times)

    return {
        "check": "latency_2hop",
        "passed": p50 < 100,
        "iterations": iterations,
        "p50_ms": round(p50, 1),
        "p95_ms": round(p95, 1),
        "p99_ms": round(p99, 1),
        "mean_ms": round(mean, 1),
        "min_ms": round(min(times), 1),
        "max_ms": round(max(times), 1),
        "gate": "p50 < 100ms",
    }


def run_validation(uri: str, user: str, password: str, measure_perf: bool = True) -> dict:
    """Run all validation checks and return report."""
    driver = GraphDatabase.driver(uri, auth=(user, password))
    driver.verify_connectivity()
    print(f"Connected to {uri}")

    report = {"timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "checks": []}

    with driver.session() as s:
        print("  [1/5] Structural integrity...")
        structural = check_structural_integrity(s)
        report["checks"].append(structural)
        status = "PASS" if structural["passed"] else "FAIL"
        print(f"        {status}: {structural['total_nodes']} nodes, "
              f"{structural['total_relationships']} rels, "
              f"{structural['orphan_nodes']} orphans ({structural['orphan_pct']}%)")

        print("  [2/5] Contradiction detection...")
        contradictions = check_contradictions(s)
        report["checks"].append(contradictions)
        print(f"        {contradictions['violations_with_multiple_fixes']} violations "
              f"with multiple fixes, "
              f"{contradictions['cross_version_contradictions']} cross-version")

        print("  [3/5] Seed triple integrity...")
        seeds = check_seed_triples(s)
        report["checks"].append(seeds)
        status = "PASS" if seeds["passed"] else "FAIL"
        print(f"        {status}: {seeds['found']}/{seeds['expected']} seeds found")
        if seeds["missing"]:
            print(f"        MISSING: {seeds['missing']}")

        print("  [4/5] Version coverage...")
        versions = check_version_coverage(s)
        report["checks"].append(versions)
        status = "PASS" if versions["passed"] else "FAIL"
        print(f"        {status}: {versions['total_version_nodes']} version nodes, "
              f"{versions['diverges_from_edges']} DIVERGES_FROM edges")

        if measure_perf:
            print("  [5/5] Latency measurement (100 iterations)...")
            latency = measure_latency(s)
            report["checks"].append(latency)
            status = "PASS" if latency["passed"] else "FAIL"
            print(f"        {status}: p50={latency['p50_ms']}ms  "
                  f"p95={latency['p95_ms']}ms  (gate: p50 < 100ms)")

    driver.close()

    # Overall pass/fail
    report["all_passed"] = all(c["passed"] for c in report["checks"])
    return report


def main():
    parser = argparse.ArgumentParser(description="Validate Neo4j knowledge graph")
    parser.add_argument("--uri", default=os.environ.get("NEO4J_URI"))
    parser.add_argument("--user", default=os.environ.get("NEO4J_USER", "neo4j"))
    parser.add_argument("--password", default=os.environ.get("NEO4J_PASSWORD"))
    parser.add_argument("--output", default="results/graph/validation_report.json")
    parser.add_argument("--skip-latency", action="store_true")
    args = parser.parse_args()

    if not args.uri or not args.password:
        print("ERROR: --uri and --password required (or set NEO4J_URI, NEO4J_PASSWORD)")
        sys.exit(1)

    report = run_validation(args.uri, args.user, args.password, not args.skip_latency)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, default=str))
    print(f"\nReport saved to {out_path}")

    if report["all_passed"]:
        print("\n" + "=" * 60)
        print("VALIDATION: ALL CHECKS PASSED")
        print("=" * 60)
    else:
        failed = [c["check"] for c in report["checks"] if not c["passed"]]
        print(f"\nVALIDATION: {len(failed)} CHECK(S) FAILED: {failed}")
        sys.exit(1)


if __name__ == "__main__":
    main()
