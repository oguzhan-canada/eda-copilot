"""
Neo4j schema loader and seed triple ingester.

Idempotent: safe to run multiple times. Uses MERGE and IF NOT EXISTS
throughout so re-runs produce no duplicates or errors.

Usage:
    python -m pipeline.graph.load_neo4j [--uri bolt://localhost:7687] [--schema configs/graph_schema.cypher]

Environment:
    NEO4J_URI       — bolt:// or neo4j+s:// connection string
    NEO4J_USER      — username (default: neo4j)
    NEO4J_PASSWORD  — password
"""

import argparse
import os
import sys
from pathlib import Path

from neo4j import GraphDatabase
from neo4j.exceptions import ServiceUnavailable, AuthError


def get_driver(uri: str, user: str, password: str):
    """Create and verify Neo4j driver connection."""
    driver = GraphDatabase.driver(uri, auth=(user, password))
    driver.verify_connectivity()
    return driver


def split_cypher_statements(text: str) -> list:
    """Split Cypher text on semicolons, respecting string literals."""
    statements = []
    current = []
    in_string = False
    escape_next = False
    
    # Remove single-line comments first
    lines = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("//"):
            continue
        # Remove inline comments (outside strings)
        clean = []
        in_sq = False
        i = 0
        while i < len(line):
            ch = line[i]
            if ch == "'" and (i == 0 or line[i-1] != "\\"):
                in_sq = not in_sq
            if not in_sq and i + 1 < len(line) and line[i:i+2] == "//":
                break
            clean.append(ch)
            i += 1
        lines.append("".join(clean))
    
    text = "\n".join(lines)
    
    # Now split on ; outside of single-quoted strings
    in_sq = False
    buf = []
    for ch in text:
        if ch == "'" and not escape_next:
            in_sq = not in_sq
        if ch == "\\" and in_sq:
            escape_next = True
            buf.append(ch)
            continue
        escape_next = False
        if ch == ";" and not in_sq:
            stmt = "".join(buf).strip()
            if stmt:
                statements.append(stmt)
            buf = []
        else:
            buf.append(ch)
    
    # Last statement (no trailing semicolon)
    stmt = "".join(buf).strip()
    if stmt:
        statements.append(stmt)
    
    return statements


def load_schema(driver, schema_path: Path) -> dict:
    """Execute schema Cypher file statement-by-statement.
    
    Returns dict with counts of constraints/indexes/merges applied.
    """
    text = schema_path.read_text(encoding="utf-8")
    statements = split_cypher_statements(text)
    
    stats = {"constraints": 0, "indexes": 0, "merges": 0, "total": 0}
    
    with driver.session() as session:
        for stmt in statements:
            upper = stmt.upper()
            try:
                session.run(stmt)
                stats["total"] += 1
                if "CREATE CONSTRAINT" in upper:
                    stats["constraints"] += 1
                elif "CREATE INDEX" in upper:
                    stats["indexes"] += 1
                elif "MERGE" in upper:
                    stats["merges"] += 1
            except Exception as e:
                print(f"  WARNING: Statement failed: {str(e)[:100]}")
                print(f"  Statement: {stmt[:80]}...")
                # Continue — IF NOT EXISTS handles most cases
                continue
    
    return stats


def load_triples(driver, triples_path: Path, batch_size: int = 500) -> dict:
    """Bulk load triples from a JSONL file into Neo4j.
    
    Each triple becomes a MERGE of subject node, object node, and relationship.
    Uses UNWIND for batch efficiency.
    
    Returns dict with loading statistics.
    """
    import json
    
    triples = []
    with open(triples_path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                triples.append(json.loads(line))
    
    stats = {"nodes_merged": 0, "rels_merged": 0, "errors": 0, "batches": 0}
    
    # Batch loading using UNWIND for efficiency
    for i in range(0, len(triples), batch_size):
        batch = triples[i:i + batch_size]
        stats["batches"] += 1
        
        with driver.session() as session:
            for triple in batch:
                subj_id = triple.get("subject_id", "")
                subj_label = triple.get("subject_label", "Entity")
                pred = triple.get("predicate", "RELATED_TO")
                obj_id = triple.get("object_id", "")
                obj_label = triple.get("object_label", "Entity")
                props = triple.get("properties", {})
                
                if not subj_id or not obj_id:
                    stats["errors"] += 1
                    continue
                
                # Sanitize label names (must be valid Cypher identifiers)
                subj_label = _sanitize_label(subj_label)
                obj_label = _sanitize_label(obj_label)
                pred = _sanitize_rel_type(pred)
                
                # Build node MERGE with required properties for constrained labels
                subj_props = _default_node_props(subj_label, subj_id, props)
                obj_props = _default_node_props(obj_label, obj_id, props)
                
                query = (
                    f"MERGE (s:{subj_label} {{id: $subj_id}}) "
                    f"ON CREATE SET s += $subj_props "
                    f"MERGE (o:{obj_label} {{id: $obj_id}}) "
                    f"ON CREATE SET o += $obj_props "
                    f"MERGE (s)-[r:{pred}]->(o) "
                    f"SET r += $props"
                )
                
                try:
                    session.run(
                        query,
                        subj_id=subj_id,
                        obj_id=obj_id,
                        props=props,
                        subj_props=subj_props,
                        obj_props=obj_props,
                    )
                    stats["nodes_merged"] += 2
                    stats["rels_merged"] += 1
                except Exception as e:
                    stats["errors"] += 1
                    if stats["errors"] <= 5:
                        print(f"  WARNING: Triple load failed: {str(e)[:100]}")
                        print(f"  Triple: ({subj_id})-[{pred}]->({obj_id})")
        
        if stats["batches"] % 10 == 0 or i + batch_size >= len(triples):
            print(f"  Batch {stats['batches']}: {i + len(batch)}/{len(triples)} triples processed")
    
    return stats


def _default_node_props(label: str, node_id: str, triple_props: dict) -> dict:
    """Generate default properties for nodes with existence constraints."""
    defaults = {"id": node_id}
    
    if label == "Version":
        defaults["version_tag"] = triple_props.get("version", node_id)
        defaults["source_tool"] = triple_props.get("source_tool", "OpenROAD")
        defaults["capture_date"] = triple_props.get("capture_date", "2026-01-01")
    
    return defaults


def _sanitize_label(label: str) -> str:
    """Ensure label is a valid Cypher identifier."""
    valid_labels = {
        "Design", "Module", "Violation", "Rule", "TimingPath",
        "ToolRun", "Report", "Version", "Fix", "PDK", "Metric", "Entity",
    }
    if label in valid_labels:
        return label
    # Try case-insensitive match
    for vl in valid_labels:
        if label.lower() == vl.lower():
            return vl
    return "Entity"


def _sanitize_rel_type(rel_type: str) -> str:
    """Ensure relationship type is a valid Cypher identifier."""
    # Replace any non-alphanumeric/underscore chars
    import re
    clean = re.sub(r"[^A-Za-z0-9_]", "_", rel_type)
    return clean.upper() if clean else "RELATED_TO"


def verify_seed_triples(driver) -> list:
    """Verify the 4 MLCAD seed bug triples exist.
    
    Returns list of verification results.
    """
    checks = [
        {
            "name": "ED-001: SDC override fix",
            "query": "MATCH (f:Fix {id:'variant_sdc_override'})-[:FIXES]->(v:Violation {id:'ed_001_sdc_override'}) RETURN f.id AS fix, v.id AS violation",
        },
        {
            "name": "ED-002: Version divergence",
            "query": "MATCH (ver1:Version {id:'orfs_v3_0'})-[:DIVERGES_FROM]->(ver2:Version {id:'orfs_26q1'}) RETURN ver1.version_tag AS v1, ver2.version_tag AS v2",
        },
        {
            "name": "ED-003: Unit mismatch fix",
            "query": "MATCH (f:Fix {id:'prepend_unit_declaration'})-[:FIXES]->(v:Violation {id:'ed_003_unit_mismatch'}) RETURN f.id AS fix, v.id AS violation",
        },
        {
            "name": "ED-004: DEF naming fix",
            "query": "MATCH (f:Fix {id:'automated_name_normalization'})-[:FIXES]->(v:Violation {id:'ed_004_def_naming'}) RETURN f.id AS fix, v.id AS violation",
        },
    ]
    
    results = []
    with driver.session() as session:
        for check in checks:
            result = session.run(check["query"]).single()
            passed = result is not None
            results.append({"name": check["name"], "passed": passed})
    
    return results


def get_node_counts(driver) -> dict:
    """Get counts of all node labels in the graph."""
    with driver.session() as session:
        result = session.run(
            "MATCH (n) RETURN labels(n)[0] AS label, count(n) AS count ORDER BY count DESC"
        )
        return {record["label"]: record["count"] for record in result}


def main():
    parser = argparse.ArgumentParser(description="Load Neo4j schema and seed triples")
    parser.add_argument(
        "--uri",
        default=os.environ.get("NEO4J_URI", "bolt://localhost:7687"),
        help="Neo4j connection URI",
    )
    parser.add_argument(
        "--user",
        default=os.environ.get("NEO4J_USER", "neo4j"),
        help="Neo4j username",
    )
    parser.add_argument(
        "--password",
        default=os.environ.get("NEO4J_PASSWORD", ""),
        help="Neo4j password",
    )
    parser.add_argument(
        "--schema",
        default="configs/graph_schema.cypher",
        help="Path to schema Cypher file",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Only verify seed triples, don't load schema",
    )
    parser.add_argument(
        "--input",
        default=None,
        help="Path to triples JSONL file for bulk loading",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=500,
        help="Batch size for bulk triple loading (default: 500)",
    )
    args = parser.parse_args()

    if not args.password:
        print("ERROR: NEO4J_PASSWORD environment variable or --password required")
        sys.exit(1)

    # Connect
    print(f"Connecting to {args.uri} as {args.user}...")
    try:
        driver = get_driver(args.uri, args.user, args.password)
    except ServiceUnavailable:
        print(f"ERROR: Cannot reach Neo4j at {args.uri}")
        sys.exit(1)
    except AuthError:
        print("ERROR: Authentication failed — check NEO4J_USER/NEO4J_PASSWORD")
        sys.exit(1)

    print("  Connected successfully.\n")

    if not args.verify_only:
        # Load schema
        schema_path = Path(args.schema)
        if not schema_path.exists():
            print(f"ERROR: Schema file not found: {schema_path}")
            sys.exit(1)

        print(f"Loading schema from {schema_path}...")
        stats = load_schema(driver, schema_path)
        print(f"  Applied: {stats['constraints']} constraints, {stats['indexes']} indexes, {stats['merges']} merges")
        print(f"  Total statements: {stats['total']}\n")

    # Bulk load triples if --input provided
    if args.input:
        input_path = Path(args.input)
        if not input_path.exists():
            print(f"ERROR: Triples file not found: {input_path}")
            sys.exit(1)
        
        line_count = sum(1 for _ in open(input_path, encoding="utf-8") if _.strip())
        print(f"Loading {line_count} triples from {input_path} (batch_size={args.batch_size})...")
        load_stats = load_triples(driver, input_path, batch_size=args.batch_size)
        print(f"  Nodes merged: {load_stats['nodes_merged']}")
        print(f"  Relationships merged: {load_stats['rels_merged']}")
        print(f"  Errors: {load_stats['errors']}")
        print(f"  Batches: {load_stats['batches']}\n")

    # Verify seed triples
    print("Verifying seed triples...")
    results = verify_seed_triples(driver)
    all_passed = True
    for r in results:
        status = "PASS" if r["passed"] else "FAIL"
        print(f"  [{status}] {r['name']}")
        if not r["passed"]:
            all_passed = False

    # Node counts
    print("\nNode counts:")
    counts = get_node_counts(driver)
    for label, count in counts.items():
        print(f"  {label}: {count}")

    driver.close()

    if not all_passed:
        print("\nERROR: Some seed triple checks failed!")
        sys.exit(1)
    
    print("\n" + "=" * 60)
    print("SCHEMA LOAD COMPLETE — All seed triples verified")
    print("=" * 60)


if __name__ == "__main__":
    main()
