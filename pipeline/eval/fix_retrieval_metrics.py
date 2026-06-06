"""
Recompute retrieval precision on existing eval results by re-running
retrieval only (no synthesis or judging) against the fixed benchmark.

Merges updated retrieval_precision and source_recall into existing results,
preserving judge_scores and system_answer from the original run.

Usage:
    python fix_retrieval_metrics.py \
        --benchmark data/edabench/edabench_v1.jsonl \
        --results results/eval/system_eval_v2.json \
        --output results/eval/system_eval_v3.json

    python fix_retrieval_metrics.py \
        --benchmark data/edabench/edabench_v1.jsonl \
        --results results/eval/baseline_vector_only_v2.json \
        --output results/eval/baseline_vector_only_v3.json \
        --ablation vector_only

    python fix_retrieval_metrics.py \
        --benchmark data/edabench/edabench_v1.jsonl \
        --results results/eval/baseline_no_retrieval_v2.json \
        --output results/eval/baseline_no_retrieval_v3.json \
        --ablation no_retrieval
"""
import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median


def evaluate_retrieval_precision(graph_facts: list, expected_nodes: list) -> float:
    """Score: what fraction of expected graph nodes were retrieved?"""
    expected = set(expected_nodes)
    if not expected:
        return None  # Return None instead of 1.0 — caller decides how to handle

    retrieved_nodes = set()
    for fact in graph_facts:
        if isinstance(fact, dict):
            for key in ["id", "node_id", "entity", "entity_id"]:
                if key in fact:
                    retrieved_nodes.add(str(fact[key]).lower())
            data = fact.get("data", {})
            if isinstance(data, dict):
                if "edges" in data:
                    for edge in data.get("edges", []):
                        retrieved_nodes.add(str(edge.get("source", "")).lower())
                        retrieved_nodes.add(str(edge.get("target", "")).lower())
                if "versions" in data:
                    for v in data.get("versions", []):
                        for vk in ["design_id", "version_id"]:
                            if vk in v:
                                retrieved_nodes.add(str(v[vk]).lower())
        elif isinstance(fact, str):
            retrieved_nodes.add(fact.lower())

    expected_lower = {n.lower() for n in expected}
    found = expected_lower & retrieved_nodes
    return len(found) / len(expected_lower)


def evaluate_source_recall(chunks: list, expected_sources: list) -> float:
    """Score: what fraction of expected sources appear in retrieved chunks?"""
    if not expected_sources:
        return None

    chunk_sources = []
    for chunk in chunks:
        sf = chunk.get("source_file", "")
        if sf:
            chunk_sources.append(sf.lower())

    found = 0
    for src in expected_sources:
        src_lower = src.lower()
        if any(src_lower in cs for cs in chunk_sources):
            found += 1

    return found / len(expected_sources)


def main():
    parser = argparse.ArgumentParser(description="Fix retrieval metrics on existing results")
    parser.add_argument("--benchmark", required=True, help="Fixed EDABench JSONL")
    parser.add_argument("--results", required=True, help="Existing results JSON to update")
    parser.add_argument("--output", required=True, help="Output path for updated results")
    parser.add_argument("--ablation", default=None, choices=["vector_only", "no_retrieval"])
    args = parser.parse_args()

    # Load benchmark
    items = []
    with open(args.benchmark, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                items.append(json.loads(line))
    item_map = {item["id"]: item for item in items}
    print(f"Loaded {len(items)} benchmark items")

    # Load existing results
    existing = json.load(open(args.results))
    results = existing.get("results", existing.get("items", []))
    print(f"Loaded {len(results)} existing results from {args.results}")

    # Import retriever
    if args.ablation == "no_retrieval":
        print("ABLATION: no_retrieval — skipping retrieval, setting precision=N/A")
        for r in results:
            item = item_map.get(r["id"], {})
            r["retrieval_precision"] = None
            r["retrieval_precision_valid"] = False
            r["source_recall"] = None
            r["graph_facts"] = []
            r["graph_facts_count"] = 0
    else:
        try:
            sys.path.insert(0, os.getcwd())
            from apps.api.services.fusion_retriever import FusionRetriever
            retriever = FusionRetriever()
            print("FusionRetriever initialized")
        except Exception as e:
            print(f"ERROR: Could not initialize retriever: {e}")
            return

        for i, r in enumerate(results):
            item = item_map.get(r["id"], {})
            query = r["query"]
            print(f"  [{i+1}/{len(results)}] {r['id']}: {query[:50]}...", end="", flush=True)

            try:
                t0 = time.perf_counter()
                retrieval = retriever.retrieve(query)
                latency = time.perf_counter() - t0

                if args.ablation == "vector_only":
                    retrieval["graph_facts"] = []

                graph_facts = retrieval.get("graph_facts", [])
                chunks = retrieval.get("chunks", [])

                # Compute fixed metrics
                expected_nodes = item.get("expected_graph_nodes", [])
                expected_sources = item.get("expected_sources", [])

                ret_prec = evaluate_retrieval_precision(graph_facts, expected_nodes)
                src_recall = evaluate_source_recall(chunks, expected_sources)

                # Update result
                r["retrieval_precision"] = ret_prec
                r["retrieval_precision_valid"] = ret_prec is not None
                r["source_recall"] = src_recall if src_recall is not None else r.get("source_recall", 0)
                r["graph_facts_count"] = len(graph_facts)
                r["chunks_count"] = len(chunks)
                r["chunk_sources"] = [c.get("source_file", "") for c in chunks if c.get("source_file")]
                # Store graph fact entity_ids for verification
                r["graph_fact_ids"] = list(set(
                    str(f.get("entity_id", f.get("id", "")))
                    for f in graph_facts if isinstance(f, dict)
                ))

                print(f" prec={ret_prec} facts={len(graph_facts)} ({latency:.1f}s)", flush=True)

            except Exception as e:
                print(f" ERROR: {e}", flush=True)
                r["retrieval_precision"] = None
                r["retrieval_precision_valid"] = False

    # Recompute aggregate report
    # Only include items with valid retrieval precision
    valid_prec = [r for r in results if r.get("retrieval_precision") is not None]
    all_prec = [r["retrieval_precision"] for r in valid_prec]

    report = existing.get("report", {})
    report["retrieval_v3"] = {
        "items_with_expected_nodes": len(valid_prec),
        "items_without_expected_nodes": len(results) - len(valid_prec),
        "mean_precision_on_grounded": round(mean(all_prec), 4) if all_prec else None,
        "note": "Retrieval precision computed only on items with expected_graph_nodes"
    }

    # Per-category with valid precision
    by_cat = {}
    for r in valid_prec:
        by_cat.setdefault(r["task_category"], []).append(r["retrieval_precision"])
    report["retrieval_precision_by_category"] = {
        cat: {"mean": round(mean(scores), 4), "n": len(scores)}
        for cat, scores in sorted(by_cat.items())
    }

    # Write updated results
    output = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "benchmark_file": args.benchmark,
        "base_results_file": args.results,
        "num_items": len(results),
        "report": report,
        "results": results,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(output, indent=2))
    print(f"\nUpdated results written to {args.output}")

    # Print summary
    print(f"\n{'='*60}")
    print(f"RETRIEVAL PRECISION FIX SUMMARY")
    print(f"{'='*60}")
    print(f"Items with expected nodes: {len(valid_prec)}/{len(results)}")
    if all_prec:
        print(f"Mean precision (grounded items only): {mean(all_prec):.4f}")
        print(f"\nBy category:")
        for cat, info in report["retrieval_precision_by_category"].items():
            print(f"  {cat:30s} {info['mean']:.4f} (n={info['n']})")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
