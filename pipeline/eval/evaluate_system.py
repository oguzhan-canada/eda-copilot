"""
Component 4.2 — EDABench system evaluation.

Evaluates the full GraphRAG pipeline on each benchmark item, scoring on 4 axes:
  1. Retrieval precision: expected graph nodes found in retrieval results
  2. Source recall: expected source files appear in citations
  3. Answer correctness: Claude-judged quality (4-dimension rubric)
  4. Latency: wall-clock time for full pipeline

Usage:
    python -m pipeline.eval.evaluate_system \
        --benchmark data/edabench/edabench_v1.jsonl \
        --output results/eval/edabench_results.json

    # Batch mode for answer judging (50% cheaper)
    python -m pipeline.eval.evaluate_system \
        --benchmark data/edabench/edabench_v1.jsonl \
        --output results/eval/edabench_results.json \
        --batch
"""
import argparse
import json
import os
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median

try:
    import anthropic
except ImportError:
    anthropic = None


JUDGE_SYSTEM_PROMPT = """You are evaluating an EDA copilot's answer against a ground-truth answer.
Score on 4 dimensions, each 0.0 to 1.0:

1. **factual_accuracy**: Is the answer factually correct for EDA domain?
2. **completeness**: Does it cover the key points from the ground truth?
3. **actionability**: Does it provide clear, actionable guidance?
4. **specificity**: Does it reference specific tools, metrics, versions, or files?

Respond with valid JSON only:
{
  "factual_accuracy": 0.0-1.0,
  "completeness": 0.0-1.0,
  "actionability": 0.0-1.0,
  "specificity": 0.0-1.0,
  "overall": 0.0-1.0,
  "reasoning": "1-2 sentence explanation"
}"""


def evaluate_retrieval_precision(result: dict, item: dict) -> float:
    """Score: what fraction of expected graph nodes were retrieved?"""
    expected = set(item.get("expected_graph_nodes", []))
    if not expected:
        return 1.0  # No graph node expectations = pass

    retrieved_nodes = set()
    for fact in result.get("graph_facts", []):
        if isinstance(fact, dict):
            for key in ["id", "node_id", "entity", "entity_id"]:
                if key in fact:
                    retrieved_nodes.add(str(fact[key]).lower())
            # Also check nested data for node references
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
    return len(found) / len(expected_lower) if expected_lower else 1.0


def evaluate_source_recall(answer: dict, item: dict,
                           retrieval: dict = None) -> float:
    """Score: what fraction of expected sources appear in citations or retrieved chunks?

    Checks three places for source matches:
    1. Synthesizer citations list
    2. Answer text (inline mentions)
    3. Retrieved chunk source_file fields (retrieval-level recall)
    """
    expected = item.get("expected_sources", [])
    if not expected:
        return 1.0

    citations = answer.get("citations", [])
    answer_text = answer.get("answer", "").lower()

    # Collect all chunk source_files from retrieval
    chunk_sources = []
    if retrieval:
        for chunk in retrieval.get("chunks", []):
            sf = chunk.get("source_file", "")
            if sf:
                chunk_sources.append(sf.lower())

    found = 0
    for src in expected:
        src_lower = src.lower()
        if any(src_lower in str(c).lower() for c in citations):
            found += 1
        elif src_lower in answer_text:
            found += 1
        elif any(src_lower in cs for cs in chunk_sources):
            found += 1

    return found / len(expected) if expected else 1.0


def judge_answer(client, query: str, ground_truth: str,
                 system_answer: str, model: str) -> dict:
    """Use Claude to judge answer quality against ground truth."""
    prompt = f"""**Query:** {query}

**Ground Truth Answer:**
{ground_truth}

**System's Answer:**
{system_answer}

Score the system's answer against the ground truth. Respond with valid JSON only."""

    try:
        response = client.messages.create(
            model=model,
            max_tokens=512,
            system=JUDGE_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
        )
        text = response.content[0].text.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1]
            if text.endswith("```"):
                text = text[:-3]
            text = text.strip()
        return json.loads(text)
    except Exception as e:
        return {
            "factual_accuracy": 0.0,
            "completeness": 0.0,
            "actionability": 0.0,
            "specificity": 0.0,
            "overall": 0.0,
            "reasoning": f"Judge error: {str(e)}",
        }


def build_judge_batch_requests(items: list[dict], system_answers: list[dict],
                                model: str) -> list[dict]:
    """Build Claude batch API requests for judging answers."""
    requests = []
    for item, ans in zip(items, system_answers):
        prompt = f"""**Query:** {item['query']}

**Ground Truth Answer:**
{item['ground_truth_answer']}

**System's Answer:**
{ans.get('answer', 'No answer generated')}

Score the system's answer against the ground truth. Respond with valid JSON only."""

        requests.append({
            "custom_id": item["id"],
            "params": {
                "model": model,
                "max_tokens": 512,
                "system": JUDGE_SYSTEM_PROMPT,
                "messages": [{"role": "user", "content": prompt}],
            }
        })
    return requests


def run_evaluation(args):
    """Run full evaluation pipeline."""
    benchmark_path = Path(args.benchmark)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load benchmark
    items = []
    with open(benchmark_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                items.append(json.loads(line))

    print(f"Loaded {len(items)} benchmark items", flush=True)

    # Import pipeline components
    ablation = getattr(args, 'ablation', None)
    pipeline_available = False
    retriever = None
    synthesizer = None

    if ablation == "no_retrieval":
        # Direct LLM — no retrieval, just Claude answering raw
        print(f"ABLATION: no_retrieval (direct LLM, no context)", flush=True)
        try:
            from apps.api.services.synthesizer import Synthesizer
            synthesizer = Synthesizer()
            pipeline_available = True
        except ImportError:
            print("WARNING: Synthesizer not available.", flush=True)
    elif ablation == "vector_only":
        # Vector-only RAG — disable graph facts
        print(f"ABLATION: vector_only (no KG, vector retrieval only)", flush=True)
        try:
            from apps.api.services.fusion_retriever import FusionRetriever
            from apps.api.services.synthesizer import Synthesizer
            retriever = FusionRetriever()
            synthesizer = Synthesizer()
            pipeline_available = True
        except ImportError:
            print("WARNING: Pipeline not available.", flush=True)
    else:
        # Full system
        try:
            from apps.api.services.fusion_retriever import FusionRetriever
            from apps.api.services.synthesizer import Synthesizer
            retriever = FusionRetriever()
            synthesizer = Synthesizer()
            pipeline_available = True
        except ImportError:
            print("WARNING: Pipeline not available. Running in judge-only mode.", flush=True)

    results = []
    latencies = []

    for i, item in enumerate(items):
        print(f"  [{i+1}/{len(items)}] {item['id']}: {item['query'][:60]}...", flush=True)

        if pipeline_available:
            t0 = time.perf_counter()

            if ablation == "no_retrieval":
                # No retrieval — send query directly to Claude
                retrieval = {
                    "graph_facts": [],
                    "chunks": [],
                    "task_category": item["task_category"],
                    "confidence": 0.0,
                    "entities_found": [],
                }
                answer = synthesizer.synthesize(item["query"], retrieval)
            elif ablation == "vector_only":
                # Full retrieval but zero out graph facts
                retrieval = retriever.retrieve(item["query"])
                retrieval["graph_facts"] = []  # Strip KG contribution
                answer = synthesizer.synthesize(item["query"], retrieval)
            else:
                # Full system
                retrieval = retriever.retrieve(item["query"])
                answer = synthesizer.synthesize(item["query"], retrieval)

            latency = time.perf_counter() - t0
        else:
            retrieval = {"graph_facts": [], "chunks": [], "task_category": item["task_category"]}
            answer = {"answer": "", "citations": []}
            latency = 0.0

        # Retrieval precision
        ret_precision = evaluate_retrieval_precision(retrieval, item)

        # Source recall (now includes retrieval-level chunk matching)
        src_recall = evaluate_source_recall(answer, item, retrieval)

        # Category match
        cat_match = retrieval.get("task_category") == item["task_category"]

        # Collect chunk source_files for traceability
        chunk_sources = [c.get("source_file", "") for c in retrieval.get("chunks", []) if c.get("source_file")]

        result = {
            "id": item["id"],
            "query": item["query"],
            "task_category": item["task_category"],
            "difficulty": item.get("difficulty", "medium"),
            "retrieval_precision": ret_precision,
            "source_recall": src_recall,
            "category_match": cat_match,
            "latency_s": round(latency, 3),
            "graph_facts_count": len(retrieval.get("graph_facts", [])),
            "chunks_count": len(retrieval.get("chunks", [])),
            "chunk_sources": chunk_sources,
            "system_answer": answer.get("answer", ""),
            "judge_scores": None,  # Filled in by judging step
        }
        results.append(result)
        latencies.append(latency)

    # Judge answers
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if api_key and pipeline_available:
        judge_model = args.judge_model
        if args.batch:
            # Batch judging
            client = anthropic.Anthropic(api_key=api_key)
            requests = build_judge_batch_requests(
                items,
                [{"answer": r["system_answer"]} for r in results],
                judge_model,
            )
            batch = client.messages.batches.create(requests=requests)
            print(f"\nJudge batch submitted: {batch.id}", flush=True)
            print(f"Retrieve with: --batch-retrieve {batch.id}", flush=True)
        else:
            # Sync judging
            client = anthropic.Anthropic(api_key=api_key)
            for i, (item, result) in enumerate(zip(items, results)):
                if result["system_answer"]:
                    scores = judge_answer(
                        client, item["query"], item["ground_truth_answer"],
                        result["system_answer"], judge_model,
                    )
                    result["judge_scores"] = scores
                    print(f"    Judged: overall={scores.get('overall', 0):.2f}", flush=True)

    # Compute aggregate metrics
    report = compute_report(results, items, latencies)

    # Write results
    output = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "benchmark_file": str(benchmark_path),
        "num_items": len(items),
        "report": report,
        "results": results,
    }
    output_path.write_text(json.dumps(output, indent=2))
    print(f"\nResults written to {output_path}", flush=True)

    # Print summary
    print_summary(report)


def compute_report(results: list[dict], items: list[dict],
                   latencies: list[float]) -> dict:
    """Compute aggregate evaluation metrics."""
    judged = [r for r in results if r.get("judge_scores")]

    report = {
        "retrieval": {
            "mean_precision": round(mean(r["retrieval_precision"] for r in results), 4),
            "mean_source_recall": round(mean(r["source_recall"] for r in results), 4),
            "category_accuracy": round(
                sum(1 for r in results if r["category_match"]) / len(results), 4
            ) if results else 0,
        },
        "latency": {
            "mean_s": round(mean(latencies), 3) if latencies else 0,
            "median_s": round(median(latencies), 3) if latencies else 0,
            "p95_s": round(sorted(latencies)[int(0.95 * len(latencies))], 3) if latencies else 0,
        },
    }

    if judged:
        dims = ["factual_accuracy", "completeness", "actionability", "specificity", "overall"]
        report["answer_quality"] = {}
        for dim in dims:
            scores = [r["judge_scores"].get(dim, 0) for r in judged]
            report["answer_quality"][dim] = round(mean(scores), 4) if scores else 0

    # Per-category breakdown
    by_cat: dict[str, list] = {}
    for r in results:
        by_cat.setdefault(r["task_category"], []).append(r)

    report["by_category"] = {}
    for cat, cat_results in by_cat.items():
        cat_report = {
            "count": len(cat_results),
            "mean_retrieval_precision": round(mean(r["retrieval_precision"] for r in cat_results), 4),
            "mean_source_recall": round(mean(r["source_recall"] for r in cat_results), 4),
        }
        cat_judged = [r for r in cat_results if r.get("judge_scores")]
        if cat_judged:
            cat_report["mean_overall_score"] = round(
                mean(r["judge_scores"].get("overall", 0) for r in cat_judged), 4
            )
        report["by_category"][cat] = cat_report

    # Per-difficulty breakdown
    by_diff: dict[str, list] = {}
    for r in results:
        by_diff.setdefault(r["difficulty"], []).append(r)

    report["by_difficulty"] = {}
    for diff, diff_results in by_diff.items():
        diff_report = {
            "count": len(diff_results),
            "mean_retrieval_precision": round(mean(r["retrieval_precision"] for r in diff_results), 4),
        }
        diff_judged = [r for r in diff_results if r.get("judge_scores")]
        if diff_judged:
            diff_report["mean_overall_score"] = round(
                mean(r["judge_scores"].get("overall", 0) for r in diff_judged), 4
            )
        report["by_difficulty"][diff] = diff_report

    return report


def print_summary(report: dict):
    """Print evaluation summary to console."""
    print(f"\n{'='*60}", flush=True)
    print(f"EDABench EVALUATION SUMMARY", flush=True)
    print(f"{'='*60}", flush=True)

    ret = report["retrieval"]
    print(f"\nRetrieval:", flush=True)
    print(f"  Mean precision:      {ret['mean_precision']:.4f}", flush=True)
    print(f"  Mean source recall:  {ret['mean_source_recall']:.4f}", flush=True)
    print(f"  Category accuracy:   {ret['category_accuracy']:.4f}", flush=True)

    lat = report["latency"]
    print(f"\nLatency:", flush=True)
    print(f"  Mean:    {lat['mean_s']:.3f}s", flush=True)
    print(f"  Median:  {lat['median_s']:.3f}s", flush=True)
    print(f"  p95:     {lat['p95_s']:.3f}s", flush=True)

    if "answer_quality" in report:
        aq = report["answer_quality"]
        print(f"\nAnswer Quality:", flush=True)
        for dim, score in aq.items():
            print(f"  {dim:20s} {score:.4f}", flush=True)

    print(f"\nBy Category:", flush=True)
    for cat, data in report.get("by_category", {}).items():
        score_str = f" overall={data['mean_overall_score']:.3f}" if "mean_overall_score" in data else ""
        print(f"  {cat:25s} n={data['count']:3d}  prec={data['mean_retrieval_precision']:.3f}"
              f"  recall={data['mean_source_recall']:.3f}{score_str}", flush=True)

    print(f"\nBy Difficulty:", flush=True)
    for diff, data in report.get("by_difficulty", {}).items():
        score_str = f" overall={data['mean_overall_score']:.3f}" if "mean_overall_score" in data else ""
        print(f"  {diff:10s} n={data['count']:3d}  prec={data['mean_retrieval_precision']:.3f}{score_str}", flush=True)

    print(f"{'='*60}", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Evaluate system on EDABench")
    parser.add_argument("--benchmark", required=True, help="EDABench JSONL file")
    parser.add_argument("--output", default="results/eval/edabench_results.json")
    parser.add_argument("--judge-model", default="claude-sonnet-4-20250514")
    parser.add_argument("--batch", action="store_true", help="Use batch API for judging")
    parser.add_argument("--ablation", default=None,
                        choices=["vector_only", "no_retrieval", "lora_only"],
                        help="Ablation mode: vector_only (no KG), no_retrieval (direct LLM), lora_only")
    args = parser.parse_args()

    run_evaluation(args)


if __name__ == "__main__":
    main()
