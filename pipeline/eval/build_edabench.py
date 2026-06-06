"""
Component 4.1 — EDABench construction pipeline.

Builds a 120-item evaluation benchmark from:
  1. Anchor seeds (mlcad_seeds.yaml — 7 verified items)
  2. High-quality holdout Q&A (test.jsonl — not seen during LoRA training)
  3. KG-derived questions (Claude generates from known graph triples)

Each item includes ground-truth retrieval expectations for KG-grounded evaluation.

Usage:
    # Step 1: Generate candidates via Claude batch API
    python -m pipeline.eval.build_edabench \
        --mode generate \
        --seeds data/edabench/seeds/mlcad_seeds.yaml \
        --holdout data/train/test.jsonl \
        --output data/edabench/edabench_candidates.jsonl \
        --batch

    # Step 2: Retrieve batch results
    python -m pipeline.eval.build_edabench \
        --mode retrieve \
        --batch-id BATCH_ID \
        --output data/edabench/edabench_candidates.jsonl

    # Step 3: Assemble final benchmark (contamination check + balance)
    python -m pipeline.eval.build_edabench \
        --mode assemble \
        --candidates data/edabench/edabench_candidates.jsonl \
        --train-data data/train/train.jsonl \
        --output data/edabench/edabench_v1.jsonl
"""
import argparse
import json
import os
import random
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import yaml

try:
    import anthropic
except ImportError:
    print("ERROR: anthropic SDK not installed. Run: pip install anthropic")
    sys.exit(1)

try:
    from datasketch import MinHash, MinHashLSH
except ImportError:
    print("ERROR: datasketch not installed. Run: pip install datasketch")
    sys.exit(1)


# Target distribution
CATEGORY_TARGETS = {
    "error_diagnosis": 48,
    "rtl_qa": 18,
    "constraint_generation": 18,
    "drc_rule_lookup": 18,
    "cross_tool_knowledge": 18,
}
TOTAL_TARGET = 120

# MinHash contamination settings
NUM_PERM = 128
CONTAMINATION_THRESHOLD = 0.70

# KG triples for question generation (from graph_schema.cypher + known nodes)
KG_FACTS = [
    {
        "category": "error_diagnosis",
        "fact": "Violation ed_001_sdc_override: FLOW_VARIANT does not override SDC clock period; all sweep runs produce identical PPA. Fix: Create per-variant SDC files.",
        "graph_nodes": ["ed_001_sdc_override", "variant_sdc_override"],
        "difficulty": "hard",
    },
    {
        "category": "error_diagnosis",
        "fact": "Violation ed_002_version_divergence: ORFS v3.0 to 26Q1 migration produces >10% PPA divergence. JPEG WNS flips from +13.7ps to -12.8ps. Fix: Version-tag all training data.",
        "graph_nodes": ["ed_002_version_divergence", "version_tag_training_data", "orfs_v3_0", "orfs_26q1"],
        "difficulty": "hard",
    },
    {
        "category": "error_diagnosis",
        "fact": "Violation ed_003_unit_mismatch: SDC written in ps, tool expects ns. Produces implausible WNS values like +1244ns. Fix: Prepend set_units -time ns.",
        "graph_nodes": ["ed_003_unit_mismatch", "prepend_unit_declaration"],
        "difficulty": "medium",
    },
    {
        "category": "error_diagnosis",
        "fact": "Violation ed_004_def_naming: CircuitNet DEF instance names don't match OpenROAD expectations. Fix: Run automated name normalization.",
        "graph_nodes": ["ed_004_def_naming", "automated_name_normalization"],
        "difficulty": "medium",
    },
    {
        "category": "error_diagnosis",
        "fact": "Violation ed_005_ibex_26q1_sigsegv: OpenROAD 26Q1 crashes with SIGSEGV during global routing on ibex design with OPENROAD_HIERARCHICAL=1. Peak memory 717MB. Fix: Pin to ORFS v3.0 or disable hierarchical routing.",
        "graph_nodes": ["ed_005_ibex_26q1_sigsegv", "orfs_26q1", "orfs_ibex_26q1"],
        "difficulty": "high",
    },
    {
        "category": "cross_tool_knowledge",
        "fact": "ORFS v3.0 and 26Q1 diverge in timing engine and detailed routing, causing >10% absolute PPA shifts. However, Bayesian optimization candidates remain on Pareto front across versions.",
        "graph_nodes": ["orfs_v3_0", "orfs_26q1"],
        "difficulty": "expert",
    },
    {
        "category": "constraint_generation",
        "fact": "SDC constraints for OpenROAD require nanosecond time units. A 500MHz clock needs create_clock -period 2.0 -name clk [get_ports clk].",
        "graph_nodes": [],
        "difficulty": "easy",
    },
    {
        "category": "drc_rule_lookup",
        "fact": "SKY130 metal1 minimum width is 0.14um. DRC violations occur when routing produces wires below this threshold.",
        "graph_nodes": [],
        "difficulty": "easy",
    },
    {
        "category": "drc_rule_lookup",
        "fact": "ASAP7 has different DRC rules than SKY130. Via enclosure rules differ between PDKs and cause violations when switching.",
        "graph_nodes": [],
        "difficulty": "medium",
    },
    {
        "category": "rtl_qa",
        "fact": "Yosys synthesis errors on width mismatches in module instantiations. The port width in the module definition must match the connecting signal width.",
        "graph_nodes": [],
        "difficulty": "easy",
    },
]

GENERATE_SYSTEM_PROMPT = """You are creating evaluation benchmark items for an EDA copilot system. 
Each benchmark item tests whether the system can correctly answer EDA questions using a 
knowledge graph and document retrieval.

Generate a Q&A pair for the given task category. The response MUST be valid JSON with:
- "query": A natural question a chip designer would ask (50-150 words)
- "ground_truth_answer": The correct, detailed answer (100-300 words)
- "expected_sources": List of plausible source file types/names that should be cited
- "difficulty": "easy", "medium", "hard", or "expert"

Guidelines:
- Questions must be specific and testable — not vague or open-ended
- Answers must be factually correct for the EDA domain
- Include tool-specific details (OpenROAD, Yosys, SKY130, ASAP7, etc.)
- Reference realistic metrics (WNS, TNS, slack, DRC counts, cell areas)
- Vary question styles and complexity levels"""

ANNOTATE_SYSTEM_PROMPT = """You are annotating an EDA benchmark item with ground-truth metadata.
Given a Q&A pair, add structured annotations for evaluation.

Your response MUST be valid JSON with:
- "expected_graph_nodes": List of knowledge graph entity IDs that should appear in retrieval 
  results (e.g., violation IDs, tool version nodes, design names). Use lowercase_underscore format.
- "expected_sources": List of source file paths/types that the system should cite
- "difficulty": "easy", "medium", "hard", or "expert"
- "reasoning": Brief explanation of why these graph nodes are expected (1-2 sentences)

Known graph node patterns: ed_001_*, ed_002_*, violation IDs, fix IDs, 
orfs_v3_0, orfs_26q1, design names (jpeg, ibex, aes, gcd, spm_unit, swerv_wrapper).
Tool names: OpenROAD, Yosys, Magic, KLayout. PDKs: SKY130, ASAP7."""


def load_seeds(seeds_path: Path) -> list[dict]:
    """Load anchor seed items from YAML."""
    with open(seeds_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    items = []
    for seed in raw:
        items.append({
            "id": seed["id"],
            "query": seed["question"].strip(),
            "ground_truth_answer": seed["gold_answer"].strip(),
            "expected_graph_nodes": _seed_to_graph_nodes(seed["id"]),
            "expected_sources": _seed_to_sources(seed),
            "task_category": seed["category"],
            "difficulty": seed["difficulty"],
            "seed_bug_id": seed["id"],
            "source": "mlcad_seed",
            "verified": True,
        })
    return items


def _seed_to_graph_nodes(seed_id: str) -> list[str]:
    """Map seed IDs to expected KG node IDs."""
    mapping = {
        "ED-001": ["ed_001_sdc_override", "variant_sdc_override"],
        "ED-002": ["ed_002_version_divergence", "version_tag_training_data", "orfs_v3_0", "orfs_26q1"],
        "ED-003": ["ed_003_unit_mismatch", "prepend_unit_declaration"],
        "ED-004": ["ed_004_def_naming", "automated_name_normalization"],
        "ED-005": ["ed_005_ibex_26q1_sigsegv", "orfs_26q1", "orfs_ibex_26q1"],
        "ML-001": [],
        "ML-002": ["orfs_v3_0", "orfs_26q1"],
    }
    return mapping.get(seed_id, [])


def _seed_to_sources(seed: dict) -> list[str]:
    """Map seed items to expected source files."""
    sources = []
    if seed.get("tool") == "OpenROAD":
        sources.extend(["6_finish.rpt", "timing_report"])
    if "SDC" in seed.get("question", "") or "sdc" in seed.get("question", "").lower():
        sources.append("constraints.sdc")
    if "DEF" in seed.get("question", ""):
        sources.append(".def")
    return sources if sources else ["documentation"]


def sample_holdout(holdout_path: Path, category_needs: dict[str, int],
                   seed: int = 42) -> list[dict]:
    """Sample high-quality items from the test holdout split."""
    rng = random.Random(seed)
    records = []
    with open(holdout_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rec = json.loads(line)
                if rec.get("judge_score", 0) >= 0.90:
                    records.append(rec)

    by_category: dict[str, list] = {}
    for rec in records:
        cat = rec["task_category"]
        # Map general_eda to closest match
        if cat == "general_eda":
            cat = "cross_tool_knowledge"
        by_category.setdefault(cat, []).append(rec)

    sampled = []
    eb_counter = 1

    for cat, need in category_needs.items():
        pool = by_category.get(cat, [])
        rng.shuffle(pool)
        # Take up to 60% from holdout, rest from generation
        holdout_take = min(len(pool), int(need * 0.6))
        for rec in pool[:holdout_take]:
            sampled.append({
                "id": f"EB-{eb_counter:03d}",
                "query": rec["input"].strip(),
                "ground_truth_answer": rec["output"].strip(),
                "expected_graph_nodes": [],  # Will be annotated later
                "expected_sources": [rec.get("source_file", "documentation")],
                "task_category": cat,
                "difficulty": "medium",  # Will be annotated later
                "seed_bug_id": "",
                "source": "holdout_test_split",
                "verified": False,
            })
            eb_counter += 1

    return sampled


def build_generation_requests(category_needs: dict[str, int],
                              model: str = "claude-sonnet-4-20250514") -> list[dict]:
    """Build Claude batch API requests for generating new benchmark items."""
    requests = []
    req_counter = 0

    for cat, need in category_needs.items():
        # Find matching KG facts
        cat_facts = [f for f in KG_FACTS if f["category"] == cat]

        for i in range(need):
            fact = cat_facts[i % len(cat_facts)] if cat_facts else None
            prompt_parts = [f"Generate a benchmark Q&A item for category: {cat}"]

            if fact:
                prompt_parts.append(f"\nRelevant KG fact for inspiration (vary the question, don't copy):\n{fact['fact']}")
                prompt_parts.append(f"Difficulty target: {fact['difficulty']}")
            else:
                difficulties = ["easy", "medium", "hard"]
                prompt_parts.append(f"Difficulty target: {difficulties[i % 3]}")

            prompt_parts.append(f"\nItem number {i+1} of {need} for this category — ensure diversity.")

            requests.append({
                "custom_id": f"gen_{cat}_{req_counter:04d}",
                "params": {
                    "model": model,
                    "max_tokens": 1024,
                    "system": GENERATE_SYSTEM_PROMPT,
                    "messages": [{"role": "user", "content": "\n".join(prompt_parts)}],
                }
            })
            req_counter += 1

    return requests


def build_annotation_requests(items: list[dict],
                              model: str = "claude-sonnet-4-20250514") -> list[dict]:
    """Build Claude batch API requests for annotating items with ground truth."""
    requests = []
    for item in items:
        if item.get("expected_graph_nodes"):  # Already annotated (seeds)
            continue
        prompt = f"""Annotate this EDA benchmark item:

Category: {item['task_category']}
Query: {item['query']}
Answer: {item['ground_truth_answer'][:500]}

Respond with valid JSON only — no markdown fences."""

        requests.append({
            "custom_id": item["id"],
            "params": {
                "model": model,
                "max_tokens": 512,
                "system": ANNOTATE_SYSTEM_PROMPT,
                "messages": [{"role": "user", "content": prompt}],
            }
        })
    return requests


def text_to_minhash(text: str, num_perm: int = NUM_PERM) -> MinHash:
    """MinHash from text using word-level 5-grams."""
    m = MinHash(num_perm=num_perm)
    words = text.lower().split()
    for i in range(max(1, len(words) - 4)):
        shingle = " ".join(words[i:i + 5])
        m.update(shingle.encode("utf-8"))
    return m


def contamination_check(candidates: list[dict], train_path: Path,
                         threshold: float = CONTAMINATION_THRESHOLD) -> tuple[list[dict], list[dict]]:
    """Check benchmark candidates against training data. Returns (clean, contaminated)."""
    # Build LSH index from training data
    lsh = MinHashLSH(threshold=threshold, num_perm=NUM_PERM)
    train_hashes = {}

    print(f"Building LSH index from training data...", flush=True)
    with open(train_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if line.strip():
                rec = json.loads(line)
                text = f"{rec.get('input', '')} {rec.get('output', '')}"
                mh = text_to_minhash(text)
                key = f"train_{idx}"
                train_hashes[key] = mh
                try:
                    lsh.insert(key, mh)
                except ValueError:
                    pass

    print(f"LSH index built with {len(train_hashes):,} training records", flush=True)

    clean = []
    contaminated = []

    for item in candidates:
        item_text = f"{item['query']} {item['ground_truth_answer']}"
        item_mh = text_to_minhash(item_text)
        matches = lsh.query(item_mh)

        if matches:
            best_sim = max(
                item_mh.jaccard(train_hashes[m]) for m in matches
                if m in train_hashes
            )
            if best_sim >= threshold:
                item["contamination"] = {
                    "best_jaccard": round(best_sim, 4),
                    "num_matches": len(matches),
                }
                contaminated.append(item)
                continue

        clean.append(item)

    return clean, contaminated


def assemble_benchmark(candidates: list[dict], category_targets: dict[str, int]) -> list[dict]:
    """Select final benchmark items to match target distribution."""
    by_category: dict[str, list] = {}
    for item in candidates:
        cat = item["task_category"]
        by_category.setdefault(cat, []).append(item)

    final = []
    for cat, target in category_targets.items():
        pool = by_category.get(cat, [])
        # Prioritize: seeds first, then holdout, then generated
        seeds = [i for i in pool if i.get("source") == "mlcad_seed"]
        holdout = [i for i in pool if i.get("source") == "holdout_test_split"]
        generated = [i for i in pool if i.get("source") == "generated"]

        selected = seeds[:]
        remaining = target - len(selected)

        if remaining > 0:
            selected.extend(holdout[:remaining])
            remaining = target - len(selected)

        if remaining > 0:
            selected.extend(generated[:remaining])

        final.extend(selected[:target])

    # Re-number sequentially
    seed_ids = {i["id"] for i in final if i.get("source") == "mlcad_seed"}
    eb_counter = 1
    for item in final:
        if item["id"] not in seed_ids:
            item["id"] = f"EB-{eb_counter:03d}"
            eb_counter += 1

    return final


def submit_batch(api_key: str, requests: list[dict], purpose: str) -> str:
    """Submit requests to Claude batch API."""
    client = anthropic.Anthropic(api_key=api_key)

    print(f"Submitting {len(requests)} {purpose} requests to batch API...", flush=True)
    batch = client.messages.batches.create(requests=requests)
    batch_id = batch.id

    jobs_dir = Path("results/batch_jobs")
    jobs_dir.mkdir(parents=True, exist_ok=True)
    job_meta = {
        "batch_id": batch_id,
        "purpose": purpose,
        "submitted_at": datetime.now(timezone.utc).isoformat(),
        "num_requests": len(requests),
        "status": "processing",
    }
    job_file = jobs_dir / f"{batch_id}.json"
    job_file.write_text(json.dumps(job_meta, indent=2))

    print(f"\n{'='*60}", flush=True)
    print(f"BATCH SUBMITTED — {purpose}", flush=True)
    print(f"{'='*60}", flush=True)
    print(f"Batch ID: {batch_id}", flush=True)
    print(f"Requests: {len(requests):,}", flush=True)
    print(f"Retrieve: python -m pipeline.eval.build_edabench --mode retrieve --batch-id {batch_id} --output <path>", flush=True)
    print(f"{'='*60}", flush=True)

    return batch_id


def retrieve_batch(api_key: str, batch_id: str, output_path: Path) -> list[dict]:
    """Retrieve batch results and parse into benchmark items."""
    client = anthropic.Anthropic(api_key=api_key)

    print(f"Retrieving batch {batch_id}...", flush=True)
    batch = client.messages.batches.retrieve(batch_id)
    print(f"Status: {batch.processing_status}", flush=True)

    if batch.processing_status != "ended":
        counts = batch.request_counts
        print(f"  Processing: {counts.processing}", flush=True)
        print(f"  Succeeded:  {counts.succeeded}", flush=True)
        print(f"  Errored:    {counts.errored}", flush=True)
        return []

    items = []
    success = 0
    failed = 0

    for result in client.messages.batches.results(batch_id):
        if result.result.type == "succeeded":
            text = result.result.message.content[0].text.strip()
            if text.startswith("```"):
                text = text.split("\n", 1)[1]
                if text.endswith("```"):
                    text = text[:-3]
                text = text.strip()

            try:
                data = json.loads(text)
                custom_id = result.custom_id

                if custom_id.startswith("gen_"):
                    # Generation result
                    parts = custom_id.split("_")
                    cat = "_".join(parts[1:-1])
                    item = {
                        "id": custom_id,
                        "query": data["query"],
                        "ground_truth_answer": data["ground_truth_answer"],
                        "expected_graph_nodes": data.get("expected_graph_nodes", []),
                        "expected_sources": data.get("expected_sources", []),
                        "task_category": cat,
                        "difficulty": data.get("difficulty", "medium"),
                        "seed_bug_id": "",
                        "source": "generated",
                        "verified": False,
                    }
                    items.append(item)
                else:
                    # Annotation result — store for merging
                    items.append({"_annotation_id": custom_id, **data})

                success += 1
            except (json.JSONDecodeError, KeyError):
                failed += 1
        else:
            failed += 1

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"Retrieved: {success} succeeded, {failed} failed", flush=True)
    print(f"Output: {output_path}", flush=True)
    return items


def mode_generate(args):
    """Generate benchmark candidates."""
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("ERROR: ANTHROPIC_API_KEY not set")
        sys.exit(1)

    seeds_path = Path(args.seeds)
    holdout_path = Path(args.holdout)
    output_path = Path(args.output)

    # Step 1: Load seeds as anchors
    seeds = load_seeds(seeds_path)
    print(f"Loaded {len(seeds)} seed anchor items", flush=True)

    # Count remaining needs per category
    seed_counts = Counter(s["task_category"] for s in seeds)
    category_needs = {}
    for cat, target in CATEGORY_TARGETS.items():
        category_needs[cat] = target - seed_counts.get(cat, 0)

    print(f"Category needs after seeds: {category_needs}", flush=True)

    # Step 2: Sample from holdout
    holdout_items = sample_holdout(holdout_path, category_needs)
    print(f"Sampled {len(holdout_items)} items from holdout", flush=True)

    # Update needs
    holdout_counts = Counter(h["task_category"] for h in holdout_items)
    gen_needs = {}
    for cat, need in category_needs.items():
        remaining = need - holdout_counts.get(cat, 0)
        if remaining > 0:
            gen_needs[cat] = remaining

    print(f"Generation needs: {gen_needs}", flush=True)

    # Step 3: Save seeds + holdout items
    all_items = seeds + holdout_items
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for item in all_items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"Wrote {len(all_items)} items to {output_path}", flush=True)

    # Step 4: Generate remaining via Claude
    if sum(gen_needs.values()) > 0:
        gen_requests = build_generation_requests(gen_needs, args.model)
        print(f"Need to generate {len(gen_requests)} more items", flush=True)

        if args.batch:
            batch_id = submit_batch(api_key, gen_requests, "edabench_generation")
            print(f"\nAfter batch completes, run:", flush=True)
            print(f"  python -m pipeline.eval.build_edabench --mode retrieve --batch-id {batch_id} --output {output_path}", flush=True)
        else:
            # Sync mode — call Claude directly
            client = anthropic.Anthropic(api_key=api_key)
            generated = []
            for req in gen_requests:
                try:
                    response = client.messages.create(**req["params"])
                    text = response.content[0].text.strip()
                    if text.startswith("```"):
                        text = text.split("\n", 1)[1]
                        if text.endswith("```"):
                            text = text[:-3]
                        text = text.strip()
                    data = json.loads(text)
                    parts = req["custom_id"].split("_")
                    cat = "_".join(parts[1:-1])
                    item = {
                        "id": req["custom_id"],
                        "query": data["query"],
                        "ground_truth_answer": data["ground_truth_answer"],
                        "expected_graph_nodes": data.get("expected_graph_nodes", []),
                        "expected_sources": data.get("expected_sources", []),
                        "task_category": cat,
                        "difficulty": data.get("difficulty", "medium"),
                        "seed_bug_id": "",
                        "source": "generated",
                        "verified": False,
                    }
                    generated.append(item)
                    print(f"  Generated {req['custom_id']}: {data['query'][:60]}...", flush=True)
                except Exception as e:
                    print(f"  FAILED {req['custom_id']}: {e}", flush=True)

            with open(output_path, "a", encoding="utf-8") as f:
                for item in generated:
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")
            print(f"Generated {len(generated)} additional items", flush=True)

    # Step 5: Build annotation requests for holdout items
    items_needing_annotation = [i for i in holdout_items if not i.get("expected_graph_nodes")]
    if items_needing_annotation:
        ann_requests = build_annotation_requests(items_needing_annotation, args.model)
        if args.batch and ann_requests:
            ann_batch_id = submit_batch(api_key, ann_requests, "edabench_annotation")
            print(f"Annotation batch submitted: {ann_batch_id}", flush=True)


def mode_retrieve(args):
    """Retrieve batch results."""
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("ERROR: ANTHROPIC_API_KEY not set")
        sys.exit(1)

    output_path = Path(args.output)
    items = retrieve_batch(api_key, args.batch_id, output_path)
    print(f"Retrieved {len(items)} items", flush=True)


def mode_assemble(args):
    """Assemble final benchmark from candidates with contamination check."""
    candidates_path = Path(args.candidates)
    train_path = Path(args.train_data)
    output_path = Path(args.output)

    # Load all candidates
    candidates = []
    with open(candidates_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rec = json.loads(line)
                if "_annotation_id" not in rec:
                    candidates.append(rec)

    print(f"Loaded {len(candidates)} candidate items", flush=True)

    # Contamination check
    clean, contaminated = contamination_check(candidates, train_path)
    print(f"\nContamination check: {len(clean)} clean, {len(contaminated)} contaminated", flush=True)

    if contaminated:
        quarantine_path = output_path.parent / "edabench_contaminated.jsonl"
        with open(quarantine_path, "w", encoding="utf-8") as f:
            for item in contaminated:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(f"Quarantined {len(contaminated)} items to {quarantine_path}", flush=True)

    # Assemble to target distribution
    final = assemble_benchmark(clean, CATEGORY_TARGETS)

    # Ensure difficulty distribution
    difficulty_counts = Counter(i["difficulty"] for i in final)
    hard_count = difficulty_counts.get("hard", 0) + difficulty_counts.get("expert", 0)

    # Write final benchmark
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for item in final:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    # Summary
    cat_counts = Counter(i["task_category"] for i in final)
    print(f"\n{'='*60}", flush=True)
    print(f"EDABench v1 ASSEMBLED", flush=True)
    print(f"{'='*60}", flush=True)
    print(f"Total items:        {len(final)}", flush=True)
    print(f"Target:             {TOTAL_TARGET}", flush=True)
    print(f"Contaminated:       {len(contaminated)} (removed)", flush=True)
    print(f"\nCategory distribution:", flush=True)
    for cat in CATEGORY_TARGETS:
        actual = cat_counts.get(cat, 0)
        target = CATEGORY_TARGETS[cat]
        status = "OK" if actual >= target else f"SHORT ({target - actual} needed)"
        print(f"  {cat:25s} {actual:3d}/{target:3d} {status}", flush=True)
    print(f"\nDifficulty distribution:", flush=True)
    for diff in ["easy", "medium", "hard", "expert"]:
        print(f"  {diff:10s} {difficulty_counts.get(diff, 0):3d}", flush=True)
    print(f"  Hard+Expert:   {hard_count:3d} (target >=20)", flush=True)
    print(f"\nOutput: {output_path}", flush=True)
    print(f"{'='*60}", flush=True)

    # Save assembly report
    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "total_items": len(final),
        "target": TOTAL_TARGET,
        "contaminated_removed": len(contaminated),
        "category_distribution": dict(cat_counts),
        "difficulty_distribution": dict(difficulty_counts),
        "sources": dict(Counter(i.get("source", "unknown") for i in final)),
    }
    report_path = output_path.parent / "edabench_assembly_report.json"
    report_path.write_text(json.dumps(report, indent=2))
    print(f"Report: {report_path}", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Build EDABench evaluation benchmark")
    parser.add_argument("--mode", required=True, choices=["generate", "retrieve", "assemble"],
                        help="Pipeline mode")
    parser.add_argument("--seeds", default="data/edabench/seeds/mlcad_seeds.yaml")
    parser.add_argument("--holdout", default="data/train/test.jsonl")
    parser.add_argument("--output", default="data/edabench/edabench_candidates.jsonl")
    parser.add_argument("--candidates", help="Candidates file for assemble mode")
    parser.add_argument("--train-data", default="data/train/train.jsonl",
                        help="Training data for contamination check")
    parser.add_argument("--model", default="claude-sonnet-4-20250514")
    parser.add_argument("--batch", action="store_true",
                        help="Use batch API (50%% cheaper, 24h turnaround)")
    parser.add_argument("--batch-id", help="Batch ID for retrieve mode")
    args = parser.parse_args()

    if args.mode == "generate":
        mode_generate(args)
    elif args.mode == "retrieve":
        if not args.batch_id:
            print("ERROR: --batch-id required for retrieve mode")
            sys.exit(1)
        mode_retrieve(args)
    elif args.mode == "assemble":
        if not args.candidates:
            print("ERROR: --candidates required for assemble mode")
            sys.exit(1)
        mode_assemble(args)


if __name__ == "__main__":
    main()
