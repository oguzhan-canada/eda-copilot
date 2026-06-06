"""
Component 1.2 — Generate Q&A pairs from injected violation cases.

Calls Claude API to expand each injected case into a structured Q&A pair
with question, answer, evidence_span, and task_category.

Supports two modes:
  - async: Concurrent API calls for fast iteration (~10x faster than sequential)
  - batch: Anthropic Message Batches API for 50% cost savings (24h turnaround)

Usage:
    # Fast mode (dev/testing)
    python -m pipeline.synth_qa.generate_qa \
        --input data/synthetic/injected_cases.jsonl \
        --output data/synthetic/qa_raw.jsonl \
        --model claude-sonnet-4-20250514 \
        --concurrency 10

    # Batch mode (production, 50% cheaper)
    python -m pipeline.synth_qa.generate_qa \
        --input data/synthetic/injected_cases.jsonl \
        --output data/synthetic/qa_raw.jsonl \
        --model claude-sonnet-4-20250514 \
        --batch

    # Check/retrieve batch results
    python -m pipeline.synth_qa.generate_qa \
        --batch-retrieve BATCH_ID \
        --output data/synthetic/qa_raw.jsonl
"""
import argparse
import asyncio
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

try:
    import anthropic
except ImportError:
    print("ERROR: anthropic SDK not installed. Run: pip install anthropic")
    sys.exit(1)


SYSTEM_PROMPT = """You are an expert EDA (Electronic Design Automation) engineer generating 
training data for an EDA copilot system. Given a violation scenario in an EDA file, generate 
a realistic Q&A pair that a chip designer might encounter and need help with.

Your response must be valid JSON with exactly these fields:
- "question": A natural, realistic question a designer would ask (50-150 words)
- "answer": A detailed, actionable answer with root cause and fix (100-300 words)  
- "evidence_span": The specific snippet from the file that shows the violation (20-100 words)
- "task_category": One of: error_diagnosis, constraint_generation, rtl_qa, cross_tool_knowledge, general_eda

Guidelines:
- Questions should sound natural, not formulaic
- Answers must include: what went wrong, why it matters, and how to fix it
- Evidence spans should quote the specific problematic code/config
- Vary question styles: "Why does...", "After running...", "My design shows...", "How do I fix..."
- Include tool-specific details (OpenROAD, Yosys, STA, DRC checker names)
- Reference realistic metrics (WNS, TNS, slack values, DRC counts)"""


def build_user_prompt(case: dict) -> str:
    """Build the user prompt from an injected violation case."""
    return f"""Generate a Q&A pair for this EDA violation scenario:

**Violation Family**: {case['violation_family']}
**Task Category**: {case['task_category']}
**Source File**: {case['source_file']}
**Violation Detail**: {case['violation_detail']}

**Injected Code Snippet** (the file with the error):
```
{case.get('injected_snippet', '')[:1500]}
```

**Expected Diagnosis**: {case['expected_diagnosis']}

**Expected Fix**: {case['expected_fix']}

{f"**Seed Bug Reference**: {case['seed_bug_id']} (from verified MLCAD project findings)" if case.get('seed_bug_id') else ""}

Generate a realistic Q&A pair. Respond with valid JSON only — no markdown fences."""


def call_claude(client: anthropic.Anthropic, case: dict, model: str,
                max_retries: int = 3) -> dict | None:
    """Call Claude API to generate a Q&A pair from an injected case."""
    for attempt in range(max_retries):
        try:
            response = client.messages.create(
                model=model,
                max_tokens=1024,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": build_user_prompt(case)}],
            )

            text = response.content[0].text.strip()
            # Strip markdown fences if present
            if text.startswith("```"):
                text = text.split("\n", 1)[1]
                if text.endswith("```"):
                    text = text[:-3]
                text = text.strip()

            qa = json.loads(text)

            # Validate required fields
            required = {"question", "answer", "evidence_span", "task_category"}
            if not required.issubset(qa.keys()):
                missing = required - qa.keys()
                continue

            # Add metadata
            qa["case_id"] = case["case_id"]
            qa["violation_family"] = case["violation_family"]
            qa["source_file"] = case["source_file"]
            qa["seed_bug_id"] = case.get("seed_bug_id", "")
            qa["model_used"] = model
            qa["input_tokens"] = response.usage.input_tokens
            qa["output_tokens"] = response.usage.output_tokens

            return qa

        except json.JSONDecodeError:
            time.sleep(1)
        except anthropic.RateLimitError:
            wait = 2 ** (attempt + 1)
            time.sleep(wait)
        except anthropic.APIError as e:
            time.sleep(2)

    return None


async def call_claude_async(aclient: anthropic.AsyncAnthropic, case: dict, model: str,
                            semaphore: asyncio.Semaphore, max_retries: int = 3) -> dict | None:
    """Async version of call_claude for concurrent processing."""
    async with semaphore:
        for attempt in range(max_retries):
            try:
                response = await aclient.messages.create(
                    model=model,
                    max_tokens=1024,
                    system=SYSTEM_PROMPT,
                    messages=[{"role": "user", "content": build_user_prompt(case)}],
                )

                text = response.content[0].text.strip()
                if text.startswith("```"):
                    text = text.split("\n", 1)[1]
                    if text.endswith("```"):
                        text = text[:-3]
                    text = text.strip()

                qa = json.loads(text)

                required = {"question", "answer", "evidence_span", "task_category"}
                if not required.issubset(qa.keys()):
                    continue

                qa["case_id"] = case["case_id"]
                qa["violation_family"] = case["violation_family"]
                qa["source_file"] = case["source_file"]
                qa["seed_bug_id"] = case.get("seed_bug_id", "")
                qa["model_used"] = model
                qa["input_tokens"] = response.usage.input_tokens
                qa["output_tokens"] = response.usage.output_tokens

                return qa

            except json.JSONDecodeError:
                await asyncio.sleep(1)
            except anthropic.RateLimitError:
                wait = 2 ** (attempt + 1)
                await asyncio.sleep(wait)
            except anthropic.APIError:
                await asyncio.sleep(2)

    return None


async def process_batch_async(aclient: anthropic.AsyncAnthropic, cases: list,
                               model: str, concurrency: int) -> list:
    """Process a batch of cases concurrently."""
    semaphore = asyncio.Semaphore(concurrency)
    tasks = [call_claude_async(aclient, case, model, semaphore) for case in cases]
    return await asyncio.gather(*tasks)


def main():
    parser = argparse.ArgumentParser(description="Generate Q&A from injected violations")
    parser.add_argument("--input", help="Injected cases JSONL")
    parser.add_argument("--output", default="data/synthetic/qa_raw.jsonl")
    parser.add_argument("--model", default="claude-sonnet-4-20250514")
    parser.add_argument("--batch-size", type=int, default=50,
                        help="Batch size for concurrent processing")
    parser.add_argument("--concurrency", type=int, default=10,
                        help="Max concurrent API calls")
    parser.add_argument("--limit", type=int, default=0,
                        help="Process only first N records (0 = all)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from existing output file")
    parser.add_argument("--batch", action="store_true",
                        help="Use Anthropic Message Batches API (50%% cheaper, 24h turnaround)")
    parser.add_argument("--batch-retrieve", type=str, default=None,
                        help="Retrieve results for a batch ID")
    args = parser.parse_args()

    # Check API key
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("ERROR: ANTHROPIC_API_KEY environment variable not set.")
        print("Set it with: $env:ANTHROPIC_API_KEY = 'sk-ant-...'")
        sys.exit(1)

    # Batch retrieve mode
    if args.batch_retrieve:
        retrieve_batch_results(api_key, args.batch_retrieve, Path(args.output))
        return

    if not args.input:
        print("ERROR: --input required (or use --batch-retrieve)")
        sys.exit(1)

    # Batch submission mode
    if args.batch:
        submit_batch(api_key, args)
        return

    # Async streaming mode (original behavior)
    run_async_mode(api_key, args)


def submit_batch(api_key: str, args):
    """Submit a batch job to Anthropic Message Batches API (50% cost savings)."""
    client = anthropic.Anthropic(api_key=api_key)
    input_path = Path(args.input)
    output_path = Path(args.output)

    # Load cases
    cases = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                cases.append(json.loads(line))

    if args.limit > 0:
        cases = cases[:args.limit]

    # Resume support
    done_ids = set()
    if args.resume and output_path.exists():
        with open(output_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rec = json.loads(line)
                    done_ids.add(rec.get("case_id"))
        cases = [c for c in cases if c["case_id"] not in done_ids]

    if not cases:
        print("No new cases to process. Done.", flush=True)
        return

    print(f"Submitting {len(cases)} cases to Batch API...", flush=True)
    print(f"Model: {args.model} | Mode: BATCH (50% discount)", flush=True)

    # Build batch requests
    requests = []
    for case in cases:
        requests.append({
            "custom_id": case["case_id"],
            "params": {
                "model": args.model,
                "max_tokens": 1024,
                "system": SYSTEM_PROMPT,
                "messages": [{"role": "user", "content": build_user_prompt(case)}],
            }
        })

    # Submit
    batch = client.messages.batches.create(requests=requests)
    batch_id = batch.id

    # Save batch job metadata
    jobs_dir = Path("results/batch_jobs")
    jobs_dir.mkdir(parents=True, exist_ok=True)
    job_meta = {
        "batch_id": batch_id,
        "submitted_at": datetime.now(timezone.utc).isoformat(),
        "input_file": str(input_path),
        "output_file": str(output_path),
        "model": args.model,
        "num_requests": len(requests),
        "status": "processing",
    }
    job_file = jobs_dir / f"{batch_id}.json"
    with open(job_file, "w", encoding="utf-8") as f:
        json.dump(job_meta, f, indent=2)

    print(f"\n{'='*60}", flush=True)
    print(f"BATCH SUBMITTED", flush=True)
    print(f"{'='*60}", flush=True)
    print(f"Batch ID:       {batch_id}", flush=True)
    print(f"Requests:       {len(requests):,}", flush=True)
    print(f"Job metadata:   {job_file}", flush=True)
    print(f"", flush=True)
    print(f"Results available in up to 24 hours.", flush=True)
    print(f"Retrieve with:", flush=True)
    print(f"  python -m pipeline.synth_qa.generate_qa --batch-retrieve {batch_id} --output {output_path}", flush=True)
    print(f"{'='*60}", flush=True)


def retrieve_batch_results(api_key: str, batch_id: str, output_path: Path):
    """Retrieve and process results from a completed batch job."""
    client = anthropic.Anthropic(api_key=api_key)

    print(f"Retrieving batch {batch_id}...", flush=True)
    batch = client.messages.batches.retrieve(batch_id)
    print(f"Status: {batch.processing_status}", flush=True)

    if batch.processing_status != "ended":
        print(f"\nBatch not yet complete. Check back later.", flush=True)
        counts = batch.request_counts
        print(f"  Processing: {counts.processing}", flush=True)
        print(f"  Succeeded:  {counts.succeeded}", flush=True)
        print(f"  Errored:    {counts.errored}", flush=True)
        return

    # Stream results
    output_path.parent.mkdir(parents=True, exist_ok=True)
    success = 0
    failed = 0
    total_input_tokens = 0
    total_output_tokens = 0

    with open(output_path, "a", encoding="utf-8") as out_f:
        for item in client.messages.batches.results(batch_id):
            if item.result.type == "succeeded":
                msg = item.result.message
                text = msg.content[0].text.strip()
                if text.startswith("```"):
                    text = text.split("\n", 1)[1]
                    if text.endswith("```"):
                        text = text[:-3]
                    text = text.strip()

                try:
                    qa = json.loads(text)
                    required = {"question", "answer", "evidence_span", "task_category"}
                    if required.issubset(qa.keys()):
                        qa["case_id"] = item.custom_id
                        qa["model_used"] = "batch"
                        qa["input_tokens"] = msg.usage.input_tokens
                        qa["output_tokens"] = msg.usage.output_tokens
                        out_f.write(json.dumps(qa, ensure_ascii=False) + "\n")
                        total_input_tokens += msg.usage.input_tokens
                        total_output_tokens += msg.usage.output_tokens
                        success += 1
                    else:
                        failed += 1
                except json.JSONDecodeError:
                    failed += 1
            else:
                failed += 1

    cost_est = (total_input_tokens * 1.5 + total_output_tokens * 7.5) / 1_000_000  # batch pricing
    print(f"\n{'='*60}", flush=True)
    print(f"BATCH RESULTS RETRIEVED", flush=True)
    print(f"{'='*60}", flush=True)
    print(f"Succeeded:      {success:,}", flush=True)
    print(f"Failed:         {failed}", flush=True)
    print(f"Estimated cost: ${cost_est:.2f} (batch pricing)", flush=True)
    print(f"Output:         {output_path}", flush=True)
    print(f"{'='*60}", flush=True)

    # Update job metadata
    job_file = Path(f"results/batch_jobs/{batch_id}.json")
    if job_file.exists():
        with open(job_file, "r", encoding="utf-8") as f:
            meta = json.load(f)
        meta["status"] = "completed"
        meta["completed_at"] = datetime.now(timezone.utc).isoformat()
        meta["succeeded"] = success
        meta["failed"] = failed
        meta["cost_estimate"] = cost_est
        with open(job_file, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)


def run_async_mode(api_key: str, args):
    """Original async streaming mode for fast iteration."""
    aclient = anthropic.AsyncAnthropic(api_key=api_key)
    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load input cases
    cases = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                cases.append(json.loads(line))

    if args.limit > 0:
        cases = cases[:args.limit]

    print(f"Loaded {len(cases)} injected cases", flush=True)
    print(f"Model: {args.model} | Concurrency: {args.concurrency}", flush=True)

    # Resume support
    done_ids = set()
    if args.resume and output_path.exists():
        with open(output_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rec = json.loads(line)
                    done_ids.add(rec.get("case_id"))
        print(f"Resuming — {len(done_ids)} already completed", flush=True)
        cases = [c for c in cases if c["case_id"] not in done_ids]

    if not cases:
        print("No new cases to process. Done.", flush=True)
        return

    print(f"Processing {len(cases)} remaining cases...", flush=True)

    # Process in batches
    mode = "a" if args.resume else "w"
    total_input_tokens = 0
    total_output_tokens = 0
    success = 0
    failed = 0
    t0 = time.time()

    with open(output_path, mode, encoding="utf-8") as out_f:
        for batch_start in range(0, len(cases), args.batch_size):
            batch = cases[batch_start:batch_start + args.batch_size]

            results = asyncio.run(process_batch_async(
                aclient, batch, args.model, args.concurrency
            ))

            for qa in results:
                if qa:
                    out_f.write(json.dumps(qa, ensure_ascii=False) + "\n")
                    total_input_tokens += qa.get("input_tokens", 0)
                    total_output_tokens += qa.get("output_tokens", 0)
                    success += 1
                else:
                    failed += 1

            out_f.flush()
            elapsed = time.time() - t0
            total_done = batch_start + len(batch)
            rate = total_done / elapsed if elapsed > 0 else 0
            eta_min = (len(cases) - total_done) / rate / 60 if rate > 0 else 0
            cost_est = (total_input_tokens * 3 + total_output_tokens * 15) / 1_000_000
            print(f"  Batch {total_done}/{len(cases)} | "
                  f"OK: {success} | Fail: {failed} | "
                  f"Rate: {rate:.1f}/s | ETA: {eta_min:.0f}min | "
                  f"Cost: ${cost_est:.2f}", flush=True)

    # Summary
    elapsed = time.time() - t0
    cost_est = (total_input_tokens * 3 + total_output_tokens * 15) / 1_000_000
    print(f"\n{'='*60}", flush=True)
    print(f"Q&A GENERATION SUMMARY", flush=True)
    print(f"{'='*60}", flush=True)
    print(f"Input cases:        {len(cases) + len(done_ids):,}", flush=True)
    print(f"Generated:          {success + len(done_ids):,}", flush=True)
    print(f"Failed:             {failed}", flush=True)
    print(f"Input tokens:       {total_input_tokens:,}", flush=True)
    print(f"Output tokens:      {total_output_tokens:,}", flush=True)
    print(f"Estimated cost:     ${cost_est:.2f}", flush=True)
    print(f"Wall time:          {elapsed/3600:.1f} hours", flush=True)
    print(f"Output:             {output_path}", flush=True)
    print(f"{'='*60}", flush=True)


if __name__ == "__main__":
    main()
