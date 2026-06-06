"""
Component 1.2 — Judge and filter Q&A pairs (async version).

Second Claude API call scoring each Q&A on a 0-1 rubric across four
dimensions: factual correctness, specificity, actionability, format
compliance. Rejects anything below the minimum threshold.

Uses async concurrency (default 10) for ~10x speedup over sequential.

Usage:
    python -m pipeline.synth_qa.judge_qa \
        --input data/synthetic/qa_raw.jsonl \
        --output data/synthetic/qa_train.jsonl \
        --scores data/synthetic/judge_scores.parquet \
        --min-score 0.90
"""
import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path

import pandas as pd

try:
    import anthropic
except ImportError:
    print("ERROR: anthropic SDK not installed. Run: pip install anthropic")
    sys.exit(1)


JUDGE_SYSTEM_PROMPT = """You are a quality judge for EDA (Electronic Design Automation) training data.
Score each Q&A pair on four dimensions, each 0.0 to 1.0:

1. **factual_correctness**: Is the answer technically correct for the EDA domain? 
   Would an experienced chip designer agree with the diagnosis and fix?
   - 1.0: Completely correct, no errors
   - 0.5: Mostly correct, minor inaccuracies
   - 0.0: Incorrect diagnosis or wrong fix

2. **specificity**: Does the answer reference specific tools, commands, file formats, 
   and values rather than generic advice?
   - 1.0: Mentions exact tool names, commands, values
   - 0.5: Some specifics, some generic statements
   - 0.0: Entirely generic advice

3. **actionability**: Can a designer follow the fix instructions step-by-step?
   - 1.0: Clear steps, specific commands/edits
   - 0.5: General direction but missing details
   - 0.0: Vague or unhelpful

4. **format_compliance**: Does the Q&A follow the expected structure?
   - 1.0: Well-structured question and answer, proper evidence span
   - 0.5: Minor formatting issues
   - 0.0: Missing fields or garbled output

Respond with valid JSON only:
{"factual_correctness": 0.0-1.0, "specificity": 0.0-1.0, "actionability": 0.0-1.0, "format_compliance": 0.0-1.0, "overall": 0.0-1.0, "rationale": "brief explanation"}

The "overall" score should be the minimum of the four dimension scores (not the average)."""


def build_judge_prompt(qa: dict) -> str:
    return f"""Score this EDA Q&A pair:

**Question**: {qa.get('question', '')}

**Answer**: {qa.get('answer', '')}

**Evidence Span**: {qa.get('evidence_span', '')}

**Task Category**: {qa.get('task_category', '')}
**Violation Family**: {qa.get('violation_family', '')}

Respond with valid JSON only -- no markdown fences."""


async def judge_qa_async(client: anthropic.AsyncAnthropic, qa: dict, model: str,
                         semaphore: asyncio.Semaphore,
                         max_retries: int = 3) -> dict | None:
    async with semaphore:
        for attempt in range(max_retries):
            try:
                response = await client.messages.create(
                    model=model,
                    max_tokens=512,
                    system=JUDGE_SYSTEM_PROMPT,
                    messages=[{"role": "user", "content": build_judge_prompt(qa)}],
                )

                text = response.content[0].text.strip()
                if text.startswith("```"):
                    text = text.split("\n", 1)[1]
                    if text.endswith("```"):
                        text = text[:-3]
                    text = text.strip()

                scores = json.loads(text)

                required = {"factual_correctness", "specificity", "actionability",
                            "format_compliance", "overall"}
                if not required.issubset(scores.keys()):
                    continue

                for key in required - {"rationale"}:
                    if key in scores:
                        scores[key] = max(0.0, min(1.0, float(scores[key])))

                scores["case_id"] = qa.get("case_id", "")
                scores["input_tokens"] = response.usage.input_tokens
                scores["output_tokens"] = response.usage.output_tokens
                return scores

            except json.JSONDecodeError:
                await asyncio.sleep(1)
            except anthropic.RateLimitError:
                await asyncio.sleep(2 ** (attempt + 1))
            except anthropic.APIError as e:
                print(f"  API error: {e}")
                await asyncio.sleep(2)

    return None


async def run_judge(args):
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("ERROR: ANTHROPIC_API_KEY environment variable not set.")
        sys.exit(1)

    client = anthropic.AsyncAnthropic(api_key=api_key)
    semaphore = asyncio.Semaphore(args.concurrency)

    input_path = Path(args.input)
    output_path = Path(args.output)
    scores_path = Path(args.scores)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    scores_path.parent.mkdir(parents=True, exist_ok=True)

    # Load input
    qa_records = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                qa_records.append(json.loads(line))

    if args.limit > 0:
        qa_records = qa_records[:args.limit]

    print(f"Loaded {len(qa_records)} Q&A records to judge")
    print(f"Model: {args.model} | Concurrency: {args.concurrency}")
    print(f"Min score threshold: {args.min_score}")

    # Resume support — check both output and scores for already-judged IDs
    done_ids = set()
    if args.resume:
        if output_path.exists():
            with open(output_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        done_ids.add(json.loads(line).get("case_id"))
        # Also check rejected records from scores parquet
        if scores_path.exists():
            try:
                existing_df = pd.read_parquet(scores_path)
                done_ids.update(existing_df["case_id"].tolist())
            except Exception:
                pass
        if done_ids:
            print(f"Resuming - {len(done_ids)} already judged")
            qa_records = [r for r in qa_records if r.get("case_id") not in done_ids]

    if not qa_records:
        print("No records to judge.")
        return

    print(f"Processing {len(qa_records)} remaining records...")

    # Process in batches for progress tracking
    all_scores = []
    accepted = 0
    rejected = 0
    failed = 0
    total_input_tokens = 0
    total_output_tokens = 0
    start_time = time.time()

    mode = "a" if args.resume else "w"
    out_f = open(output_path, mode, encoding="utf-8")

    batch_size = args.batch_size
    total = len(qa_records)

    for batch_start in range(0, total, batch_size):
        batch = qa_records[batch_start:batch_start + batch_size]
        tasks = [judge_qa_async(client, qa, args.model, semaphore) for qa in batch]
        results = await asyncio.gather(*tasks)

        for qa, scores in zip(batch, results):
            if scores is None:
                failed += 1
                all_scores.append({
                    "case_id": qa.get("case_id", ""),
                    "overall": 0.0,
                    "factual_correctness": 0.0,
                    "specificity": 0.0,
                    "actionability": 0.0,
                    "format_compliance": 0.0,
                    "status": "api_failure",
                })
                continue

            total_input_tokens += scores.get("input_tokens", 0)
            total_output_tokens += scores.get("output_tokens", 0)

            score_record = {
                "case_id": scores["case_id"],
                "overall": scores["overall"],
                "factual_correctness": scores["factual_correctness"],
                "specificity": scores["specificity"],
                "actionability": scores["actionability"],
                "format_compliance": scores["format_compliance"],
                "rationale": scores.get("rationale", ""),
                "status": "accepted" if scores["overall"] >= args.min_score else "rejected",
            }
            all_scores.append(score_record)

            if scores["overall"] >= args.min_score:
                qa["judge_score"] = scores["overall"]
                qa["judge_scores"] = {
                    "factual_correctness": scores["factual_correctness"],
                    "specificity": scores["specificity"],
                    "actionability": scores["actionability"],
                    "format_compliance": scores["format_compliance"],
                }
                out_f.write(json.dumps(qa, ensure_ascii=False) + "\n")
                accepted += 1
            else:
                rejected += 1

        out_f.flush()
        done_so_far = batch_start + len(batch)
        elapsed = time.time() - start_time
        rate = done_so_far / elapsed if elapsed > 0 else 0
        eta = int((total - done_so_far) / rate / 60) if rate > 0 else 0
        cost = (total_input_tokens * 3 + total_output_tokens * 15) / 1_000_000
        print(f"  Batch {done_so_far}/{total} | "
              f"Accept: {accepted} | Reject: {rejected} | Fail: {failed} | "
              f"Rate: {rate:.1f}/s | ETA: {eta}min | Cost: ${cost:.2f}")

    out_f.close()

    # Save scores parquet (merge with existing if resuming)
    if all_scores:
        new_df = pd.DataFrame(all_scores)
        if args.resume and scores_path.exists():
            try:
                existing_df = pd.read_parquet(scores_path)
                df = pd.concat([existing_df, new_df], ignore_index=True)
            except Exception:
                df = new_df
        else:
            df = new_df
        df.to_parquet(scores_path, index=False)

    # Summary
    cost = (total_input_tokens * 3 + total_output_tokens * 15) / 1_000_000
    total_judged = accepted + rejected + failed
    accept_rate = accepted / (accepted + rejected) * 100 if (accepted + rejected) > 0 else 0
    wall_time = (time.time() - start_time) / 3600

    print(f"\n{'='*60}")
    print(f"JUDGE SUMMARY")
    print(f"{'='*60}")
    print(f"Total judged:       {total_judged + len(done_ids):,}")
    print(f"Accepted (>={args.min_score}): {accepted + len(done_ids):,}")
    print(f"Rejected:           {rejected}")
    print(f"API failures:       {failed}")
    print(f"Accept rate:        {accept_rate:.1f}%")
    print(f"Input tokens:       {total_input_tokens:,}")
    print(f"Output tokens:      {total_output_tokens:,}")
    print(f"Estimated cost:     ${cost:.2f}")
    print(f"Wall time:          {wall_time:.1f} hours")
    print(f"Output:             {output_path}")
    print(f"Scores:             {scores_path}")

    if all_scores:
        scores_vals = [s["overall"] for s in all_scores if s["overall"] > 0]
        if scores_vals:
            scores_vals.sort()
            mean_score = sum(scores_vals) / len(scores_vals)
            p10 = scores_vals[len(scores_vals) // 10] if len(scores_vals) >= 10 else scores_vals[0]
            print(f"Mean score:         {mean_score:.3f}")
            print(f"P10 score:          {p10:.3f}")

    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(description="Judge and filter Q&A pairs (async)")
    parser.add_argument("--input", required=True, help="Raw Q&A JSONL")
    parser.add_argument("--output", default="data/synthetic/qa_train.jsonl")
    parser.add_argument("--scores", default="data/synthetic/judge_scores.parquet")
    parser.add_argument("--min-score", type=float, default=0.90)
    parser.add_argument("--model", default="claude-sonnet-4-20250514")
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--concurrency", type=int, default=10)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    asyncio.run(run_judge(args))


if __name__ == "__main__":
    main()
