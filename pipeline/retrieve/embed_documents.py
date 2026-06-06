"""
Embedding pipeline using Voyage AI voyage-code-2.

Reads chunks from data/chunks/chunks.jsonl, embeds in batches,
writes to data/embeddings/embeddings.parquet.

Cost optimization (from AI Inference Cost Autopsy):
  - Batch requests (128 per batch, API limit)
  - Idempotent: skip already-embedded chunk_ids on re-run
  - Cost tracking per batch to results/costs/embedding_costs.jsonl
  - Pre-flight cost estimate before any API calls

Usage:
    python -m pipeline.retrieve.embed_documents \
        --input data/chunks/chunks.jsonl \
        --output data/embeddings/embeddings.parquet \
        --estimate-only   # dry run — show cost without calling API
"""

import argparse
import json
import os
import sys
import time
from hashlib import md5
from pathlib import Path

import voyageai


VOYAGE_MODEL = 'voyage-code-2'
BATCH_SIZE = 128  # Payment method added, 300 RPM unlocked  # Reduced for 10K TPM limit; increase to 128 with payment method
COST_PER_MILLION_TOKENS = 0.12  # USD


def make_chunk_id(chunk: dict) -> str:
    """Deterministic chunk ID for idempotent processing."""
    key = f"{chunk['source_id']}_{chunk['chunk_index']}_{chunk.get('content_type', '')}"
    return md5(key.encode()).hexdigest()[:16]


def estimate_cost(chunks: list) -> dict:
    """Pre-flight cost estimation without API calls."""
    total_tokens = sum(c['token_count'] for c in chunks)
    cost = (total_tokens / 1_000_000) * COST_PER_MILLION_TOKENS
    return {
        'total_chunks': len(chunks),
        'total_tokens': total_tokens,
        'estimated_cost_usd': round(cost, 2),
        'batches': (len(chunks) + BATCH_SIZE - 1) // BATCH_SIZE,
        'model': VOYAGE_MODEL,
    }


def embed_batch(client, texts: list, input_type: str = 'document') -> list:
    """Embed a batch of texts, return list of embeddings."""
    result = client.embed(texts, model=VOYAGE_MODEL, input_type=input_type)
    return result.embeddings


def _save_embeddings(results, existing_ids, out_path):
    """Incremental save of embeddings to parquet."""
    import pandas as pd
    if not results:
        return
    df = pd.DataFrame(results)
    if existing_ids and out_path.exists():
        existing_df = pd.read_parquet(out_path)
        df = pd.concat([existing_df, df], ignore_index=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)


def _save_cost_log(cost_log, cost_path):
    """Incremental save of cost log."""
    cost_path = Path(cost_path)
    cost_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cost_path, 'w') as f:
        for entry in cost_log:
            f.write(json.dumps(entry) + '\n')


def run_embedding(
    input_path: str,
    output_path: str,
    cost_log_path: str,
    api_key: str,
    max_chunks: int = None,
    estimate_only: bool = False,
):
    """Main embedding pipeline."""
    # Load chunks
    chunks = []
    with open(input_path, encoding='utf-8') as f:
        for line in f:
            chunks.append(json.loads(line))

    if max_chunks:
        chunks = chunks[:max_chunks]

    # Add chunk IDs
    for c in chunks:
        c['chunk_id'] = make_chunk_id(c)

    # Pre-flight estimate
    est = estimate_cost(chunks)
    print(f"Pre-flight estimate:")
    print(f"  Chunks: {est['total_chunks']:,}")
    print(f"  Tokens: {est['total_tokens']:,}")
    print(f"  Batches: {est['batches']}")
    print(f"  Estimated cost: ${est['estimated_cost_usd']:.2f}")
    print(f"  Model: {est['model']}")

    if estimate_only:
        print("\n--estimate-only flag set. Exiting without API calls.")
        return est

    # Check for existing embeddings (idempotent)
    out_path = Path(output_path)
    existing_ids = set()
    if out_path.exists():
        import pyarrow.parquet as pq
        existing_df = pq.read_table(out_path).to_pandas()
        existing_ids = set(existing_df['chunk_id'].tolist())
        print(f"\nExisting embeddings: {len(existing_ids)} (will skip)")

    new_chunks = [c for c in chunks if c['chunk_id'] not in existing_ids]
    if not new_chunks:
        print("All chunks already embedded. Nothing to do.")
        return est

    print(f"\nNew chunks to embed: {len(new_chunks)}")
    new_est = estimate_cost(new_chunks)
    print(f"Cost for new chunks: ${new_est['estimated_cost_usd']:.2f}")

    # Initialize Voyage client
    client = voyageai.Client(api_key=api_key)

    # Embed in batches
    all_results = []
    cost_log = []
    total_api_tokens = 0

    for batch_idx in range(0, len(new_chunks), BATCH_SIZE):
        batch = new_chunks[batch_idx:batch_idx + BATCH_SIZE]
        texts = [c['text'][:8000] for c in batch]  # Cap text length for API

        batch_num = batch_idx // BATCH_SIZE + 1
        total_batches = (len(new_chunks) + BATCH_SIZE - 1) // BATCH_SIZE

        try:
            start_time = time.time()
            embeddings = embed_batch(client, texts)
            elapsed = time.time() - start_time

            batch_tokens = sum(c['token_count'] for c in batch)
            total_api_tokens += batch_tokens
            batch_cost = (batch_tokens / 1_000_000) * COST_PER_MILLION_TOKENS

            for chunk, embedding in zip(batch, embeddings):
                all_results.append({
                    'chunk_id': chunk['chunk_id'],
                    'source_id': chunk['source_id'],
                    'source_file': chunk['source_file'],
                    'content_type': chunk['content_type'],
                    'chunk_index': chunk['chunk_index'],
                    'token_count': chunk['token_count'],
                    'text': chunk['text'][:2000],  # Truncate for storage
                    'embedding': embedding,
                })

            cost_log.append({
                'batch': batch_num,
                'chunks': len(batch),
                'tokens': batch_tokens,
                'cost_usd': round(batch_cost, 4),
                'elapsed_s': round(elapsed, 2),
                'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
            })

            if batch_num % 10 == 0 or batch_num == total_batches:
                running_cost = (total_api_tokens / 1_000_000) * COST_PER_MILLION_TOKENS
                print(f"  Batch {batch_num}/{total_batches}: "
                      f"{len(all_results)} embedded, ${running_cost:.2f} spent, "
                      f"{elapsed:.1f}s", flush=True)

                # Incremental save every 50 batches
                if batch_num % 50 == 0:
                    _save_embeddings(all_results, existing_ids, out_path)
                    _save_cost_log(cost_log, cost_log_path)

        except Exception as e:
            err_msg = str(e)
            print(f"  ERROR batch {batch_num}: {err_msg[:120]}", flush=True)
            cost_log.append({
                'batch': batch_num,
                'error': err_msg,
                'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
            })
            # Rate limit errors need full cooldown, not short backoff
            if 'rate' in err_msg.lower() or 'RPM' in err_msg or 'TPM' in err_msg:
                print(f"  Rate limited — cooling down 60s", flush=True)
                time.sleep(60)
            else:
                time.sleep(21)  # Default backoff = same as rate limit interval
            continue

        # Rate limiting: respect TPM and RPM limits
        # Reduced limits (no payment): 3 RPM, 10K TPM
        # Standard limits: 300 RPM
        min_interval = 21  # 3 RPM = 1 request per 20s, add 1s buffer
        if elapsed < min_interval:
            time.sleep(min_interval - elapsed)

    # Write embeddings to parquet
    if all_results:
        _save_embeddings(all_results, existing_ids, out_path)
        print(f"\nEmbeddings saved: {len(all_results)} new to {out_path}", flush=True)

    # Write cost log
    _save_cost_log(cost_log, Path(cost_log_path))

    total_cost = (total_api_tokens / 1_000_000) * COST_PER_MILLION_TOKENS
    print(f"\nTotal API tokens: {total_api_tokens:,}")
    print(f"Total cost: ${total_cost:.2f}")
    print(f"Cost log: {cost_log_path}")

    return {
        'total_embedded': len(all_results),
        'total_tokens': total_api_tokens,
        'total_cost_usd': round(total_cost, 2),
        'batches_completed': len([c for c in cost_log if 'error' not in c]),
        'batches_failed': len([c for c in cost_log if 'error' in c]),
    }


def main():
    parser = argparse.ArgumentParser(description="Embed document chunks with Voyage AI")
    parser.add_argument('--input', default='data/chunks/chunks.jsonl')
    parser.add_argument('--output', default='data/embeddings/embeddings.parquet')
    parser.add_argument('--cost-log', default='results/costs/embedding_costs.jsonl')
    parser.add_argument('--api-key', default=os.environ.get('VOYAGE_API_KEY'))
    parser.add_argument('--max-chunks', type=int, default=None,
                        help='Limit number of chunks (for testing)')
    parser.add_argument('--estimate-only', action='store_true',
                        help='Show cost estimate without calling API')
    args = parser.parse_args()

    if not args.api_key and not args.estimate_only:
        print("ERROR: --api-key or VOYAGE_API_KEY required")
        sys.exit(1)

    result = run_embedding(
        args.input, args.output, args.cost_log,
        args.api_key, args.max_chunks, args.estimate_only,
    )
    print(f"\nResult: {json.dumps(result, indent=2)}")


if __name__ == '__main__':
    main()
