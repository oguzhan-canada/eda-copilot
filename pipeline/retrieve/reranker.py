"""
Cross-encoder reranking of hybrid search results.

Takes top-K candidates from hybrid_search and re-scores with a
cross-encoder model for higher-precision final ranking.

Model: cross-encoder/ms-marco-MiniLM-L-6-v2
  - Lightweight (~80MB), runs on CPU in <100ms for 20 candidates
  - Trained on MS MARCO passage ranking — good fit for technical Q&A

Usage:
    from pipeline.retrieve.reranker import rerank

    candidates = engine.search("JPEG timing violation", top_k=20, mode="hybrid")
    top5 = rerank("JPEG timing violation", candidates, top_n=5)
"""

from sentence_transformers import CrossEncoder

_model = None
MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def get_model() -> CrossEncoder:
    """Lazy-load the cross-encoder model (cached after first call)."""
    global _model
    if _model is None:
        _model = CrossEncoder(MODEL_NAME)
    return _model


def rerank(
    query: str,
    candidates: list[dict],
    top_n: int = 5,
    text_key: str = "text",
) -> list[dict]:
    """
    Rerank candidate chunks using cross-encoder scoring.

    Args:
        query: The user's natural language query
        candidates: List of dicts from hybrid_search, each must have a 'text' field
        top_n: Number of top results to return after reranking
        text_key: Key in candidate dicts containing the text to score

    Returns:
        Top-N candidates sorted by cross-encoder score (descending),
        each augmented with a 'rerank_score' field.
    """
    if not candidates:
        return []

    model = get_model()

    # Build query-document pairs for cross-encoder
    pairs = [(query, c.get(text_key, "")) for c in candidates]
    scores = model.predict(pairs)

    # Pair scores with candidates, sort descending
    scored = sorted(
        zip(scores, candidates),
        key=lambda x: x[0],
        reverse=True,
    )

    results = []
    for score, candidate in scored[:top_n]:
        result = dict(candidate)
        result["rerank_score"] = float(score)
        results.append(result)

    return results


def rerank_with_stats(
    query: str,
    candidates: list[dict],
    top_n: int = 5,
    text_key: str = "text",
) -> dict:
    """
    Rerank with diagnostic stats for evaluation.

    Returns dict with:
        - results: reranked top-N candidates
        - stats: score distribution, rank changes
    """
    if not candidates:
        return {"results": [], "stats": {}}

    model = get_model()
    pairs = [(query, c.get(text_key, "")) for c in candidates]
    scores = model.predict(pairs)

    # Track original positions
    scored = []
    for i, (score, candidate) in enumerate(zip(scores, candidates)):
        entry = dict(candidate)
        entry["rerank_score"] = float(score)
        entry["original_rank"] = i + 1
        scored.append(entry)

    scored.sort(key=lambda x: x["rerank_score"], reverse=True)

    # Assign new ranks
    for i, entry in enumerate(scored):
        entry["reranked_rank"] = i + 1

    top_results = scored[:top_n]

    # Compute stats
    rank_changes = [
        abs(r["original_rank"] - r["reranked_rank"]) for r in top_results
    ]
    all_scores = [float(s) for s in scores]

    stats = {
        "candidates_scored": len(candidates),
        "top_n": top_n,
        "score_min": min(all_scores),
        "score_max": max(all_scores),
        "score_mean": sum(all_scores) / len(all_scores),
        "avg_rank_change": sum(rank_changes) / len(rank_changes) if rank_changes else 0,
        "top1_original_rank": top_results[0]["original_rank"] if top_results else None,
    }

    return {"results": top_results, "stats": stats}
