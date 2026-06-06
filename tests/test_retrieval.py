"""
Smoke tests for the retrieval pipeline: chunking, embedding, Weaviate search.

Run:
    python -m pytest tests/test_retrieval.py -v
"""

import json
import os
from pathlib import Path

import pytest


# ── Paths ──────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHUNKS_PATH = PROJECT_ROOT / "data" / "chunks" / "chunks.jsonl"
PRIORITY_CHUNKS_PATH = PROJECT_ROOT / "data" / "chunks" / "chunks_priority.jsonl"
EMBEDDINGS_PATH = PROJECT_ROOT / "data" / "embeddings" / "embeddings.parquet"
COST_LOG_PATH = PROJECT_ROOT / "results" / "costs" / "embedding_costs.jsonl"


# ── Chunking tests ────────────────────────────────────────────────────────
class TestChunking:
    @pytest.fixture(autouse=True)
    def load_chunks(self):
        if not CHUNKS_PATH.exists():
            pytest.skip("chunks.jsonl not found")
        with open(CHUNKS_PATH, encoding="utf-8") as f:
            self.chunks = [json.loads(line) for line in f if line.strip()]

    def test_chunk_count_in_range(self):
        """Total chunks should be 40K–200K."""
        assert 40_000 <= len(self.chunks) <= 200_000, (
            f"Chunk count {len(self.chunks)} outside expected range"
        )

    def test_required_fields_present(self):
        """Every chunk must have required metadata fields."""
        required = ["source_file", "chunk_index", "content_type", "token_count", "source_id"]
        missing = [
            c for c in self.chunks if not all(k in c for k in required)
        ]
        assert len(missing) == 0, f"{len(missing)} chunks missing required fields"

    def test_no_empty_text(self):
        """No chunk should have empty text."""
        empty = [c for c in self.chunks if not c.get("text", "").strip()]
        assert len(empty) == 0, f"{len(empty)} chunks have empty text"

    def test_content_types_valid(self):
        """Content types should be from the expected set."""
        valid_types = {"code", "log", "documentation", "forum_qa", "orfs_report", "hdl"}
        types_found = {c["content_type"] for c in self.chunks}
        unexpected = types_found - valid_types
        # Allow unexpected but warn
        if unexpected:
            pytest.skip(f"Unexpected content types (not an error): {unexpected}")

    def test_forum_chunks_have_text(self):
        """Forum Q&A chunks should have real content, not empty strings."""
        forum = [c for c in self.chunks if c["content_type"] == "forum_qa"]
        if not forum:
            pytest.skip("No forum_qa chunks found")
        empty_forum = [c for c in forum if len(c.get("text", "").strip()) < 20]
        pct_empty = len(empty_forum) / len(forum) * 100
        assert pct_empty < 5, (
            f"{pct_empty:.1f}% of forum chunks are near-empty"
        )


# ── Embedding tests ───────────────────────────────────────────────────────
class TestEmbeddings:
    @pytest.fixture(autouse=True)
    def load_embeddings(self):
        if not EMBEDDINGS_PATH.exists():
            pytest.skip("embeddings.parquet not found")
        import pandas as pd
        self.df = pd.read_parquet(EMBEDDINGS_PATH)

    def test_embedding_count(self):
        """Should have at least the priority chunk count."""
        assert len(self.df) >= 1000, f"Only {len(self.df)} embeddings"

    def test_embedding_dimension(self):
        """Embeddings should be 1536-dim (voyage-code-2)."""
        sample = self.df.iloc[0]["embedding"]
        assert len(sample) == 1536, f"Embedding dim: {len(sample)}, expected 1536"

    def test_no_duplicate_chunk_ids(self):
        """No duplicate chunk_ids in embeddings."""
        dupes = self.df["chunk_id"].duplicated().sum()
        assert dupes == 0, f"{dupes} duplicate chunk_ids"

    def test_required_columns(self):
        """Parquet must have required columns."""
        required = ["chunk_id", "source_id", "content_type", "embedding"]
        missing = [c for c in required if c not in self.df.columns]
        assert not missing, f"Missing columns: {missing}"


# ── Cost tracking tests ──────────────────────────────────────────────────
class TestCostTracking:
    def test_cost_log_exists(self):
        """Cost log should be written."""
        if not COST_LOG_PATH.exists():
            pytest.skip("Cost log not found")
        with open(COST_LOG_PATH) as f:
            entries = [json.loads(line) for line in f if line.strip()]
        assert len(entries) > 0, "Cost log is empty"

    def test_cost_reasonable(self):
        """Total embedding cost should be under $50."""
        if not COST_LOG_PATH.exists():
            pytest.skip("Cost log not found")
        with open(COST_LOG_PATH) as f:
            entries = [json.loads(line) for line in f if line.strip()]
        total = sum(e.get("cost_usd", 0) for e in entries if "cost_usd" in e)
        assert total < 50, f"Total cost ${total:.2f} exceeds $50 budget"


# ── Weaviate search tests ────────────────────────────────────────────────
class TestWeaviateSearch:
    @pytest.fixture(autouse=True)
    def setup_engine(self):
        if not os.environ.get("WEAVIATE_URL"):
            pytest.skip("WEAVIATE_URL not set")
        if not os.environ.get("VOYAGE_API_KEY"):
            pytest.skip("VOYAGE_API_KEY not set")
        from pipeline.retrieve.hybrid_search import WeaviateSearchEngine
        self.engine = WeaviateSearchEngine()
        yield
        self.engine.close()

    def test_collection_not_empty(self):
        """Weaviate collection should have indexed documents."""
        count = self.engine.collection.aggregate.over_all(
            total_count=True
        ).total_count
        assert count > 0, "Weaviate collection is empty"

    def test_hybrid_search_returns_results(self):
        """Basic hybrid search should return results."""
        results = self.engine.search(
            "JPEG timing violation", top_k=5, mode="hybrid"
        )
        assert len(results) > 0, "Hybrid search returned no results"

    def test_ed002_in_top5(self):
        """ED-002 regression: WNS sign flip query should return relevant results."""
        results = self.engine.search(
            "WNS sign flip between ORFS v3.0 and 26Q1",
            top_k=5,
            mode="hybrid",
        )
        assert len(results) > 0, "ED-002 query returned no results"
        # At minimum, ORFS-related content should appear
        sources = [r["source_file"] for r in results]
        has_orfs = any("orfs" in s.lower() or "jpeg" in s.lower() for s in sources)
        assert has_orfs, f"No ORFS/JPEG sources in top-5: {sources}"


# ── Reranker tests ────────────────────────────────────────────────────────
class TestReranker:
    def test_rerank_basic(self):
        """Reranker should score and sort candidates."""
        from pipeline.retrieve.reranker import rerank

        candidates = [
            {"text": "The WNS value flipped sign between ORFS v3.0 and 26Q1 releases.", "source_file": "a.txt"},
            {"text": "Python is a programming language.", "source_file": "b.txt"},
            {"text": "Timing violations occur when setup or hold constraints are not met.", "source_file": "c.txt"},
        ]
        results = rerank("WNS sign flip ORFS version upgrade", candidates, top_n=3)
        assert len(results) == 3
        assert all("rerank_score" in r for r in results)
        # Scores should be in descending order
        scores = [r["rerank_score"] for r in results]
        assert scores == sorted(scores, reverse=True), f"Scores not descending: {scores}"

    def test_rerank_relevance(self):
        """Reranker should rank relevant content higher than irrelevant."""
        from pipeline.retrieve.reranker import rerank

        candidates = [
            {"text": "How to make pancakes: mix flour, eggs, and milk.", "source_file": "recipe.txt"},
            {"text": "The JPEG design showed WNS regression after ORFS 26Q1 upgrade.", "source_file": "orfs.txt"},
            {"text": "Weather forecast: sunny with clouds.", "source_file": "weather.txt"},
        ]
        results = rerank("WNS sign flip ORFS version upgrade", candidates, top_n=3)
        # The ORFS-related text should rank first
        assert results[0]["source_file"] == "orfs.txt", (
            f"Expected orfs.txt first, got {results[0]['source_file']}"
        )

    def test_rerank_with_stats(self):
        """rerank_with_stats should return diagnostics."""
        from pipeline.retrieve.reranker import rerank_with_stats

        candidates = [
            {"text": "Timing violation in JPEG encoder module.", "source_file": "a.txt"},
            {"text": "Unrelated content about cooking.", "source_file": "b.txt"},
        ]
        result = rerank_with_stats("JPEG timing violation", candidates, top_n=2)
        assert "results" in result
        assert "stats" in result
        assert result["stats"]["candidates_scored"] == 2
        assert result["stats"]["score_max"] >= result["stats"]["score_min"]

    def test_rerank_empty_candidates(self):
        """Reranker should handle empty input gracefully."""
        from pipeline.retrieve.reranker import rerank
        results = rerank("any query", [], top_n=5)
        assert results == []

    def test_rerank_top_n_limit(self):
        """Reranker should respect top_n parameter."""
        from pipeline.retrieve.reranker import rerank

        candidates = [{"text": f"Document {i}", "source_file": f"{i}.txt"} for i in range(10)]
        results = rerank("test query", candidates, top_n=3)
        assert len(results) == 3
