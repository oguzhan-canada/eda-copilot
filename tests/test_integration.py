"""
Integration tests for the full GraphRAG pipeline.

Tests the fusion retriever + synthesizer end-to-end with 5 seed queries.

Run:
    python -m pytest tests/test_integration.py -v

Requires:
    NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD
    WEAVIATE_URL, WEAVIATE_API_KEY
    VOYAGE_API_KEY (for hybrid/dense search)
    ANTHROPIC_API_KEY (for synthesis tests only)
"""

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# Skip all tests if Neo4j not configured
pytestmark = pytest.mark.skipif(
    not os.environ.get("NEO4J_URI"),
    reason="NEO4J_URI not set",
)


@pytest.fixture(scope="module")
def retriever():
    from apps.api.services.fusion_retriever import FusionRetriever
    r = FusionRetriever()
    yield r
    r.close()


@pytest.fixture(scope="module")
def synthesizer():
    if not os.environ.get("ANTHROPIC_API_KEY"):
        pytest.skip("ANTHROPIC_API_KEY not set")
    from apps.api.services.synthesizer import Synthesizer
    return Synthesizer()


# ── Entity Extraction ────────────────────────────────────────────────────

class TestEntityExtraction:
    def test_jpeg_design_found(self, retriever):
        entities = retriever.extract_entities(
            "Why does JPEG timing fail after upgrading ORFS?"
        )
        entity_ids = [e.lower() for e in entities]
        assert any("jpeg" in e for e in entity_ids), f"No JPEG entity: {entities}"

    def test_ibex_design_found(self, retriever):
        entities = retriever.extract_entities(
            "OpenROAD crashes with SIGSEGV on ibex"
        )
        entity_ids = [e.lower() for e in entities]
        assert any("ibex" in e for e in entity_ids), f"No ibex entity: {entities}"

    def test_no_false_positives_generic_query(self, retriever):
        entities = retriever.extract_entities(
            "How do I write an SDC constraint for a 500MHz clock?"
        )
        # Should not match random KG nodes
        assert len(entities) < 3, f"Too many entities for generic query: {entities}"

    def test_ed_pattern_matching(self, retriever):
        entities = retriever.extract_entities("What causes ED-002?")
        entity_ids = [e.lower() for e in entities]
        assert any("ed_002" in e for e in entity_ids), f"No ED-002: {entities}"


# ── Query Routing ────────────────────────────────────────────────────────

class TestQueryRouting:
    SEED_QUERIES = [
        ("error_diagnosis", "Why does JPEG timing fail after upgrading from ORFS v3.0 to 26Q1?"),
        ("cross_tool_knowledge", "What changed between ORFS v3.0 and 26Q1 that affects timing?"),
        ("error_diagnosis", "OpenROAD crashes with SIGSEGV on ibex in global routing"),
        ("constraint_generation", "How do I write an SDC constraint for a 500MHz clock?"),
        ("error_diagnosis", "SDC time unit mismatch producing implausible WNS values"),
    ]

    @pytest.mark.parametrize("expected,query", SEED_QUERIES)
    def test_routing_accuracy(self, retriever, expected, query):
        result = retriever.retrieve(query, top_k=1, search_mode="sparse")
        assert result["task_category"] == expected, (
            f"Expected {expected}, got {result['task_category']}"
        )


# ── Graph Retrieval ──────────────────────────────────────────────────────

class TestGraphRetrieval:
    def test_jpeg_returns_version_data(self, retriever):
        """ED-002: JPEG query should return version divergence facts."""
        result = retriever.retrieve(
            "Why does JPEG timing fail after upgrading from ORFS v3.0 to 26Q1?",
            top_k=3, search_mode="sparse",
        )
        assert len(result["graph_facts"]) > 0, "No graph facts for JPEG query"
        # Should have design_jpeg with version data
        jpeg_facts = [
            f for f in result["graph_facts"]
            if "jpeg" in f.get("entity_id", "").lower()
        ]
        assert len(jpeg_facts) > 0, "No JPEG-specific graph facts"

    def test_ibex_returns_graph_data(self, retriever):
        """ED-005: ibex query should return graph subgraph."""
        result = retriever.retrieve(
            "OpenROAD crashes with SIGSEGV on ibex in global routing",
            top_k=3, search_mode="sparse",
        )
        assert len(result["graph_facts"]) > 0, "No graph facts for ibex query"
        ibex_facts = [
            f for f in result["graph_facts"]
            if "ibex" in f.get("entity_id", "").lower()
        ]
        assert len(ibex_facts) > 0, "No ibex-specific graph facts"

    def test_generic_query_returns_chunks(self, retriever):
        """Generic query with no entity matches should still return chunks."""
        result = retriever.retrieve(
            "How do I write an SDC constraint for a 500MHz clock?",
            top_k=3, search_mode="sparse",
        )
        assert len(result["chunks"]) > 0, "No chunks for SDC query"


# ── Context Formatting ───────────────────────────────────────────────────

class TestContextFormatting:
    def test_format_context_produces_text(self, retriever):
        result = retriever.retrieve(
            "JPEG timing failure ORFS upgrade",
            top_k=3, search_mode="sparse",
        )
        context = retriever.format_context(result)
        assert len(context) > 100, f"Context too short: {len(context)} chars"
        assert "Knowledge Graph Facts" in context or "Retrieved Documents" in context


# ── Synthesis (requires ANTHROPIC_API_KEY) ────────────────────────────────

class TestSynthesis:
    def test_ed002_synthesis(self, retriever, synthesizer):
        """Full E2E: JPEG timing query → Claude answer with citations."""
        result = retriever.retrieve(
            "Why does JPEG timing fail after upgrading from ORFS v3.0 to 26Q1?",
            top_k=5, search_mode="sparse",
        )
        answer = synthesizer.synthesize(
            "Why does JPEG timing fail after upgrading from ORFS v3.0 to 26Q1?",
            result,
        )
        assert "answer" in answer, "No 'answer' key in response"
        assert len(answer["answer"]) > 50, f"Answer too short: {answer['answer'][:50]}"
        assert "confidence" in answer, "No confidence in response"
        assert "usage" in answer, "No usage stats in response"
