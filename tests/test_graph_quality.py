"""
Graph quality tests — automated regression tests for Neo4j KG.

Run after every reload to catch schema regressions, data loss, and
performance degradation.

Usage:
    python -m pytest tests/test_graph_quality.py -v
"""
import os
import statistics
import time

import pytest
from neo4j import GraphDatabase

URI = os.environ.get('NEO4J_URI', '')
USER = os.environ.get('NEO4J_USER', '')
PASSWORD = os.environ.get('NEO4J_PASSWORD', '')

SEED_IDS = [
    'ed_001_sdc_override',
    'ed_002_version_divergence',
    'ed_003_unit_mismatch',
    'ed_004_def_naming',
    'ed_005_ibex_26q1_sigsegv',
]

CANONICAL_VERSIONS = ['version_ORFS_v3.0', 'version_ORFS_26Q1']


@pytest.fixture(scope='module')
def neo4j_session():
    driver = GraphDatabase.driver(URI, auth=(USER, PASSWORD))
    driver.verify_connectivity()
    session = driver.session()
    yield session
    session.close()
    driver.close()


class TestDesignNodes:
    def test_all_designs_have_name(self, neo4j_session):
        """All Design nodes must have a name property."""
        result = neo4j_session.run(
            'MATCH (d:Design) WHERE d.name IS NULL RETURN count(d) AS c'
        ).single()
        assert result['c'] == 0, f"{result['c']} Design nodes missing name property"

    def test_canonical_designs_present(self, neo4j_session):
        """All 6 canonical designs must exist."""
        for name in ['aes', 'ibex', 'jpeg', 'riscv32i', 'swerv_wrapper', 'gcd']:
            result = neo4j_session.run(
                'MATCH (d:Design) WHERE d.name = $name RETURN count(d) AS c',
                name=name
            ).single()
            assert result['c'] > 0, f"Canonical design '{name}' not found"


class TestMetricNodes:
    def test_orfs_metrics_have_values(self, neo4j_session):
        """At least 80% of Metric nodes with parseable IDs have value property."""
        total = neo4j_session.run(
            'MATCH (m:Metric) RETURN count(m) AS c'
        ).single()['c']
        with_val = neo4j_session.run(
            'MATCH (m:Metric) WHERE m.value IS NOT NULL RETURN count(m) AS c'
        ).single()['c']
        pct = with_val / total if total > 0 else 0
        assert pct >= 0.70, f"Only {with_val}/{total} ({pct:.0%}) Metric nodes have value"

    def test_wns_values_are_numeric(self, neo4j_session):
        """All Metric nodes with value must have numeric value."""
        result = neo4j_session.run(
            'MATCH (m:Metric) WHERE m.value IS NOT NULL '
            'AND NOT toFloat(toString(m.value)) = m.value '
            'RETURN count(m) AS c'
        ).single()
        assert result['c'] == 0, f"{result['c']} Metric nodes have non-numeric value"


class TestVersionNodes:
    def test_canonical_versions_exist(self, neo4j_session):
        """Both canonical ORFS versions must exist."""
        for vid in CANONICAL_VERSIONS:
            result = neo4j_session.run(
                'MATCH (v:Version {id: $id}) RETURN count(v) AS c', id=vid
            ).single()
            assert result['c'] == 1, f"Canonical version '{vid}' count={result['c']}, expected 1"

    def test_no_duplicate_canonical_versions(self, neo4j_session):
        """No duplicate nodes for canonical ORFS version IDs."""
        # Check that the old aliases don't exist
        for alias in ['orfs_v3_0', 'orfs_26q1']:
            result = neo4j_session.run(
                'MATCH (v:Version {id: $id}) RETURN count(v) AS c', id=alias
            ).single()
            assert result['c'] == 0, f"Alias version '{alias}' still exists (should be merged)"

    def test_diverges_from_edge_exists(self, neo4j_session):
        """DIVERGES_FROM edge between canonical versions."""
        result = neo4j_session.run(
            'MATCH (a:Version)-[:DIVERGES_FROM]->(b:Version) '
            'WHERE a.id IN $ids AND b.id IN $ids '
            'RETURN count(*) AS c',
            ids=CANONICAL_VERSIONS
        ).single()
        assert result['c'] > 0, "No DIVERGES_FROM edge between canonical versions"


class TestSeedTriples:
    def test_all_seeds_present(self, neo4j_session):
        """All ED-001 through ED-005 seed triples must be present."""
        result = neo4j_session.run(
            'MATCH (v:Violation) WHERE v.id IN $ids RETURN collect(v.id) AS found',
            ids=SEED_IDS
        ).single()
        found = set(result['found'])
        missing = set(SEED_IDS) - found
        assert len(missing) == 0, f"Missing seed triples: {missing}"

    def test_ed002_has_fix(self, neo4j_session):
        """ED-002 must have at least one Fix relationship."""
        result = neo4j_session.run(
            'MATCH (v:Violation {id: "ed_002_version_divergence"})<-[:FIXES]-(f:Fix) '
            'RETURN count(f) AS c'
        ).single()
        assert result['c'] > 0, "ED-002 has no FIXES relationship"


class TestGraphConnectivity:
    def test_no_orphan_nodes(self, neo4j_session):
        """No orphan nodes (nodes without any relationships)."""
        result = neo4j_session.run(
            'MATCH (n) WHERE NOT (n)--() RETURN count(n) AS c'
        ).single()
        assert result['c'] == 0, f"{result['c']} orphan nodes found"

    def test_violation_fix_2hop_paths(self, neo4j_session):
        """At least 1,000 Violation nodes reachable to Fix within 2 hops."""
        result = neo4j_session.run(
            'MATCH (v:Violation)-[*1..2]-(f:Fix) '
            'RETURN count(DISTINCT v) AS violations'
        ).single()
        assert result['violations'] >= 1000, \
            f"Only {result['violations']} violations with 2-hop Fix path (need >= 1000)"

    def test_minimum_node_count(self, neo4j_session):
        """Graph must have at least 15,000 nodes."""
        result = neo4j_session.run('MATCH (n) RETURN count(n) AS c').single()
        assert result['c'] >= 15000, f"Only {result['c']} nodes (need >= 15000)"

    def test_minimum_relationship_count(self, neo4j_session):
        """Graph must have at least 15,000 relationships."""
        result = neo4j_session.run(
            'MATCH ()-[r]->() RETURN count(r) AS c'
        ).single()
        assert result['c'] >= 15000, f"Only {result['c']} relationships (need >= 15000)"


class TestLatency:
    def test_2hop_latency_p50_under_100ms(self, neo4j_session):
        """p50 latency for 2-hop Violation→Fix query must be < 100ms."""
        # Warm up
        neo4j_session.run(
            'MATCH (v:Violation)-[:CAUSES|FIXES*1..2]-(n) '
            'RETURN v.id, collect(n.id) LIMIT 10'
        ).consume()

        times = []
        for _ in range(20):
            start = time.perf_counter()
            neo4j_session.run(
                'MATCH (v:Violation)-[:CAUSES|FIXES*1..2]-(n) '
                'RETURN v.id, collect(n.id) LIMIT 10'
            ).consume()
            times.append((time.perf_counter() - start) * 1000)

        p50 = statistics.median(times)
        assert p50 < 100, f"p50 latency {p50:.1f}ms exceeds 100ms gate"
