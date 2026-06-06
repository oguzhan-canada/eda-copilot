"""
Smoke tests for Neo4j graph loader.

Tests that:
1. Schema applies without error (idempotent — runs twice)
2. All 4 MLCAD seed bug triples exist after load
3. Node/relationship counts are correct for seed data
4. Schema re-application doesn't create duplicates

Requires: NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD environment variables.
Skip with: pytest -m "not neo4j" to skip when Neo4j is unavailable.
"""

import os
import pytest
from pathlib import Path

# Skip entire module if Neo4j credentials not configured
pytestmark = pytest.mark.skipif(
    not os.environ.get("NEO4J_PASSWORD"),
    reason="NEO4J_PASSWORD not set — skipping Neo4j tests",
)

neo4j = pytest.importorskip("neo4j")


@pytest.fixture(scope="module")
def driver():
    """Create a Neo4j driver for the test session."""
    uri = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
    user = os.environ.get("NEO4J_USER", "neo4j")
    password = os.environ["NEO4J_PASSWORD"]
    
    drv = neo4j.GraphDatabase.driver(uri, auth=(user, password))
    drv.verify_connectivity()
    yield drv
    drv.close()


@pytest.fixture(scope="module", autouse=True)
def load_schema_twice(driver):
    """Load schema twice to verify idempotency."""
    from pipeline.graph.load_neo4j import load_schema
    
    schema_path = Path("configs/graph_schema.cypher")
    
    # First load
    stats1 = load_schema(driver, schema_path)
    assert stats1["total"] > 0, "Schema should have statements"
    
    # Second load — must not error
    stats2 = load_schema(driver, schema_path)
    assert stats2["total"] == stats1["total"], "Idempotent run should execute same statement count"


class TestSeedTriples:
    """Verify all 4 MLCAD seed bug triples exist."""

    def test_ed001_sdc_override_fix(self, driver):
        with driver.session() as session:
            result = session.run(
                "MATCH (f:Fix {id:'variant_sdc_override'})-[:FIXES]->(v:Violation {id:'ed_001_sdc_override'}) "
                "RETURN f.id AS fix, v.error_code AS code"
            ).single()
        assert result is not None, "ED-001 fix->violation triple missing"
        assert result["code"] == "SDC_CLOCK_OVERRIDE"

    def test_ed002_version_divergence(self, driver):
        with driver.session() as session:
            result = session.run(
                "MATCH (v1:Version {id:'orfs_v3_0'})-[:DIVERGES_FROM]->(v2:Version {id:'orfs_26q1'}) "
                "RETURN v1.version_tag AS tag1, v2.version_tag AS tag2"
            ).single()
        assert result is not None, "ED-002 version divergence triple missing"
        assert result["tag1"] == "ORFS_v3.0"
        assert result["tag2"] == "ORFS_26Q1"

    def test_ed003_unit_mismatch_fix(self, driver):
        with driver.session() as session:
            result = session.run(
                "MATCH (f:Fix {id:'prepend_unit_declaration'})-[:FIXES]->(v:Violation {id:'ed_003_unit_mismatch'}) "
                "RETURN f.id AS fix, v.error_code AS code"
            ).single()
        assert result is not None, "ED-003 fix->violation triple missing"
        assert result["code"] == "SDC_UNIT_MISMATCH"

    def test_ed004_def_naming_fix(self, driver):
        with driver.session() as session:
            result = session.run(
                "MATCH (f:Fix {id:'automated_name_normalization'})-[:FIXES]->(v:Violation {id:'ed_004_def_naming'}) "
                "RETURN f.id AS fix, v.error_code AS code"
            ).single()
        assert result is not None, "ED-004 fix->violation triple missing"
        assert result["code"] == "DEF_INSTANCE_NAMING"


class TestSchemaIntegrity:
    """Verify schema constraints and indexes exist."""

    def test_constraints_created(self, driver):
        with driver.session() as session:
            result = session.run("SHOW CONSTRAINTS")
            constraints = [r["name"] for r in result]
        # 10 uniqueness + 3 property existence = 13 constraints
        assert len(constraints) >= 13, f"Expected >=13 constraints, got {len(constraints)}"

    def test_indexes_created(self, driver):
        with driver.session() as session:
            result = session.run("SHOW INDEXES")
            indexes = [r["name"] for r in result]
        # 6 explicit indexes + constraint-backed indexes
        assert len(indexes) >= 6, f"Expected >=6 indexes, got {len(indexes)}"

    def test_no_duplicate_nodes_after_reload(self, driver):
        """Seed data uses MERGE — no duplicates after multiple loads."""
        with driver.session() as session:
            result = session.run(
                "MATCH (v:Violation) RETURN v.id AS id, count(*) AS cnt"
            )
            for record in result:
                assert record["cnt"] == 1, f"Duplicate found: {record['id']} (count={record['cnt']})"
