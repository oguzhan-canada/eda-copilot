"""
GraphRAG retrieval engine — 2-hop subgraph retrieval from Neo4j.

Provides task-specific retrieval methods aligned with query_router categories:
  - retrieve_error_diagnosis:    Violation → Fix paths with context
  - retrieve_rtl_context:        Module → Port → Design subgraph
  - retrieve_version_divergence: Version comparison across ORFS releases
  - retrieve_drc_rules:          Rule → PDK lookups
  - retrieve_subgraph:           Generic N-hop entity retrieval

Each method returns structured dicts ready for LLM prompt injection.

Usage:
    engine = GraphRAGEngine(uri, user, password)
    fixes = engine.retrieve_error_diagnosis("ed_002_version_divergence")
    engine.close()
"""

import os
from typing import Optional

from neo4j import GraphDatabase


class GraphRAGEngine:
    """2-hop knowledge graph retrieval for EDA diagnostic queries."""

    def __init__(
        self,
        uri: Optional[str] = None,
        user: Optional[str] = None,
        password: Optional[str] = None,
    ):
        self.uri = uri or os.environ.get("NEO4J_URI")
        self.user = user or os.environ.get("NEO4J_USER", "neo4j")
        self.password = password or os.environ.get("NEO4J_PASSWORD")
        self.driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))
        self.driver.verify_connectivity()

    def close(self):
        self.driver.close()

    def retrieve_subgraph(self, entity_id: str, max_hops: int = 2, limit: int = 50) -> dict:
        """Generic N-hop retrieval around any entity."""
        # Neo4j doesn't allow parameters in variable-length ranges
        cypher = (
            f"MATCH path = (n {{id: $entity_id}})-[*1..{max_hops}]-(m) "
            "UNWIND relationships(path) AS r "
            "WITH n, m, r, startNode(r) AS src, endNode(r) AS tgt "
            "RETURN DISTINCT "
            "  src.id AS source_id, labels(src) AS source_labels, "
            "  type(r) AS rel_type, "
            "  tgt.id AS target_id, labels(tgt) AS target_labels "
            "LIMIT $limit"
        )
        with self.driver.session() as session:
            result = session.run(
                cypher,
                entity_id=entity_id,
                limit=limit,
            )
            edges = []
            node_set = set()
            for record in result:
                edges.append({
                    "source": record["source_id"],
                    "source_labels": record["source_labels"],
                    "relation": record["rel_type"],
                    "target": record["target_id"],
                    "target_labels": record["target_labels"],
                })
                node_set.add(record["source_id"])
                node_set.add(record["target_id"])
            return {"entity": entity_id, "nodes": len(node_set), "edges": edges}

    def retrieve_error_diagnosis(self, violation_id: str) -> dict:
        """Retrieve Violation → Fix paths with documentation context.

        Returns the violation, all known fixes, and any linked reports
        or tool runs for context injection into LLM prompts.
        """
        with self.driver.session() as session:
            violation = session.run(
                "MATCH (v:Violation {id: $vid}) "
                "RETURN v.id AS id, v.description AS description, "
                "       v.error_code AS error_code, v.severity AS severity, "
                "       v.source AS source",
                vid=violation_id,
            ).single()

            if not violation:
                return {"violation_id": violation_id, "found": False}

            fixes = session.run(
                "MATCH (v:Violation {id: $vid})<-[:FIXES]-(f:Fix) "
                "OPTIONAL MATCH (f)-[:DOCUMENTED_IN]->(r:Report) "
                "RETURN f.id AS fix_id, f.description AS fix_description, "
                "       f.source AS fix_source, "
                "       collect(DISTINCT r.source_file) AS doc_files",
                vid=violation_id,
            ).data()

            causes = session.run(
                "MATCH (v:Violation {id: $vid})<-[:CAUSES]-(t:ToolRun) "
                "OPTIONAL MATCH (t)-[:VERSION_OF|RAN_ON]->(ver:Version) "
                "RETURN t.id AS run_id, t.design AS design, "
                "       ver.version_tag AS version",
                vid=violation_id,
            ).data()

            return {
                "violation_id": violation_id,
                "found": True,
                "violation": dict(violation),
                "fixes": fixes,
                "causes": causes,
            }

    def retrieve_violation_fixes(self, violation_id: str) -> list:
        """Simple fix lookup for a violation — returns list of fix dicts."""
        with self.driver.session() as session:
            result = session.run(
                "MATCH (v:Violation {id: $vid})<-[:FIXES]-(f:Fix) "
                "OPTIONAL MATCH (f)-[:DOCUMENTED_IN]->(r:Report) "
                "RETURN f.id AS fix_id, f.description AS description, "
                "       f.source AS source, r.source_file AS doc_file",
                vid=violation_id,
            )
            return [dict(r) for r in result]

    def retrieve_version_divergence(self, design: str) -> list:
        """Retrieve version comparison data for a design.

        Graph topology: Design -[:RAN_ON]-> Version -[:DIVERGES_FROM]-> Version
        ToolRun -[:HAS_TIMING|HAS_METRIC]-> Metric (linked by ID convention)
        """
        with self.driver.session() as session:
            # Design→Version via RAN_ON
            result = session.run(
                "MATCH (d:Design)-[:RAN_ON]->(v:Version) "
                "WHERE toLower(d.id) CONTAINS toLower($design) "
                "OPTIONAL MATCH (v)-[:DIVERGES_FROM]->(v2:Version) "
                "RETURN d.id AS design_id, "
                "       v.id AS version_id, v.version_tag AS version_tag, "
                "       v2.id AS diverges_to, v2.version_tag AS diverges_to_tag",
                design=design,
            )
            versions = [dict(r) for r in result]

            # ToolRun→Metric for this design (by ID convention)
            metrics = session.run(
                "MATCH (t:ToolRun)-[:HAS_TIMING|HAS_METRIC]->(m:Metric) "
                "WHERE toLower(t.id) CONTAINS toLower($design) "
                "RETURN t.id AS run_id, collect(m.id) AS metric_ids",
                design=design,
            ).data()

            return {"versions": versions, "tool_runs": metrics}

    def retrieve_rtl_context(self, module_name: str, limit: int = 30) -> dict:
        """Retrieve module context: ports, instances, design hierarchy."""
        with self.driver.session() as session:
            result = session.run(
                "MATCH (m:Module {id: $mid})-[r]-(n) "
                "RETURN type(r) AS rel, labels(n) AS labels, "
                "       n.id AS neighbor_id, n.name AS neighbor_name "
                "LIMIT $limit",
                mid=module_name,
                limit=limit,
            )
            neighbors = [dict(r) for r in result]
            return {"module": module_name, "context": neighbors}

    def retrieve_drc_rules(self, rule_id: str) -> dict:
        """Retrieve DRC rule details with linked PDK and layer info."""
        with self.driver.session() as session:
            result = session.run(
                "MATCH (r:Rule {id: $rid}) "
                "OPTIONAL MATCH (r)-[:DEFINED_IN]->(p:PDK) "
                "OPTIONAL MATCH (v:Violation)-[:VIOLATES]->(r) "
                "RETURN r.id AS rule_id, r.description AS description, "
                "       collect(DISTINCT p.id) AS pdks, "
                "       count(DISTINCT v) AS violation_count",
                rid=rule_id,
            )
            data = result.single()
            if not data:
                return {"rule_id": rule_id, "found": False}
            return {"rule_id": rule_id, "found": True, **dict(data)}

    def retrieve_by_category(self, category: str, entity_id: str) -> dict:
        """Dispatch retrieval based on query_router category."""
        dispatch = {
            "error_diagnosis": lambda: self.retrieve_error_diagnosis(entity_id),
            "rtl_qa": lambda: self.retrieve_rtl_context(entity_id),
            "constraint_generation": lambda: self.retrieve_subgraph(entity_id),
            "drc_rule_lookup": lambda: self.retrieve_drc_rules(entity_id),
            "cross_tool_knowledge": lambda: self.retrieve_subgraph(entity_id),
        }
        handler = dispatch.get(category, lambda: self.retrieve_subgraph(entity_id))
        return handler()
