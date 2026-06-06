"""
Fusion retriever — combines Neo4j graph retrieval and Weaviate vector search.

Retrieval flow:
  1. Route query → task_category (query_router.py)
  2. Extract entity mentions from query (match against cached KG node IDs)
  3. PARALLEL:
     a. Weaviate hybrid search → top-20 chunks
     b. Neo4j 2-hop subgraph for each entity mention → structured facts
  4. Rerank hybrid search results → top-N chunks
  5. Format context: reranked chunks + graph facts → prompt context block
  6. Return: {chunks, graph_facts, task_category, entities_found}

Usage:
    retriever = FusionRetriever()
    result = retriever.retrieve("Why does JPEG timing fail after ORFS upgrade?")
    # result = {
    #   "task_category": "error_diagnosis",
    #   "confidence": 0.85,
    #   "entities_found": ["ed_002_version_divergence", "design_jpeg"],
    #   "graph_facts": [...],
    #   "chunks": [...],
    # }
"""

import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

from apps.api.services.graph_rag import GraphRAGEngine
from apps.api.services.query_router import (
    TaskCategory,
    route_query_with_confidence,
)
from pipeline.retrieve.reranker import rerank


class FusionRetriever:
    """Parallel graph + vector retrieval with reranking."""

    def __init__(
        self,
        neo4j_uri: Optional[str] = None,
        neo4j_user: Optional[str] = None,
        neo4j_password: Optional[str] = None,
    ):
        self.graph = GraphRAGEngine(
            uri=neo4j_uri,
            user=neo4j_user,
            password=neo4j_password,
        )
        self._weaviate_engine = None
        self._known_ids: Optional[set] = None

    @property
    def weaviate_engine(self):
        if self._weaviate_engine is None:
            from pipeline.retrieve.hybrid_search import WeaviateSearchEngine
            self._weaviate_engine = WeaviateSearchEngine()
        return self._weaviate_engine

    def _load_known_ids(self) -> set:
        """Cache all KG node IDs for entity extraction."""
        if self._known_ids is not None:
            return self._known_ids

        with self.graph.driver.session() as session:
            result = session.run(
                "MATCH (n) WHERE n.id IS NOT NULL RETURN n.id AS id"
            )
            self._known_ids = {r["id"] for r in result}
        return self._known_ids

    def extract_entities(self, query: str) -> list[str]:
        """Match query tokens against known KG node IDs.

        Uses word-boundary-aware matching to avoid false positives.
        Returns IDs sorted by length (longer = more specific = higher priority).
        """
        known_ids = self._load_known_ids()
        q_lower = query.lower()
        q_words = set(re.findall(r'\b\w+\b', q_lower))
        matched = []

        # Priority labels: only match these prefixed node types
        priority_prefixes = ["design_", "version_", "tool_", "ed_"]
        # High-value node names to match by suffix (word-boundary check)
        design_names = {"jpeg", "ibex", "aes", "gcd", "riscv32i", "swerv", "swerv_wrapper"}
        tool_names = {"orfs", "yosys", "openroad", "openlane"}
        version_patterns = {"v3.0", "v3_0", "26q1", "26Q1"}

        for nid in known_ids:
            nid_lower = nid.lower()

            # Direct full-ID match (e.g., query mentions "ed_002")
            if nid_lower in q_lower:
                matched.append(nid)
                continue

            # Match known design names as word boundaries
            for prefix in priority_prefixes:
                if not nid_lower.startswith(prefix):
                    continue
                suffix = nid_lower[len(prefix):]

                # Design/tool names: exact word match in query
                if suffix in design_names and suffix in q_words:
                    matched.append(nid)
                    break
                if suffix in tool_names and suffix in q_words:
                    matched.append(nid)
                    break
                # Version patterns
                if suffix in version_patterns and suffix in q_lower:
                    matched.append(nid)
                    break

        # Also match ED-NNN patterns
        ed_matches = re.findall(r'\b(ed[_-]?\d{3})\b', q_lower)
        for ed in ed_matches:
            # Normalize ed-002 → find matching KG ID
            normalized = ed.replace("-", "_")
            candidates = [nid for nid in known_ids if normalized in nid.lower()]
            matched.extend(candidates)

        # Deduplicate, sort longest first (most specific)
        seen = set()
        unique = []
        for m in matched:
            if m not in seen:
                seen.add(m)
                unique.append(m)
        return sorted(unique, key=len, reverse=True)

    def _retrieve_graph_facts(
        self, category: TaskCategory, entities: list[str]
    ) -> list[dict]:
        """Retrieve structured facts from Neo4j for each entity.

        Smart dispatch: uses entity ID prefix to choose retrieval method
        rather than blindly using category dispatch.
        """
        facts = []

        for entity_id in entities[:5]:  # Cap at 5 entities to bound latency
            try:
                eid_lower = entity_id.lower()

                # Choose retrieval method based on entity type, not just category
                if eid_lower.startswith("ed_") or eid_lower.startswith("violation_"):
                    result = self.graph.retrieve_error_diagnosis(entity_id)
                elif eid_lower.startswith("design_"):
                    # For designs, get the 2-hop subgraph + version divergence
                    design_name = entity_id.replace("design_", "")
                    result = self.graph.retrieve_version_divergence(design_name)
                    if not result.get("versions"):
                        result = self.graph.retrieve_subgraph(entity_id)
                elif eid_lower.startswith("rule_") or eid_lower.startswith("constraint_"):
                    result = self.graph.retrieve_drc_rules(entity_id)
                elif eid_lower.startswith("version_") or eid_lower.startswith("tool_"):
                    result = self.graph.retrieve_subgraph(entity_id)
                else:
                    result = self.graph.retrieve_by_category(category.value, entity_id)

                if result:
                    facts.append({
                        "entity_id": entity_id,
                        "category": category.value,
                        "data": result,
                    })
            except Exception as e:
                facts.append({
                    "entity_id": entity_id,
                    "category": category.value,
                    "error": str(e),
                })

        # For cross_tool_knowledge, also try version divergence
        if category == TaskCategory.CROSS_TOOL_KNOWLEDGE:
            design_entities = [
                e for e in entities
                if e.lower().startswith("design_") or len(e) <= 10
            ]
            for de in design_entities[:2]:
                name = de.replace("design_", "")
                try:
                    divergence = self.graph.retrieve_version_divergence(name)
                    if divergence.get("versions"):
                        facts.append({
                            "entity_id": de,
                            "category": "version_divergence",
                            "data": divergence,
                        })
                except Exception:
                    pass

        return facts

    def _retrieve_vector_chunks(
        self, query: str, top_k: int = 20, mode: str = "hybrid"
    ) -> list[dict]:
        """Retrieve candidate chunks from Weaviate."""
        try:
            return self.weaviate_engine.search(
                query, top_k=top_k, mode=mode
            )
        except Exception as e:
            # Fallback to BM25 if hybrid fails (e.g., Voyage rate limit)
            try:
                return self.weaviate_engine.search(
                    query, top_k=top_k, mode="sparse"
                )
            except Exception:
                return [{"error": str(e), "source_file": "N/A", "text": ""}]

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        vector_candidates: int = 10,  # 10 candidates balances quality vs rerank latency
        search_mode: str = "hybrid",
    ) -> dict:
        """Full fusion retrieval: route → extract → parallel fetch → rerank.

        Args:
            query: Natural language query
            top_k: Final number of reranked chunks to return
            vector_candidates: Number of candidates to fetch before reranking
            search_mode: "hybrid", "dense", or "sparse"

        Returns:
            Dict with task_category, confidence, entities_found,
            graph_facts, and reranked chunks.
        """
        # Step 1: Route query
        category, confidence = route_query_with_confidence(query)

        # Step 2: Extract entities
        entities = self.extract_entities(query)

        # Step 3: Parallel retrieval
        graph_facts = []
        vector_chunks = []

        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = {}

            if entities:
                futures["graph"] = executor.submit(
                    self._retrieve_graph_facts, category, entities
                )

            futures["vector"] = executor.submit(
                self._retrieve_vector_chunks, query, vector_candidates, search_mode
            )

            for key, future in futures.items():
                try:
                    result = future.result(timeout=30)
                    if key == "graph":
                        graph_facts = result
                    elif key == "vector":
                        vector_chunks = result
                except Exception as e:
                    if key == "graph":
                        graph_facts = [{"error": str(e)}]
                    else:
                        vector_chunks = [{"error": str(e), "text": ""}]

        # Step 4: Rerank vector chunks
        reranked = []
        if vector_chunks and "error" not in vector_chunks[0]:
            try:
                reranked = rerank(query, vector_chunks, top_n=top_k)
            except Exception:
                reranked = vector_chunks[:top_k]
        else:
            reranked = vector_chunks[:top_k]

        return {
            "task_category": category.value,
            "confidence": confidence,
            "entities_found": entities,
            "graph_facts": graph_facts,
            "chunks": reranked,
        }

    def format_context(self, result: dict) -> str:
        """Format retrieval result into a context string for LLM prompt.

        Produces a structured text block with graph facts and document chunks
        suitable for injection into a Claude system/user prompt.
        """
        sections = []

        # Graph facts section
        if result["graph_facts"]:
            sections.append("## Knowledge Graph Facts")
            for fact in result["graph_facts"]:
                if "error" in fact:
                    continue
                entity = fact.get("entity_id", "unknown")
                data = fact.get("data", {})
                sections.append(f"\n### Entity: {entity}")

                if isinstance(data, dict):
                    # Error diagnosis format
                    if "violation" in data and data.get("found"):
                        v = data["violation"]
                        sections.append(f"- Violation: {v.get('description', 'N/A')}")
                        for fix in data.get("fixes", []):
                            sections.append(
                                f"  - Fix: {fix.get('fix_description', fix.get('fix_id', 'N/A'))}"
                            )
                            if fix.get("doc_files"):
                                sections.append(f"    Source: {fix['doc_files']}")

                    # Subgraph format
                    elif "edges" in data:
                        for edge in data["edges"][:15]:
                            sections.append(
                                f"- {edge['source']} --[{edge['relation']}]--> {edge['target']}"
                            )

                    # Version divergence format
                    elif "versions" in data:
                        for v in data["versions"]:
                            line = f"- {v.get('design_id', '?')} → {v.get('version_id', '?')}"
                            if v.get("diverges_to"):
                                line += f" (diverges from {v['diverges_to']})"
                            sections.append(line)

        # Document chunks section
        if result["chunks"]:
            sections.append("\n## Retrieved Documents")
            for i, chunk in enumerate(result["chunks"], 1):
                if "error" in chunk:
                    continue
                source = chunk.get("source_file", "unknown")
                score = chunk.get("rerank_score", chunk.get("score", 0))
                text = chunk.get("text", "")
                sections.append(f"\n### [{i}] {source} (score: {score:.3f})")
                sections.append(text)

        return "\n".join(sections)

    def close(self):
        self.graph.close()
        if self._weaviate_engine:
            self._weaviate_engine.close()
