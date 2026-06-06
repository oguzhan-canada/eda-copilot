"""
Claude API synthesizer — generates answers from fusion retriever context.

Prompt structure:
  [SYSTEM] EDA copilot instructions with citation/version requirements
  [CONTEXT] Graph facts + reranked document chunks from FusionRetriever
  [QUERY]  User question

Output: answer + citations + confidence (high/medium/low)

Usage:
    from apps.api.services.synthesizer import Synthesizer
    synth = Synthesizer()
    answer = synth.synthesize(query, retrieval_result)
"""

import json
import os
from typing import Optional

import anthropic


SYSTEM_PROMPT = """You are an EDA (Electronic Design Automation) copilot specializing in OpenROAD, ORFS, and ASIC design flows.

Answer the user's question using ONLY the provided context (knowledge graph facts and retrieved documents). Do not use external knowledge.

Rules:
1. Always cite your sources by referencing the source_file paths from the retrieved documents.
2. If the context contains version-specific information, explicitly state which tool version applies.
3. If the context is insufficient to answer fully, say so and indicate what additional information would be needed.
4. For error diagnosis queries, structure your answer as: Root Cause → Evidence → Fix → Caveats.
5. For version comparison queries, clearly separate findings per version.

Output your response as JSON with this structure:
{
    "answer": "Your detailed answer here",
    "citations": ["source_file_1", "source_file_2"],
    "confidence": "high|medium|low",
    "reasoning": "Brief explanation of why this confidence level"
}"""

DEFAULT_MODEL = "claude-sonnet-4-20250514"  # Update when newer model available
MAX_CONTEXT_TOKENS = 6000  # Cap context to control cost


class Synthesizer:
    """Thin wrapper around Claude API for answer generation."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
    ):
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        self.model = model or DEFAULT_MODEL
        self.client = anthropic.Anthropic(api_key=self.api_key)

    def synthesize(
        self,
        query: str,
        retrieval_result: dict,
        max_tokens: int = 1024,
    ) -> dict:
        """Generate an answer from fusion retriever output.

        Args:
            query: Original user query
            retrieval_result: Output from FusionRetriever.retrieve()
            max_tokens: Max output tokens for Claude response

        Returns:
            Dict with answer, citations, confidence, task_category,
            and token usage stats.
        """
        from apps.api.services.fusion_retriever import FusionRetriever

        # Build context from retrieval result
        # Use a temporary instance just for formatting if needed
        context = self._format_context(retrieval_result)

        # Truncate context if too long (rough char estimate)
        max_chars = MAX_CONTEXT_TOKENS * 4
        if len(context) > max_chars:
            context = context[:max_chars] + "\n\n[Context truncated for length]"

        user_message = f"""## Context

{context}

## Query

{query}

Respond with JSON only."""

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_message}],
            )

            raw_text = response.content[0].text
            usage = {
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
                "model": self.model,
            }

            # Parse JSON response
            parsed = self._parse_response(raw_text)
            parsed["task_category"] = retrieval_result.get("task_category", "unknown")
            parsed["entities_found"] = retrieval_result.get("entities_found", [])
            parsed["usage"] = usage
            return parsed

        except Exception as e:
            return {
                "answer": f"Error generating answer: {str(e)}",
                "citations": [],
                "confidence": "low",
                "reasoning": f"API error: {str(e)}",
                "task_category": retrieval_result.get("task_category", "unknown"),
                "entities_found": retrieval_result.get("entities_found", []),
                "usage": {},
            }

    def _format_context(self, retrieval_result: dict) -> str:
        """Format retrieval result into context string."""
        sections = []

        # Graph facts
        graph_facts = retrieval_result.get("graph_facts", [])
        if graph_facts:
            sections.append("### Knowledge Graph Facts\n")
            for fact in graph_facts:
                if "error" in fact:
                    continue
                entity = fact.get("entity_id", "unknown")
                data = fact.get("data", {})
                sections.append(f"**Entity: {entity}**")

                if isinstance(data, dict):
                    if "violation" in data and data.get("found"):
                        v = data["violation"]
                        sections.append(f"- Violation: {v.get('description', 'N/A')}")
                        for fix in data.get("fixes", []):
                            desc = fix.get("fix_description", fix.get("fix_id", "N/A"))
                            sections.append(f"  - Fix: {desc}")
                    elif "edges" in data:
                        for edge in data["edges"][:10]:
                            sections.append(
                                f"- {edge['source']} --[{edge['relation']}]--> {edge['target']}"
                            )
                    elif "versions" in data:
                        for v in data["versions"]:
                            line = f"- {v.get('design_id', '?')} runs on {v.get('version_id', '?')}"
                            if v.get("diverges_to"):
                                line += f" (diverges from {v['diverges_to']})"
                            sections.append(line)
                sections.append("")

        # Document chunks
        chunks = retrieval_result.get("chunks", [])
        if chunks:
            sections.append("### Retrieved Documents\n")
            for i, chunk in enumerate(chunks, 1):
                if "error" in chunk:
                    continue
                source = chunk.get("source_file", "unknown")
                text = chunk.get("text", "")
                score = chunk.get("rerank_score", chunk.get("score", 0))
                sections.append(f"**[{i}] {source}** (relevance: {score:.3f})")
                sections.append(text)
                sections.append("")

        return "\n".join(sections) if sections else "No context available."

    def _parse_response(self, raw_text: str) -> dict:
        """Parse Claude's JSON response, handling markdown fences."""
        text = raw_text.strip()
        # Strip markdown code fences if present
        if text.startswith("```"):
            lines = text.split("\n")
            lines = [l for l in lines if not l.strip().startswith("```")]
            text = "\n".join(lines)

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            # If JSON parsing fails, return raw text as answer
            return {
                "answer": raw_text,
                "citations": [],
                "confidence": "low",
                "reasoning": "Response was not valid JSON",
            }
