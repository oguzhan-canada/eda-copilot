"""
Query router — classifies incoming EDA queries into task categories.

Each category maps to a different retrieval strategy in the GraphRAG engine:
  - error_diagnosis:        2-hop Violation→Fix subgraph
  - rtl_qa:                 Module→Port→Design subgraph
  - constraint_generation:  Rule→SDC→Design subgraph
  - drc_rule_lookup:        Rule→PDK→Layer subgraph
  - cross_tool_knowledge:   Version→DIVERGES_FROM→Version subgraph

Current implementation: keyword-based scoring (stub).
Week 11: replace with fine-tuned classifier on synthetic Q&A data.
"""

from enum import Enum
from typing import Optional


class TaskCategory(str, Enum):
    ERROR_DIAGNOSIS = "error_diagnosis"
    RTL_QA = "rtl_qa"
    CONSTRAINT_GENERATION = "constraint_generation"
    DRC_RULE_LOOKUP = "drc_rule_lookup"
    CROSS_TOOL_KNOWLEDGE = "cross_tool_knowledge"


CATEGORY_SIGNALS: dict[TaskCategory, list[str]] = {
    TaskCategory.ERROR_DIAGNOSIS: [
        "error", "crash", "violation", "failed", "fail", "sigsegv",
        "segfault", "wns", "tns", "slack", "drc", "timing", "setup",
        "hold", "bug", "fix", "wrong", "broken", "issue",
    ],
    TaskCategory.RTL_QA: [
        "module", "port", "wire", "reg", "always", "assign", "verilog",
        "rtl", "input", "output", "inout", "instance", "net", "signal",
        "flip-flop", "latch", "mux", "decoder",
    ],
    TaskCategory.CONSTRAINT_GENERATION: [
        "sdc", "create_clock", "set_false_path", "set_multicycle_path",
        "constraint", "clock", "period", "uncertainty", "set_input_delay",
        "set_output_delay", "set_max_delay", "group_path",
    ],
    TaskCategory.DRC_RULE_LOOKUP: [
        "spacing", "enclosure", "layer", "metal", "via", "rule",
        "width", "density", "antenna", "min_area", "overlap",
        "pdk", "sky130", "asap7", "nangate",
    ],
    TaskCategory.CROSS_TOOL_KNOWLEDGE: [
        "version", "orfs", "yosys", "openroad", "openlane", "diverge",
        "upgrade", "migration", "compatibility", "tool", "release",
        "regression", "difference", "change",
    ],
}


def route_query(query: str, default: TaskCategory = TaskCategory.ERROR_DIAGNOSIS) -> TaskCategory:
    """Classify a query into a task category using keyword scoring.

    Returns the category with the highest keyword match count.
    Falls back to default if no keywords match.
    """
    q = query.lower()
    scores = {}
    for cat, keywords in CATEGORY_SIGNALS.items():
        score = sum(1 for kw in keywords if kw in q)
        scores[cat] = score

    max_score = max(scores.values())
    if max_score == 0:
        return default

    return max(scores, key=scores.get)


def route_query_with_confidence(query: str) -> tuple[TaskCategory, float]:
    """Route query and return confidence score (0.0–1.0).

    Confidence is the proportion of the winning category's keywords
    that matched, capped by a minimum keyword count threshold.
    """
    q = query.lower()
    scores = {}
    for cat, keywords in CATEGORY_SIGNALS.items():
        matched = sum(1 for kw in keywords if kw in q)
        scores[cat] = matched

    max_score = max(scores.values())
    if max_score == 0:
        return TaskCategory.ERROR_DIAGNOSIS, 0.0

    winner = max(scores, key=scores.get)
    total_keywords = len(CATEGORY_SIGNALS[winner])
    confidence = min(1.0, max_score / max(3, total_keywords * 0.3))
    return winner, round(confidence, 2)
