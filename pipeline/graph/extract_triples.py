"""
Component 2.2 — Tiered Triple Extraction Pipeline.

Three-tier extraction architecture optimized for cost:
  Tier 1 (free): Regex-based extraction from structured EDA reports/logs
  Tier 2 (free): spaCy NER for entity recognition in semi-structured text
  Tier 3 (paid): Claude Batch API for ambiguous forum/doc content (~15% of triples)

Usage:
    # Extract from deduped corpus
    python -m pipeline.graph.extract_triples \
        --input C:\\eda-kg-data\\corpus\\staging\\dedup \
        --output data/graph/triples_raw.jsonl \
        --llm-queue data/graph/llm_queue.jsonl

    # After batch completes, merge LLM-extracted triples
    python -m pipeline.graph.extract_triples \
        --merge-llm data/graph/llm_results.jsonl \
        --output data/graph/triples_raw.jsonl
"""

import argparse
import json
import re
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional


@dataclass
class Triple:
    """A single knowledge graph triple."""
    subject_id: str
    subject_label: str  # Node label: Design, Module, Violation, Rule, etc.
    predicate: str      # Relationship type: CAUSES, FIXES, VIOLATES, etc.
    object_id: str
    object_label: str
    properties: dict    # Additional properties on the relationship
    source_file: str
    extraction_tier: int  # 1=regex, 2=NER, 3=LLM
    confidence: float     # 0.0-1.0


# ═══════════════════════════════════════════════════════════════════════════════
# TIER 1: Regex-based extraction (structured reports/logs)
# ═══════════════════════════════════════════════════════════════════════════════

# Timing violation patterns (STA reports)
TIMING_SLACK_RE = re.compile(
    r"(?:slack|WNS|TNS)\s*[=:]\s*([+-]?\d+\.?\d*)\s*(ps|ns)",
    re.IGNORECASE,
)

TIMING_PATH_RE = re.compile(
    r"(?:Startpoint|Endpoint)\s*:\s*(\S+)",
    re.IGNORECASE,
)

CLOCK_RE = re.compile(
    r"(?:create_clock|set_clock)\s+.*?-(?:period|name)\s+(\S+)",
    re.IGNORECASE,
)

# DRC violation patterns
DRC_VIOLATION_RE = re.compile(
    r"(?:violation|error)\s*:\s*(\S+)\s+(?:on\s+)?(?:layer\s+)?(\w+).*?(?:at\s+)?\((\d+\.?\d*)\s*,\s*(\d+\.?\d*)\)",
    re.IGNORECASE,
)

DRC_RULE_RE = re.compile(
    r"(?:rule|check)\s*[=:]\s*(\w+(?:\.\w+)*)",
    re.IGNORECASE,
)

# SDC directive patterns
SDC_CLOCK_PERIOD_RE = re.compile(
    r"create_clock\s+.*?-period\s+(\d+\.?\d*)",
)

SDC_FALSE_PATH_RE = re.compile(
    r"set_false_path\s+(.*)",
)

SDC_UNITS_RE = re.compile(
    r"set_units\s+.*?-time\s+(\w+)",
)

# Tool version patterns
TOOL_VERSION_RE = re.compile(
    r"(?:OpenROAD|Yosys|OpenSTA|TritonRoute|OpenDP|KLayout)\s+(?:v|version\s+)?(\d+\.\d+(?:\.\d+)?)",
    re.IGNORECASE,
)

# ORFS flow patterns
ORFS_DESIGN_RE = re.compile(
    r"DESIGN_NAME\s*[=:]\s*(\w+)",
)

ORFS_PLATFORM_RE = re.compile(
    r"PLATFORM\s*[=:]\s*(\w+)",
)

# Module hierarchy
MODULE_INSTANCE_RE = re.compile(
    r"(?:COMPONENTS|INSTANCES)\s+(\d+)\s*;",
)

VERILOG_MODULE_RE = re.compile(
    r"module\s+(\w+)\s*[#(]",
)

VERILOG_INSTANCE_RE = re.compile(
    r"(\w+)\s+(\w+)\s*\(",  # cell_type instance_name (
)


def extract_regex_triples(filepath: Path, content: str) -> list[Triple]:
    """Tier 1: Extract triples using regex patterns from structured files."""
    triples = []
    fname = filepath.name.lower()
    source = str(filepath)

    # ORFS JSON report (6_report.json) — structured PPA metrics
    if fname == "6_report.json" and content.strip().startswith("{"):
        triples += _extract_orfs_report_triples(content, source, filepath)
        return triples  # ORFS JSON fully handled, skip regex

    # Detect file type by name/extension
    if any(x in fname for x in ["timing", "sta", "slack", "report"]):
        triples += _extract_timing_triples(content, source)
    
    if any(x in fname for x in ["drc", "violation", "error"]):
        triples += _extract_drc_triples(content, source)
    
    if fname.endswith(".sdc") or "constraint" in fname:
        triples += _extract_sdc_triples(content, source)
    
    if any(x in fname for x in ["makefile", "config", "flow", ".mk"]):
        triples += _extract_flow_triples(content, source)
    
    if fname.endswith(".v") or fname.endswith(".sv"):
        triples += _extract_rtl_triples(content, source)

    # Always check for tool versions
    triples += _extract_tool_versions(content, source)

    return triples


def _extract_orfs_report_triples(content: str, source: str, filepath: Path) -> list[Triple]:
    """Extract PPA triples from ORFS 6_report.json files."""
    triples = []
    try:
        report = json.loads(content)
    except json.JSONDecodeError:
        return triples

    # Infer design name and version from path
    # Expected: .../v3_0/<design>/...  or .../26q1/<design>/...
    parts = filepath.parts
    design_name = None
    orfs_version = None
    for i, p in enumerate(parts):
        if p in ("v3_0", "v3.0"):
            orfs_version = "ORFS_v3.0"
            if i + 1 < len(parts):
                design_name = parts[i + 1]
        elif p in ("26q1", "26Q1"):
            orfs_version = "ORFS_26Q1"
            if i + 1 < len(parts):
                design_name = parts[i + 1]

    if not design_name:
        # Try from report keys (some have design__ prefix)
        for k in report:
            if "design" in k.lower():
                design_name = k.split("__")[-1] if "__" in k else "unknown"
                break
        if not design_name:
            design_name = "unknown"

    design_id = f"design_{design_name}"
    version_id = f"version_{orfs_version}" if orfs_version else "version_unknown"
    run_id = f"run_{design_name}_{orfs_version or 'unknown'}"

    # Design -> Version relationship
    if orfs_version:
        triples.append(Triple(
            subject_id=design_id,
            subject_label="Design",
            predicate="RAN_ON",
            object_id=version_id,
            object_label="Version",
            properties={"design": design_name, "version": orfs_version},
            source_file=source,
            extraction_tier=1,
            confidence=0.99,
        ))

    # WNS (setup worst slack)
    wns = report.get("finish__timing__setup__ws")
    if wns is not None:
        triples.append(Triple(
            subject_id=run_id,
            subject_label="ToolRun",
            predicate="HAS_TIMING",
            object_id=f"metric_wns_{design_name}_{orfs_version}",
            object_label="Metric",
            properties={"wns_ps": wns, "design": design_name, "version": orfs_version},
            source_file=source,
            extraction_tier=1,
            confidence=0.99,
        ))

        # If WNS is negative, it's a timing violation
        if wns < 0:
            triples.append(Triple(
                subject_id=run_id,
                subject_label="ToolRun",
                predicate="VIOLATES",
                object_id=f"constraint_setup_{design_name}",
                object_label="Rule",
                properties={"wns_ps": wns, "violation_type": "setup_timing"},
                source_file=source,
                extraction_tier=1,
                confidence=0.95,
            ))

    # TNS (total negative slack)
    tns = report.get("finish__timing__setup__tns")
    if tns is not None:
        triples.append(Triple(
            subject_id=run_id,
            subject_label="ToolRun",
            predicate="HAS_METRIC",
            object_id=f"metric_tns_{design_name}_{orfs_version}",
            object_label="Metric",
            properties={"tns_ps": tns, "metric_type": "tns"},
            source_file=source,
            extraction_tier=1,
            confidence=0.99,
        ))

    # Area
    area = report.get("finish__design__instance__area")
    if area is not None:
        triples.append(Triple(
            subject_id=run_id,
            subject_label="ToolRun",
            predicate="HAS_METRIC",
            object_id=f"metric_area_{design_name}_{orfs_version}",
            object_label="Metric",
            properties={"area_um2": area, "metric_type": "area"},
            source_file=source,
            extraction_tier=1,
            confidence=0.99,
        ))

    # Utilization
    util = report.get("finish__design__instance__utilization")
    if util is not None:
        triples.append(Triple(
            subject_id=run_id,
            subject_label="ToolRun",
            predicate="HAS_METRIC",
            object_id=f"metric_util_{design_name}_{orfs_version}",
            object_label="Metric",
            properties={"utilization": util, "metric_type": "utilization"},
            source_file=source,
            extraction_tier=1,
            confidence=0.99,
        ))

    # DRV violations
    drv_setup = report.get("finish__timing__drv__setup_violation_count")
    drv_hold = report.get("finish__timing__drv__hold_violation_count")
    if drv_setup is not None and drv_setup > 0:
        triples.append(Triple(
            subject_id=run_id,
            subject_label="ToolRun",
            predicate="VIOLATES",
            object_id=f"rule_drv_setup_{design_name}",
            object_label="Rule",
            properties={"count": drv_setup, "violation_type": "drv_setup"},
            source_file=source,
            extraction_tier=1,
            confidence=0.95,
        ))
    if drv_hold is not None and drv_hold > 0:
        triples.append(Triple(
            subject_id=run_id,
            subject_label="ToolRun",
            predicate="VIOLATES",
            object_id=f"rule_drv_hold_{design_name}",
            object_label="Rule",
            properties={"count": drv_hold, "violation_type": "drv_hold"},
            source_file=source,
            extraction_tier=1,
            confidence=0.95,
        ))

    # Power
    power_total = report.get("finish__power__total")
    if power_total is not None:
        triples.append(Triple(
            subject_id=run_id,
            subject_label="ToolRun",
            predicate="HAS_METRIC",
            object_id=f"metric_power_{design_name}_{orfs_version}",
            object_label="Metric",
            properties={"power_w": power_total, "metric_type": "power"},
            source_file=source,
            extraction_tier=1,
            confidence=0.99,
        ))

    return triples


def _extract_timing_triples(content: str, source: str) -> list[Triple]:
    """Extract timing path and slack triples from STA reports."""
    triples = []
    
    for m in TIMING_SLACK_RE.finditer(content):
        slack_val = float(m.group(1))
        unit = m.group(2)
        tp_id = f"tp_{abs(hash(content[:200]))}_{m.start()}"
        
        triples.append(Triple(
            subject_id=tp_id,
            subject_label="TimingPath",
            predicate="HAS_SLACK",
            object_id=f"slack_{slack_val}{unit}",
            object_label="Metric",
            properties={"slack": slack_val, "unit": unit},
            source_file=source,
            extraction_tier=1,
            confidence=0.95,
        ))

    # Extract path endpoints
    endpoints = TIMING_PATH_RE.findall(content)
    for i in range(0, len(endpoints) - 1, 2):
        start_pin = endpoints[i]
        end_pin = endpoints[i + 1] if i + 1 < len(endpoints) else None
        if end_pin:
            triples.append(Triple(
                subject_id=f"pin_{start_pin}",
                subject_label="Module",
                predicate="TIMING_PATH_TO",
                object_id=f"pin_{end_pin}",
                object_label="Module",
                properties={"startpoint": start_pin, "endpoint": end_pin},
                source_file=source,
                extraction_tier=1,
                confidence=0.90,
            ))

    return triples


def _extract_drc_triples(content: str, source: str) -> list[Triple]:
    """Extract DRC violation triples."""
    triples = []
    
    for m in DRC_VIOLATION_RE.finditer(content):
        rule_name = m.group(1)
        layer = m.group(2)
        x, y = m.group(3), m.group(4)
        viol_id = f"drc_{rule_name}_{x}_{y}"
        
        triples.append(Triple(
            subject_id=viol_id,
            subject_label="Violation",
            predicate="VIOLATES",
            object_id=f"rule_{rule_name}",
            object_label="Rule",
            properties={"layer": layer, "x": float(x), "y": float(y)},
            source_file=source,
            extraction_tier=1,
            confidence=0.92,
        ))

    return triples


def _extract_sdc_triples(content: str, source: str) -> list[Triple]:
    """Extract SDC constraint triples."""
    triples = []
    
    for m in SDC_CLOCK_PERIOD_RE.finditer(content):
        period = float(m.group(1))
        triples.append(Triple(
            subject_id=f"clk_{source}",
            subject_label="Rule",
            predicate="DEFINES_CLOCK",
            object_id=f"period_{period}",
            object_label="Metric",
            properties={"period": period, "directive": "create_clock"},
            source_file=source,
            extraction_tier=1,
            confidence=0.98,
        ))
    
    # Unit declarations
    for m in SDC_UNITS_RE.finditer(content):
        unit = m.group(1)
        triples.append(Triple(
            subject_id=f"sdc_{source}",
            subject_label="Report",
            predicate="USES_UNIT",
            object_id=f"unit_{unit}",
            object_label="Version",
            properties={"time_unit": unit},
            source_file=source,
            extraction_tier=1,
            confidence=0.99,
        ))

    return triples


def _extract_flow_triples(content: str, source: str) -> list[Triple]:
    """Extract ORFS flow configuration triples."""
    triples = []
    
    designs = ORFS_DESIGN_RE.findall(content)
    platforms = ORFS_PLATFORM_RE.findall(content)
    
    for design in designs:
        triples.append(Triple(
            subject_id=f"design_{design}",
            subject_label="Design",
            predicate="TARGETS",
            object_id=f"pdk_{platforms[0]}" if platforms else "pdk_unknown",
            object_label="PDK",
            properties={"design_name": design},
            source_file=source,
            extraction_tier=1,
            confidence=0.95,
        ))

    return triples


def _extract_rtl_triples(content: str, source: str) -> list[Triple]:
    """Extract module hierarchy triples from Verilog."""
    triples = []
    
    modules = VERILOG_MODULE_RE.findall(content)
    for mod in modules[:10]:  # Cap to avoid huge files
        triples.append(Triple(
            subject_id=f"module_{mod}",
            subject_label="Module",
            predicate="DEFINED_IN",
            object_id=f"file_{source}",
            object_label="Report",
            properties={"module_name": mod},
            source_file=source,
            extraction_tier=1,
            confidence=0.98,
        ))

    return triples


def _extract_tool_versions(content: str, source: str) -> list[Triple]:
    """Extract tool version triples from any file."""
    triples = []
    seen = set()
    
    for m in TOOL_VERSION_RE.finditer(content):
        tool = m.group(0).split()[0]
        version = m.group(1)
        key = (tool, version)
        if key in seen:
            continue
        seen.add(key)
        
        triples.append(Triple(
            subject_id=f"tool_{tool.lower()}_{version}",
            subject_label="Version",
            predicate="VERSION_OF",
            object_id=f"tool_{tool.lower()}",
            object_label="ToolRun",
            properties={"tool_name": tool, "tool_version": version},
            source_file=source,
            extraction_tier=1,
            confidence=0.97,
        ))

    return triples


# ═══════════════════════════════════════════════════════════════════════════════
# TIER 2: NER-based extraction (semi-structured text)
# ═══════════════════════════════════════════════════════════════════════════════

def extract_ner_triples(content: str, source: str) -> list[Triple]:
    """Tier 2: Extract triples using pattern-based NER for EDA entities.
    
    Identifies module names, net names, cell names, and their relationships
    without requiring a full NLP model (uses EDA-specific heuristics).
    """
    triples = []
    
    # EDA-specific entity patterns (more contextual than pure regex)
    # Cell library references
    cell_re = re.compile(r"\b(sky130_fd_sc_\w+|asap7sc\w+|NAND\d+X\d+|INV\w+|BUF\w+|DFF\w+)\b")
    for m in cell_re.finditer(content):
        cell = m.group(1)
        triples.append(Triple(
            subject_id=f"cell_{cell}",
            subject_label="Module",
            predicate="INSTANCE_OF",
            object_id="library_standard_cells",
            object_label="PDK",
            properties={"cell_name": cell},
            source_file=source,
            extraction_tier=2,
            confidence=0.85,
        ))
    
    # Net names with direction indicators
    net_re = re.compile(r"\b(clk|reset|rst|data_in|data_out|valid|ready|enable)\b", re.IGNORECASE)
    # Only extract from structured contexts (port maps, signal lists)
    if "port" in content.lower() or "signal" in content.lower() or ".v" in source:
        for m in net_re.finditer(content[:5000]):  # First 5K only
            net = m.group(1)
            triples.append(Triple(
                subject_id=f"net_{net}_{hash(source) % 10000}",
                subject_label="Module",
                predicate="HAS_PORT",
                object_id=f"port_{net}",
                object_label="Module",
                properties={"net_name": net},
                source_file=source,
                extraction_tier=2,
                confidence=0.75,
            ))

    return triples


# ═══════════════════════════════════════════════════════════════════════════════
# TIER 3: LLM extraction queue (forum/documentation content)
# ═══════════════════════════════════════════════════════════════════════════════

def should_use_llm(filepath: Path, content: str, regex_triples: list) -> bool:
    """Determine if this document needs LLM extraction.
    
    Criteria:
    - Source is forum post or documentation (not structured report)
    - Regex coverage is low (< 60% of content matched)
    - Content has enough substance (> 200 chars)
    """
    fname = filepath.name.lower()
    source_type = "structured"
    
    if any(x in str(filepath).lower() for x in ["forum", "discussion", "issue", "qa_pair"]):
        source_type = "forum"
    elif any(x in fname for x in ["readme", "doc", "guide", "tutorial", ".md", ".rst"]):
        source_type = "documentation"
    
    if source_type == "structured":
        return False
    
    if len(content) < 200:
        return False
    
    # Calculate regex coverage
    matched_chars = sum(
        len(t.properties.get("evidence", "")) for t in regex_triples
    ) if regex_triples else 0
    coverage = matched_chars / len(content) if len(content) > 0 else 1.0
    
    return coverage < 0.60


def build_llm_extraction_request(filepath: Path, content: str) -> dict:
    """Build a request for the LLM extraction batch queue."""
    return {
        "custom_id": f"extract_{hash(str(filepath)) % 1_000_000}",
        "source_file": str(filepath),
        "content_preview": content[:3000],
        "prompt": f"""Extract knowledge graph triples from this EDA forum/documentation content.

Content:
{content[:3000]}

Extract triples in this JSON format:
[
  {{
    "subject": "entity name",
    "subject_label": "one of: Design, Module, Violation, Rule, TimingPath, ToolRun, Report, Version, Fix, PDK",
    "predicate": "one of: CAUSES, VIOLATES, FIXES, DEPENDS_ON, EQUIVALENT_TO, DOCUMENTED_IN, DIVERGES_FROM, INCOMPATIBLE_WITH, CONTAINS, PRODUCED_BY, TARGETS",
    "object": "entity name",
    "object_label": "node label",
    "confidence": 0.0-1.0
  }}
]

Only extract triples you are confident about. Prefer precision over recall.
Respond with valid JSON array only.""",
    }


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════

def process_file(filepath) -> tuple[list["Triple"], Optional[dict]]:
    """Process a single file through the tiered extraction pipeline.
    
    Returns:
        (triples, llm_request) — triples from Tier 1+2, and optionally a Tier 3 queue item
    """
    filepath = Path(filepath) if not isinstance(filepath, Path) else filepath
    try:
        content = filepath.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return [], None
    
    if len(content) < 50:
        return [], None

    # Tier 1: Regex
    triples = extract_regex_triples(filepath, content)
    
    # Tier 2: NER (on residual)
    ner_triples = extract_ner_triples(content, str(filepath))
    triples.extend(ner_triples)
    
    # Tier 3: Queue for LLM if needed
    llm_request = None
    if should_use_llm(filepath, content, triples):
        llm_request = build_llm_extraction_request(filepath, content)
    
    return triples, llm_request


def main():
    parser = argparse.ArgumentParser(description="Tiered triple extraction pipeline")
    parser.add_argument("--input", help="Input directory (deduped corpus)")
    parser.add_argument("--output", default="data/graph/triples_raw.jsonl",
                        help="Output JSONL file for extracted triples")
    parser.add_argument("--llm-queue", default="data/graph/llm_queue.jsonl",
                        help="Output queue file for Tier 3 LLM extraction")
    parser.add_argument("--merge-llm", help="Merge LLM extraction results into output")
    parser.add_argument("--max-files", type=int, default=0,
                        help="Process only first N files (0 = all)")
    args = parser.parse_args()

    if args.merge_llm:
        merge_llm_results(Path(args.merge_llm), Path(args.output))
        return

    if not args.input:
        print("ERROR: --input required")
        sys.exit(1)

    input_dir = Path(args.input)
    output_path = Path(args.output)
    llm_queue_path = Path(args.llm_queue)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    llm_queue_path.parent.mkdir(parents=True, exist_ok=True)

    # Collect files
    files = sorted(input_dir.rglob("*"))
    files = [f for f in files if f.is_file() and f.stat().st_size < 512_000]
    
    if args.max_files > 0:
        files = files[:args.max_files]

    print(f"Processing {len(files)} files from {input_dir}", flush=True)
    print(f"Output: {output_path}", flush=True)
    print(f"LLM queue: {llm_queue_path}", flush=True)

    total_triples = 0
    llm_queued = 0
    tier_counts = {1: 0, 2: 0}

    with open(output_path, "w", encoding="utf-8") as out_f, \
         open(llm_queue_path, "w", encoding="utf-8") as llm_f:
        
        for i, filepath in enumerate(files):
            triples, llm_request = process_file(filepath)
            
            for t in triples:
                out_f.write(json.dumps(asdict(t), ensure_ascii=False) + "\n")
                tier_counts[t.extraction_tier] = tier_counts.get(t.extraction_tier, 0) + 1
            
            total_triples += len(triples)
            
            if llm_request:
                llm_f.write(json.dumps(llm_request, ensure_ascii=False) + "\n")
                llm_queued += 1
            
            if (i + 1) % 500 == 0:
                print(f"  Processed {i+1}/{len(files)} files | "
                      f"Triples: {total_triples:,} | LLM queued: {llm_queued}",
                      flush=True)

    print(f"\n{'='*60}", flush=True)
    print(f"TRIPLE EXTRACTION SUMMARY", flush=True)
    print(f"{'='*60}", flush=True)
    print(f"Files processed:    {len(files):,}", flush=True)
    print(f"Total triples:      {total_triples:,}", flush=True)
    print(f"  Tier 1 (regex):   {tier_counts.get(1, 0):,}", flush=True)
    print(f"  Tier 2 (NER):     {tier_counts.get(2, 0):,}", flush=True)
    print(f"  Tier 3 (LLM):     queued {llm_queued:,} documents", flush=True)
    print(f"Output:             {output_path}", flush=True)
    print(f"LLM queue:          {llm_queue_path}", flush=True)
    print(f"{'='*60}", flush=True)

    if llm_queued > 0:
        print(f"\nNext step: Submit LLM queue to Anthropic Batch API:", flush=True)
        print(f"  python -m pipeline.graph.submit_llm_extraction --input {llm_queue_path}", flush=True)


def merge_llm_results(llm_results_path: Path, output_path: Path):
    """Merge LLM extraction results into the main triples file."""
    new_triples = 0
    
    with open(output_path, "a", encoding="utf-8") as out_f:
        with open(llm_results_path, "r", encoding="utf-8") as llm_f:
            for line in llm_f:
                if not line.strip():
                    continue
                result = json.loads(line)
                source = result.get("source_file", "")
                
                for t in result.get("triples", []):
                    triple = Triple(
                        subject_id=f"llm_{t['subject'].lower().replace(' ', '_')}",
                        subject_label=t.get("subject_label", "Module"),
                        predicate=t.get("predicate", "RELATED_TO"),
                        object_id=f"llm_{t['object'].lower().replace(' ', '_')}",
                        object_label=t.get("object_label", "Module"),
                        properties={},
                        source_file=source,
                        extraction_tier=3,
                        confidence=t.get("confidence", 0.70),
                    )
                    out_f.write(json.dumps(asdict(triple), ensure_ascii=False) + "\n")
                    new_triples += 1
    
    print(f"Merged {new_triples:,} LLM-extracted triples into {output_path}", flush=True)


if __name__ == "__main__":
    main()
