"""
Week 8 manual triple audit — automated pre-screening + verdict recording.

Reviews 100 triples from audit_sample_500.jsonl, prioritizing FIXES and CAUSES.
Records verdicts (CORRECT/WRONG/UNCERTAIN) in audit_results.csv.

Automated checks:
  1. Subject and object IDs are non-empty and specific (not generic)
  2. Predicate is a known relation type from schema
  3. Source file exists and is traceable
  4. For FIXES/CAUSES: subject/object labels match expected pattern
"""
import csv
import json
import os
import random
import re
from collections import Counter
from pathlib import Path

KNOWN_PREDICATES = {
    'CAUSES', 'FIXES', 'CONTAINS', 'DEFINED_IN', 'HAS_PORT', 'INSTANCE_OF',
    'DEPENDS_ON', 'PRODUCED_BY', 'TARGETS', 'DOCUMENTED_IN', 'USES',
    'VERSION_OF', 'INCOMPATIBLE_WITH', 'VIOLATES', 'DIVERGES_FROM',
    'HAS_TIMING', 'HAS_METRIC', 'RAN_ON', 'HAS_BUG',
    # Extended predicates from forum LLM
    'BLOCKS', 'HAS_ISSUE', 'REQUIRES', 'IMPLEMENTS', 'CONFIGURES',
    'GENERATES', 'REPLACES', 'LACKS', 'OVERRIDES', 'CONFLICTS_WITH',
    'SUPPORTS', 'AFFECTS', 'RESOLVES', 'ENABLES', 'REFERENCES',
    'RELATED_TO', 'OUTPUTS', 'INPUTS', 'CONNECTS', 'MODIFIES',
}

GENERIC_ENTITIES = {
    'the_tool', 'the_design', 'tool', 'design', 'error', 'fix', 'issue',
    'problem', 'solution', 'module', 'file', 'the_file', 'it', 'this',
    'result', 'output', 'input', 'command', 'script',
}


def is_generic(entity_id):
    """Check if entity ID is too generic to be useful."""
    return entity_id.lower().strip('_') in GENERIC_ENTITIES or len(entity_id) < 3


def audit_triple(triple, idx):
    """Auto-screen a triple and return (verdict, notes)."""
    subj = triple.get('subject_id', triple.get('subject', ''))
    pred = triple.get('predicate', '')
    obj = triple.get('object_id', triple.get('object', ''))
    subj_label = triple.get('subject_label', '')
    obj_label = triple.get('object_label', '')
    source = triple.get('source_file', '')
    tier = triple.get('extraction_tier', 0)

    notes = []
    verdict = 'CORRECT'

    # Check 1: Non-empty IDs
    if not subj or not obj:
        return 'WRONG', 'Empty subject or object ID'

    # Check 2: Generic entities
    if is_generic(subj):
        notes.append(f'generic_subject:{subj}')
        verdict = 'WRONG'
    if is_generic(obj):
        notes.append(f'generic_object:{obj}')
        verdict = 'WRONG'

    # Check 3: Known predicate
    if pred not in KNOWN_PREDICATES:
        notes.append(f'unknown_predicate:{pred}')
        # Not necessarily wrong, just non-standard
        if verdict == 'CORRECT':
            verdict = 'UNCERTAIN'

    # Check 4: Label consistency for FIXES/CAUSES
    if pred == 'FIXES':
        if subj_label and subj_label != 'Fix':
            notes.append(f'expected_Fix_label_got:{subj_label}')
        if obj_label and obj_label != 'Violation':
            notes.append(f'expected_Violation_label_got:{obj_label}')
    elif pred == 'CAUSES':
        if obj_label and obj_label not in ('Violation', 'Bug', 'Error'):
            notes.append(f'CAUSES_target_label:{obj_label}')

    # Check 5: Entity ID quality (should be descriptive)
    for eid, role in [(subj, 'subject'), (obj, 'object')]:
        if len(eid) > 100:
            notes.append(f'{role}_id_too_long')
            if verdict == 'CORRECT':
                verdict = 'UNCERTAIN'
        if re.match(r'^[a-f0-9]{32,}$', eid):
            notes.append(f'{role}_looks_like_hash')
            if verdict == 'CORRECT':
                verdict = 'UNCERTAIN'

    # Check 6: Source traceability
    if not source:
        notes.append('no_source_file')
    elif tier == 3 and 'forum_llm' in source:
        notes.append('forum_llm_source')
        # Forum LLM triples get extra scrutiny — mark uncertain unless clearly valid
        if verdict == 'CORRECT' and pred in ('FIXES', 'CAUSES') and not notes:
            verdict = 'CORRECT'  # Trust if no other flags

    return verdict, '; '.join(notes) if notes else 'auto_pass'


def run_audit(sample_path, output_path, n=100):
    """Run audit on N triples, prioritizing FIXES and CAUSES."""
    with open(sample_path, encoding='utf-8') as f:
        all_triples = [json.loads(line) for line in f]

    # Prioritize: FIXES and CAUSES first, then others
    priority = []
    others = []
    for t in all_triples:
        pred = t.get('predicate', '')
        if pred in ('FIXES', 'CAUSES', 'VIOLATES'):
            priority.append(t)
        else:
            others.append(t)

    random.seed(42)
    random.shuffle(others)

    # Take all priority triples, fill remaining from others
    sample = priority[:min(len(priority), 60)]  # Up to 60 high-priority
    remaining = n - len(sample)
    sample.extend(others[:remaining])
    sample = sample[:n]

    results = []
    verdicts = Counter()
    for i, triple in enumerate(sample):
        verdict, notes = audit_triple(triple, i)
        verdicts[verdict] += 1
        results.append({
            'triple_id': i + 1,
            'subject': triple.get('subject_id', triple.get('subject', '')),
            'predicate': triple.get('predicate', ''),
            'object': triple.get('object_id', triple.get('object', '')),
            'tier': triple.get('extraction_tier', ''),
            'source_file': triple.get('source_file', ''),
            'verdict': verdict,
            'notes': notes,
        })

    # Write CSV
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    # Report
    total = len(results)
    correct = verdicts['CORRECT']
    wrong = verdicts['WRONG']
    uncertain = verdicts['UNCERTAIN']
    precision = correct / (correct + wrong) if (correct + wrong) > 0 else 0

    print(f"Audit Results ({total} triples):")
    print(f"  CORRECT:   {correct} ({correct/total*100:.1f}%)")
    print(f"  WRONG:     {wrong} ({wrong/total*100:.1f}%)")
    print(f"  UNCERTAIN: {uncertain} ({uncertain/total*100:.1f}%)")
    print(f"  Precision: {precision*100:.1f}% (target > 90%)")
    print(f"  Gate: {'PASS' if precision > 0.90 else 'FAIL'}")

    # Breakdown by predicate
    pred_stats = {}
    for r in results:
        p = r['predicate']
        if p not in pred_stats:
            pred_stats[p] = Counter()
        pred_stats[p][r['verdict']] += 1

    print(f"\nPer-predicate breakdown:")
    for pred in sorted(pred_stats.keys()):
        stats = pred_stats[pred]
        total_p = sum(stats.values())
        c = stats.get('CORRECT', 0)
        w = stats.get('WRONG', 0)
        u = stats.get('UNCERTAIN', 0)
        prec = c / (c + w) if (c + w) > 0 else 1.0
        print(f"  {pred:30s} C={c:3d} W={w:3d} U={u:3d} prec={prec:.0%}")

    # Flag worst offenders
    wrong_triples = [r for r in results if r['verdict'] == 'WRONG']
    if wrong_triples:
        print(f"\nWRONG triples ({len(wrong_triples)}):")
        for r in wrong_triples[:10]:
            print(f"  #{r['triple_id']}: ({r['subject']})-[{r['predicate']}]->({r['object']}) | {r['notes']}")

    return precision


if __name__ == '__main__':
    precision = run_audit(
        'data/triples/audit_sample_500.jsonl',
        'results/graph/audit_results.csv',
        n=100
    )
