"""Week 8 schema fixes: Design names, Metric values, Version dedup."""
from neo4j import GraphDatabase
import json
import re

URI = os.environ.get('NEO4J_URI', '')
AUTH = (os.environ.get('NEO4J_USER', ''), os.environ.get('NEO4J_PASSWORD', ''))
CANONICAL_DESIGNS = ['aes', 'ibex', 'jpeg', 'riscv32i', 'swerv_wrapper', 'gcd']


def fix_design_names(driver):
    """Fix 1: Add name property to all Design nodes."""
    print("=== FIX 1: Design node name property ===")
    with driver.session() as s:
        designs = s.run(
            'MATCH (d:Design) WHERE d.name IS NULL RETURN d.id AS did'
        ).data()
        print(f"Design nodes without name: {len(designs)}")

        fixed = 0
        for d in designs:
            did = d['did']
            name = did.replace('design_', '', 1) if did.startswith('design_') else did
            matched = None
            for canon in CANONICAL_DESIGNS:
                if name == canon or name.startswith(canon + '_') or canon in name:
                    matched = canon
                    break
            set_name = matched if matched else name
            s.run('MATCH (d:Design {id: $did}) SET d.name = $name',
                  did=did, name=set_name)
            fixed += 1

        with_name = s.run(
            'MATCH (d:Design) WHERE d.name IS NOT NULL RETURN count(d) AS c'
        ).single()['c']
        total = s.run('MATCH (d:Design) RETURN count(d) AS c').single()['c']
        print(f"Fixed: {fixed}, With name: {with_name}/{total}")
        for canon in CANONICAL_DESIGNS:
            cnt = s.run(
                'MATCH (d:Design) WHERE d.name = $n RETURN count(d) AS c',
                n=canon
            ).single()['c']
            print(f"  {canon}: {cnt} nodes")


def fix_metric_values(driver, orfs_data_path=None):
    """Fix 2: Add value/unit to Metric nodes from ORFS data."""
    print("\n=== FIX 2: Metric value/unit properties ===")

    # Load ORFS report data to get actual metric values
    orfs_metrics = {}
    orfs_base = r'C:\eda-kg-data\orfs\runs'
    import pathlib
    for report_path in pathlib.Path(orfs_base).rglob('6_report.json'):
        try:
            report = json.loads(report_path.read_text())
            parts = report_path.parts
            # Extract version and design from path
            for i, p in enumerate(parts):
                if p in ('v3_0', 'v3.0'):
                    version = 'ORFS_v3.0'
                    design = parts[i + 1] if i + 1 < len(parts) else None
                    break
                elif p in ('26q1', '26Q1'):
                    version = 'ORFS_26Q1'
                    design = parts[i + 1] if i + 1 < len(parts) else None
                    break
            else:
                continue

            if design:
                wns = report.get('finish__timing__setup__ws')
                tns = report.get('finish__timing__setup__tns')
                area = report.get('finish__design__instance__area')
                util = report.get('finish__design__instance__utilization')
                power = report.get('finish__power__total')

                orfs_metrics[f'metric_wns_{design}_{version}'] = (wns, 'ps')
                orfs_metrics[f'metric_tns_{design}_{version}'] = (tns, 'ps')
                orfs_metrics[f'metric_area_{design}_{version}'] = (area, 'um2')
                orfs_metrics[f'metric_util_{design}_{version}'] = (util, '%')
                if power is not None:
                    orfs_metrics[f'metric_power_{design}_{version}'] = (power, 'mW')
        except Exception as e:
            print(f"  Warning: {report_path}: {e}")

    print(f"ORFS metrics parsed: {len(orfs_metrics)}")
    for k, v in list(orfs_metrics.items())[:5]:
        print(f"  {k}: {v}")

    with driver.session() as s:
        metrics = s.run('MATCH (m:Metric) RETURN m.id AS mid').data()
        print(f"Total Metric nodes: {len(metrics)}")

        fixed = 0
        for m in metrics:
            mid = m['mid']
            if mid in orfs_metrics:
                val, unit = orfs_metrics[mid]
                if val is not None:
                    try:
                        s.run(
                            'MATCH (m:Metric {id: $mid}) '
                            'SET m.value = toFloat($val), m.unit = $unit',
                            mid=mid, val=str(val), unit=unit
                        )
                        fixed += 1
                    except Exception as e:
                        print(f"  Error setting {mid}: {e}")

        has_val = s.run(
            'MATCH (m:Metric) WHERE m.value IS NOT NULL RETURN count(m) AS c'
        ).single()['c']
        print(f"Metrics with value: {has_val}/{len(metrics)} (fixed: {fixed})")


def fix_version_dedup(driver):
    """Fix 3: Merge duplicate Version nodes."""
    print("\n=== FIX 3: Version node dedup ===")
    canonical_map = {
        'orfs_v3_0': 'version_ORFS_v3.0',
        'orfs_26q1': 'version_ORFS_26Q1',
    }

    with driver.session() as s:
        versions = s.run(
            'MATCH (v:Version) RETURN v.id AS vid, v.version_tag AS vtag'
        ).data()
        print(f"Total Version nodes: {len(versions)}")

        for alias_id, canon_id in canonical_map.items():
            alias_cnt = s.run(
                'MATCH (v:Version {id: $id}) RETURN count(v) AS c', id=alias_id
            ).single()['c']
            canon_cnt = s.run(
                'MATCH (v:Version {id: $id}) RETURN count(v) AS c', id=canon_id
            ).single()['c']
            print(f"  {alias_id}: {alias_cnt}, {canon_id}: {canon_cnt}")

            if alias_cnt > 0 and canon_cnt > 0:
                rels = s.run(
                    'MATCH (alias:Version {id: $aid})-[r]-(n) '
                    'RETURN type(r) AS rtype, '
                    '       startNode(r).id AS src, endNode(r).id AS tgt',
                    aid=alias_id
                ).data()
                print(f"    Relationships to re-point: {len(rels)}")

                for rel in rels:
                    rtype = rel['rtype']
                    if rel['src'] == alias_id:
                        s.run(
                            f'MATCH (canon:Version {{id: $cid}}), (tgt {{id: $tid}}) '
                            f'MERGE (canon)-[:{rtype}]->(tgt)',
                            cid=canon_id, tid=rel['tgt']
                        )
                    else:
                        s.run(
                            f'MATCH (src {{id: $sid}}), (canon:Version {{id: $cid}}) '
                            f'MERGE (src)-[:{rtype}]->(canon)',
                            sid=rel['src'], cid=canon_id
                        )

                s.run(
                    'MATCH (alias:Version {id: $aid}) DETACH DELETE alias',
                    aid=alias_id
                )
                print(f"    Merged {alias_id} -> {canon_id}")

        remaining = s.run('MATCH (v:Version) RETURN count(v) AS c').single()['c']
        print(f"Version nodes after dedup: {remaining}")

        for cid in ['version_ORFS_v3.0', 'version_ORFS_26Q1']:
            rel_cnt = s.run(
                'MATCH (v:Version {id: $id})-[r]-() RETURN count(r) AS c',
                id=cid
            ).single()['c']
            print(f"  {cid}: {rel_cnt} relationships")


if __name__ == '__main__':
    driver = GraphDatabase.driver(URI, auth=AUTH)
    driver.verify_connectivity()
    print(f"Connected to {URI}\n")

    fix_design_names(driver)
    fix_metric_values(driver)
    fix_version_dedup(driver)

    driver.close()
    print("\n=== ALL SCHEMA FIXES COMPLETE ===")
