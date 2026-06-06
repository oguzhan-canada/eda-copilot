// ─────────────────────────────────────────────────────────────────────────────
// graph_schema.cypher — EDA Knowledge Graph Ontology
// ─────────────────────────────────────────────────────────────────────────────
// Idempotent schema migration. Run on every deployment.
// Matches the paper §5.1 ER diagram.
//
// Node labels: Design, Module, Violation, Rule, TimingPath, ToolRun,
//              Report, Version, Fix, PDK
//
// Relationship types: CAUSES, VIOLATES, FIXES, DEPENDS_ON, EQUIVALENT_TO,
//                     DOCUMENTED_IN, DIVERGES_FROM, INCOMPATIBLE_WITH,
//                     CONTAINS, PRODUCED_BY, TARGETS

// ── Uniqueness constraints ───────────────────────────────────────────────────
CREATE CONSTRAINT design_id IF NOT EXISTS
  FOR (d:Design) REQUIRE d.id IS UNIQUE;

CREATE CONSTRAINT module_id IF NOT EXISTS
  FOR (m:Module) REQUIRE m.id IS UNIQUE;

CREATE CONSTRAINT violation_id IF NOT EXISTS
  FOR (v:Violation) REQUIRE v.id IS UNIQUE;

CREATE CONSTRAINT rule_id IF NOT EXISTS
  FOR (r:Rule) REQUIRE r.id IS UNIQUE;

CREATE CONSTRAINT timing_path_id IF NOT EXISTS
  FOR (tp:TimingPath) REQUIRE tp.id IS UNIQUE;

CREATE CONSTRAINT tool_run_id IF NOT EXISTS
  FOR (tr:ToolRun) REQUIRE tr.id IS UNIQUE;

CREATE CONSTRAINT report_id IF NOT EXISTS
  FOR (rp:Report) REQUIRE rp.id IS UNIQUE;

CREATE CONSTRAINT version_id IF NOT EXISTS
  FOR (v:Version) REQUIRE v.id IS UNIQUE;

CREATE CONSTRAINT fix_id IF NOT EXISTS
  FOR (f:Fix) REQUIRE f.id IS UNIQUE;

CREATE CONSTRAINT pdk_id IF NOT EXISTS
  FOR (p:PDK) REQUIRE p.id IS UNIQUE;

// ── Indexes for retrieval performance ────────────────────────────────────────
CREATE INDEX violation_error_code IF NOT EXISTS
  FOR (v:Violation) ON (v.error_code);

CREATE INDEX rule_name IF NOT EXISTS
  FOR (r:Rule) ON (r.name);

CREATE INDEX version_tag IF NOT EXISTS
  FOR (v:Version) ON (v.version_tag);

CREATE INDEX tool_run_tool_version IF NOT EXISTS
  FOR (tr:ToolRun) ON (tr.tool_name, tr.tool_version);

CREATE INDEX design_name IF NOT EXISTS
  FOR (d:Design) ON (d.name);

CREATE INDEX timing_path_slack IF NOT EXISTS
  FOR (tp:TimingPath) ON (tp.slack);

// ── Property existence constraints ───────────────────────────────────────────
// Version nodes must carry provenance
CREATE CONSTRAINT version_provenance IF NOT EXISTS
  FOR (v:Version) REQUIRE v.version_tag IS NOT NULL;

CREATE CONSTRAINT version_source IF NOT EXISTS
  FOR (v:Version) REQUIRE v.source_tool IS NOT NULL;

CREATE CONSTRAINT version_date IF NOT EXISTS
  FOR (v:Version) REQUIRE v.capture_date IS NOT NULL;

// ── Seed data: 4 verified ORFS bug triples (ground truth) ───────────────────
// ED-001: SDC clock period override
MERGE (v1:Violation {id: 'ed_001_sdc_override'})
  SET v1.error_code = 'SDC_CLOCK_OVERRIDE',
      v1.description = 'FLOW_VARIANT does not override SDC clock period; all sweep runs produce identical PPA',
      v1.severity = 'hard',
      v1.source = 'MLCAD_ED001'
MERGE (f1:Fix {id: 'variant_sdc_override'})
  SET f1.description = 'Create per-variant SDC files explicitly via SDC_FILE=<variant.sdc>; regex-replace set clk_period at sweep launch',
      f1.source = 'MLCAD_ED001'
MERGE (f1)-[:FIXES]->(v1);

// ED-002: ORFS version divergence + WNS sign flip
MERGE (v2:Violation {id: 'ed_002_version_divergence'})
  SET v2.error_code = 'ORFS_VERSION_DIVERGENCE',
      v2.description = 'ORFS v3.0→26Q1 migration produces >10% PPA divergence; JPEG WNS sign flips from +13.7ps to -12.8ps',
      v2.severity = 'hard',
      v2.source = 'MLCAD_ED002'
MERGE (f2:Fix {id: 'version_tag_training_data'})
  SET f2.description = 'Version-tag all training data; re-validate Pareto candidates on target tool version',
      f2.source = 'MLCAD_ED002'
MERGE (f2)-[:FIXES]->(v2)
MERGE (ver_v3:Version {id: 'orfs_v3_0'})
  SET ver_v3.version_tag = 'ORFS_v3.0', ver_v3.source_tool = 'OpenROAD', ver_v3.capture_date = '2025-01-01'
MERGE (ver_26q1:Version {id: 'orfs_26q1'})
  SET ver_26q1.version_tag = 'ORFS_26Q1', ver_26q1.source_tool = 'OpenROAD', ver_26q1.capture_date = '2025-04-01'
MERGE (ver_v3)-[:DIVERGES_FROM]->(ver_26q1);

// ED-003: SDC time-unit mismatch
MERGE (v3:Violation {id: 'ed_003_unit_mismatch'})
  SET v3.error_code = 'SDC_UNIT_MISMATCH',
      v3.description = 'SDC written in ps, tool expects ns; produces implausible WNS values (e.g., +1244 ps interpreted as +1244 ns)',
      v3.severity = 'medium',
      v3.source = 'MLCAD_ED003'
MERGE (f3:Fix {id: 'prepend_unit_declaration'})
  SET f3.description = 'Prepend set_units -time ns directive to all SDC files',
      f3.source = 'MLCAD_ED003'
MERGE (f3)-[:FIXES]->(v3);

// ED-004: CircuitNet DEF naming mismatch
MERGE (v4:Violation {id: 'ed_004_def_naming'})
  SET v4.error_code = 'DEF_INSTANCE_NAMING',
      v4.description = 'CircuitNet DEF instance names do not match OpenROAD expectations; causes placement extraction failures',
      v4.severity = 'medium',
      v4.source = 'MLCAD_ED004'
MERGE (f4:Fix {id: 'automated_name_normalization'})
  SET f4.description = 'Run fix_def_instances.py automated name normalization pipeline before any OpenROAD run on CircuitNet data',
      f4.source = 'MLCAD_ED004'
MERGE (f4)-[:FIXES]->(v4);

// ED-005: OpenROAD 26Q1 SIGSEGV in ibex global routing
MERGE (v5:Violation {id: 'ed_005_ibex_26q1_sigsegv'})
  SET v5.error_code = 'OPENROAD_SIGSEGV_GRT',
      v5.description = 'OpenROAD 26Q1-2900-gdf79404cd8 crashes with SIGSEGV during global routing (do-5_1_grt) on ibex design with OPENROAD_HIERARCHICAL=1. Peak memory 717MB (not OOM). Crash is in _start of openroad binary.',
      v5.severity = 'high',
      v5.source = 'ORFS_sweep_2026'
MERGE (ver26q1:Version {id: 'orfs_26q1'})
MERGE (run_ibex:ToolRun {id: 'orfs_ibex_26q1'})
  SET run_ibex.design = 'ibex',
      run_ibex.platform = 'asap7',
      run_ibex.exit_stage = '5_1_grt',
      run_ibex.signal = 'SIGSEGV',
      run_ibex.peak_memory_kb = 717240
MERGE (ver26q1)-[:HAS_BUG]->(v5)
MERGE (run_ibex)-[:CAUSES]->(v5)
MERGE (run_ibex)-[:RAN_ON]->(ver26q1);
