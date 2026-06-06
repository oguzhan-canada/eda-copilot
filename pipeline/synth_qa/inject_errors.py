"""
Component 1.2 — Synthetic violation injection.

Reads deduped corpus files and injects synthetic EDA violations
across 10 families. Each injected case includes the violation snippet,
expected diagnosis, expected fix, and (if applicable) the seed bug ID.

Usage:
    python -m pipeline.synth_qa.inject_errors \
        --input C:/eda-kg-data/corpus/staging/dedup \
        --seeds data/edabench/seeds/mlcad_seeds.yaml \
        --output data/synthetic/injected_cases.jsonl \
        --families all \
        --max-per-source 3
"""
import argparse
import json
import random
import re
from datetime import datetime, timezone
from pathlib import Path

import yaml


# ── Violation Family Definitions ─────────────────────────────────────────────
# Each family defines: target file types, injection function, expected fields.

VIOLATION_FAMILIES = {
    "setup_violation": {
        "extensions": [".sdc", ".tcl"],
        "description": "Setup timing violation — clock period too tight for combinational depth",
        "task_category": "error_diagnosis",
    },
    "hold_violation": {
        "extensions": [".sdc", ".tcl"],
        "description": "Hold timing violation — insufficient hold margin on clock domain crossing",
        "task_category": "error_diagnosis",
    },
    "drc_spacing": {
        "extensions": [".def", ".lef"],
        "description": "DRC spacing violation — metal layer minimum spacing violated",
        "task_category": "error_diagnosis",
    },
    "drc_enclosure": {
        "extensions": [".def", ".lef"],
        "description": "DRC enclosure violation — via enclosure rule violated",
        "task_category": "error_diagnosis",
    },
    "lvs_mismatch": {
        "extensions": [".v", ".sv"],
        "description": "LVS mismatch — netlist port count differs from schematic",
        "task_category": "error_diagnosis",
    },
    "sdc_unit_mismatch": {
        "extensions": [".sdc", ".tcl"],
        "description": "SDC unit mismatch — time values in ps but tool expects ns",
        "task_category": "error_diagnosis",
        "seed_bug_id": "ED-003",
    },
    "missing_clock": {
        "extensions": [".sdc", ".tcl"],
        "description": "Missing clock definition — register paths are unconstrained",
        "task_category": "error_diagnosis",
    },
    "false_path_misuse": {
        "extensions": [".sdc", ".tcl"],
        "description": "False path misuse — valid timing paths incorrectly marked as false",
        "task_category": "constraint_generation",
    },
    "def_naming_mismatch": {
        "extensions": [".def"],
        "description": "DEF naming mismatch — instance names don't match expected convention",
        "task_category": "error_diagnosis",
        "seed_bug_id": "ED-004",
    },
    "version_drift": {
        "extensions": [".tcl", ".sdc", ".py", ".mk", ".sh"],
        "description": "Version drift — flow behavior changes between tool versions cause PPA divergence",
        "task_category": "cross_tool_knowledge",
        "seed_bug_id": "ED-001,ED-002",
    },
    "rtl_port_mismatch": {
        "extensions": [".v", ".sv"],
        "description": "RTL port/signal mismatch — module instantiation uses wrong port widths or missing connections",
        "task_category": "rtl_qa",
    },
    "ppa_knob_misconfiguration": {
        "extensions": [".tcl", ".sdc", ".mk", ".sh"],
        "description": "PPA knob misconfiguration — synthesis/P&R knobs set to suboptimal values hurting area/power/timing",
        "task_category": "optimization_advisory",
    },
}

# ── Injection Templates ──────────────────────────────────────────────────────

def inject_setup_violation(text: str, filepath: str) -> dict | None:
    """Inject a setup violation by making clock period unreasonably tight."""
    # Look for clock period definitions
    m = re.search(r'(create_clock.*?-period\s+)([\d.]+)', text)
    if not m:
        m = re.search(r'(set\s+clk_period\s+)([\d.]+)', text)
    if not m:
        return None

    orig_period = float(m.group(2))
    # Make period 10x smaller — guaranteed setup violations
    bad_period = round(orig_period / 10, 3)
    injected = text[:m.start(2)] + str(bad_period) + text[m.end(2):]

    return {
        "injected_snippet": injected[:2000],
        "original_snippet": text[:2000],
        "violation_detail": f"Clock period changed from {orig_period} to {bad_period} (10x reduction)",
        "expected_diagnosis": (
            f"The clock period is set to {bad_period}, which is 10x smaller than the original "
            f"{orig_period}. This creates setup timing violations on all register-to-register "
            f"paths because the combinational logic delay exceeds the available clock period. "
            f"The WNS will be severely negative."
        ),
        "expected_fix": (
            f"Restore the clock period to {orig_period} or a value that accommodates the "
            f"longest combinational path in the design. Run STA to verify WNS >= 0 after correction."
        ),
    }


def inject_hold_violation(text: str, filepath: str) -> dict | None:
    """Inject hold violation by removing hold margin constraints."""
    if "set_clock_uncertainty" not in text and "create_clock" in text:
        # Add a problematic hold uncertainty
        injected = text + "\n# ERROR: negative hold uncertainty forces hold violations\n"
        injected += "set_clock_uncertainty -hold -0.5 [all_clocks]\n"
        return {
            "injected_snippet": injected[:2000],
            "original_snippet": text[:2000],
            "violation_detail": "Added negative hold uncertainty (-0.5) to all clocks",
            "expected_diagnosis": (
                "A negative hold uncertainty of -0.5 is applied to all clocks. This is "
                "physically impossible — hold uncertainty must be non-negative. It forces "
                "the timing engine to report false hold violations on every path."
            ),
            "expected_fix": (
                "Remove the negative hold uncertainty. If hold margin is needed, use a "
                "positive value: `set_clock_uncertainty -hold 0.1 [all_clocks]`. Typical "
                "hold uncertainty is 0.05–0.2 ns depending on PVT variation."
            ),
        }
    return None


def inject_drc_spacing(text: str, filepath: str) -> dict | None:
    """Inject DRC spacing violation in DEF/LEF by corrupting spacing values."""
    if ".def" in filepath:
        m = re.search(r'(UNITS DISTANCE MICRONS\s+)(\d+)', text)
        if m:
            orig_units = m.group(2)
            # Change units to create apparent spacing violations
            bad_units = str(int(orig_units) // 10)
            injected = text[:m.start(2)] + bad_units + text[m.end(2):]
            return {
                "injected_snippet": injected[:2000],
                "original_snippet": text[:2000],
                "violation_detail": f"DISTANCE MICRONS changed from {orig_units} to {bad_units}",
                "expected_diagnosis": (
                    f"The DEF distance unit was changed from {orig_units} to {bad_units}, "
                    f"scaling all coordinates by 10x. This makes metal tracks appear 10x "
                    f"closer together, triggering spacing DRC violations across the entire design."
                ),
                "expected_fix": (
                    f"Restore `UNITS DISTANCE MICRONS {orig_units}` to match the technology "
                    f"library scale. All placement and routing coordinates are defined relative "
                    f"to this unit factor."
                ),
            }
    elif ".lef" in filepath:
        m = re.search(r'(SPACING\s+)([\d.]+)', text)
        if m:
            orig_spacing = m.group(2)
            bad_spacing = str(round(float(orig_spacing) * 5, 4))
            injected = text[:m.start(2)] + bad_spacing + text[m.end(2):]
            return {
                "injected_snippet": injected[:2000],
                "original_snippet": text[:2000],
                "violation_detail": f"SPACING changed from {orig_spacing} to {bad_spacing}",
                "expected_diagnosis": (
                    f"The LEF SPACING rule was changed from {orig_spacing} to {bad_spacing} (5x). "
                    f"This creates an artificially tight spacing constraint that existing routes "
                    f"cannot satisfy, causing widespread spacing DRC violations."
                ),
                "expected_fix": (
                    f"Restore the SPACING value to {orig_spacing} to match the PDK design rules. "
                    f"Verify against the foundry DRM document."
                ),
            }
    return None


def inject_drc_enclosure(text: str, filepath: str) -> dict | None:
    """Inject DRC enclosure violation by modifying via enclosure rules."""
    m = re.search(r'(ENCLOSURE\s+\S+\s+)([\d.]+)(\s+)([\d.]+)', text)
    if not m:
        m = re.search(r'(WIDTH\s+)([\d.]+)', text)
    if not m:
        return None

    orig_val = m.group(2)
    bad_val = str(round(float(orig_val) * 3, 4))
    injected = text[:m.start(2)] + bad_val + text[m.end(2):]

    return {
        "injected_snippet": injected[:2000],
        "original_snippet": text[:2000],
        "violation_detail": f"Enclosure/width value changed from {orig_val} to {bad_val}",
        "expected_diagnosis": (
            f"The enclosure or width value was changed from {orig_val} to {bad_val} (3x increase). "
            f"This creates enclosure DRC violations because existing vias no longer meet the "
            f"inflated minimum enclosure requirement."
        ),
        "expected_fix": (
            f"Restore the value to {orig_val}. Cross-reference with the PDK via definition "
            f"rules to ensure consistency between LEF and technology files."
        ),
    }


def inject_lvs_mismatch(text: str, filepath: str) -> dict | None:
    """Inject LVS mismatch by adding an extra port to a module."""
    m = re.search(r'(module\s+\w+\s*\()(.*?)(\))', text, re.DOTALL)
    if not m:
        return None

    ports = m.group(2)
    # Add a phantom port
    injected = text[:m.end(2)] + ", phantom_debug_port" + text[m.end(2):]

    return {
        "injected_snippet": injected[:2000],
        "original_snippet": text[:2000],
        "violation_detail": "Added phantom_debug_port to module port list",
        "expected_diagnosis": (
            "The module has an extra port `phantom_debug_port` that does not exist in the "
            "schematic or the instantiation hierarchy. This causes an LVS port count mismatch — "
            "the netlist has N+1 ports while the schematic expects N."
        ),
        "expected_fix": (
            "Remove `phantom_debug_port` from the module port list. If this was intended as a "
            "debug port, it must also be added to the schematic and all parent instantiations."
        ),
    }


def inject_sdc_unit_mismatch(text: str, filepath: str) -> dict | None:
    """Inject SDC unit mismatch — ps values where ns expected (ED-003 pattern)."""
    m = re.search(r'(create_clock.*?-period\s+)([\d.]+)', text)
    if not m:
        m = re.search(r'(set\s+clk_period\s+)([\d.]+)', text)
    if not m:
        return None

    orig_period = float(m.group(2))
    # Multiply by 1000 to simulate ps-instead-of-ns error
    ps_period = round(orig_period * 1000, 1)
    injected = text[:m.start(2)] + str(ps_period) + text[m.end(2):]

    return {
        "injected_snippet": injected[:2000],
        "original_snippet": text[:2000],
        "violation_detail": f"Clock period changed from {orig_period} to {ps_period} (ps/ns confusion)",
        "expected_diagnosis": (
            f"The clock period is set to {ps_period}, which appears to be in picoseconds "
            f"but the tool interprets it as nanoseconds. This results in an implausibly long "
            f"clock period ({ps_period} ns = {ps_period/1e3:.1f} us), making all timing paths "
            f"appear to have enormous positive slack (WNS >> 0). The design appears to meet "
            f"timing trivially, masking real violations."
        ),
        "expected_fix": (
            f"Either add `set_units -time ns` at the top of the SDC file and use the value "
            f"{orig_period}, or convert the period to nanoseconds: {orig_period} ns. "
            f"Verify that WNS is realistic after correction."
        ),
    }


def inject_missing_clock(text: str, filepath: str) -> dict | None:
    """Inject missing clock by commenting out create_clock statements."""
    if "create_clock" not in text:
        return None

    injected = re.sub(
        r'^(\s*create_clock\b.*?)$',
        r'# REMOVED: \1',
        text,
        count=1,
        flags=re.MULTILINE,
    )
    if injected == text:
        return None

    return {
        "injected_snippet": injected[:2000],
        "original_snippet": text[:2000],
        "violation_detail": "Commented out create_clock statement",
        "expected_diagnosis": (
            "The primary clock definition (create_clock) has been removed or commented out. "
            "Without a clock definition, all register-to-register paths are unconstrained. "
            "The timing engine will not report setup or hold violations, giving a false "
            "impression that the design meets timing."
        ),
        "expected_fix": (
            "Restore the create_clock statement. Every synchronous design must have at least "
            "one clock defined. Run `report_checks -unconstrained` to verify all register "
            "paths are properly constrained after restoration."
        ),
    }


def inject_false_path_misuse(text: str, filepath: str) -> dict | None:
    """Inject false path on a real data path, masking timing violations."""
    if "create_clock" not in text:
        return None

    # Add a blanket false_path that disables all timing
    injected = text + (
        "\n# ERROR: blanket false path disables all timing analysis\n"
        "set_false_path -from [all_inputs] -to [all_outputs]\n"
        "set_false_path -from [all_registers] -to [all_registers]\n"
    )

    return {
        "injected_snippet": injected[:2000],
        "original_snippet": text[:2000],
        "violation_detail": "Added blanket set_false_path on all input-to-output and register-to-register paths",
        "expected_diagnosis": (
            "Two set_false_path constraints disable all timing analysis: one marks all "
            "input-to-output paths as false, and another marks all register-to-register "
            "paths as false. This means NO timing checks are performed — the design will "
            "always appear to meet timing regardless of actual violations."
        ),
        "expected_fix": (
            "Remove the blanket false_path constraints. False paths should only be applied "
            "to genuinely asynchronous paths (e.g., between independent clock domains with "
            "proper synchronizers). Use `set_false_path -from [get_clocks clkA] -to "
            "[get_clocks clkB]` for specific CDC paths only."
        ),
    }


def inject_def_naming_mismatch(text: str, filepath: str) -> dict | None:
    """Inject DEF naming mismatch — instance names with wrong convention (ED-004)."""
    if "COMPONENTS" not in text:
        # Try simpler DEF structure
        m = re.search(r'(DESIGN\s+)(\w+)', text)
        if not m:
            return None
        orig_name = m.group(2)
        bad_name = f"circuitnet_{orig_name}_v2"
        injected = text[:m.start(2)] + bad_name + text[m.end(2):]
    else:
        # Rename instances with CircuitNet-style prefix
        injected = re.sub(
            r'\b(PLACED|FIXED)\b',
            r'\1',
            text,
        )
        # Add CircuitNet-style prefix to instance names after '-'
        injected = re.sub(
            r'^(\s*-\s+)(\w+)',
            lambda m: m.group(1) + "cn_" + m.group(2),
            injected,
            count=5,
            flags=re.MULTILINE,
        )

    return {
        "injected_snippet": injected[:2000],
        "original_snippet": text[:2000],
        "violation_detail": "Instance/design names changed to CircuitNet-incompatible convention",
        "expected_diagnosis": (
            "The DEF file uses instance names with a 'cn_' prefix (CircuitNet convention) "
            "that OpenROAD does not expect. When OpenROAD performs instance lookup during "
            "placement extraction, it cannot find the expected instance names, causing "
            "name-resolution failures."
        ),
        "expected_fix": (
            "Run a name normalization pipeline on the DEF file to strip or remap the "
            "CircuitNet-style prefixes to match OpenROAD's expected naming convention. "
            "Tools like `normalize_def_instances.py` can automate this transformation."
        ),
    }


def inject_version_drift(text: str, filepath: str) -> dict | None:
    """Inject version drift scenario — hardcoded version references (ED-001/ED-002)."""
    # Look for any version-like references
    m = re.search(r'(v3\.0|v3_0|version.*3)', text, re.IGNORECASE)
    if not m:
        # Try to inject a version comment as context
        if any(kw in text.lower() for kw in ["flow", "orfs", "openroad", "make"]):
            injected = (
                "# WARNING: This script was validated against ORFS v3.0 only.\n"
                "# After upgrading to ORFS 26Q1, PPA metrics diverged by >10%.\n"
                "# The JPEG benchmark WNS flipped from +13.7 ps to -12.8 ps.\n"
                "# FLOW_VARIANT does not override SDC clock period.\n"
                + text
            )
            return {
                "injected_snippet": injected[:2000],
                "original_snippet": text[:2000],
                "violation_detail": "Added version drift warning — ORFS v3.0 to 26Q1 migration",
                "expected_diagnosis": (
                    "The script/config was validated only against ORFS v3.0. After migration "
                    "to ORFS 26Q1, internal flow behavior changed (detailed routing, timing "
                    "engine), causing >10% PPA divergence. The JPEG WNS sign flip (+13.7 to "
                    "-12.8 ps) indicates a design that met timing now violates it. Additionally, "
                    "FLOW_VARIANT does not override the SDC clock period — sweeps across variants "
                    "produce identical PPA because the hardcoded SDC value dominates."
                ),
                "expected_fix": (
                    "1) Version-tag all data/configs with the ORFS version used. "
                    "2) Re-validate Pareto candidates on the target tool version. "
                    "3) For FLOW_VARIANT sweeps, use per-variant SDC files with explicit "
                    "clock period values via `SDC_FILE=<variant.sdc>`. "
                    "4) Retrain or recalibrate surrogate models on version-matched data."
                ),
            }
    else:
        old_ver = m.group(0)
        injected = text.replace(old_ver, "26Q1", 1)
        return {
            "injected_snippet": injected[:2000],
            "original_snippet": text[:2000],
            "violation_detail": f"Changed version reference from '{old_ver}' to '26Q1'",
            "expected_diagnosis": (
                "A version reference was changed without updating the associated constraints "
                "or data. ORFS version upgrades change internal flow behavior, causing PPA "
                "divergence that invalidates previously trained models and timing closures."
            ),
            "expected_fix": (
                "When migrating tool versions: 1) version-tag all data, 2) re-run STA on "
                "critical designs, 3) verify Pareto front stability, 4) retrain models if "
                "PPA divergence exceeds 5%."
            ),
        }
    return None


def inject_rtl_port_mismatch(text: str, filepath: str) -> dict | None:
    """Inject RTL port width mismatch in module instantiation."""
    # Look for module instantiation patterns
    m = re.search(r'(\.\w+\s*\(\s*)(\w+)(\[(\d+):(\d+)\])?\s*\)', text)
    if not m:
        return None

    port_name = m.group(2)
    if m.group(3):
        hi = int(m.group(4))
        lo = int(m.group(5))
        # Widen the bus by 1 to create mismatch
        new_hi = hi + 1
        injected = text[:m.start(3)] + f"[{new_hi}:{lo}]" + text[m.end(3):]
        return {
            "injected_snippet": injected[:2000],
            "original_snippet": text[:2000],
            "violation_detail": f"Port width changed from [{hi}:{lo}] to [{new_hi}:{lo}]",
            "expected_diagnosis": (
                f"Signal `{port_name}` is connected with width [{new_hi}:{lo}] ({new_hi-lo+1} bits) "
                f"but the module port expects [{hi}:{lo}] ({hi-lo+1} bits). This width mismatch "
                f"causes either truncation or an elaboration error depending on the simulator."
            ),
            "expected_fix": (
                f"Correct the bus width to [{hi}:{lo}] to match the module port definition. "
                f"Run `yosys -p 'read_verilog <file>; hierarchy -check'` to verify port widths."
            ),
        }
    else:
        # Inject a width where there was none (scalar to vector mismatch)
        injected = text[:m.end(2)] + f"[1:0]" + text[m.end(2):]
        return {
            "injected_snippet": injected[:2000],
            "original_snippet": text[:2000],
            "violation_detail": f"Scalar signal `{port_name}` changed to 2-bit vector [1:0]",
            "expected_diagnosis": (
                f"Signal `{port_name}` is connected as a 2-bit vector [1:0] but the module port "
                f"expects a scalar (1-bit). This creates a width mismatch warning and likely "
                f"incorrect behavior — only bit 0 will be used."
            ),
            "expected_fix": (
                f"Remove the [1:0] index to connect `{port_name}` as a scalar signal matching "
                f"the module port definition."
            ),
        }


def inject_ppa_knob_misconfiguration(text: str, filepath: str) -> dict | None:
    """Inject PPA-degrading knob settings in synthesis/P&R scripts."""
    # Look for common EDA flow knobs
    patterns = [
        (r'(set\s+::env\(CORE_UTILIZATION\)\s+)([\d.]+)', 'CORE_UTILIZATION'),
        (r'(CORE_UTILIZATION\s*[=:]\s*)([\d.]+)', 'CORE_UTILIZATION'),
        (r'(export\s+CORE_UTILIZATION\s*=\s*)([\d.]+)', 'CORE_UTILIZATION'),
        (r'(set\s+::env\(PLACE_DENSITY\)\s+)([\d.]+)', 'PLACE_DENSITY'),
        (r'(PLACE_DENSITY\s*[=:]\s*)([\d.]+)', 'PLACE_DENSITY'),
        (r'(set\s+::env\(CTS_BUF_DISTANCE\)\s+)([\d.]+)', 'CTS_BUF_DISTANCE'),
        (r'(CTS_BUF_DISTANCE\s*[=:]\s*)([\d.]+)', 'CTS_BUF_DISTANCE'),
    ]

    for pat, knob_name in patterns:
        m = re.search(pat, text)
        if m:
            orig_val = float(m.group(2))
            if knob_name == "CORE_UTILIZATION":
                bad_val = 95  # way too high — causes congestion
                diagnosis = (
                    f"CORE_UTILIZATION is set to {bad_val}%, which is extreme. "
                    f"Above 80%, the placer cannot find legal positions, causing massive "
                    f"congestion, routing DRC violations, and up to 3x runtime increase. "
                    f"Original value was {orig_val}%."
                )
                fix = (
                    f"Lower CORE_UTILIZATION to 40-70% (typical for digital designs). "
                    f"Start at {min(orig_val, 60)}% and increase in 5% steps while monitoring "
                    f"congestion via `report_design_area` and routing overflow."
                )
            elif knob_name == "PLACE_DENSITY":
                bad_val = 0.99
                diagnosis = (
                    f"PLACE_DENSITY is set to {bad_val}, forcing near-maximum cell packing. "
                    f"This leaves no room for buffer insertion during CTS and timing optimization, "
                    f"degrading both timing and power. Original value was {orig_val}."
                )
                fix = (
                    f"Set PLACE_DENSITY to 0.5-0.7 for reasonable buffer insertion headroom. "
                    f"Monitor post-CTS timing with `report_checks -path_delay max`."
                )
            else:
                bad_val = 1
                diagnosis = (
                    f"CTS_BUF_DISTANCE is set to {bad_val}, which is too small. "
                    f"This inserts excessive clock buffers, increasing clock power and skew. "
                    f"Original value was {orig_val}."
                )
                fix = (
                    f"Set CTS_BUF_DISTANCE to 60-120 (microns). Check clock skew with "
                    f"`report_clock_skew` after CTS."
                )

            injected = text[:m.start(2)] + str(bad_val) + text[m.end(2):]
            return {
                "injected_snippet": injected[:2000],
                "original_snippet": text[:2000],
                "violation_detail": f"{knob_name} changed from {orig_val} to {bad_val}",
                "expected_diagnosis": diagnosis,
                "expected_fix": fix,
            }

    return None


INJECTION_FUNCTIONS = {
    "setup_violation": inject_setup_violation,
    "hold_violation": inject_hold_violation,
    "drc_spacing": inject_drc_spacing,
    "drc_enclosure": inject_drc_enclosure,
    "lvs_mismatch": inject_lvs_mismatch,
    "sdc_unit_mismatch": inject_sdc_unit_mismatch,
    "missing_clock": inject_missing_clock,
    "false_path_misuse": inject_false_path_misuse,
    "def_naming_mismatch": inject_def_naming_mismatch,
    "version_drift": inject_version_drift,
    "rtl_port_mismatch": inject_rtl_port_mismatch,
    "ppa_knob_misconfiguration": inject_ppa_knob_misconfiguration,
}


def generate_templates(family_name: str, count: int) -> list[dict]:
    """Generate synthetic template-based cases for underrepresented families."""
    templates = []

    # Design/clock parameters to vary
    designs = ["aes_cipher", "ibex_core", "jpeg_encoder", "riscv32i", "spm_unit",
               "fir_filter", "uart_tx", "spi_master", "i2c_controller", "alu_top",
               "fifo_async", "arbiter_rr", "dma_engine", "cache_ctrl", "pll_top"]
    clock_periods = [2.0, 5.0, 8.0, 10.0, 12.5, 15.0, 20.0, 25.0, 50.0, 100.0]
    metals = ["met1", "met2", "met3", "met4", "met5"]
    spacings = [0.14, 0.16, 0.20, 0.28, 0.36, 0.42, 0.50]
    widths = [0.14, 0.16, 0.20, 0.28, 0.36]

    for i in range(min(count, 1500)):
        design = random.choice(designs)
        period = random.choice(clock_periods)
        metal = random.choice(metals)
        spacing = random.choice(spacings)

        if family_name == "setup_violation":
            bad_period = round(period / random.uniform(5, 20), 3)
            templates.append({
                "source_file": f"synthetic/{design}/constraints.sdc",
                "injected_snippet": f"create_clock -period {bad_period} -name clk [get_ports clk]\nset_input_delay [expr {bad_period} * 0.3] -clock clk [all_inputs]",
                "original_snippet": f"create_clock -period {period} -name clk [get_ports clk]\nset_input_delay [expr {period} * 0.3] -clock clk [all_inputs]",
                "violation_detail": f"Clock period reduced from {period} to {bad_period} ns for {design}",
                "expected_diagnosis": f"The clock period is {bad_period} ns, which is too tight for the {design} design's combinational depth. This creates setup violations on critical paths. Original target was {period} ns.",
                "expected_fix": f"Restore clock period to {period} ns. If faster timing is needed, optimize the critical path logic depth or insert pipeline stages.",
            })

        elif family_name == "hold_violation":
            uncertainty = round(-random.uniform(0.1, 0.8), 2)
            templates.append({
                "source_file": f"synthetic/{design}/constraints.sdc",
                "injected_snippet": f"create_clock -period {period} -name clk [get_ports clk]\nset_clock_uncertainty -hold {uncertainty} [all_clocks]",
                "original_snippet": f"create_clock -period {period} -name clk [get_ports clk]",
                "violation_detail": f"Negative hold uncertainty {uncertainty} ns applied to {design}",
                "expected_diagnosis": f"Hold uncertainty is set to {uncertainty} ns (negative), which is physically impossible and forces false hold violations on all paths in {design}.",
                "expected_fix": f"Remove the negative hold uncertainty or set to a realistic positive value (0.05-0.2 ns). Run hold analysis: report_checks -path_delay min.",
            })

        elif family_name == "drc_spacing":
            bad_spacing = round(spacing * random.uniform(0.3, 0.6), 4)
            templates.append({
                "source_file": f"synthetic/{design}/tech.lef",
                "injected_snippet": f"LAYER {metal}\n  TYPE ROUTING ;\n  SPACING {bad_spacing} ;",
                "original_snippet": f"LAYER {metal}\n  TYPE ROUTING ;\n  SPACING {spacing} ;",
                "violation_detail": f"{metal} spacing reduced from {spacing} to {bad_spacing} um",
                "expected_diagnosis": f"The {metal} layer spacing is {bad_spacing} um, below the PDK minimum of {spacing} um. This will trigger spacing DRC violations on every {metal} route.",
                "expected_fix": f"Restore {metal} SPACING to {spacing} um per the PDK design rules. Re-run DRC after correction.",
            })

        elif family_name == "drc_enclosure":
            enc = round(random.choice(widths) * random.uniform(2, 4), 4)
            orig_enc = round(enc / random.uniform(2, 4), 4)
            via = f"via{random.randint(1,4)}"
            templates.append({
                "source_file": f"synthetic/{design}/tech.lef",
                "injected_snippet": f"VIA {via}\n  ENCLOSURE {metal} {enc} {enc} ;",
                "original_snippet": f"VIA {via}\n  ENCLOSURE {metal} {orig_enc} {orig_enc} ;",
                "violation_detail": f"{via} enclosure inflated from {orig_enc} to {enc} um",
                "expected_diagnosis": f"The {via} enclosure on {metal} is {enc} um, which is larger than the original {orig_enc} um. Existing vias no longer satisfy this inflated rule, causing enclosure DRC violations.",
                "expected_fix": f"Restore ENCLOSURE to {orig_enc} um. Verify against the PDK via definition in the technology LEF.",
            })

        elif family_name == "lvs_mismatch":
            extra_port = random.choice(["debug_out", "scan_enable", "test_mode", "spare_io", "bist_clk"])
            templates.append({
                "source_file": f"synthetic/{design}/{design}.v",
                "injected_snippet": f"module {design} (\n  input clk, rst,\n  input [{random.randint(7,31)}:0] data_in,\n  output [{random.randint(7,31)}:0] data_out,\n  output {extra_port}\n);",
                "original_snippet": f"module {design} (\n  input clk, rst,\n  input [{random.randint(7,31)}:0] data_in,\n  output [{random.randint(7,31)}:0] data_out\n);",
                "violation_detail": f"Added phantom port '{extra_port}' to {design} module",
                "expected_diagnosis": f"Module {design} has an extra port `{extra_port}` not present in the schematic or parent instantiation, causing an LVS port count mismatch.",
                "expected_fix": f"Remove `{extra_port}` from the module port list, or add it to the schematic and all parent instantiations if it's intentional.",
            })

        elif family_name == "sdc_unit_mismatch":
            ps_period = round(period * 1000, 1)
            templates.append({
                "source_file": f"synthetic/{design}/constraints.sdc",
                "injected_snippet": f"# Clock constraint for {design}\ncreate_clock -period {ps_period} -name clk [get_ports clk]",
                "original_snippet": f"# Clock constraint for {design}\ncreate_clock -period {period} -name clk [get_ports clk]",
                "violation_detail": f"Period {ps_period} ps used where tool expects ns ({period} ns intended)",
                "expected_diagnosis": f"Clock period is {ps_period} (intended as picoseconds) but tool interprets as nanoseconds = {ps_period/1000:.1f} us. All timing paths show implausibly large positive slack.",
                "expected_fix": f"Add `set_units -time ns` and use {period} for the period, or divide by 1000: {period} ns.",
            })

        elif family_name == "missing_clock":
            templates.append({
                "source_file": f"synthetic/{design}/constraints.sdc",
                "injected_snippet": f"# Constraints for {design}\n# create_clock -period {period} -name clk [get_ports clk]\nset_input_delay 2.0 -clock clk [all_inputs]\nset_output_delay 2.0 -clock clk [all_outputs]",
                "original_snippet": f"# Constraints for {design}\ncreate_clock -period {period} -name clk [get_ports clk]\nset_input_delay 2.0 -clock clk [all_inputs]\nset_output_delay 2.0 -clock clk [all_outputs]",
                "violation_detail": f"create_clock commented out in {design} constraints",
                "expected_diagnosis": f"The create_clock statement for {design} is commented out. All register paths are unconstrained — timing analysis reports no violations (false negative).",
                "expected_fix": f"Uncomment: `create_clock -period {period} -name clk [get_ports clk]`. Run `report_checks -unconstrained` to verify all paths are constrained.",
            })

        elif family_name == "false_path_misuse":
            templates.append({
                "source_file": f"synthetic/{design}/constraints.sdc",
                "injected_snippet": f"create_clock -period {period} -name clk [get_ports clk]\nset_false_path -from [all_inputs] -to [all_outputs]\nset_false_path -from [all_registers] -to [all_registers]",
                "original_snippet": f"create_clock -period {period} -name clk [get_ports clk]",
                "violation_detail": f"Blanket false_path disables all timing in {design}",
                "expected_diagnosis": f"Two set_false_path constraints disable ALL timing analysis in {design}: input-to-output and reg-to-reg. No timing checks are performed — design appears to meet timing regardless of real violations.",
                "expected_fix": f"Remove blanket false_paths. Apply only to genuinely asynchronous CDC paths: `set_false_path -from [get_clocks clkA] -to [get_clocks clkB]`.",
            })

        elif family_name == "def_naming_mismatch":
            prefix = random.choice(["cn_", "circuitnet_", "CN_", "net_"])
            templates.append({
                "source_file": f"synthetic/{design}/{design}.def",
                "injected_snippet": f"DESIGN {design} ;\nCOMPONENTS 4 ;\n  - {prefix}U1 NAND2_X1 ;\n  - {prefix}U2 INV_X1 ;\n  - {prefix}U3 BUF_X2 ;\n  - {prefix}U4 DFF_X1 ;\nEND COMPONENTS",
                "original_snippet": f"DESIGN {design} ;\nCOMPONENTS 4 ;\n  - U1 NAND2_X1 ;\n  - U2 INV_X1 ;\n  - U3 BUF_X2 ;\n  - U4 DFF_X1 ;\nEND COMPONENTS",
                "violation_detail": f"Instance names prefixed with '{prefix}' (incompatible convention)",
                "expected_diagnosis": f"DEF instance names use '{prefix}' prefix (CircuitNet convention) that OpenROAD cannot resolve during placement extraction, causing name lookup failures.",
                "expected_fix": f"Run normalize_def_instances.py to strip '{prefix}' prefixes. Expected pattern: bare instance names (U1, U2, ...) matching the netlist hierarchy.",
            })

        elif family_name == "version_drift":
            old_ver = random.choice(["v3.0", "v2.5", "2023.06", "22Q4"])
            new_ver = random.choice(["26Q1", "v4.0", "2025.01", "25Q2"])
            ppa_shift = round(random.uniform(8, 25), 1)
            templates.append({
                "source_file": f"synthetic/{design}/Makefile",
                "injected_snippet": f"# {design} flow configuration\nORFS_VERSION = {new_ver}\n# WARNING: migrated from {old_ver}, PPA shifted {ppa_shift}%\n# WNS sign may have flipped",
                "original_snippet": f"# {design} flow configuration\nORFS_VERSION = {old_ver}",
                "violation_detail": f"ORFS version changed from {old_ver} to {new_ver} without re-validation",
                "expected_diagnosis": f"ORFS version upgraded from {old_ver} to {new_ver} causing {ppa_shift}% PPA divergence in {design}. Surrogate models trained on {old_ver} data are invalid. WNS sign may have flipped.",
                "expected_fix": f"1) Version-tag all data with ORFS version. 2) Re-run STA on {design}. 3) Retrain models on {new_ver} data if PPA shift > 5%. 4) Use per-variant SDC files.",
            })

        elif family_name == "rtl_port_mismatch":
            width_a = random.randint(7, 31)
            width_b = width_a + random.choice([1, 2, -1])
            port = random.choice(["data_in", "data_out", "addr", "wdata", "rdata", "mask"])
            submod = random.choice(["mem_ctrl", "alu_unit", "decoder", "fifo_rd", "spi_shift"])
            templates.append({
                "source_file": f"synthetic/{design}/{design}.v",
                "injected_snippet": (
                    f"module {design} (\n  input clk, rst,\n"
                    f"  input [{width_a}:0] {port}\n);\n\n"
                    f"  {submod} u_{submod} (\n"
                    f"    .{port}({port}[{width_b}:0])\n  );"
                ),
                "original_snippet": (
                    f"module {design} (\n  input clk, rst,\n"
                    f"  input [{width_a}:0] {port}\n);\n\n"
                    f"  {submod} u_{submod} (\n"
                    f"    .{port}({port}[{width_a}:0])\n  );"
                ),
                "violation_detail": f"Port `{port}` connected as [{width_b}:0] but submodule expects [{width_a}:0]",
                "expected_diagnosis": (
                    f"Signal `{port}` is connected with width [{width_b}:0] ({width_b+1} bits) "
                    f"to submodule `{submod}`, but the port expects [{width_a}:0] ({width_a+1} bits). "
                    f"This width mismatch causes truncation or zero-extension depending on the tool, "
                    f"leading to functional bugs."
                ),
                "expected_fix": (
                    f"Change connection to `{port}[{width_a}:0]` to match the submodule port width. "
                    f"Run `yosys -p 'read_verilog {design}.v; hierarchy -check'` to detect remaining mismatches."
                ),
            })

        elif family_name == "ppa_knob_misconfiguration":
            knob = random.choice(["CORE_UTILIZATION", "PLACE_DENSITY", "CTS_BUF_DISTANCE",
                                  "GPL_TIMING_DRIVEN", "CELL_PAD_IN_SITES_GLOBAL_PLACEMENT"])
            if knob == "CORE_UTILIZATION":
                bad_val, good_val = random.choice([(92, 55), (95, 60), (88, 50)])
                templates.append({
                    "source_file": f"synthetic/{design}/config.mk",
                    "injected_snippet": f"DESIGN_CONFIG = {design}\nCORE_UTILIZATION = {bad_val}\nPLACE_DENSITY = 0.65",
                    "original_snippet": f"DESIGN_CONFIG = {design}\nCORE_UTILIZATION = {good_val}\nPLACE_DENSITY = 0.65",
                    "violation_detail": f"CORE_UTILIZATION set to {bad_val}% (was {good_val}%)",
                    "expected_diagnosis": f"CORE_UTILIZATION at {bad_val}% is too aggressive for {design}. Above 80%, placement congestion causes routing DRC violations and up to 3x runtime. Target was {good_val}%.",
                    "expected_fix": f"Lower CORE_UTILIZATION to {good_val}%. Increase in 5% increments while checking `report_design_area` and routing overflow.",
                })
            elif knob == "PLACE_DENSITY":
                bad_val = round(random.uniform(0.92, 0.99), 2)
                good_val = round(random.uniform(0.55, 0.70), 2)
                templates.append({
                    "source_file": f"synthetic/{design}/config.mk",
                    "injected_snippet": f"DESIGN_CONFIG = {design}\nPLACE_DENSITY = {bad_val}",
                    "original_snippet": f"DESIGN_CONFIG = {design}\nPLACE_DENSITY = {good_val}",
                    "violation_detail": f"PLACE_DENSITY set to {bad_val} (was {good_val})",
                    "expected_diagnosis": f"PLACE_DENSITY at {bad_val} leaves no headroom for CTS buffer insertion in {design}. Post-CTS timing will degrade significantly vs the {good_val} baseline.",
                    "expected_fix": f"Set PLACE_DENSITY to {good_val}. Verify post-CTS slack with `report_checks -path_delay max`.",
                })
            elif knob == "CTS_BUF_DISTANCE":
                bad_val = random.choice([1, 2, 5])
                good_val = random.choice([80, 100, 120])
                templates.append({
                    "source_file": f"synthetic/{design}/config.mk",
                    "injected_snippet": f"DESIGN_CONFIG = {design}\nCTS_BUF_DISTANCE = {bad_val}",
                    "original_snippet": f"DESIGN_CONFIG = {design}\nCTS_BUF_DISTANCE = {good_val}",
                    "violation_detail": f"CTS_BUF_DISTANCE set to {bad_val} (was {good_val})",
                    "expected_diagnosis": f"CTS_BUF_DISTANCE at {bad_val} is too small for {design}, inserting excessive clock buffers. This increases clock network power by 2-5x and skew.",
                    "expected_fix": f"Set CTS_BUF_DISTANCE to {good_val}. Check `report_clock_skew` after CTS.",
                })
            else:
                templates.append({
                    "source_file": f"synthetic/{design}/config.mk",
                    "injected_snippet": f"DESIGN_CONFIG = {design}\nGPL_TIMING_DRIVEN = 0\nCELL_PAD_IN_SITES_GLOBAL_PLACEMENT = 0",
                    "original_snippet": f"DESIGN_CONFIG = {design}\nGPL_TIMING_DRIVEN = 1\nCELL_PAD_IN_SITES_GLOBAL_PLACEMENT = 2",
                    "violation_detail": f"Timing-driven placement disabled, cell padding removed in {design}",
                    "expected_diagnosis": f"GPL_TIMING_DRIVEN=0 disables timing-aware global placement — critical paths will not be optimized during placement. CELL_PAD=0 removes inter-cell spacing, causing DRC violations and signal integrity issues.",
                    "expected_fix": f"Set GPL_TIMING_DRIVEN=1 and CELL_PAD_IN_SITES_GLOBAL_PLACEMENT=2. Re-run placement and verify WNS improvement.",
                })

    return templates


def read_text_file(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="ignore")
    except (PermissionError, OSError):
        return ""


def load_seeds(seeds_path: Path) -> dict:
    """Load MLCAD seed bugs for reference."""
    if not seeds_path.exists():
        return {}
    with open(seeds_path, "r", encoding="utf-8") as f:
        seeds = yaml.safe_load(f)
    return {s["id"]: s for s in seeds}


def main():
    parser = argparse.ArgumentParser(description="Inject synthetic EDA violations")
    parser.add_argument("--input", required=True, help="Deduped corpus directory")
    parser.add_argument("--seeds", default="data/edabench/seeds/mlcad_seeds.yaml")
    parser.add_argument("--output", default="data/synthetic/injected_cases.jsonl")
    parser.add_argument("--families", default="all",
                        help="Comma-separated family names or 'all'")
    parser.add_argument("--max-per-source", type=int, default=3,
                        help="Max injection cases per source file for diversity")
    parser.add_argument("--target-count", type=int, default=15000,
                        help="Target total injected cases")
    args = parser.parse_args()

    input_root = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    seeds = load_seeds(Path(args.seeds))

    # Select families
    if args.families == "all":
        families = list(VIOLATION_FAMILIES.keys())
    else:
        families = [f.strip() for f in args.families.split(",")]

    # Per-family target
    per_family_target = args.target_count // len(families)
    print(f"Target: {args.target_count} total, ~{per_family_target} per family")
    print(f"Families: {', '.join(families)}")

    # Index corpus files by extension
    print("Indexing corpus files...")
    ext_index: dict[str, list[Path]] = {}
    for f in input_root.rglob("*"):
        if f.is_file() and ".git" not in f.parts:
            ext = f.suffix.lower()
            ext_index.setdefault(ext, []).append(f)

    for ext, files in sorted(ext_index.items()):
        if ext in {".sdc", ".tcl", ".def", ".lef", ".v", ".sv", ".py", ".mk", ".sh"}:
            print(f"  {ext}: {len(files)} files")

    # Generate injections per family
    all_cases = []
    source_usage: dict[str, dict[str, int]] = {}  # source -> {family -> count}

    for family_name in families:
        family_info = VIOLATION_FAMILIES[family_name]
        inject_fn = INJECTION_FUNCTIONS[family_name]
        target_exts = family_info["extensions"]
        seed_bug_id = family_info.get("seed_bug_id", "")

        # Gather candidate files
        candidates = []
        for ext in target_exts:
            candidates.extend(ext_index.get(ext, []))

        random.shuffle(candidates)
        family_cases = []

        for fpath in candidates:
            if len(family_cases) >= per_family_target:
                break

            # Enforce max-per-source PER FAMILY (not global)
            fkey = str(fpath)
            fam_usage = source_usage.setdefault(fkey, {})
            if fam_usage.get(family_name, 0) >= args.max_per_source:
                continue

            text = read_text_file(fpath)
            if len(text) < 50:
                continue

            result = inject_fn(text, str(fpath))
            if result is None:
                continue

            case = {
                "case_id": f"{family_name}_{len(family_cases):04d}",
                "source_file": str(fpath.relative_to(input_root)),
                "violation_family": family_name,
                "task_category": family_info["task_category"],
                "seed_bug_id": seed_bug_id,
                **result,
            }
            family_cases.append(case)
            fam_usage[family_name] = fam_usage.get(family_name, 0) + 1

        # Template augmentation for underrepresented families
        if len(family_cases) < per_family_target:
            templates = generate_templates(family_name, per_family_target - len(family_cases))
            for t in templates:
                t["case_id"] = f"{family_name}_{len(family_cases):04d}"
                t["violation_family"] = family_name
                t["task_category"] = family_info["task_category"]
                t["seed_bug_id"] = seed_bug_id
                family_cases.append(t)

        all_cases.extend(family_cases)
        print(f"  {family_name}: {len(family_cases)} cases (target: {per_family_target})")

    # Shuffle for training diversity
    random.shuffle(all_cases)

    # Write output
    with open(output_path, "w", encoding="utf-8") as f:
        for case in all_cases:
            f.write(json.dumps(case, ensure_ascii=False) + "\n")

    # Summary
    print(f"\n{'='*60}")
    print(f"INJECTION SUMMARY")
    print(f"{'='*60}")
    print(f"Total cases:        {len(all_cases):,}")
    print(f"Families used:      {len(families)}")
    print(f"Target:             {args.target_count:,}")

    from collections import Counter
    family_dist = Counter(c["violation_family"] for c in all_cases)
    cat_dist = Counter(c["task_category"] for c in all_cases)

    print(f"\nPer-family distribution:")
    for fam, cnt in sorted(family_dist.items()):
        pct = cnt / len(all_cases) * 100
        print(f"  {fam:30s} {cnt:>6,} ({pct:5.1f}%)")

    print(f"\nPer-category distribution:")
    for cat, cnt in sorted(cat_dist.items()):
        pct = cnt / len(all_cases) * 100
        print(f"  {cat:30s} {cnt:>6,} ({pct:5.1f}%)")

    seed_cases = sum(1 for c in all_cases if c["seed_bug_id"])
    print(f"\nSeed-bug-linked cases:  {seed_cases:,}")
    print(f"Output: {output_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
