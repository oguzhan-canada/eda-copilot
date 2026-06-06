#!/usr/bin/env python3
"""
scripts/run_openroad.py

Run OpenROAD-flow-scripts on 5 designs × multiple clock targets.
Produces placed DEF, STA reports, and power reports for each configuration.

Features:
  - Parallel execution (configurable max_jobs)
  - Resume support: skips completed runs
  - Per-run status tracking (JSON log)

Usage:
  python scripts/run_openroad.py --output data/raw/openroad_runs/
  python scripts/run_openroad.py --output data/raw/openroad_runs/ --max-jobs 4
  python scripts/run_openroad.py --designs ibex aes128 --clocks 2.0 3.0 4.0

Prerequisites:
  - OpenROAD-flow-scripts installed (OPENROAD_BIN, YOSYS_BIN, PDK_ROOT set)
  - ASAP7 PDK available
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm


DEFAULT_DESIGNS = ["ibex", "swerv_wrapper", "riscv32i", "jpeg", "aes"]

# Native SDC clock periods (ps) per ASAP7 design — used for design-relative sweeps
NATIVE_CLOCK_PS = {
    "aes": 380,
    "ibex": 1000,
    "jpeg": 680,
    "riscv32i": 950,
    "swerv_wrapper": 700,
}

# Multipliers for design-relative clock sweep (straddle the timing knee)
CLOCK_MULTIPLIERS = [0.85, 1.0, 1.15, 1.5, 2.0, 3.0]


def generate_clock_targets_absolute(start=2.0, stop=6.0, step=0.4):
    """Generate absolute clock targets in ns (legacy mode)."""
    targets = []
    val = start
    while val <= stop + 1e-9:
        targets.append(round(val, 2))
        val += step
    return targets


def generate_design_relative_clocks(design: str) -> list:
    """Generate clock targets relative to the design's native period.

    Returns clock periods in nanoseconds.
    """
    native_ps = NATIVE_CLOCK_PS.get(design, 1000)
    targets = []
    for mult in CLOCK_MULTIPLIERS:
        period_ps = native_ps * mult
        period_ns = round(period_ps / 1000.0, 3)
        targets.append(period_ns)
    return targets


def create_variant_sdc(original_sdc: Path, clock_ps: float, variant_dir: Path, variant_name: str) -> Path:
    """Create a variant SDC file with the target clock period (in ps).

    ORFS designs use ``set clk_period <value>`` in their SDC (picoseconds for
    ASAP7).  We copy the file, replacing that value with the sweep target so
    each FLOW_VARIANT is synthesised at the correct frequency.

    Each variant gets a uniquely-named file to avoid parallel write races.
    """
    variant_dir.mkdir(parents=True, exist_ok=True)
    stem = original_sdc.stem
    variant_sdc = variant_dir / f"{stem}_{variant_name}.sdc"

    # Skip if already exists with correct content
    if variant_sdc.exists():
        return variant_sdc

    content = original_sdc.read_text()

    # Replace 'set clk_period <number>' with the target value
    new_content, count = re.subn(
        r'(set\s+clk_period\s+)\d+\.?\d*',
        rf'\g<1>{clock_ps:.1f}',
        content,
    )

    if count == 0:
        raise ValueError(
            f"No 'set clk_period' found in {original_sdc}. "
            f"Cannot override clock period for this design."
        )

    variant_sdc.write_text(new_content)
    return variant_sdc


def generate_clock_targets(start=2.0, stop=6.0, step=0.4):
    targets = []
    val = start
    while val <= stop + 1e-9:
        targets.append(round(val, 2))
        val += step
    return targets


def check_tools():
    """Verify EDA tools are accessible."""
    tools = {
        "openroad": os.environ.get("OPENROAD_BIN", "openroad"),
        "yosys": os.environ.get("YOSYS_BIN", "yosys"),
    }
    for name, binary in tools.items():
        try:
            result = subprocess.run(
                [binary, "--version"], capture_output=True, timeout=10
            )
            print(f"  ✓ {name}: {result.stdout.decode().strip()[:60]}")
        except (FileNotFoundError, subprocess.TimeoutExpired):
            print(f"  ✗ {name}: NOT FOUND at {binary}")
            sys.exit(1)


def _format_clock(clock_ns: float) -> str:
    """Format clock value for run_id/variant names (remove trailing zeros)."""
    return f"{clock_ns:.3f}".rstrip("0").rstrip(".")


def run_single_flow(
    design: str,
    clock_ns: float,
    output_dir: Path,
    flow_dir: str,
    force: bool = False,
) -> dict:
    """
    Run a single OpenROAD flow. Returns a status dict.
    """
    clk_str = _format_clock(clock_ns)
    run_id = f"{design}_clk{clk_str}"
    run_dir = output_dir / run_id
    status_file = run_dir / "run_status.json"

    # Resume: skip if already completed (unless forced)
    if not force and status_file.exists():
        status = json.loads(status_file.read_text())
        if status.get("status") == "success":
            return status

    run_dir.mkdir(parents=True, exist_ok=True)
    start_time = time.time()

    # Clean stale ORFS variant artifacts when forcing re-run
    variant = f"clk{clk_str}".replace(".", "p")
    if force:
        for subdir in ["results", "logs", "objects"]:
            stale = Path(flow_dir) / subdir / "asap7" / design / variant
            if stale.exists():
                shutil.rmtree(stale, ignore_errors=True)

    try:
        env = os.environ.copy()
        env["DESIGN_NAME"] = design
        env["CLOCK_PERIOD"] = str(clock_ns)
        env["FLOW_HOME"] = flow_dir

        # Create a variant SDC file with the target clock period (in ps for ASAP7)
        clock_ps = clock_ns * 1000.0
        design_dir = Path(flow_dir) / "designs" / "asap7" / design
        sdc_candidates = list(design_dir.glob("*.sdc"))
        if not sdc_candidates:
            raise FileNotFoundError(f"No SDC file found in {design_dir}")

        # Pick the SDC referenced by config.mk or the first one found
        config_mk = design_dir / "config.mk"
        original_sdc = sdc_candidates[0]
        if config_mk.exists():
            mk_text = config_mk.read_text()
            for sdc in sdc_candidates:
                if sdc.name in mk_text:
                    original_sdc = sdc
                    break

        variant_sdc_dir = Path(flow_dir) / "designs" / "asap7" / design / "sdc_variants"
        variant_sdc = create_variant_sdc(original_sdc, clock_ps, variant_sdc_dir, variant)

        # Resolve tool paths — ORFS Makefile defaults use $(FLOW_HOME)/../tools/install/
        # which breaks when flow_dir is a symlink (.. resolves to symlink parent, not target parent)
        tools_base = os.environ.get("ORFS_TOOLS_DIR",
                                     os.path.join(os.path.dirname(flow_dir), "tools", "install"))
        openroad_exe = os.environ.get("OPENROAD_EXE",
                                       os.path.join(tools_base, "OpenROAD", "bin", "openroad"))
        yosys_exe = os.environ.get("YOSYS_EXE",
                                    os.path.join(tools_base, "yosys", "bin", "yosys"))

        # Target the 6_report log directly — this chains all deps (synth→place→route→fill→report)
        # without requiring klayout/GDS generation that the "finish" target needs
        report_target = f"logs/asap7/{design}/{variant}/6_report.log"
        cmd = [
            "make", "-C", flow_dir,
            f"DESIGN_CONFIG=designs/asap7/{design}/config.mk",
            f"CLOCK_PERIOD={clock_ns}",
            f"SDC_FILE={variant_sdc}",
            f"FLOW_VARIANT={variant}",
            f"OPENROAD_EXE={openroad_exe}",
            f"YOSYS_EXE={yosys_exe}",
            report_target,
        ]

        result = subprocess.run(
            cmd,
            capture_output=True,
            timeout=14400,  # 4 hr timeout (swerv_wrapper routing needs ~2hrs with swap)
            env=env,
            cwd=flow_dir,
        )

        elapsed = time.time() - start_time

        if result.returncode == 0:
            # Collect output files from the variant-specific directory
            variant_dir = Path(flow_dir) / "results" / "asap7" / design / variant
            results_glob = variant_dir if variant_dir.exists() else Path(flow_dir) / "results" / "asap7" / design
            timing_rpt = list(results_glob.rglob("*6_report*"))
            power_rpt = list(results_glob.rglob("*power*"))
            placed_def = list(results_glob.rglob("*.def"))
            metrics_json = list(results_glob.rglob("6_report.json"))
            synth_verilog = list(results_glob.rglob("1_synth.v"))
            final_odb = list(results_glob.rglob("6_final.odb"))

            # Also grab the logs JSON for structured metrics
            log_dir = Path(flow_dir) / "logs" / "asap7" / design / variant
            log_json = list(log_dir.rglob("6_report.json")) if log_dir.exists() else []

            status = {
                "run_id": run_id,
                "design": design,
                "clock_ns": clock_ns,
                "flow_variant": variant,
                "status": "success",
                "elapsed_s": round(elapsed, 1),
                "metrics_json": str(metrics_json[0]) if metrics_json else (str(log_json[0]) if log_json else None),
                "timing_rpt": str(timing_rpt[0]) if timing_rpt else None,
                "power_rpt": str(power_rpt[0]) if power_rpt else None,
                "placed_def": str(placed_def[0]) if placed_def else None,
                "synth_verilog": str(synth_verilog[0]) if synth_verilog else None,
                "final_odb": str(final_odb[0]) if final_odb else None,
                "stdout_tail": result.stdout.decode()[-500:],
            }

            # Copy key outputs to run_dir for portability
            for src in (metrics_json[:1] + log_json[:1] + timing_rpt[:1] + power_rpt[:1] + placed_def[:1] + synth_verilog[:1] + final_odb[:1]):
                dst = run_dir / src.name
                if not dst.exists():
                    shutil.copy2(src, dst)
        else:
            status = {
                "run_id": run_id,
                "design": design,
                "clock_ns": clock_ns,
                "status": "failed",
                "elapsed_s": round(elapsed, 1),
                "returncode": result.returncode,
                "stderr_tail": result.stderr.decode()[-500:],
            }
    except subprocess.TimeoutExpired:
        status = {
            "run_id": run_id,
            "design": design,
            "clock_ns": clock_ns,
            "status": "timeout",
            "elapsed_s": 1800,
        }
    except Exception as e:
        status = {
            "run_id": run_id,
            "design": design,
            "clock_ns": clock_ns,
            "status": "error",
            "error": str(e),
        }

    # Save status
    status_file.write_text(json.dumps(status, indent=2))
    return status


def main():
    parser = argparse.ArgumentParser(description="Run OpenROAD design sweep")
    parser.add_argument("--output", type=str, default="data/raw/openroad_runs",
                        help="Output directory for run results")
    parser.add_argument("--flow-dir", type=str,
                        default="/opt/OpenROAD-flow-scripts/flow",
                        help="OpenROAD-flow-scripts flow directory")
    parser.add_argument("--designs", nargs="+", default=DEFAULT_DESIGNS,
                        help="Designs to run")
    parser.add_argument("--clocks", nargs="+", type=float, default=None,
                        help="Specific clock targets (overrides sweep)")
    parser.add_argument("--max-jobs", type=int, default=4,
                        help="Max parallel jobs")
    parser.add_argument("--check-only", action="store_true",
                        help="Only check tool availability")
    parser.add_argument("--force", action="store_true",
                        help="Force re-run of completed jobs")
    parser.add_argument("--design-relative", action="store_true",
                        help="Use design-relative clock targets instead of absolute")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Check tools
    print("Checking EDA tools...")
    check_tools()
    if args.check_only:
        return

    # Generate run matrix
    if args.clocks:
        # Explicit clock targets — same for all designs
        runs = [(d, c) for d in args.designs for c in args.clocks]
        print(f"\nPlanned runs: {len(runs)} ({len(args.designs)} designs × {len(args.clocks)} clocks)")
    elif args.design_relative:
        # Design-relative targets — different per design
        runs = []
        for d in args.designs:
            clocks = generate_design_relative_clocks(d)
            runs.extend([(d, c) for c in clocks])
            print(f"  {d}: native={NATIVE_CLOCK_PS.get(d,'?')} ps → {[f'{c:.3f}' for c in clocks]} ns")
        print(f"\nPlanned runs: {len(runs)}")
    else:
        clocks = generate_clock_targets_absolute()
        runs = [(d, c) for d in args.designs for c in clocks]
        print(f"\nPlanned runs: {len(runs)} ({len(args.designs)} designs × {len(clocks)} clocks)")

    # Check for completed runs
    completed = 0
    for d, c in runs:
        clk_str = _format_clock(c)
        status_file = output_dir / f"{d}_clk{clk_str}" / "run_status.json"
        if status_file.exists():
            s = json.loads(status_file.read_text())
            if s.get("status") == "success":
                completed += 1
    print(f"Already completed: {completed}/{len(runs)}")

    if completed == len(runs):
        print("All runs already complete!")
        return

    # Execute
    print(f"\nRunning with max {args.max_jobs} parallel jobs...")
    all_status = []

    with ProcessPoolExecutor(max_workers=args.max_jobs) as executor:
        futures = {
            executor.submit(run_single_flow, d, c, output_dir, args.flow_dir, args.force): (d, c)
            for d, c in runs
        }
        for future in tqdm(as_completed(futures), total=len(futures), desc="OpenROAD"):
            design, clock = futures[future]
            try:
                status = future.result()
                all_status.append(status)
                if status["status"] != "success":
                    print(f"  ⚠ {status['run_id']}: {status['status']}")
            except Exception as e:
                print(f"  ✗ {design}@{clock}ns: {e}")

    # Summary
    success = sum(1 for s in all_status if s.get("status") == "success")
    failed = sum(1 for s in all_status if s.get("status") != "success")
    print(f"\nComplete: {success} success, {failed} failed out of {len(runs)} total")

    # Save master log
    log_path = output_dir / "run_log.json"
    log_path.write_text(json.dumps(all_status, indent=2))
    print(f"Run log: {log_path}")


if __name__ == "__main__":
    main()
