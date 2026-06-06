#!/usr/bin/env python3
"""
scripts/extract_timing.py

Extract timing features from OpenROAD STA reports.

Extracts: WNS, TNS, slack percentiles (p0, p10, p50, p90),
          violation count, achieved frequency, clock period.

Usage:
  python scripts/extract_timing.py \
    --input data/raw/openroad_runs/ \
    --output features/timing_stats.csv
"""

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm


def parse_timing_report(rpt_path: str) -> dict:
    """
    Parse an OpenROAD/OpenSTA timing report.
    Returns timing statistics dict.
    """
    with open(rpt_path) as f:
        content = f.read()

    result = {
        "timing_rpt_path": rpt_path,
        "wns": np.nan,
        "tns": np.nan,
        "violation_count": 0,
        "clock_period": np.nan,
        "freq_mhz": np.nan,
        "slack_p0": np.nan,
        "slack_p10": np.nan,
        "slack_p50": np.nan,
        "slack_p90": np.nan,
    }

    # Extract all slack values
    slack_pattern = re.compile(r'slack\s+\((?:MET|VIOLATED)\)\s+([+-]?[0-9]+\.?[0-9]*)', re.IGNORECASE)
    slacks = [float(m.group(1)) for m in slack_pattern.finditer(content)]

    if not slacks:
        # Alternative format: just "slack" followed by a number
        alt_pattern = re.compile(r'slack\s+([+-]?[0-9]+\.[0-9]+)')
        slacks = [float(m.group(1)) for m in alt_pattern.finditer(content)]

    if slacks:
        slacks_arr = np.array(slacks)
        result["wns"] = float(slacks_arr.min())
        result["tns"] = float(slacks_arr[slacks_arr < 0].sum()) if (slacks_arr < 0).any() else 0.0
        result["violation_count"] = int((slacks_arr < 0).sum())
        result["slack_p0"] = float(np.percentile(slacks_arr, 0))
        result["slack_p10"] = float(np.percentile(slacks_arr, 10))
        result["slack_p50"] = float(np.percentile(slacks_arr, 50))
        result["slack_p90"] = float(np.percentile(slacks_arr, 90))

    # Extract clock period
    clk_pattern = re.compile(r'clock\s+(\w+)\s+\(rise|fall\)\s+([0-9]+\.?[0-9]*)')
    clk_match = clk_pattern.search(content)
    if clk_match:
        result["clock_period"] = float(clk_match.group(2))
    else:
        # Alternative: look for "Period" or "period"
        period_pattern = re.compile(r'[Pp]eriod\s*[:=]\s*([0-9]+\.?[0-9]*)')
        pm = period_pattern.search(content)
        if pm:
            result["clock_period"] = float(pm.group(1))

    # Compute frequency
    if not np.isnan(result["clock_period"]) and result["clock_period"] > 0:
        result["freq_mhz"] = 1000.0 / result["clock_period"]

    return result


def parse_power_report(rpt_path: str) -> dict:
    """Parse an OpenROAD power report."""
    with open(rpt_path) as f:
        content = f.read()

    result = {
        "total_power": np.nan,
    }

    # Pattern: Total ... power_value
    total_pattern = re.compile(r'[Tt]otal\s+.*?([0-9]+\.?[0-9]*(?:[eE][+-]?\d+)?)\s*(?:mW|W|uW)', re.MULTILINE)
    m = total_pattern.search(content)
    if m:
        power = float(m.group(1))
        # Normalise to mW
        if "uW" in content[m.start():m.end()+5]:
            power /= 1000.0
        elif "W" in content[m.start():m.end()+5] and "mW" not in content[m.start():m.end()+5]:
            power *= 1000.0
        result["total_power"] = power

    return result


def parse_area_report(rpt_path: str) -> dict:
    """Parse area from OpenROAD report."""
    with open(rpt_path) as f:
        content = f.read()

    result = {"cell_area": np.nan}

    # Pattern: cell_area or total area
    area_pattern = re.compile(r'(?:cell_area|total\s+area|Design\s+area)\s*[:=]?\s*([0-9]+\.?[0-9]*)', re.IGNORECASE)
    m = area_pattern.search(content)
    if m:
        result["cell_area"] = float(m.group(1))

    return result


def parse_orfs_json(json_path: str) -> dict:
    """Parse ORFS 6_report.json for PPA metrics."""
    with open(json_path) as f:
        data = json.load(f)

    result = {
        "timing_rpt_path": json_path,
        "wns": np.nan,
        "tns": np.nan,
        "violation_count": 0,
        "clock_period": np.nan,
        "freq_mhz": np.nan,
        "slack_p0": np.nan,
        "slack_p10": np.nan,
        "slack_p50": np.nan,
        "slack_p90": np.nan,
        "total_power": np.nan,
        "cell_area": np.nan,
    }

    # Timing: WNS / TNS (setup)
    wns = data.get("finish__timing__setup__ws")
    tns = data.get("finish__timing__setup__tns")
    if wns is not None:
        result["wns"] = float(wns)
        result["slack_p0"] = float(wns)
    if tns is not None:
        result["tns"] = float(tns)

    # Violation count from endpoint slack histogram
    viol = 0
    for k, v in data.items():
        if "setup__wns" in k or "setup__ws" in k:
            if isinstance(v, (int, float)) and v < 0:
                viol += 1
    result["violation_count"] = viol if viol > 0 else (1 if result["wns"] < 0 else 0) if not np.isnan(result["wns"]) else 0

    # Frequency
    for k, v in data.items():
        if "fmax" in k and v is not None:
            result["freq_mhz"] = float(v) / 1e6
            break

    # Clock period from run_status.json is more reliable, but try from metrics
    for k, v in data.items():
        if "clock_period" in k.lower() and v is not None:
            result["clock_period"] = float(v)
            break

    # Total power (internal + switching + leakage)
    int_pwr = data.get("finish__power__internal__total", 0) or 0
    sw_pwr = data.get("finish__power__switching__total", 0) or 0
    leak_pwr = data.get("finish__power__leakage__total", 0) or 0
    total = float(int_pwr) + float(sw_pwr) + float(leak_pwr)
    if total > 0:
        result["total_power"] = total * 1000  # W to mW

    # Cell area
    area = data.get("finish__design__instance__area")
    if area is not None:
        result["cell_area"] = float(area)

    # Slack percentiles from WNS (single value — use it for all percentiles)
    if not np.isnan(result["wns"]):
        result["slack_p10"] = result["wns"]
        result["slack_p50"] = result["wns"]
        result["slack_p90"] = result["wns"]

    return result


def main():
    parser = argparse.ArgumentParser(description="Extract timing features")
    parser.add_argument("--input", type=str, required=True,
                        help="Directory with OpenROAD run results")
    parser.add_argument("--output", type=str, default="features/timing_stats.csv",
                        help="Output CSV path")
    args = parser.parse_args()

    input_dir = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []

    # Process each run directory
    run_dirs = sorted([d for d in input_dir.iterdir() if d.is_dir()])
    print(f"Found {len(run_dirs)} run directories")

    for run_dir in tqdm(run_dirs, desc="Extracting timing"):
        row = {"run_id": run_dir.name}

        # Try ORFS 6_report.json first (most reliable source)
        orfs_json = run_dir / "6_report.json"
        if orfs_json.exists():
            try:
                row.update(parse_orfs_json(str(orfs_json)))
            except Exception as e:
                print(f"Warning: Failed to parse {orfs_json}: {e}")

            # Get clock_period from run_status.json if available
            status_json = run_dir / "run_status.json"
            if status_json.exists():
                try:
                    status = json.loads(status_json.read_text())
                    row["clock_period"] = float(status.get("clock_ns", 0))
                    if row["clock_period"] > 0:
                        row["freq_mhz"] = 1000.0 / row["clock_period"]
                except Exception:
                    pass
        else:
            # Fallback: parse .rpt text files
            timing_files = list(run_dir.rglob("*timing*")) + list(run_dir.rglob("*report*.rpt"))
            power_files = list(run_dir.rglob("*power*"))
            area_files = list(run_dir.rglob("*area*"))

            for tf in timing_files:
                try:
                    row.update(parse_timing_report(str(tf)))
                    break
                except Exception:
                    continue

            for pf in power_files:
                try:
                    row.update(parse_power_report(str(pf)))
                    break
                except Exception:
                    continue

            for af in area_files:
                try:
                    row.update(parse_area_report(str(af)))
                    break
                except Exception:
                    continue

        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    print(f"Saved timing stats: {output_path} ({len(df)} rows)")


if __name__ == "__main__":
    main()
