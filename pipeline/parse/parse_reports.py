"""
scripts/parse_logs.py

Parses OpenROAD timing (.rpt) and power (.pwr) report files to extract
ground-truth PPA labels and timing features.

Extracted fields:
  - wns          : Worst Negative Slack (ns)
  - tns          : Total Negative Slack (ns)
  - clock_period : Target clock period (ns)
  - freq_mhz     : Achieved frequency (MHz) = 1000 / (clock_period - wns) if wns < 0
  - violation_count : Number of timing paths with negative slack
  - path_slacks  : list of top-k path slack values
  - slack_p0/p10/p50/p90 : slack percentiles
  - total_power  : Total power (mW)
  - leakage_power: Static leakage (mW)
  - dynamic_power: Dynamic switching power (mW)
  - cell_area    : Total cell area (um²)

Usage:
  python parse_logs.py --timing report_timing.rpt \
                       --power  report_power.pwr \
                       --output metrics.csv \
                       [--design-id my_design]
"""

import re
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


TOP_K_PATHS = 10   # number of individual path slack values to record


# ─────────────────────────────── Timing parser ────────────────────────────────

def parse_timing_report(rpt_path: str) -> dict:
    """
    Parse OpenROAD report_timing output.
    Handles both single-path and multi-path report formats.
    """
    metrics = {
        "wns": 0.0,
        "tns": 0.0,
        "clock_period": None,
        "violation_count": 0,
        "path_slacks": [],
        "slack_p0": 0.0,
        "slack_p10": 0.0,
        "slack_p50": 0.0,
        "slack_p90": 0.0,
        "freq_mhz": None,
    }

    if not Path(rpt_path).exists():
        print(f"[WARNING] Timing report not found: {rpt_path}")
        return metrics

    with open(rpt_path) as f:
        content = f.read()

    # ── Clock period ──────────────────────────────────────────────────────────
    cp_match = re.search(r'clock period\s*[=:]\s*([0-9.]+)', content, re.IGNORECASE)
    if not cp_match:
        # Try "Period: X.XX"
        cp_match = re.search(r'Period\s*:\s*([0-9.]+)', content, re.IGNORECASE)
    if cp_match:
        metrics["clock_period"] = float(cp_match.group(1))

    # ── WNS / TNS (summary lines from report_checks) ─────────────────────────
    wns_match = re.search(r'wns\s+([+-]?[0-9.]+)', content, re.IGNORECASE)
    tns_match = re.search(r'tns\s+([+-]?[0-9.]+)', content, re.IGNORECASE)
    if wns_match:
        metrics["wns"] = float(wns_match.group(1))
    if tns_match:
        metrics["tns"] = float(tns_match.group(1))

    # ── Individual path slacks ────────────────────────────────────────────────
    # Matches lines like:  slack (VIOLATED)    -0.123
    #                      slack (MET)          0.456
    slack_pattern = re.compile(
        r'slack\s+\((VIOLATED|MET)\)\s+([+-]?[0-9.]+)', re.IGNORECASE
    )
    slacks = [float(m.group(2)) for m in slack_pattern.finditer(content)]

    if not slacks:
        # Fallback: look for bare "slack = X.XX" lines
        slacks = [float(m.group(1)) for m in
                  re.finditer(r'slack\s*=\s*([+-]?[0-9.]+)', content)]

    if slacks:
        metrics["path_slacks"] = sorted(slacks)[:TOP_K_PATHS]
        metrics["violation_count"] = sum(1 for s in slacks if s < 0)

        arr = np.array(slacks)
        # Update WNS/TNS from path list if not found in summary
        if metrics["wns"] == 0.0:
            metrics["wns"] = float(arr.min())
        if metrics["tns"] == 0.0:
            metrics["tns"] = float(arr[arr < 0].sum()) if (arr < 0).any() else 0.0

        metrics["slack_p0"]  = float(np.percentile(arr, 0))
        metrics["slack_p10"] = float(np.percentile(arr, 10))
        metrics["slack_p50"] = float(np.percentile(arr, 50))
        metrics["slack_p90"] = float(np.percentile(arr, 90))

    # ── Derived frequency ─────────────────────────────────────────────────────
    cp = metrics["clock_period"]
    wns = metrics["wns"]
    if cp is not None:
        achievable_period = cp - wns if wns < 0 else cp
        metrics["freq_mhz"] = round(1000.0 / achievable_period, 2) if achievable_period > 0 else 0.0

    return metrics


# ─────────────────────────────── Power parser ─────────────────────────────────

def parse_power_report(pwr_path: str) -> dict:
    """
    Parse OpenROAD report_power output.

    Expected format (OpenROAD):
        Group       Internal  Switching    Leakage       Total
        ...
        Total       X.XXe-03  X.XXe-03   X.XXe-06   X.XXe-03  100%
    All values in Watts; we convert to mW.
    """
    power = {
        "total_power":   0.0,
        "leakage_power": 0.0,
        "dynamic_power": 0.0,
    }

    if not Path(pwr_path).exists():
        print(f"[WARNING] Power report not found: {pwr_path}")
        return power

    with open(pwr_path) as f:
        content = f.read()

    sci = r'[+-]?[0-9]*\.?[0-9]+(?:[eE][+-]?[0-9]+)?'

    # Total row: Internal  Switching  Leakage  Total
    total_match = re.search(
        rf'Total\s+({sci})\s+({sci})\s+({sci})\s+({sci})',
        content, re.IGNORECASE
    )
    if total_match:
        internal  = float(total_match.group(1))
        switching = float(total_match.group(2))
        leakage   = float(total_match.group(3))
        total     = float(total_match.group(4))
        # Convert W → mW
        power["total_power"]   = total   * 1e3
        power["leakage_power"] = leakage * 1e3
        power["dynamic_power"] = (internal + switching) * 1e3
    else:
        # Fallback: look for bare "Total power: X.XX mW"
        m = re.search(r'[Tt]otal\s+[Pp]ower\s*[:=]\s*(' + sci + r')\s*(m?W)',
                      content)
        if m:
            val = float(m.group(1))
            unit = m.group(2)
            power["total_power"] = val if unit == "mW" else val * 1e3

    return power


# ─────────────────────────────── Area parser ──────────────────────────────────

def parse_area_report(rpt_path: str) -> dict:
    """
    Parse OpenROAD report_design_area or yosys stat output for cell area.
    """
    area = {"cell_area": 0.0}
    if not rpt_path or not Path(rpt_path).exists():
        return area

    with open(rpt_path) as f:
        content = f.read()

    # OpenROAD: "Design area  X.XX u^2"  or  "Chip area for module ...: X.XX"
    m = re.search(r'[Dd]esign\s+[Aa]rea\s+([0-9.eE+\-]+)', content)
    if not m:
        m = re.search(r'[Cc]hip\s+[Aa]rea\s+[^\:]*:\s*([0-9.eE+\-]+)', content)
    if m:
        area["cell_area"] = float(m.group(1))

    return area


# ─────────────────────────────── Main assembler ───────────────────────────────

def parse_all(timing_path: str, power_path: str,
              area_path: str | None = None,
              design_id: str = "unknown") -> dict:
    """Combine timing + power + area into a single metrics dict."""
    metrics = {"design_id": design_id}
    metrics.update(parse_timing_report(timing_path))
    metrics.update(parse_power_report(power_path))
    metrics.update(parse_area_report(area_path))

    # Flatten path_slacks list → individual columns (for CSV storage)
    for i, s in enumerate(metrics.pop("path_slacks", [])):
        metrics[f"path_slack_{i}"] = s

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Parse EDA logs → PPA metrics CSV")
    parser.add_argument("--timing",    required=True, help="Timing .rpt file")
    parser.add_argument("--power",     required=True, help="Power .pwr file")
    parser.add_argument("--area",      default=None,  help="Area report file (optional)")
    parser.add_argument("--output",    required=True, help="Output .csv path")
    parser.add_argument("--design-id", default="unknown", help="Design identifier")
    args = parser.parse_args()

    metrics = parse_all(args.timing, args.power, args.area, args.design_id)
    pd.DataFrame([metrics]).to_csv(args.output, index=False)

    print(f"Design: {args.design_id}")
    print(f"  WNS={metrics['wns']:.3f} ns | TNS={metrics['tns']:.3f} ns "
          f"| Freq={metrics.get('freq_mhz','N/A')} MHz")
    print(f"  Total power={metrics['total_power']:.2f} mW | "
          f"Cell area={metrics['cell_area']:.1f} um²")
    print(f"  Violations={metrics['violation_count']} | "
          f"Saved → {args.output}")


if __name__ == "__main__":
    main()
