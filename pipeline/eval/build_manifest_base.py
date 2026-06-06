#!/usr/bin/env python3
"""
scripts/build_manifest.py

Assemble manifest_real.csv from all data sources.
Merges partial manifests from OpenABC-D, CircuitNet, and OpenROAD runs
into a single file following the schema in configs/default.yaml.

Usage:
  python scripts/build_manifest.py \
    --openabc data/raw/openabc_d/openabc_manifest_partial.csv \
    --circuitnet data/raw/circuitnet/circuitnet_manifest_partial.csv \
    --openroad data/raw/openroad_runs/ \
    --timing features/timing_stats.csv \
    --spatial features/spatial/spatial_stats.csv \
    --output data/manifest_real.csv
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def load_openroad_manifest(
    runs_dir: str,
    timing_csv: str | None = None,
    spatial_csv: str | None = None,
) -> pd.DataFrame:
    """Build manifest rows from OpenROAD run results."""
    runs_path = Path(runs_dir)
    rows = []

    # Load timing stats if available
    timing_df = None
    if timing_csv and Path(timing_csv).exists():
        timing_df = pd.read_csv(timing_csv)
        timing_df = timing_df.set_index("run_id")

    # Load spatial stats if available
    spatial_df = None
    if spatial_csv and Path(spatial_csv).exists():
        spatial_df = pd.read_csv(spatial_csv)

    for run_dir in sorted(runs_path.iterdir()):
        if not run_dir.is_dir():
            continue

        status_file = run_dir / "run_status.json"
        if not status_file.exists():
            continue

        status = json.loads(status_file.read_text())
        if status.get("status") != "success":
            continue

        run_id = status["run_id"]
        design = status["design"]
        clock_ns = status["clock_ns"]

        row = {
            "sample_id": run_id,
            "source": "openroad",
            "design_name": design,
            "tech_node": "7nm",
            "clock_target_ns": clock_ns,
            "graph_path": "",
            "spatial_path": "",
            "timing_rpt_path": status.get("timing_rpt", ""),
            "run_status": "success",
            "tool_version": "openroad_4b4c5a7",
        }

        # Merge timing stats
        if timing_df is not None and run_id in timing_df.index:
            ts = timing_df.loc[run_id]
            for col in ["wns", "tns", "freq_mhz", "slack_p0", "slack_p10",
                        "slack_p50", "slack_p90", "violation_count", "clock_period",
                        "total_power", "cell_area", "timing_rpt_path"]:
                if col in ts.index:
                    row[col] = ts[col]

        # Check for graph file
        graph_candidates = [
            Path(f"features/graph/{run_id}.pt"),
            Path(f"features/graph/{design}.pt"),
        ]
        for gc in graph_candidates:
            if gc.exists():
                row["graph_path"] = str(gc)
                break

        # Check for spatial file
        spatial_candidates = [
            Path(f"features/spatial/{run_id}.npy"),
        ]
        for sc in spatial_candidates:
            if sc.exists():
                row["spatial_path"] = str(sc)
                break

        # Set availability flags
        row["has_delay"] = not (pd.isna(row.get("freq_mhz")) or row.get("freq_mhz") is None)
        row["has_power"] = not (pd.isna(row.get("total_power")) or row.get("total_power") is None)

        rows.append(row)

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="Build unified manifest")
    parser.add_argument("--openabc", type=str, default=None,
                        help="OpenABC-D partial manifest CSV")
    parser.add_argument("--circuitnet", type=str, default=None,
                        help="CircuitNet partial manifest CSV")
    parser.add_argument("--openroad", type=str, default=None,
                        help="OpenROAD runs directory")
    parser.add_argument("--timing", type=str, default=None,
                        help="Timing stats CSV")
    parser.add_argument("--spatial", type=str, default=None,
                        help="Spatial stats CSV")
    parser.add_argument("--output", type=str, default="data/manifest_real.csv",
                        help="Output manifest path")
    args = parser.parse_args()

    dfs = []

    # OpenABC-D
    if args.openabc and Path(args.openabc).exists():
        df_abc = pd.read_csv(args.openabc)
        print(f"OpenABC-D: {len(df_abc)} samples")
        dfs.append(df_abc)

    # CircuitNet
    if args.circuitnet and Path(args.circuitnet).exists():
        df_cn = pd.read_csv(args.circuitnet)
        print(f"CircuitNet: {len(df_cn)} samples")
        dfs.append(df_cn)

    # OpenROAD
    if args.openroad:
        df_or = load_openroad_manifest(args.openroad, args.timing, args.spatial)
        print(f"OpenROAD: {len(df_or)} samples")
        dfs.append(df_or)

    if not dfs:
        print("ERROR: No data sources provided. Use --openabc, --circuitnet, or --openroad")
        return

    # Merge all sources
    manifest = pd.concat(dfs, ignore_index=True)

    # Fill missing columns with defaults
    for col in ["graph_path", "spatial_path", "timing_rpt_path"]:
        if col not in manifest.columns:
            manifest[col] = ""

    for col in ["delay", "total_power", "freq_mhz", "cell_area",
                 "wns", "tns", "slack_p0", "slack_p10", "slack_p50",
                 "slack_p90", "clock_period"]:
        if col not in manifest.columns:
            manifest[col] = np.nan

    for col in ["violation_count", "node_count", "total_edges",
                 "max_fan_out", "register_count", "hotspot_tile_count"]:
        if col not in manifest.columns:
            manifest[col] = 0

    for col in ["avg_path_depth", "edge_density", "mean_density",
                 "std_density", "peak_density", "utilization_estimate"]:
        if col not in manifest.columns:
            manifest[col] = 0.0

    for col in ["has_delay", "has_power"]:
        if col not in manifest.columns:
            manifest[col] = False

    if "run_status" not in manifest.columns:
        manifest["run_status"] = "success"

    if "clock_target_ns" not in manifest.columns:
        manifest["clock_target_ns"] = manifest.get("clock_period", np.nan)

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(output_path, index=False)

    # Summary
    print(f"\n{'='*60}")
    print(f"Manifest saved: {output_path}")
    print(f"Total samples: {len(manifest)}")
    if 'source' in manifest.columns:
        print(f"Sources: {manifest['source'].value_counts().to_dict()}")
    if 'has_delay' in manifest.columns:
        print(f"Has delay: {manifest['has_delay'].sum()}")
    if 'has_power' in manifest.columns:
        print(f"Has power: {manifest['has_power'].sum()}")
    if 'design_name' in manifest.columns:
        print(f"Designs: {manifest['design_name'].nunique()}")
    print(f"Columns: {list(manifest.columns)}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
