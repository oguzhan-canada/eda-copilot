"""
scripts/parse_def.py

Parses an OpenROAD placement DEF file and produces:
  - A 2D cell-density numpy array  (grid_size × grid_size)
  - A summary stats dict (for tabular GBDT features)

Usage:
  python parse_def.py --def path/to/placed.def \
                      --output path/to/density.npy \
                      --grid 100
"""

import re
import argparse
import numpy as np
from pathlib import Path


def parse_def(def_path: str, grid_size: int = 100) -> tuple[np.ndarray, dict]:
    """
    Parse a placed DEF file.

    Parameters
    ----------
    def_path   : path to the .def file
    grid_size  : number of bins per dimension (default 100)

    Returns
    -------
    density_map : ndarray [grid_size, grid_size]  — normalized cell density
    stats       : dict with scalar summary statistics for tabular features
    """
    with open(def_path) as f:
        content = f.read()

    # ── Extract die area ──────────────────────────────────────────────────────
    die_match = re.search(
        r'DIEAREA\s*\(\s*(-?\d+)\s+(-?\d+)\s*\)\s*\(\s*(-?\d+)\s+(-?\d+)\s*\)',
        content
    )
    if not die_match:
        raise ValueError(f"DIEAREA not found in {def_path}")

    x_min, y_min = float(die_match.group(1)), float(die_match.group(2))
    x_max, y_max = float(die_match.group(3)), float(die_match.group(4))
    die_w = x_max - x_min
    die_h = y_max - y_min

    if die_w <= 0 or die_h <= 0:
        raise ValueError(f"Invalid die area: ({x_min},{y_min}) → ({x_max},{y_max})")

    # ── Extract component placements ──────────────────────────────────────────
    # Format: - inst_name cell_type + PLACED ( x y ) orientation ;
    comp_pattern = re.compile(
        r'-\s+(\w+)\s+(\w+)\s+\+\s+(?:PLACED|FIXED)\s+'
        r'\(\s*(-?\d+)\s+(-?\d+)\s*\)',
    )

    density = np.zeros((grid_size, grid_size), dtype=np.float32)
    placed_count = 0

    for m in comp_pattern.finditer(content):
        cx, cy = float(m.group(3)), float(m.group(4))
        # Map to grid bin
        bx = int((cx - x_min) / die_w * grid_size)
        by = int((cy - y_min) / die_h * grid_size)
        bx = min(bx, grid_size - 1)
        by = min(by, grid_size - 1)
        density[by, bx] += 1
        placed_count += 1

    if placed_count == 0:
        raise ValueError(f"No PLACED/FIXED components found in {def_path}. "
                         "Ensure this is a post-placement DEF.")

    # Normalize to [0, 1]
    density /= max(density.max(), 1.0)

    # ── Compute summary statistics ─────────────────────────────────────────────
    flat = density.flatten()
    hotspot_threshold = np.percentile(flat[flat > 0], 90) if flat[flat > 0].size else 0
    stats = {
        "placed_cells":         placed_count,
        "die_area_um2":         die_w * die_h,          # raw DEF units (check PDK scale)
        "mean_density":         float(density.mean()),
        "std_density":          float(density.std()),
        "peak_density":         float(density.max()),
        "hotspot_tile_count":   int((density >= hotspot_threshold).sum()),
        "utilization_estimate": float(flat[flat > 0].size / flat.size),
    }

    return density, stats


def main():
    parser = argparse.ArgumentParser(description="Parse DEF → cell density map")
    parser.add_argument("--def",    dest="def_path", required=True,
                        help="Path to placed .def file")
    parser.add_argument("--output", required=True,
                        help="Output .npy path for density array")
    parser.add_argument("--grid",   type=int, default=100,
                        help="Grid resolution (default: 100)")
    parser.add_argument("--stats",  default=None,
                        help="Optional: output .csv path for summary stats")
    args = parser.parse_args()

    density, stats = parse_def(args.def_path, args.grid)
    np.save(args.output, density)
    print(f"Saved density map {density.shape} → {args.output}")
    print(f"Stats: cells={stats['placed_cells']}, "
          f"peak={stats['peak_density']:.3f}, "
          f"utilization≈{stats['utilization_estimate']:.2%}")

    if args.stats:
        import pandas as pd
        pd.DataFrame([stats]).to_csv(args.stats, index=False)
        print(f"Saved stats → {args.stats}")


if __name__ == "__main__":
    main()
