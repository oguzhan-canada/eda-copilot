#!/usr/bin/env python3
"""
scripts/train_gbdt.py

Train XGBoost GBDT baseline with leave-one-design-out CV.
Uses only full-label samples (all 3 PPA values available).

Usage:
  python scripts/train_gbdt.py \
    --manifest data/manifest_real.csv \
    --output models/gbdt/ \
    --seed 42
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.gbdt_ppa import GBDTPPAModel, build_feature_matrix, TARGET_COLS, TARGET_NAMES


def main():
    parser = argparse.ArgumentParser(description="Train GBDT (Real Data)")
    parser.add_argument("--manifest", type=str, required=True)
    parser.add_argument("--output", type=str, default="models/gbdt")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.manifest)

    # Filter to full-label samples only
    if "has_power" in df.columns and "has_delay" in df.columns:
        df = df[(df["has_power"].astype(bool)) & (df["has_delay"].astype(bool))]
    print(f"Full-label samples: {len(df)}")

    if len(df) < 10:
        print("ERROR: Not enough full-label samples for GBDT training")
        sys.exit(1)

    # Build features and targets
    X = build_feature_matrix(df)

    # Handle target column naming
    target_cols = []
    for col in TARGET_COLS:
        if col in df.columns:
            target_cols.append(col)
        elif col == "freq_mhz" and "freq_mhz_label" in df.columns:
            target_cols.append("freq_mhz_label")
        else:
            target_cols.append(col)
    y = df[target_cols].values.astype(np.float32)

    designs = df["design_name"].values if "design_name" in df.columns else None

    # Grid search with leave-one-design-out
    model = GBDTPPAModel({"random_state": args.seed})
    print("Running hyperparameter search...")
    search_result = model.cv_search(X, y, designs=designs)

    print(f"Best params: {search_result['best_params']}")
    print(f"Best CV MAE: {search_result['best_cv_mae']:.4f}")

    # Save CV results
    cv_df = pd.DataFrame(search_result["all_results"])
    cv_df.to_csv(output_dir / "gbdt_cv_results.csv", index=False)

    # Train final model with best params on all data
    print("Training final model on all full-label data...")
    model.train(X, y, params=search_result["best_params"])

    # Per-design evaluation
    if designs is not None:
        print("\nPer-design evaluation:")
        for design in np.unique(designs):
            mask = designs == design
            if mask.sum() > 0:
                metrics = model.evaluate(X[mask], y[mask])
                print(f"  {design:20s} MAE={metrics['avg_mae']:.4f}")

    # Save
    model.save(str(output_dir))

    # Feature importance
    fi = model.feature_importance()
    fi.to_csv(output_dir / "gbdt_feature_importance.csv", index=False)

    # Eval report
    eval_report = {
        "best_params": search_result["best_params"],
        "best_cv_mae": search_result["best_cv_mae"],
        "n_samples": len(df),
        "n_designs": len(np.unique(designs)) if designs is not None else 1,
        "seed": args.seed,
    }
    with open(output_dir / "gbdt_eval_report.json", "w") as f:
        json.dump(eval_report, f, indent=2)

    print(f"\nModel saved to {output_dir}")


if __name__ == "__main__":
    main()
