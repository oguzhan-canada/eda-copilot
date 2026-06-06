#!/usr/bin/env python3
"""
scripts/train_gat.py

Train GAT model with leave-one-design-out cross-validation.

Features:
  - 5-fold CV (one fold per design)
  - Masked multi-task loss for heterogeneous labels
  - Per-fold checkpoints and history
  - Best model selection across folds

Usage:
  # Single fold (hold out ibex)
  python scripts/train_gat.py \
    --manifest data/manifest_real.csv \
    --output models/real/ \
    --hold-out ibex --seed 42

  # All folds
  python scripts/train_gat.py \
    --manifest data/manifest_real.csv \
    --output models/real/ \
    --all-folds --seed 42
"""

import argparse
import json
import random
import sys

import pandas as pd
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from features.dataset import PPADataset
from models.gat_ppa import GATForPPA, MaskedPPALoss, count_parameters


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_one_fold(
    manifest: str,
    hold_out: str,
    output_dir: Path,
    seed: int,
    epochs: int,
    batch_size: int,
    lr: float,
    hidden: int,
    layers: int,
    heads: int,
    dropout: float,
    patience: int,
    use_criticality: bool,
    device: str,
):
    """Train a single fold."""
    set_seed(seed)
    tag = "crit" if use_criticality else "nocrit"
    prefix = f"gat_{tag}_holdout_{hold_out}_seed{seed}"

    print(f"\n{'='*60}")
    print(f"Training: {prefix}")
    print(f"{'='*60}")

    # Build datasets
    train_ds = PPADataset(
        manifest, hold_out_design=hold_out, split="train",
        use_criticality=use_criticality, seed=seed,
    )
    test_ds = PPADataset(
        manifest, hold_out_design=hold_out, split="test",
        normalizer=train_ds.normalizer,
        use_criticality=use_criticality, seed=seed,
    )

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        collate_fn=PPADataset.collate_fn,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        collate_fn=PPADataset.collate_fn,
    )

    # Infer dimensions from first sample
    sample = train_ds[0]
    node_feat_dim = sample["graph"].x.shape[1]
    edge_feat_dim = sample["graph"].edge_attr.shape[1] if sample["graph"].edge_attr is not None else 1

    model = GATForPPA(
        node_feat_dim=node_feat_dim,
        edge_feat_dim=edge_feat_dim,
        hidden_dim=hidden,
        num_gat_layers=layers,
        num_heads=heads,
        dropout=dropout,
        use_criticality=use_criticality,
    ).to(device)

    print(f"Parameters: {count_parameters(model):,}")

    criterion = MaskedPPALoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_mae = float("inf")
    best_epoch = 0
    no_improve = 0
    history = {"train_loss": [], "val_mae": [], "lr": []}

    for epoch in range(1, epochs + 1):
        # ── Train ──
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            graph = batch["graph"].to(device)
            spatial = batch["spatial"].to(device)
            timing_vec = batch["timing_vec"].to(device)
            label = batch["label"].to(device)
            mask = batch["label_mask"].to(device)

            pred = model(graph, spatial, timing_vec)
            loss = criterion(pred, label, mask)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()

        scheduler.step()
        avg_loss = total_loss / len(train_loader)

        # ── Validate ──
        model.eval()
        all_preds, all_labels, all_masks = [], [], []
        with torch.no_grad():
            for batch in test_loader:
                graph = batch["graph"].to(device)
                spatial = batch["spatial"].to(device)
                timing_vec = batch["timing_vec"].to(device)

                pred = model(graph, spatial, timing_vec)
                all_preds.append(pred.cpu())
                all_labels.append(batch["label"])
                all_masks.append(batch["label_mask"])

        preds = torch.cat(all_preds)
        labels = torch.cat(all_labels)
        masks = torch.cat(all_masks)

        # MAE on valid entries only
        diff = (preds - labels).abs()
        masked_diff = diff * masks.float()
        valid_counts = masks.float().sum(dim=0).clamp(min=1)
        per_metric_mae = masked_diff.sum(dim=0) / valid_counts
        val_mae = per_metric_mae.mean().item()

        history["train_loss"].append(avg_loss)
        history["val_mae"].append(val_mae)
        history["lr"].append(scheduler.get_last_lr()[0])

        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}/{epochs}  loss={avg_loss:.4f}  val_MAE={val_mae:.4f}")

        # ── Early stopping ──
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_epoch = epoch
            no_improve = 0

            output_dir.mkdir(parents=True, exist_ok=True)
            torch.save({
                "state_dict": model.state_dict(),
                "config": {
                    "node_feat_dim": node_feat_dim,
                    "edge_feat_dim": edge_feat_dim,
                    "hidden_dim": hidden,
                    "num_gat_layers": layers,
                    "num_heads": heads,
                    "dropout": dropout,
                    "use_criticality": use_criticality,
                },
                "val_mae": best_val_mae,
                "epoch": best_epoch,
                "hold_out_design": hold_out,
                "seed": seed,
                "normalizer_min": train_ds.normalizer.min_.tolist(),
                "normalizer_max": train_ds.normalizer.max_.tolist(),
            }, output_dir / f"{prefix}_best.pt")
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"  Early stopping at epoch {epoch} (patience={patience})")
                break

    print(f"  Best: epoch={best_epoch}, val_MAE={best_val_mae:.4f}")

    # Save history
    with open(output_dir / f"{prefix}_history.json", "w") as f:
        json.dump(history, f, indent=2)

    return {"hold_out": hold_out, "seed": seed, "best_val_mae": best_val_mae, "best_epoch": best_epoch}


def main():
    parser = argparse.ArgumentParser(description="Train GAT (Real Data)")
    parser.add_argument("--manifest", type=str, required=True)
    parser.add_argument("--output", type=str, default="models/real")
    parser.add_argument("--hold-out", type=str, default=None,
                        help="Design to hold out (single fold)")
    parser.add_argument("--all-folds", action="store_true",
                        help="Run all 5 folds")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--no-criticality", action="store_true")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    output_dir = Path(args.output)

    if args.all_folds:
        # Dynamically discover designs from manifest
        manifest_df = pd.read_csv(args.manifest)
        designs = sorted(manifest_df["design_name"].unique().tolist())
        print(f"Discovered {len(designs)} designs for leave-one-out CV: {designs}")
        all_results = []
        for design in designs:
            result = train_one_fold(
                args.manifest, design, output_dir, args.seed,
                args.epochs, args.batch_size, args.lr,
                args.hidden, args.layers, args.heads, args.dropout,
                args.patience, not args.no_criticality, args.device,
            )
            all_results.append(result)

        # Summary
        print(f"\n{'='*60}")
        print("Cross-Validation Summary:")
        maes = [r["best_val_mae"] for r in all_results]
        for r in all_results:
            print(f"  {r['hold_out']:20s} val_MAE={r['best_val_mae']:.4f} (epoch {r['best_epoch']})")
        print(f"  {'Mean':20s} val_MAE={np.mean(maes):.4f} ± {np.std(maes):.4f}")
        print(f"{'='*60}")

        # Save best checkpoints index
        best_ckpts = {}
        for r in all_results:
            tag = "crit" if not args.no_criticality else "nocrit"
            ckpt = str(output_dir / f"gat_{tag}_holdout_{r['hold_out']}_seed{args.seed}_best.pt")
            best_ckpts[r["hold_out"]] = ckpt

        with open(output_dir / "best_checkpoints.json", "w") as f:
            json.dump(best_ckpts, f, indent=2)

    elif args.hold_out:
        train_one_fold(
            args.manifest, args.hold_out, output_dir, args.seed,
            args.epochs, args.batch_size, args.lr,
            args.hidden, args.layers, args.heads, args.dropout,
            args.patience, not args.no_criticality, args.device,
        )
    else:
        print("ERROR: Specify --hold-out <design> or --all-folds")
        sys.exit(1)


if __name__ == "__main__":
    main()
