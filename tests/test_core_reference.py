"""
tests/test_core.py

Unit tests for the real-data PPA framework.
Tests cover: manifest validation, masked loss, dataset splits,
feature extraction, and model forward passes.

Run: python -m pytest tests/test_core.py -v
"""

import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from torch_geometric.data import Data

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ─────────────────────────── Fixtures ─────────────────────────────────────────

@pytest.fixture
def sample_manifest(tmp_path):
    """Create a minimal manifest CSV for testing."""
    rows = []
    designs = ["ibex", "swerv_wrapper", "riscv32i", "jpeg_encoder", "aes128"]
    for i, design in enumerate(designs):
        for clk in [2.0, 3.0]:
            # Create dummy graph
            graph = Data(
                x=torch.randn(20, 25),  # 22 gate types + fan_in + fan_out + depth
                edge_index=torch.randint(0, 20, (2, 40)),
                edge_attr=torch.rand(40, 1),
            )
            graph_path = tmp_path / f"graph_{design}_{clk}.pt"
            torch.save(graph, graph_path)

            # Create dummy spatial
            spatial = np.random.rand(100, 100).astype(np.float32)
            spatial_path = tmp_path / f"spatial_{design}_{clk}.npy"
            np.save(spatial_path, spatial)

            rows.append({
                "sample_id": f"{design}_clk{clk}",
                "source": "openroad",
                "design_name": design,
                "tech_node": "7nm",
                "clock_target_ns": clk,
                "graph_path": str(graph_path),
                "spatial_path": str(spatial_path),
                "timing_rpt_path": "",
                "delay": 1.5 + i * 0.1 + clk * 0.2,
                "total_power": 10.0 + i * 2.0 + clk,
                "freq_mhz": 1000.0 / clk,
                "cell_area": 50000 + i * 5000,
                "wns": -0.1 * i,
                "tns": -0.5 * i,
                "slack_p0": -0.2,
                "slack_p10": 0.0,
                "slack_p50": 0.3,
                "slack_p90": 0.8,
                "violation_count": i,
                "clock_period": clk,
                "node_count": 20,
                "total_edges": 40,
                "max_fan_out": 5,
                "register_count": 3,
                "avg_path_depth": 4.0,
                "edge_density": 2.0,
                "mean_density": 0.5,
                "std_density": 0.2,
                "peak_density": 0.9,
                "hotspot_tile_count": 5,
                "utilization_estimate": 0.7,
                "has_delay": True,
                "has_power": True,
                "run_status": "success",
                "tool_version": "test",
            })

    manifest_path = tmp_path / "manifest_test.csv"
    pd.DataFrame(rows).to_csv(manifest_path, index=False)
    return manifest_path


@pytest.fixture
def partial_manifest(tmp_path):
    """Manifest with mixed label availability (delay-only + full PPA)."""
    rows = []
    for i in range(10):
        graph = Data(
            x=torch.randn(10, 25),
            edge_index=torch.randint(0, 10, (2, 20)),
            edge_attr=torch.rand(20, 1),
        )
        graph_path = tmp_path / f"graph_{i}.pt"
        torch.save(graph, graph_path)

        has_power = i >= 5  # first 5 are delay-only
        rows.append({
            "sample_id": f"sample_{i}",
            "source": "openabc_d" if not has_power else "openroad",
            "design_name": "ibex" if i < 5 else "aes128",
            "clock_target_ns": 3.0,
            "graph_path": str(graph_path),
            "spatial_path": "",
            "timing_rpt_path": "",
            "delay": 2.0 + i * 0.1,
            "total_power": 15.0 + i if has_power else np.nan,
            "freq_mhz": 333.0 + i * 10,
            "cell_area": 60000 + i * 1000 if has_power else np.nan,
            "has_delay": True,
            "has_power": has_power,
            "run_status": "success",
        })

    path = tmp_path / "manifest_partial.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


# ─────────────────────────── Dataset Tests ────────────────────────────────────

class TestPPADataset:
    def test_load_full_manifest(self, sample_manifest):
        from features.dataset import PPADataset
        ds = PPADataset(sample_manifest, split="all")
        assert len(ds) == 10  # 5 designs × 2 clocks

    def test_leave_one_out_split(self, sample_manifest):
        from features.dataset import PPADataset
        train_ds = PPADataset(sample_manifest, hold_out_design="ibex", split="train")
        test_ds = PPADataset(
            sample_manifest, hold_out_design="ibex", split="test",
            normalizer=train_ds.normalizer,
        )
        assert len(train_ds) == 8  # 4 designs × 2 clocks
        assert len(test_ds) == 2   # ibex × 2 clocks

    def test_no_data_leakage(self, sample_manifest):
        from features.dataset import PPADataset
        train_ds = PPADataset(sample_manifest, hold_out_design="ibex", split="train")
        test_ds = PPADataset(
            sample_manifest, hold_out_design="ibex", split="test",
            normalizer=train_ds.normalizer,
        )
        train_designs = set(train_ds.df["design_name"])
        test_designs = set(test_ds.df["design_name"])
        assert train_designs & test_designs == set()

    def test_sample_shape(self, sample_manifest):
        from features.dataset import PPADataset
        ds = PPADataset(sample_manifest, split="all")
        sample = ds[0]
        assert sample["label"].shape == (3,)
        assert sample["label_mask"].shape == (3,)
        assert sample["spatial"].shape == (1, 100, 100)
        assert sample["timing_vec"].ndim == 1

    def test_label_mask_partial(self, partial_manifest):
        from features.dataset import PPADataset
        ds = PPADataset(partial_manifest, split="all")
        # First 5 samples: delay-only (has_power=False)
        for i in range(5):
            mask = ds[i]["label_mask"]
            assert mask[0].item() == False   # power
            assert mask[1].item() == True    # delay/perf
            assert mask[2].item() == False   # area

    def test_full_label_only_filter(self, partial_manifest):
        from features.dataset import PPADataset
        ds = PPADataset(partial_manifest, split="all", full_label_only=True)
        assert len(ds) == 5  # only the 5 OpenROAD samples

    def test_collate_fn(self, sample_manifest):
        from features.dataset import PPADataset
        ds = PPADataset(sample_manifest, split="all")
        batch = PPADataset.collate_fn([ds[0], ds[1]])
        assert batch["label"].shape == (2, 3)
        assert batch["label_mask"].shape == (2, 3)
        assert batch["spatial"].shape == (2, 1, 100, 100)


# ─────────────────────────── Label Normalizer ─────────────────────────────────

class TestLabelNormalizer:
    def test_fit_transform_roundtrip(self):
        from features.dataset import LabelNormalizer
        labels = np.array([[10, 100, 50000], [20, 200, 100000]], dtype=np.float32)
        norm = LabelNormalizer().fit(labels)
        transformed = norm.transform(labels)
        assert transformed.min() >= 0.0
        assert transformed.max() <= 1.0
        recovered = norm.inverse_transform(transformed)
        np.testing.assert_allclose(recovered, labels, atol=1e-3)

    def test_nan_handling(self):
        from features.dataset import LabelNormalizer
        labels = np.array([[10, 100, np.nan], [20, np.nan, 100000]], dtype=np.float32)
        mask = ~np.isnan(labels)
        norm = LabelNormalizer().fit(labels, mask)
        transformed = norm.transform(labels)
        assert not np.any(np.isnan(transformed))


# ─────────────────────────── Masked Loss ──────────────────────────────────────

class TestMaskedLoss:
    def test_full_mask(self):
        from models.gat_ppa import MaskedPPALoss
        loss_fn = MaskedPPALoss()
        pred = torch.tensor([[0.5, 0.5, 0.5]])
        target = torch.tensor([[0.3, 0.7, 0.4]])
        mask = torch.tensor([[True, True, True]])
        loss = loss_fn(pred, target, mask)
        assert loss.item() > 0

    def test_partial_mask(self):
        from models.gat_ppa import MaskedPPALoss
        loss_fn = MaskedPPALoss()
        pred = torch.tensor([[0.5, 0.5, 0.5]])
        target = torch.tensor([[0.3, 0.7, 0.4]])
        # Only perf is valid
        mask = torch.tensor([[False, True, False]])
        loss = loss_fn(pred, target, mask)
        # Should only include perf loss
        expected = (0.5 - 0.7) ** 2  # MSE for perf
        assert abs(loss.item() - expected) < 1e-5

    def test_no_mask(self):
        from models.gat_ppa import MaskedPPALoss
        loss_fn = MaskedPPALoss()
        pred = torch.tensor([[0.5, 0.5, 0.5]])
        target = torch.tensor([[0.3, 0.7, 0.4]])
        loss = loss_fn(pred, target, mask=None)
        assert loss.item() > 0

    def test_batch_normalization(self):
        """Verify loss is normalized by valid count, not batch size."""
        from models.gat_ppa import MaskedPPALoss
        loss_fn = MaskedPPALoss()
        pred = torch.tensor([[0.5, 0.5, 0.5], [0.5, 0.5, 0.5]])
        target = torch.tensor([[0.3, 0.7, 0.4], [0.3, 0.7, 0.4]])
        mask1 = torch.tensor([[True, True, True], [True, True, True]])
        mask2 = torch.tensor([[True, True, True], [False, False, False]])

        loss1 = loss_fn(pred, target, mask1)
        loss2 = loss_fn(pred, target, mask2)
        # With mask2, only first sample contributes, but normalized by 1 not 2
        # So loss should be same per-sample
        assert abs(loss1.item() - loss2.item()) < 1e-5


# ─────────────────────────── GAT Model ────────────────────────────────────────

class TestGATForPPA:
    def test_forward_shape(self):
        from models.gat_ppa import GATForPPA
        model = GATForPPA(node_feat_dim=25, edge_feat_dim=2)
        graph = Data(
            x=torch.randn(20, 25),
            edge_index=torch.randint(0, 20, (2, 40)),
            edge_attr=torch.rand(40, 2),
            batch=torch.zeros(20, dtype=torch.long),
        )
        from torch_geometric.data import Batch
        graph = Batch.from_data_list([graph])
        spatial = torch.randn(1, 1, 100, 100)
        timing = torch.randn(1, 15)  # TIMING_VEC_DIM
        out = model(graph, spatial, timing)
        assert out.shape == (1, 3)
        assert (out >= 0).all() and (out <= 1).all()

    def test_mc_dropout(self):
        from models.gat_ppa import GATForPPA
        model = GATForPPA(node_feat_dim=25, edge_feat_dim=2)
        graph = Data(
            x=torch.randn(20, 25),
            edge_index=torch.randint(0, 20, (2, 40)),
            edge_attr=torch.rand(40, 2),
            batch=torch.zeros(20, dtype=torch.long),
        )
        from torch_geometric.data import Batch
        graph = Batch.from_data_list([graph])
        spatial = torch.randn(1, 1, 100, 100)
        timing = torch.randn(1, 15)
        mean, var = model.predict_with_uncertainty(graph, spatial, timing, T=10)
        assert mean.shape == (1, 3)
        assert var.shape == (1, 3)
        assert (var >= 0).all()


# ─────────────────────────── Criticality ──────────────────────────────────────

class TestCriticality:
    def test_fallback_weights(self):
        from features.criticality import compute_criticality_weights
        graph = Data(
            x=torch.tensor([[0.0, 0.1], [0.0, 0.5], [0.0, 1.0]]),
            edge_index=torch.tensor([[0, 1], [1, 2]]),
        )
        weights = compute_criticality_weights(graph, None, 0.1)
        assert weights.shape == (2,)
        assert (weights >= 0).all() and (weights <= 1).all()

    def test_no_edges(self):
        from features.criticality import compute_criticality_weights
        graph = Data(
            x=torch.randn(5, 3),
            edge_index=torch.zeros(2, 0, dtype=torch.long),
        )
        weights = compute_criticality_weights(graph)
        assert weights.shape == (0,)


# ─────────────────────────── GBDT Model ───────────────────────────────────────

class TestGBDT:
    def test_feature_matrix(self):
        from models.gbdt_ppa import build_feature_matrix, ALL_FEATURE_COLS
        df = pd.DataFrame({col: [1.0, 2.0] for col in ALL_FEATURE_COLS})
        X = build_feature_matrix(df)
        assert X.shape == (2, len(ALL_FEATURE_COLS))

    def test_missing_columns(self):
        from models.gbdt_ppa import build_feature_matrix, ALL_FEATURE_COLS
        df = pd.DataFrame({"wns": [1.0, 2.0], "tns": [3.0, 4.0]})
        X = build_feature_matrix(df)
        assert X.shape == (2, len(ALL_FEATURE_COLS))
        # Missing columns should be 0
        assert X[:, 0].sum() == 0  # max_fan_out not in df


# ─────────────────────────── Config Loader ────────────────────────────────────

class TestConfig:
    def test_load_config(self):
        from configs.loader import load_config
        # Reset cache
        import configs.loader
        configs.loader._CONFIG_CACHE = None
        cfg = load_config()
        assert "designs" in cfg
        assert "training" in cfg

    def test_get_designs(self):
        from configs.loader import get_designs
        import configs.loader
        configs.loader._CONFIG_CACHE = None
        designs = get_designs()
        assert "ibex" in designs
        assert len(designs) == 5

    def test_get_clock_targets(self):
        from configs.loader import get_clock_targets
        import configs.loader
        configs.loader._CONFIG_CACHE = None
        targets = get_clock_targets()
        assert 2.0 in targets
        assert 6.0 in targets
        assert len(targets) == 11
