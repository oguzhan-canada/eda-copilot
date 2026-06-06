#!/usr/bin/env python3
"""
scripts/extract_graph.py

Convert post-synthesis netlists to PyG graph objects.
Works with: Verilog netlists from OpenROAD, or pre-existing PyG files.

Node features: [gate_type_onehot, fan_in, fan_out, cell_area, depth_norm]
Edge features: [is_clock_net]

Usage:
  python scripts/extract_graph.py \
    --input data/raw/openroad_runs/ \
    --output features/graph/ \
    --liberty data/liberty/asap7.lib
"""

import argparse
import re
from pathlib import Path

import numpy as np
import torch
from torch_geometric.data import Data
from tqdm import tqdm


# ASAP7 standard cell gate types
GATE_TYPES = [
    "INVx1", "INVx2", "BUFx2", "BUFx4",
    "AND2x1", "AND2x2", "OR2x1", "OR2x2",
    "NAND2x1", "NAND2x2", "NOR2x1", "NOR2x2",
    "XOR2x1", "XNOR2x1",
    "AOI21x1", "OAI21x1",
    "DFFx1", "DFFx2",
    "MUX2x1", "HA1x1", "FA1x1",
    "UNKNOWN",
]
GATE_TO_IDX = {g: i for i, g in enumerate(GATE_TYPES)}
NUM_GATE_TYPES = len(GATE_TYPES)


def parse_verilog_netlist(verilog_path: str) -> Data:
    """
    Parse a synthesized Verilog netlist into a PyG Data object.

    Extracts:
      - Cells (instances) as nodes
      - Net connectivity as edges
      - Gate types as node features
    """
    with open(verilog_path) as f:
        content = f.read()

    # Extract module instances
    inst_pattern = re.compile(
        r'(\w+)\s+(\w+)\s*\((.*?)\)\s*;',
        re.DOTALL
    )

    cells = []
    nets = {}
    cell_to_idx = {}

    for m in inst_pattern.finditer(content):
        gate_type = m.group(1)
        inst_name = m.group(2)
        ports = m.group(3)

        # Skip module/endmodule
        if gate_type in ("module", "endmodule", "input", "output", "wire", "assign"):
            continue

        idx = len(cells)
        cell_to_idx[inst_name] = idx

        # Determine gate type index
        gate_idx = GATE_TO_IDX.get(gate_type, GATE_TO_IDX["UNKNOWN"])
        # Match ASAP7 naming: strip size suffix
        for gt in GATE_TYPES[:-1]:
            if gate_type.startswith(gt.split("x")[0]):
                gate_idx = GATE_TO_IDX[gt]
                break

        cells.append({
            "name": inst_name,
            "gate_type": gate_type,
            "gate_idx": gate_idx,
        })

        # Parse port connections to build net connectivity
        port_pattern = re.compile(r'\.(\w+)\s*\(\s*(\w+)\s*\)')
        for pm in port_pattern.finditer(ports):
            pin_name = pm.group(1)
            net_name = pm.group(2)
            if net_name not in nets:
                nets[net_name] = {"drivers": [], "sinks": []}
            if pin_name in ("Y", "Z", "Q", "ZN", "CO", "S"):
                nets[net_name]["drivers"].append(idx)
            else:
                nets[net_name]["sinks"].append(idx)

    if not cells:
        raise ValueError(f"No cells found in {verilog_path}")

    N = len(cells)

    # Build edges: driver → sink for each net
    edge_src, edge_dst = [], []
    is_clock = []
    for net_name, conn in nets.items():
        clk = 1.0 if "clk" in net_name.lower() or "clock" in net_name.lower() else 0.0
        for drv in conn["drivers"]:
            for snk in conn["sinks"]:
                edge_src.append(drv)
                edge_dst.append(snk)
                is_clock.append(clk)

    # Build node features
    node_features = []
    fan_in = np.zeros(N)
    fan_out = np.zeros(N)

    for net_name, conn in nets.items():
        for drv in conn["drivers"]:
            fan_out[drv] += len(conn["sinks"])
        for snk in conn["sinks"]:
            fan_in[snk] += len(conn["drivers"])

    # Compute topological depth (BFS from primary inputs)
    depth = np.zeros(N)
    if edge_src:
        from collections import deque
        adj = [[] for _ in range(N)]
        in_degree = np.zeros(N, dtype=int)
        for s, d in zip(edge_src, edge_dst):
            adj[s].append(d)
            in_degree[d] += 1

        queue = deque()
        for i in range(N):
            if in_degree[i] == 0:
                queue.append(i)
                depth[i] = 0

        while queue:
            node = queue.popleft()
            for child in adj[node]:
                depth[child] = max(depth[child], depth[node] + 1)
                in_degree[child] -= 1
                if in_degree[child] == 0:
                    queue.append(child)

    max_depth = depth.max() if depth.max() > 0 else 1.0

    for i, cell in enumerate(cells):
        # One-hot gate type + fan_in + fan_out + depth_norm
        onehot = [0.0] * NUM_GATE_TYPES
        onehot[cell["gate_idx"]] = 1.0
        feat = onehot + [fan_in[i], fan_out[i], depth[i] / max_depth]
        node_features.append(feat)

    x = torch.tensor(node_features, dtype=torch.float32)
    edge_index = torch.tensor([edge_src, edge_dst], dtype=torch.long) if edge_src else torch.zeros(2, 0, dtype=torch.long)
    edge_attr = torch.tensor([[c] for c in is_clock], dtype=torch.float32) if is_clock else torch.zeros(0, 1)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    data.cell_names = [c["name"] for c in cells]
    data.num_nodes = N

    return data


def main():
    parser = argparse.ArgumentParser(description="Extract graphs from netlists")
    parser.add_argument("--input", type=str, required=True,
                        help="Directory with OpenROAD run results")
    parser.add_argument("--output", type=str, default="features/graph",
                        help="Output directory for PyG files")
    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find Verilog netlists in run directories
    verilog_files = list(input_dir.rglob("*.v")) + list(input_dir.rglob("*.sv"))
    print(f"Found {len(verilog_files)} netlist files")

    for vf in tqdm(verilog_files, desc="Extracting graphs"):
        try:
            data = parse_verilog_netlist(str(vf))
            out_path = output_dir / f"{vf.stem}.pt"
            torch.save(data, out_path)
        except Exception as e:
            print(f"  SKIP {vf.name}: {e}")

    print(f"Saved graphs to {output_dir}")


if __name__ == "__main__":
    main()
