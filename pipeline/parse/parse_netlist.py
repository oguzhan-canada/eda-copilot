"""
scripts/parse_netlist.py

Parses a gate-level Verilog netlist and returns a PyTorch Geometric Data object.

Node features (per cell):
  [fan_in, fan_out, cell_type_onehot (K dims), cell_area, path_depth]

Edge features:
  [is_clock_net (0/1)]

Usage:
  python parse_netlist.py --netlist path/to/netlist.v \
                          --liberty path/to/lib.lib \
                          --output path/to/graph.pt
"""

import re
import argparse
from collections import defaultdict, deque
from pathlib import Path

import torch
from torch_geometric.data import Data

# ── Known gate types (extend as needed for your PDK) ──────────────────────────
CELL_TYPES = [
    "AND2", "AND3", "OR2", "OR3", "NAND2", "NAND3", "NOR2", "NOR3",
    "XOR2", "XNOR2", "INV", "BUF", "MUX2", "DFFR", "DFFS", "DFF",
    "LATCH", "HA", "FA", "UNKNOWN"
]
CELL_TYPE_IDX = {c: i for i, c in enumerate(CELL_TYPES)}
NUM_CELL_TYPES = len(CELL_TYPES)


def _normalize_cell_type(raw: str) -> str:
    """Map PDK-specific cell names to canonical gate types."""
    raw = raw.upper()
    for ct in CELL_TYPES[:-1]:           # exclude UNKNOWN
        if ct in raw:
            return ct
    return "UNKNOWN"


def parse_liberty_areas(liberty_path: str) -> dict[str, float]:
    """
    Extract cell area values from a .lib file.
    Returns {cell_name: area_float}.  Falls back to 1.0 if file missing.
    """
    areas: dict[str, float] = {}
    if not liberty_path or not Path(liberty_path).exists():
        return areas
    cell_name = None
    with open(liberty_path) as f:
        for line in f:
            cm = re.match(r'\s*cell\s*\(\s*(\w+)\s*\)', line)
            if cm:
                cell_name = cm.group(1)
            am = re.match(r'\s*area\s*:\s*([0-9.eE+\-]+)\s*;', line)
            if am and cell_name:
                areas[cell_name] = float(am.group(1))
    return areas


def parse_netlist(netlist_path: str, liberty_path: str | None = None) -> Data:
    """
    Parse a gate-level Verilog netlist into a PyG Data object.

    Returns
    -------
    Data with attributes:
        x          : FloatTensor [N, 2 + NUM_CELL_TYPES + 2]
                     columns: [fan_in, fan_out, ...one_hot..., cell_area, path_depth]
        edge_index : LongTensor [2, E]
        edge_attr  : FloatTensor [E, 1]  (is_clock_net)
        cell_names : list[str]  (debugging)
        num_nodes  : int
    """
    areas = parse_liberty_areas(liberty_path)

    # ── Parse netlist ──────────────────────────────────────────────────────────
    cell_instances: dict[str, str] = {}       # inst_name → cell_type
    net_drivers:    dict[str, str] = {}       # net_name  → driving inst
    net_loads:      dict[str, list[str]] = defaultdict(list)  # net → [inst, ...]
    clock_nets:     set[str] = set()

    with open(netlist_path) as f:
        content = f.read()

    # Remove block comments
    content = re.sub(r'/\*.*?\*/', '', content, flags=re.DOTALL)

    # Match cell instantiations:  CellType inst_name (.port(net), ...);
    inst_pattern = re.compile(
        r'(\w+)\s+(\w+)\s*\('
        r'(.*?)'
        r'\)\s*;',
        re.DOTALL
    )
    port_pattern = re.compile(r'\.(\w+)\s*\(\s*(\w+)\s*\)')

    keywords = {"module", "endmodule", "input", "output", "wire",
                "reg", "assign", "always", "begin", "end", "if", "else"}

    for m in inst_pattern.finditer(content):
        cell_type_raw, inst_name, port_body = m.group(1), m.group(2), m.group(3)
        if cell_type_raw.lower() in keywords:
            continue
        cell_instances[inst_name] = _normalize_cell_type(cell_type_raw)
        for pm in port_pattern.finditer(port_body):
            port, net = pm.group(1), pm.group(2)
            # Heuristic: output ports usually named Q, Z, ZN, Y, CO, S
            if port.upper() in {"Q", "Z", "ZN", "Y", "CO", "S", "QN"}:
                net_drivers[net] = inst_name
            else:
                net_loads[net].append(inst_name)
            if "clk" in net.lower() or "clock" in net.lower():
                clock_nets.add(net)

    if not cell_instances:
        raise ValueError(f"No cell instances found in {netlist_path}. "
                         "Check that this is a flattened gate-level netlist.")

    # ── Build node index ───────────────────────────────────────────────────────
    node_idx = {name: i for i, name in enumerate(cell_instances)}
    N = len(node_idx)

    fan_in  = defaultdict(int)
    fan_out = defaultdict(int)
    edges: list[tuple[int, int, float]] = []   # (src, dst, is_clock)

    for net, driver in net_drivers.items():
        if driver not in node_idx:
            continue
        is_clk = float(net in clock_nets)
        src = node_idx[driver]
        for load in net_loads.get(net, []):
            if load not in node_idx:
                continue
            dst = node_idx[load]
            edges.append((src, dst, is_clk))
            fan_out[driver] += 1
            fan_in[load] += 1

    # ── Compute topological path depth (BFS from primary inputs) ──────────────
    adj: dict[int, list[int]] = defaultdict(list)
    in_degree = defaultdict(int)
    for src, dst, _ in edges:
        adj[src].append(dst)
        in_degree[dst] += 1

    depth = [0] * N
    queue = deque(n for n in range(N) if in_degree[n] == 0)
    while queue:
        n = queue.popleft()
        for nb in adj[n]:
            if depth[nb] < depth[n] + 1:
                depth[nb] = depth[n] + 1
            in_degree[nb] -= 1
            if in_degree[nb] == 0:
                queue.append(nb)

    max_depth = max(depth) if depth else 1

    # ── Build feature matrix ───────────────────────────────────────────────────
    x = torch.zeros(N, 2 + NUM_CELL_TYPES + 2)  # fan_in, fan_out, onehot, area, depth
    for name, idx in node_idx.items():
        ct = cell_instances[name]
        ct_idx = CELL_TYPE_IDX.get(ct, CELL_TYPE_IDX["UNKNOWN"])
        area = areas.get(name, areas.get(ct, 1.0))

        x[idx, 0] = float(fan_in[name])
        x[idx, 1] = float(fan_out[name])
        x[idx, 2 + ct_idx] = 1.0               # one-hot
        x[idx, 2 + NUM_CELL_TYPES] = area
        x[idx, 2 + NUM_CELL_TYPES + 1] = depth[idx] / max(max_depth, 1)

    # ── Build edge tensors ─────────────────────────────────────────────────────
    if edges:
        src_list, dst_list, clk_list = zip(*edges)
        edge_index = torch.tensor([list(src_list), list(dst_list)], dtype=torch.long)
        edge_attr  = torch.tensor([[c] for c in clk_list], dtype=torch.float)
    else:
        edge_index = torch.zeros(2, 0, dtype=torch.long)
        edge_attr  = torch.zeros(0, 1, dtype=torch.float)

    return Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        cell_names=list(cell_instances.keys()),
        num_nodes=N,
    )


def main():
    parser = argparse.ArgumentParser(description="Parse gate-level netlist to PyG graph")
    parser.add_argument("--netlist",  required=True, help="Path to gate-level Verilog")
    parser.add_argument("--liberty",  default=None,  help="Path to .lib for cell areas")
    parser.add_argument("--output",   required=True, help="Output .pt file path")
    args = parser.parse_args()

    data = parse_netlist(args.netlist, args.liberty)
    torch.save(data, args.output)
    print(f"Saved graph: {data.num_nodes} nodes, {data.edge_index.shape[1]} edges → {args.output}")


if __name__ == "__main__":
    main()
