# Generation Scripts — AI-Specific Networking

This directory contains Python scripts to generate diagrams for the AI-Specific Networking chapter.

## Scripts

| Script | Output | Description |
|--------|--------|-------------|
| `gen_network_topology.py` | `img/network_topology_comparison.png` | Compare PCIe vs NVLink topology (GPU communication paths) |
| `gen_bandwidth_comparison.py` | `img/bandwidth_comparison.png` | Bar chart comparing interconnect bandwidth (PCIe, NVLink, InfiniBand) |
| `gen_latency_heatmap.py` | `img/latency_heatmap.png` | Heatmaps showing GPU-to-GPU latency for different topologies |
| `gen_decision_tree.py` | `img/decision_tree.png` | Decision tree for choosing interconnect topology |
| `generate_all.py` | *(runs all above)* | Master script to generate all diagrams |

## Usage

### Generate all diagrams at once:
```bash
python generate_all.py
```

### Generate individual diagrams:
```bash
python gen_network_topology.py
python gen_bandwidth_comparison.py
python gen_latency_heatmap.py
python gen_decision_tree.py
```

## Requirements

All scripts require:
- Python 3.8+
- matplotlib
- numpy
- seaborn (for heatmaps)

Install dependencies:
```bash
pip install matplotlib numpy seaborn
```

## Output

All generated PNG files are saved to `../img/` directory with:
- Dark background (`#1a1a2e`) matching the notebook theme
- High resolution (150 DPI)
- Consistent color scheme:
  - 🔴 Red (#b91c1c): PCIe (slow)
  - 🟢 Green (#15803d): NVLink (fast)
  - 🔵 Blue (#1d4ed8): InfiniBand (multi-node)
  - 🟡 Yellow: Highlights and annotations

## Customization

To change diagram parameters (e.g., bandwidth values, latency estimates), edit the corresponding script:
- Bandwidth values: See `bandwidths` array in `gen_bandwidth_comparison.py`
- Latency values: See `nvlink_latency`, `pcie_latency` arrays in `gen_latency_heatmap.py`
- Topology layout: Adjust `gpu_positions` in `gen_network_topology.py`
- Decision tree: Modify node positions and text in `gen_decision_tree.py`

## Regeneration

These scripts are **deterministic** — running them multiple times produces identical output. Regenerate anytime the source data changes or to update the visual style.
