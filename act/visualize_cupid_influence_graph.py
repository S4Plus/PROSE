#!/usr/bin/env python3
"""Visualize influence graph edges from CUPID-ACT.

Outputs:
- PNG bipartite plot (matplotlib).
- Optional HTML interactive graph (vis-network via CDN).
- Optional GraphML for Gephi.
"""
from __future__ import annotations

import argparse
import csv
import pathlib
from typing import Dict, List, Tuple

import numpy as np

try:  # pragma: no cover - optional dependency
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError as exc:  # pragma: no cover
    raise SystemExit("Matplotlib is required. Install it via `pip install matplotlib`.") from exc


def _read_nodes(path: pathlib.Path) -> Dict[str, Dict[str, float]]:
    nodes: Dict[str, Dict[str, float]] = {}
    if not path.exists():
        return nodes
    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            node_id = row.get("node_id")
            if not node_id:
                continue
            nodes[node_id] = {
                "type": row.get("type", ""),
                "reward": float(row.get("reward") or 0.0),
                "weight": float(row.get("weight") or 0.0),
            }
    return nodes


def _read_edges(path: pathlib.Path) -> List[Dict[str, object]]:
    edges: List[Dict[str, object]] = []
    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            edges.append(
                {
                    "output_id": row["output_id"],
                    "train_episode_id": row["train_episode_id"],
                    "influence": float(row["influence"]),
                    "abs_influence": float(row.get("abs_influence") or abs(float(row["influence"]))),
                    "sign": row.get("sign") or ("pos" if float(row["influence"]) >= 0 else "neg"),
                }
            )
    return edges


def _select_outputs(
    nodes: Dict[str, Dict[str, float]],
    edges: List[Dict[str, object]],
    max_outputs: int,
    sort_by: str,
) -> List[str]:
    outputs = {e["output_id"] for e in edges}
    def key_fn(oid: str) -> float:
        meta = nodes.get(oid, {})
        reward = float(meta.get("reward", 0.0))
        weight = float(meta.get("weight", 0.0))
        if sort_by == "reward":
            return reward
        if sort_by == "abs_reward":
            return abs(reward)
        if sort_by == "weight":
            return weight
        return 0.0
    if sort_by == "none":
        ordered = sorted(outputs)
    else:
        ordered = sorted(outputs, key=key_fn, reverse=True)
    return ordered[:max_outputs]


def _filter_edges(
    edges: List[Dict[str, object]],
    selected_outputs: List[str],
    top_per_output: int,
    min_abs: float,
) -> List[Dict[str, object]]:
    selected_set = set(selected_outputs)
    filtered = [e for e in edges if e["output_id"] in selected_set and e["abs_influence"] >= min_abs]
    if top_per_output <= 0:
        return filtered
    out_map: Dict[str, List[Dict[str, object]]] = {}
    for e in filtered:
        out_map.setdefault(e["output_id"], []).append(e)
    kept: List[Dict[str, object]] = []
    for oid, items in out_map.items():
        items.sort(key=lambda x: x["abs_influence"], reverse=True)
        kept.extend(items[:top_per_output])
    return kept


def _layout_bipartite(outputs: List[str], trains: List[str]) -> Tuple[Dict[str, Tuple[float, float]], Dict[str, Tuple[float, float]]]:
    out_pos: Dict[str, Tuple[float, float]] = {}
    train_pos: Dict[str, Tuple[float, float]] = {}

    def spaced(n: int) -> np.ndarray:
        if n <= 1:
            return np.array([0.5])
        return np.linspace(0.05, 0.95, n)

    for y, oid in zip(spaced(len(outputs)), outputs):
        out_pos[oid] = (0.1, float(y))
    for y, tid in zip(spaced(len(trains)), trains):
        train_pos[tid] = (0.9, float(y))
    return out_pos, train_pos


def _plot(
    out_pos: Dict[str, Tuple[float, float]],
    train_pos: Dict[str, Tuple[float, float]],
    edges: List[Dict[str, object]],
    output_path: pathlib.Path,
    title: str,
) -> None:
    fig, ax = plt.subplots(figsize=(12, 8))

    # Draw edges
    if edges:
        abs_vals = np.array([e["abs_influence"] for e in edges], dtype=np.float64)
        scale = abs_vals.max() if abs_vals.size > 0 else 1.0
        for e in edges:
            x0, y0 = out_pos[e["output_id"]]
            x1, y1 = train_pos[e["train_episode_id"]]
            color = "#1f77b4" if e["sign"] == "pos" else "#d62728"
            lw = 0.5 + 2.5 * (e["abs_influence"] / max(scale, 1e-8))
            ax.plot([x0, x1], [y0, y1], color=color, alpha=0.6, linewidth=lw)

    # Draw nodes
    if out_pos:
        ox, oy = zip(*out_pos.values())
        ax.scatter(ox, oy, s=30, color="#2ca02c", label="outputs", zorder=3)
    if train_pos:
        tx, ty = zip(*train_pos.values())
        ax.scatter(tx, ty, s=25, color="#9467bd", label="train", zorder=3)

    ax.set_title(title)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.legend(loc="upper center", ncol=2, frameon=False)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _write_graphml(
    nodes: Dict[str, Dict[str, float]],
    edges: List[Dict[str, object]],
    output_path: pathlib.Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    def esc(val: object) -> str:
        text = str(val)
        return (
            text.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
            .replace("'", "&apos;")
        )

    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<graphml xmlns="http://graphml.graphdrawing.org/xmlns">',
        '  <key id="d0" for="node" attr.name="type" attr.type="string"/>',
        '  <key id="d1" for="node" attr.name="reward" attr.type="double"/>',
        '  <key id="d2" for="node" attr.name="weight" attr.type="double"/>',
        '  <key id="d3" for="edge" attr.name="influence" attr.type="double"/>',
        '  <key id="d4" for="edge" attr.name="abs_influence" attr.type="double"/>',
        '  <key id="d5" for="edge" attr.name="sign" attr.type="string"/>',
        '  <key id="d6" for="edge" attr.name="rank" attr.type="int"/>',
        '  <graph id="G" edgedefault="directed">',
    ]
    for node_id, meta in nodes.items():
        lines.append(f'    <node id="{esc(node_id)}">')
        lines.append(f'      <data key="d0">{esc(meta.get("type", ""))}</data>')
        lines.append(f'      <data key="d1">{esc(meta.get("reward", 0.0))}</data>')
        lines.append(f'      <data key="d2">{esc(meta.get("weight", 0.0))}</data>')
        lines.append("    </node>")
    for idx, edge in enumerate(edges):
        source = esc(edge["output_id"])
        target = esc(edge["train_episode_id"])
        lines.append(f'    <edge id="e{idx}" source="{source}" target="{target}">')
        lines.append(f'      <data key="d3">{esc(edge.get("influence", 0.0))}</data>')
        lines.append(f'      <data key="d4">{esc(edge.get("abs_influence", 0.0))}</data>')
        lines.append(f'      <data key="d5">{esc(edge.get("sign", ""))}</data>')
        lines.append(f'      <data key="d6">{esc(edge.get("rank", 0))}</data>')
        lines.append("    </edge>")
    lines.append("  </graph>")
    lines.append("</graphml>")

    with output_path.open("w") as f:
        f.write("\n".join(lines))


def _write_html(
    nodes: Dict[str, Dict[str, float]],
    edges: List[Dict[str, object]],
    output_path: pathlib.Path,
    title: str,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    node_items = []
    for node_id, meta in nodes.items():
        node_type = meta.get("type", "")
        color = "#2ca02c" if node_type == "output" else "#9467bd"
        label = node_id
        tooltip = (
            f"{node_id}\\n{node_type}\\nreward={meta.get('reward', 0.0):.4g}"
            f"\\nweight={meta.get('weight', 0.0):.4g}"
        )
        node_items.append(
            {
                "id": node_id,
                "label": label,
                "title": tooltip,
                "color": color,
                "shape": "dot",
            }
        )
    edge_items = []
    for edge in edges:
        color = "#1f77b4" if edge.get("sign") == "pos" else "#d62728"
        width = 1 + 4 * float(edge.get("abs_influence", 0.0))
        title = (
            f"{edge.get('output_id')} -> {edge.get('train_episode_id')}"
            f"\\ninf={edge.get('influence', 0.0):.4g}"
        )
        edge_items.append(
            {
                "from": edge.get("output_id"),
                "to": edge.get("train_episode_id"),
                "title": title,
                "color": {"color": color},
                "width": width,
                "arrows": "to",
            }
        )

    html = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>{title}</title>
  <style>
    html, body, #mynetwork {{
      width: 100%;
      height: 100%;
      margin: 0;
      padding: 0;
      background: #fff;
    }}
  </style>
  <script src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
</head>
<body>
<div id="mynetwork"></div>
<script>
  const nodes = new vis.DataSet({node_items});
  const edges = new vis.DataSet({edge_items});
  const container = document.getElementById('mynetwork');
  const data = {{ nodes: nodes, edges: edges }};
  const options = {{
    physics: {{
      stabilization: true,
      barnesHut: {{
        gravitationalConstant: -30000,
        springLength: 120,
        springConstant: 0.04
      }}
    }},
    interaction: {{
      hover: true,
      tooltipDelay: 100
    }}
  }};
  const network = new vis.Network(container, data, options);
</script>
</body>
</html>
"""
    with output_path.open("w") as f:
        f.write(html)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize CUPID influence graph (bipartite).")
    parser.add_argument("--edges-csv", type=pathlib.Path, required=True)
    parser.add_argument("--nodes-csv", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--output-html", type=pathlib.Path, default=None, help="Optional HTML output.")
    parser.add_argument("--output-graphml", type=pathlib.Path, default=None, help="Optional GraphML output.")
    parser.add_argument("--max-outputs", type=int, default=50, help="Max output nodes to show.")
    parser.add_argument("--top-per-output", type=int, default=10, help="Max edges per output.")
    parser.add_argument("--min-abs", type=float, default=0.0, help="Minimum abs influence to draw.")
    parser.add_argument(
        "--sort-by",
        type=str,
        default="abs_reward",
        choices=("abs_reward", "reward", "weight", "none"),
        help="How to select outputs when too many.",
    )
    parser.add_argument("--title", type=str, default="CUPID-ACT Influence Graph")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    nodes = _read_nodes(args.nodes_csv)
    edges = _read_edges(args.edges_csv)

    outputs = _select_outputs(nodes, edges, args.max_outputs, args.sort_by)
    edges = _filter_edges(edges, outputs, args.top_per_output, args.min_abs)

    # collect train nodes that remain
    train_ids = sorted({e["train_episode_id"] for e in edges})
    out_pos, train_pos = _layout_bipartite(outputs, train_ids)

    _plot(out_pos, train_pos, edges, args.output, args.title)
    print(f"Saved plot to {args.output}")

    if args.output_graphml is not None:
        sub_nodes = {nid: meta for nid, meta in nodes.items() if nid in set(outputs) | set(train_ids)}
        _write_graphml(sub_nodes, edges, args.output_graphml)
        print(f"Saved GraphML to {args.output_graphml}")
    if args.output_html is not None:
        sub_nodes = {nid: meta for nid, meta in nodes.items() if nid in set(outputs) | set(train_ids)}
        _write_html(sub_nodes, edges, args.output_html, args.title)
        print(f"Saved HTML to {args.output_html}")


if __name__ == "__main__":
    main()
