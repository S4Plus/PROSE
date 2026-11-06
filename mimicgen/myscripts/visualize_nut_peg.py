
import argparse

import h5py
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np

def plot_positions(dataset_path, output_path, max_lines, seed):
    with h5py.File(dataset_path, "r") as f:
        demos = list(f["data"].keys())

        paired_positions = []
        skipped = 0

        print(f"Loading {len(demos)} demonstrations from {dataset_path}...")

        for demo_name in demos:
            obs = f["data"][demo_name]["obs"]
            nut_pos = None
            peg_pos = None

            if "object" in obs:
                obj = obs["object"][0]
                nut_pos = obj[:3]  # First 3 dims store nut xyz

                # Peg position: prefer explicit key, otherwise last 3 dims.
                if "peg1_pos" in obs:
                    peg_pos = obs["peg1_pos"][0]
                elif obj.shape[-1] >= 3:
                    peg_pos = obj[-3:]

            if nut_pos is not None and peg_pos is not None:
                paired_positions.append((nut_pos, peg_pos))
            else:
                skipped += 1

    if not paired_positions:
        print("No paired nut/peg positions found, nothing to plot.")
        return

    nut_positions = np.array([pair[0] for pair in paired_positions])
    peg_positions = np.array([pair[1] for pair in paired_positions])

    fig, ax = plt.subplots(figsize=(10, 10))

    # Draw a capped number of connections to reduce visual clutter.
    total = len(paired_positions)
    draw_idx = np.arange(total)
    if max_lines is not None:
        rng = np.random.default_rng(seed)
        draw_idx = rng.choice(draw_idx, size=min(max_lines, total), replace=False)

    for idx in draw_idx:
        peg = peg_positions[idx]
        nut = nut_positions[idx]
        ax.plot([peg[0], nut[0]], [peg[1], nut[1]], color="0.5", alpha=0.25, linewidth=0.7)

    # Nut Positions
    ax.scatter(nut_positions[:, 0], nut_positions[:, 1], c="blue", alpha=0.6, s=15, label="Nut Position")
    nut_x_min, nut_x_max = -0.115, 0.115
    nut_y_min, nut_y_max = -0.255, 0.255
    rect_nut = patches.Rectangle(
        (nut_x_min, nut_y_min),
        nut_x_max - nut_x_min,
        nut_y_max - nut_y_min,
        linewidth=2,
        edgecolor="blue",
        facecolor="none",
        linestyle="--",
        label="Nut Theoretical Range",
    )
    ax.add_patch(rect_nut)

    # Peg Positions
    ax.scatter(peg_positions[:, 0], peg_positions[:, 1], c="red", alpha=0.6, s=15, marker="x", label="Peg Position")
    peg_x_min, peg_x_max = -0.1, 0.3
    peg_y_min, peg_y_max = -0.2, 0.2
    rect_peg = patches.Rectangle(
        (peg_x_min, peg_y_min),
        peg_x_max - peg_x_min,
        peg_y_max - peg_y_min,
        linewidth=2,
        edgecolor="red",
        facecolor="none",
        linestyle="--",
        label="Peg Theoretical Range",
    )
    ax.add_patch(rect_peg)

    ax.set_title("Nut & Peg Sampling with Pairing (Square_D1)")
    ax.set_xlabel("X Position (m)")
    ax.set_ylabel("Y Position (m)")
    ax.axhline(0, color="gray", linestyle=":", alpha=0.5)
    ax.axvline(0, color="gray", linestyle=":", alpha=0.5)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axis("equal")

    plt.savefig(output_path, bbox_inches="tight")
    drawn = len(draw_idx) if max_lines is not None else total
    print(f"Combined visualization saved to {output_path} (lines drawn: {drawn}/{total}, skipped {skipped} demos without pairs)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, help="Path to the hdf5 dataset")
    parser.add_argument("--output", type=str, default="visualization.png", help="Path to save plot")
    parser.add_argument("--max-lines", type=int, default=200, help="Max number of peg-nut links to draw (None for all)")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for sub-sampling links")
    args = parser.parse_args()

    plot_positions(args.dataset, args.output, args.max_lines, args.seed)
