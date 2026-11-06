import argparse
import os

import h5py
import numpy as np
import matplotlib.pyplot as plt


def _get_object_positions_from_datagen_info(h5_file, demo_name, object_name, frame_index=0):
    datagen_info = h5_file["data"][demo_name].get("datagen_info", None)
    if datagen_info is None:
        return None
    object_poses = datagen_info.get("object_poses", None)
    if object_poses is None:
        return None
    if object_name not in object_poses:
        return None
    poses = object_poses[object_name][()]
    if poses.ndim != 3 or poses.shape[1:] != (4, 4):
        return None
    frame_index = max(0, min(frame_index, poses.shape[0] - 1))
    pos = poses[frame_index, :3, 3]
    return pos


def plot_object_positions(dataset_path, object_name, frame_index, save_dir, title=None):
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(dataset_path)

    os.makedirs(save_dir, exist_ok=True)

    positions = []
    with h5py.File(dataset_path, "r") as f:
        demos = list(f["data"].keys())
        print(f"Loading {len(demos)} demonstrations from {dataset_path}...")
        for demo_name in demos:
            pos = _get_object_positions_from_datagen_info(f, demo_name, object_name, frame_index)
            if pos is None:
                continue
            positions.append(pos)

    if not positions:
        raise RuntimeError(
            f"No positions found for object '{object_name}'. "
            "Ensure datagen_info/object_poses exists in the dataset."
        )

    positions = np.array(positions)

    plt.figure(figsize=(8, 8))
    plt.scatter(positions[:, 0], positions[:, 1], alpha=0.6, s=12)
    plot_title = title if title is not None else f"{object_name} positions (frame {frame_index})"
    plt.title(plot_title)
    plt.xlabel("X Position (m)")
    plt.ylabel("Y Position (m)")
    plt.grid(True)
    plt.axis("equal")

    dataset_base = os.path.splitext(os.path.basename(dataset_path))[0]
    output_path = os.path.join(save_dir, f"{dataset_base}_{object_name}_positions.png")
    plt.savefig(output_path, dpi=200)
    print(f"Plot saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, help="Path to the hdf5 dataset")
    parser.add_argument("--object", type=str, required=True, help="Object name in datagen_info/object_poses")
    parser.add_argument("--frame", type=int, default=0, help="Frame index to visualize (default: 0)")
    parser.add_argument(
        "--save_dir",
        type=str,
        default="/workspace/liushuo/cupid/mimicgen/myscripts/vis",
        help="Directory to save plots",
    )
    parser.add_argument("--title", type=str, default=None, help="Optional plot title")
    args = parser.parse_args()

    plot_object_positions(
        dataset_path=args.dataset,
        object_name=args.object,
        frame_index=args.frame,
        save_dir=args.save_dir,
        title=args.title,
    )
