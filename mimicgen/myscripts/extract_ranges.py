
import h5py
import numpy as np
import argparse

def extract_ranges(dataset_path):
    f = h5py.File(dataset_path, "r")
    demos = list(f["data"].keys())
    
    nut_positions = []
    peg_positions = []
    
    print(f"Analyzing {len(demos)} demonstrations from {dataset_path}...")
    
    # Check keys first
    demo0 = f["data"][demos[0]]
    obs_keys = list(demo0["obs"].keys())
    print(f"Available observation keys: {obs_keys}")
    
    for demo_name in demos:
        obs = f["data"][demo_name]["obs"]
        
        # 1. Square Nut Position
        # Usually first 3 elements of 'object' or specific key
        if "object" in obs:
            # Assuming first 3 are nut pos
            nut_pos = obs["object"][0][:3]
            print("object position",obs["object"][0])
        else:
            nut_pos = None
            
        if nut_pos is not None:
            nut_positions.append(nut_pos)
            
        if "peg1_pos" in obs:
            peg_pos = obs["peg1_pos"][0]
            peg_positions.append(peg_pos)
        elif "object" in obs and obs["object"].shape[1] == 17:
             # Hypothesis: last 3 are peg1_pos
             peg_pos = obs["object"][0][-3:]
             peg_positions.append(peg_pos)

    # Analyze Nut
    if nut_positions:
        nut_positions = np.array(nut_positions)
        print("\n--- Square Nut Ranges (from object[:3]) ---")
        print(f"X: [{np.min(nut_positions[:, 0]):.4f}, {np.max(nut_positions[:, 0]):.4f}]")
        print(f"Y: [{np.min(nut_positions[:, 1]):.4f}, {np.max(nut_positions[:, 1]):.4f}]")
        print(f"Z: [{np.min(nut_positions[:, 2]):.4f}, {np.max(nut_positions[:, 2]):.4f}]")
    else:
        print("\nCould not find Square Nut positions.")

    # Analyze Peg
    if peg_positions:
        peg_positions = np.array(peg_positions)
        print("\n--- Peg 1 Ranges (from object[-3:]) ---")
        print(f"X: [{np.min(peg_positions[:, 0]):.4f}, {np.max(peg_positions[:, 0]):.4f}]")
        print(f"Y: [{np.min(peg_positions[:, 1]):.4f}, {np.max(peg_positions[:, 1]):.4f}]")
        print(f"Z: [{np.min(peg_positions[:, 2]):.4f}, {np.max(peg_positions[:, 2]):.4f}]")
        
        # Check if it was fixed
        x_spread = np.max(peg_positions[:, 0]) - np.min(peg_positions[:, 0])
        y_spread = np.max(peg_positions[:, 1]) - np.min(peg_positions[:, 1])
        if x_spread < 1e-4 and y_spread < 1e-4:
            print("(Peg 1 appears to be FIXED)")
            print(f"Fixed Position: {np.mean(peg_positions, axis=0)}")
        else:
            print("(Peg 1 appears to be RANDOMIZED)")
    else:
        print("\nCould not find Peg 1 positions.")
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="/workspace/liushuo/cupid/mimicgen/datasets/core/square_d1.hdf5")
    args = parser.parse_args()
    
    extract_ranges(args.dataset)
