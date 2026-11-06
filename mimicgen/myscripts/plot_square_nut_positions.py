
import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import argparse
import os

def plot_nut_positions(dataset_path):
    f = h5py.File(dataset_path, "r")
    demos = list(f["data"].keys())
    
    nut_positions = []
    
    print(f"Loading {len(demos)} demonstrations...")
    for demo_name in demos:
        # Load the first frame of object observations
        # The structure depends on how the dataset was recorded, usually under 'obs'
        # We look for something resembling nut pos. Assuming 'object' obs or similar exists.
        # Based on Square_D1 env code, it likely has 'square_nut_pos' or similar if object obs are on.
        # However, typically low dim obs are concatenated.
        # Let's try to infer or read model xml if available, but for plotting scatter we just need the points.
        # Often raw simulation states are stored. 
        # But a robust way for 'square_D1' tasks is usually look at model_file in demo attributes.
        
        # Taking a shortcut: "square_nut_d0" usually puts nut info in obs.
        # Let's inspect one demo to see keys if needed, but for now let's assume standard keys
        # "square_nut_pos" or similar.
        # Actually proper way might be to read joint positions if obs key is ambiguous
        # BUT key "object" often contains concatenated pos + quat.
        
        # Let's rely on standard robosuite structure: 
        # demo['obs']['square_nut_pos'][0] (first frame)
        
        # Inspect keys showed 'object' is present.
        # In NutAssembly, 'object' obs is usually [nut_pos, nut_quat, nut_to_eef_pos, nut_to_eef_quat ...]
        # For visualization we just take the first 2 elements of 'object' which are x,y of the nut.
        
        obs = f["data"][demo_name]["obs"]
        if "object" in obs:
            # object state is typically flat, take first 2 dims for x,y
            pos = obs["object"][0][:2]
        elif "SquareNut_pos" in obs:
            pos = obs["SquareNut_pos"][0][:2]
        elif "square_nut_pos" in obs:
            pos = obs["square_nut_pos"][0][:2]
        else:
             continue
        
        nut_positions.append(pos)

    nut_positions = np.array(nut_positions)
    
    plt.figure(figsize=(8, 10))
    plt.scatter(nut_positions[:, 0], nut_positions[:, 1], alpha=0.5, s=10, label="Nut Initial Pos")
    
    # Draw Collision Zones (The "Blank" Areas)
    # Peg 1 (Target)
    # From code analysis:
    # Peg 1 is at center of ranges: 
    #   Nut X range: (-0.115, 0.115) -> Center 0
    #   Nut Y range: (-0.255, 0.255) -> Center 0 
    #   (Actually code for Peg placement uses Peg range center, but logic puts it at 0,0 relative to ref if ranges symmetric)
    #   Peg X range (-0.1, 0.3) -> Center 0.1
    #   Peg Y range (-0.2, 0.2) -> Center 0
    #   Wait, let's re-read the code for Peg 1 position carefully.
    
    # Code:
    # peg_bounds["x"] = (-0.1, 0.3)
    # peg1_pos_x = ref[0] + (-0.1 + 0.3)/2 = 0 + 0.1 = 0.1
    # peg_bounds["y"] = (-0.2, 0.2)
    # peg1_pos_y = ref[1] + (-0.2 + 0.2)/2 = 0 + 0 = 0
    


            #     nut=dict(
            #     x=(-0.25, -0.15),
            #     y=(-0.25, 0.25),
            #     z_rot=(0., 0.),
            #     # NOTE: hardcoded @self.table_offset since this might be called in init function
            #     reference=np.array((0, 0, 0.82)),
            # ),
            # peg=dict(
            #     x=(0.1, 0.1),
            #     y=(0., 0.),
            #     z_rot=(0., 0.),
            #     # NOTE: hardcoded @self.table_offset since this might be called in init function
            #     refere
    peg1_center = (0.1,0.1)
    
    # Collision radius buffer
    # peg1_horizontal_radius = 0.0226 (calculated in previous turn)
    # nut_horizontal_radius = 0.1253 (calculated in previous turn)
    # Total Exclusion Radius
    exclusion_radius = 0.1326274169979695
    
    circle = patches.Circle(peg1_center, exclusion_radius, linewidth=2, edgecolor='r', facecolor='none', linestyle='--', label=f'Collision Zone (R={exclusion_radius:.3f}m)')
    
    ax = plt.gca()
    ax.add_patch(circle)
    
    # Plot Peg 1 Center
    plt.plot(peg1_center[0], peg1_center[1], 'rx', markersize=10, markeredgewidth=3, label="Peg 1 Center (0.3,0.2)")
    
    # Bounds Box
    # Nut random range
    # x=(-0.115, 0.115)
    # y=(-0.255, 0.255)
             #     x=(-0.25, -0.15),
            #     y=(-0.25, 0.25),
    rect = patches.Rectangle((-0.25, -0.25), (-0.15) - (-0.25), 0.25 - (-0.25), linewidth=1, edgecolor='k', facecolor='none', label='Sampling Bounds')
    ax.add_patch(rect)
    
    plt.title("Initial Square Nut Positions & Collision Avoidance Zone")
    plt.xlabel("X Position (m)")
    plt.ylabel("Y Position (m)")
    plt.legend()
    plt.grid(True)
    plt.axis('equal')

    # Set axis ticks to 0.05 interval
    import matplotlib.ticker as ticker
    ax.xaxis.set_major_locator(ticker.MultipleLocator(0.05))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.05))
    
    output_path = "./tmp/core_datasets/square/square_nut_distribution.png"
    plt.savefig(output_path)
    print(f"Plot saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, help="Path to the hdf5 dataset")
    args = parser.parse_args()
    
    plot_nut_positions(args.dataset)
