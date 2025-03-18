import numpy as np
import matplotlib.pyplot as plt
from ase import Atoms
from ase.io import read, write
from ase.visualize.plot import plot_atoms
import os
import importlib.util
import sys

def load_simulation_data(filename):
    """Load saved simulation data if it exists."""
    if os.path.exists(filename):
        return np.load(filename, allow_pickle=True)
    else:
        print(f"File {filename} not found.")
        return None

def analyze_time_reversibility(initial_pos, final_pos, initial_vel, filename="reversibility_analysis.png"):
    """Analyze and visualize the time reversibility of the simulation."""
    # Calculate position differences
    pos_diff = np.linalg.norm(final_pos - initial_pos, axis=1)
    
    # Statistics
    mean_diff = np.mean(pos_diff)
    max_diff = np.max(pos_diff)
    min_diff = np.min(pos_diff)
    std_diff = np.std(pos_diff)
    
    print("\nReversibility Analysis:")
    print(f"Mean position difference: {mean_diff:.6f} Å")
    print(f"Maximum difference: {max_diff:.6f} Å")
    print(f"Minimum difference: {min_diff:.6f} Å")
    print(f"Standard deviation: {std_diff:.6f} Å")
    
    # Create visualization
    plt.figure(figsize=(15, 10))
    
    # Histogram of position differences
    plt.subplot(221)
    plt.hist(pos_diff, bins=20, alpha=0.7)
    plt.axvline(mean_diff, color='r', linestyle='--', label=f'Mean: {mean_diff:.4f} Å')
    plt.xlabel('Position Difference (Å)')
    plt.ylabel('Number of Atoms')
    plt.title('Distribution of Position Differences')
    plt.legend()
    
    # 3D scatter plot of initial vs final positions
    ax = plt.subplot(222, projection='3d')
    ax.scatter(initial_pos[:, 0], initial_pos[:, 1], initial_pos[:, 2], 
               label='Initial', alpha=0.6, s=50)
    ax.scatter(final_pos[:, 0], final_pos[:, 1], final_pos[:, 2], 
               label='Final', alpha=0.6, s=50)
    ax.set_xlabel('X (Å)')
    ax.set_ylabel('Y (Å)')
    ax.set_zlabel('Z (Å)')
    ax.set_title('Initial vs Final Positions')
    ax.legend()
    
    # Plot position differences by atom index
    plt.subplot(223)
    plt.bar(range(len(pos_diff)), pos_diff, alpha=0.7)
    plt.axhline(mean_diff, color='r', linestyle='--', label=f'Mean: {mean_diff:.4f} Å')
    plt.xlabel('Atom Index')
    plt.ylabel('Position Difference (Å)')
    plt.title('Position Difference by Atom')
    plt.legend()
    
    # Error accumulation analysis
    plt.subplot(224)
    plt.scatter(np.linalg.norm(initial_vel, axis=1), pos_diff, alpha=0.7)
    plt.xlabel('Initial Velocity Magnitude (Å/fs)')
    plt.ylabel('Position Difference (Å)')
    plt.title('Error vs Initial Velocity')
    
    plt.tight_layout()
    plt.savefig(filename)
    plt.show()
    
    return mean_diff, max_diff, std_diff

def analyze_lattice_structure(positions, box_size, cutoff=4.0, filename="lattice_analysis.png"):
    """Analyze the lattice structure of the argon system."""
    # Calculate radial distribution function (RDF)
    n_atoms = len(positions)
    
    # Compute all pairwise distances
    distances = []
    for i in range(n_atoms):
        for j in range(i+1, n_atoms):
            # Calculate distance with periodic boundary conditions
            r_ij = positions[j] - positions[i]
            # Apply minimum image convention
            r_ij = r_ij - box_size * np.round(r_ij / box_size)
            dist = np.linalg.norm(r_ij)
            distances.append(dist)
    
    # Convert to array
    distances = np.array(distances)
    
    # Create RDF histogram
    bin_width = 0.1  # Å
    bins = np.arange(0, cutoff + bin_width, bin_width)
    hist, bin_edges = np.histogram(distances, bins=bins)
    
    # Normalize by expected number in ideal gas
    bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
    bin_volumes = 4/3 * np.pi * (bin_edges[1:]**3 - bin_edges[:-1]**3)
    density = n_atoms / box_size**3
    expected_counts = bin_volumes * density * n_atoms / 2
    
    rdf = hist / expected_counts
    
    # Plot RDF
    plt.figure(figsize=(10, 6))
    plt.plot(bin_centers, rdf, 'b-')
    plt.xlabel('Distance (Å)')
    plt.ylabel('g(r)')
    plt.title('Radial Distribution Function')
    plt.grid(True, alpha=0.3)
    
    # Add vertical lines for perfect FCC lattice peaks
    # First peak at nearest neighbor distance (0.707 * lattice constant for FCC)
    lattice_constant = box_size / 3  # Assuming 3x3x3 unit cells
    nn_distance = 0.707 * lattice_constant
    plt.axvline(x=nn_distance, color='r', linestyle='--', 
                label=f'Nearest Neighbor ({nn_distance:.2f} Å)')
    
    # Second peak at second nearest neighbor
    snn_distance = 1.0 * lattice_constant
    plt.axvline(x=snn_distance, color='g', linestyle='--',
                label=f'Second Nearest ({snn_distance:.2f} Å)')
    
    plt.legend()
    plt.savefig(filename)
    plt.show()
    
    return rdf, bin_centers

def analyze_trajectory(trajectory_data, time_points, filename="trajectory_analysis.png"):
    """Analyze the trajectory data from the simulation."""
    n_frames = len(trajectory_data)
    n_atoms = len(trajectory_data[0])
    mid_point = n_frames // 2
    
    print(f"Analyzing trajectory with {n_frames} frames, {n_atoms} atoms")
    print(f"Mid-point (velocity reversal) at frame {mid_point}")
    
    # Calculate MSD (Mean Square Displacement) from initial positions
    msd = np.zeros(n_frames)
    for i in range(n_frames):
        displacement = trajectory_data[i] - trajectory_data[0]
        msd[i] = np.mean(np.sum(displacement**2, axis=1))
    
    # Calculate reversibility index
    # This measures how well each frame after reversal matches the corresponding frame before reversal
    reversibility = np.zeros(mid_point)
    for i in range(mid_point):
        forward_frame = trajectory_data[i]
        backward_frame = trajectory_data[n_frames-1-i]
        frame_diff = np.mean(np.linalg.norm(backward_frame - forward_frame, axis=1))
        reversibility[i] = frame_diff
    
    # Plot results
    plt.figure(figsize=(12, 10))
    
    # Plot MSD vs time
    plt.subplot(221)
    plt.plot(time_points, msd, 'b-')
    plt.axvline(x=time_points[mid_point], color='r', linestyle='--', label='Velocity Reversal')
    plt.xlabel('Time (ps)')
    plt.ylabel('MSD (Å²)')
    plt.title('Mean Square Displacement')
    plt.legend()
    
    # Plot reversibility index
    plt.subplot(222)
    rev_times = time_points[:mid_point]
    plt.plot(rev_times, reversibility, 'g-')
    plt.xlabel('Forward Time (ps)')
    plt.ylabel('Position Difference (Å)')
    plt.title('Reversibility Index: Difference Between Symmetric Frames')
    
    # Plot trajectory of a few sample atoms (x coordinate)
    plt.subplot(223)
    sample_atoms = [0, min(10, n_atoms-1), min(20, n_atoms-1)]  # First few atoms
    for atom_idx in sample_atoms:
        plt.plot(time_points, [pos[atom_idx][0] for pos in trajectory_data], 
                 label=f'Atom {atom_idx}')
    plt.axvline(x=time_points[mid_point], color='r', linestyle='--', label='Velocity Reversal')
    plt.xlabel('Time (ps)')
    plt.ylabel('X Position (Å)')
    plt.title('Sample Atom Trajectories (X coordinate)')
    plt.legend()
    
    # Plot distance between initial and final positions for each atom
    plt.subplot(224)
    position_diff = np.linalg.norm(trajectory_data[-1] - trajectory_data[0], axis=1)
    plt.bar(range(min(50, n_atoms)), position_diff[:min(50, n_atoms)])
    plt.axhline(y=np.mean(position_diff), color='r', linestyle='--', 
                label=f'Mean: {np.mean(position_diff):.4f} Å')
    plt.xlabel('Atom Index')
    plt.ylabel('Position Difference (Å)')
    plt.title('Initial vs Final Position Difference (First 50 atoms)')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(filename)
    plt.show()
    
    return msd, reversibility

def import_from_file(module_name, file_path):
    """Import a module from file without executing it."""
    # Create a spec for the module
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None:
        return None
    
    # Create the module
    module = importlib.util.module_from_spec(spec)
    
    # Add the module to sys.modules
    sys.modules[module_name] = module
    
    # Execute the module
    spec.loader.exec_module(module)
    
    return module

def main():
    # Try to load data from the simulation data file first
    print("Looking for saved simulation data...")
    
    data = load_simulation_data("simulation_data.npz")
    if data is not None:
        print("Data loaded from simulation_data.npz")
        # Extract data from the loaded file
        initial_positions = data['initial_positions']
        backward_final_positions = data['backward_final_positions']
        initial_velocities = data['initial_velocities']
        trajectory_data = data['trajectory_data'] if 'trajectory_data' in data else None
        box_size = data['box_size'] if 'box_size' in data else None
    else:
        print("Data file not found, attempting to extract variables from script...")
        try:
            # Create a backup of the original sys.argv
            original_argv = sys.argv.copy()
            
            # Set sys.argv to an empty list to prevent the script from running
            sys.argv = ['']
            
            # Get the absolute path to the molecular_time_travel.py file
            script_path = os.path.abspath("molecular_time_travel.py")
            
            if not os.path.exists(script_path):
                print(f"Error: {script_path} not found.")
                return
                
            # Import variables from the file
            with open(script_path, 'r') as f:
                # Find if there's a main guard in the file
                has_main_guard = "__name__ == '__main__'" in f.read()
            
            if has_main_guard:
                # If it has a main guard, we can import directly
                import molecular_time_travel as mtt
                initial_positions = mtt.initial_positions
                backward_final_positions = mtt.backward_final_positions
                initial_velocities = mtt.initial_velocities
                box_size = mtt.box_size
                trajectory_data = mtt.trajectory_data if hasattr(mtt, 'trajectory_data') else None
            else:
                print("No data found. Please run molecular_time_travel.py first.")
                return
                
            # Restore the original sys.argv
            sys.argv = original_argv
            
        except Exception as e:
            print(f"Error extracting variables from script: {e}")
            print("Please run molecular_time_travel.py first to generate data.")
            return
    
    # Perform basic reversibility analysis
    mean_diff, max_diff, std_diff = analyze_time_reversibility(
        initial_positions, 
        backward_final_positions,
        initial_velocities
    )
    
    # If we have trajectory data, analyze it
    if trajectory_data is not None:
        # Create time points (assuming fixed intervals)
        n_frames = len(trajectory_data)
        # Assume 10000 steps = 50 ps between saved frames
        time_points = np.linspace(0, 100, n_frames)  # Approximate values in ps
        analyze_trajectory(trajectory_data, time_points)
    
    # If we have box size, analyze lattice structure
    if box_size is not None:
        analyze_lattice_structure(initial_positions, box_size)
    else:
        # Estimate box size from positions (very approximate)
        pos = initial_positions
        box_size = max(np.max(pos[:, 0]) - np.min(pos[:, 0]),
                      np.max(pos[:, 1]) - np.min(pos[:, 1]),
                      np.max(pos[:, 2]) - np.min(pos[:, 2]))
        print(f"Estimated box size: {box_size:.2f} Å")
        analyze_lattice_structure(initial_positions, box_size)

if __name__ == "__main__":
    main() 