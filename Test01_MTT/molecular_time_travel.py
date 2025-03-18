import numpy as np
import matplotlib.pyplot as plt
from ase import Atoms
from ase.lattice.cubic import FaceCenteredCubic
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.md.verlet import VelocityVerlet
from ase.io import write, read
from ase import units

# Parameters
num_atoms = 108  # Use a number that works well with FCC lattice (e.g., 4*3³ = 108)
temperature = 120  # Kelvin (typical for liquid argon)
timestep = 5.0 * units.fs  # 5 femtoseconds
steps_forward = 200000  # 1 ns (200000 * 5 fs = 1 ns)

# Create argon atoms in an FCC lattice
# This ensures atoms have proper initial separation and won't overlap
size = 3  # 3x3x3 unit cells
argon_atoms = FaceCenteredCubic(
    size=(size, size, size),
    symbol='Ar',
    pbc=True
)

# Get the box size from the created lattice
box_size = argon_atoms.cell[0, 0]
print(f"Box size: {box_size} Å, contains {len(argon_atoms)} Ar atoms")

# Initialize velocities according to Maxwell-Boltzmann distribution
MaxwellBoltzmannDistribution(argon_atoms, temperature_K=temperature)

# Save initial positions for comparison
initial_positions = argon_atoms.get_positions().copy()
initial_velocities = argon_atoms.get_velocities().copy()

# Set up calculator (Lennard-Jones potential for argon)
from ase.calculators.lj import LennardJones
argon_atoms.calc = LennardJones(epsilon=0.0104, sigma=3.4, rc=10.0)

def run_simulation():
    """Run the full molecular dynamics simulation with time reversal."""
    global argon_atoms, trajectory_data, forward_final_positions, forward_final_velocities, backward_final_positions
    
    # Run forward simulation
    print("Running forward simulation...")
    dyn_forward = VelocityVerlet(argon_atoms, timestep)
    
    # We'll save positions at various timesteps
    trajectory_data = []
    trajectory_data.append(argon_atoms.get_positions().copy())
    
    for i in range(steps_forward):
        dyn_forward.run(1)
        
        # Save trajectory every 10000 steps (50 ps)
        if i % 10000 == 0:
            print(f"Forward step {i}/{steps_forward}, Time: {i*timestep/units.fs} fs")
            trajectory_data.append(argon_atoms.get_positions().copy())
    
    # Save the final forward state
    forward_final_positions = argon_atoms.get_positions().copy()
    forward_final_velocities = argon_atoms.get_velocities().copy()
    
    # Reverse velocities
    print("Reversing velocities...")
    argon_atoms.set_velocities(-forward_final_velocities)
    
    # Run backward simulation
    print("Running backward simulation...")
    dyn_backward = VelocityVerlet(argon_atoms, timestep)
    
    for i in range(steps_forward):
        dyn_backward.run(1)
        
        # Save trajectory every 10000 steps
        if i % 10000 == 0:
            print(f"Backward step {i}/{steps_forward}, Time: {i*timestep/units.fs} fs")
            trajectory_data.append(argon_atoms.get_positions().copy())
    
    # Get final positions after backward simulation
    backward_final_positions = argon_atoms.get_positions().copy()
    
    # Calculate deviation from initial positions
    position_difference = np.linalg.norm(backward_final_positions - initial_positions, axis=1)
    mean_difference = np.mean(position_difference)
    max_difference = np.max(position_difference)
    
    print("\nResults:")
    print(f"Mean position difference: {mean_difference:.6f} Angstrom")
    print(f"Maximum position difference: {max_difference:.6f} Angstrom")
    
    # Plot results
    plt.figure(figsize=(12, 8))
    
    # Plot position difference for each atom
    plt.subplot(121)
    plt.bar(range(len(argon_atoms)), position_difference)
    plt.xlabel('Atom Index')
    plt.ylabel('Position Difference (Å)')
    plt.title('Deviation from Initial Positions')
    
    # Plot trajectory of first atom (x-coordinate) as example
    plt.subplot(122)
    # Convert timestep from fs to ps for plotting (1 ps = 1000 fs)
    ps_conversion = 1000.0  # 1 ps = 1000 fs
    time_points = np.arange(0, len(trajectory_data)) * 10000 * timestep / ps_conversion
    plt.plot(time_points[:len(trajectory_data)//2+1], 
             [pos[0][0] for pos in trajectory_data[:len(trajectory_data)//2+1]], 
             'b-', label='Forward')
    plt.plot(time_points[len(trajectory_data)//2:], 
             [pos[0][0] for pos in trajectory_data[len(trajectory_data)//2:]], 
             'r-', label='Backward')
    plt.axhline(y=initial_positions[0][0], color='g', linestyle='--', label='Initial Position')
    plt.xlabel('Time (ps)')
    plt.ylabel('X Position (Å)')
    plt.title('Trajectory of First Atom')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('molecular_time_travel_results.png')
    plt.show()
    
    # Save simulation data for further analysis
    print("Saving simulation data...")
    np.savez("simulation_data.npz", 
             initial_positions=initial_positions,
             initial_velocities=initial_velocities,
             forward_final_positions=forward_final_positions,
             forward_final_velocities=forward_final_velocities,
             backward_final_positions=backward_final_positions,
             trajectory_data=trajectory_data,
             box_size=box_size)
    print("Data saved to simulation_data.npz")

# Initialize these variables to make them accessible outside the function
trajectory_data = None
forward_final_positions = None
forward_final_velocities = None
backward_final_positions = None

if __name__ == "__main__":
    # Only run the simulation when this script is executed directly
    run_simulation() 