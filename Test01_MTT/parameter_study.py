import numpy as np
import matplotlib.pyplot as plt
from ase import Atoms
from ase.lattice.cubic import FaceCenteredCubic
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.md.verlet import VelocityVerlet
from ase import units
import time
import os

def run_simulation(size, temperature, timestep, steps_forward, output_prefix):
    """
    Run a molecular dynamics simulation with time reversal.
    
    Parameters:
    -----------
    size : int
        Size of the cubic FCC lattice (size x size x size)
    temperature : float
        Temperature in Kelvin
    timestep : float
        Simulation timestep in femtoseconds
    steps_forward : int
        Number of steps to run forward (and backward)
    output_prefix : str
        Prefix for output files
    
    Returns:
    --------
    dict
        Dictionary containing simulation results
    """
    # Create lattice
    argon_atoms = FaceCenteredCubic(
        size=(size, size, size),
        symbol='Ar',
        pbc=True
    )
    
    num_atoms = len(argon_atoms)
    box_size = argon_atoms.cell[0, 0]
    
    print(f"\nRunning simulation with parameters:")
    print(f"  Atoms: {num_atoms}, Box: {box_size:.2f} Å, Temp: {temperature} K")
    print(f"  Timestep: {timestep} fs, Steps: {steps_forward}")
    
    # Convert timestep to ASE units
    timestep_ase = timestep * units.fs
    
    # Initialize velocities
    MaxwellBoltzmannDistribution(argon_atoms, temperature_K=temperature)
    
    # Save initial state
    initial_positions = argon_atoms.get_positions().copy()
    
    # Set up calculator
    from ase.calculators.lj import LennardJones
    argon_atoms.calc = LennardJones(epsilon=0.0104, sigma=3.4, rc=10.0)
    
    # Run forward
    dyn_forward = VelocityVerlet(argon_atoms, timestep_ase)
    
    # Save a few trajectory points (for first atom only, to save memory)
    traj_x = [initial_positions[0, 0]]
    traj_time = [0]
    
    start_time = time.time()
    for i in range(steps_forward):
        dyn_forward.run(1)
        if i % (steps_forward // 10) == 0:
            print(f"  Forward step {i}/{steps_forward}")
            traj_x.append(argon_atoms.get_positions()[0, 0])
            traj_time.append((i+1) * timestep)
    
    # Save midpoint state and reverse velocities
    forward_final_velocities = argon_atoms.get_velocities().copy()
    argon_atoms.set_velocities(-forward_final_velocities)
    
    # Run backward
    dyn_backward = VelocityVerlet(argon_atoms, timestep_ase)
    
    for i in range(steps_forward):
        dyn_backward.run(1)
        if i % (steps_forward // 10) == 0:
            print(f"  Backward step {i}/{steps_forward}")
            traj_x.append(argon_atoms.get_positions()[0, 0])
            traj_time.append((i+1+steps_forward) * timestep)
    
    end_time = time.time()
    runtime = end_time - start_time
    
    # Get final positions and calculate differences
    backward_final_positions = argon_atoms.get_positions().copy()
    position_difference = np.linalg.norm(backward_final_positions - initial_positions, axis=1)
    mean_difference = np.mean(position_difference)
    max_difference = np.max(position_difference)
    
    print(f"  Complete. Runtime: {runtime:.2f} seconds")
    print(f"  Mean position difference: {mean_difference:.6f} Å")
    print(f"  Maximum position difference: {max_difference:.6f} Å")
    
    # Return results
    return {
        'parameters': {
            'num_atoms': num_atoms,
            'box_size': box_size,
            'temperature': temperature,
            'timestep': timestep,
            'steps_forward': steps_forward
        },
        'results': {
            'mean_difference': mean_difference,
            'max_difference': max_difference,
            'runtime': runtime
        },
        'trajectory': {
            'time': np.array(traj_time),
            'x': np.array(traj_x)
        }
    }

def timestep_study():
    """Study the effect of timestep on reversibility."""
    print("\n=== TIMESTEP STUDY ===")
    
    # Parameters to keep constant
    size = 3  # 3x3x3 unit cells (108 atoms)
    temperature = 120
    steps_forward = 10000  # Reduced for speed
    
    # Timesteps to test (in fs)
    timesteps = [1.0, 2.0, 5.0, 10.0, 20.0]
    
    results = []
    for ts in timesteps:
        result = run_simulation(
            size, temperature, ts, steps_forward,
            f"timestep_{ts}"
        )
        results.append(result)
    
    # Plot results
    plt.figure(figsize=(12, 10))
    
    # Plot 1: Mean position difference vs timestep
    plt.subplot(221)
    plt.plot(timesteps, [r['results']['mean_difference'] for r in results], 'o-')
    plt.xlabel('Timestep (fs)')
    plt.ylabel('Mean Position Difference (Å)')
    plt.title('Effect of Timestep on Reversibility')
    
    # Plot 2: Max position difference vs timestep
    plt.subplot(222)
    plt.plot(timesteps, [r['results']['max_difference'] for r in results], 'o-')
    plt.xlabel('Timestep (fs)')
    plt.ylabel('Max Position Difference (Å)')
    plt.title('Effect of Timestep on Maximum Error')
    
    # Plot 3: Runtime vs timestep
    plt.subplot(223)
    plt.plot(timesteps, [r['results']['runtime'] for r in results], 'o-')
    plt.xlabel('Timestep (fs)')
    plt.ylabel('Runtime (s)')
    plt.title('Effect of Timestep on Simulation Runtime')
    
    # Plot 4: Sample trajectory for smallest and largest timestep
    plt.subplot(224)
    plt.plot(
        results[0]['trajectory']['time'], 
        results[0]['trajectory']['x'], 
        'b-', 
        label=f"Timestep={timesteps[0]} fs"
    )
    plt.plot(
        results[-1]['trajectory']['time'], 
        results[-1]['trajectory']['x'], 
        'r-', 
        label=f"Timestep={timesteps[-1]} fs"
    )
    plt.axvline(x=results[0]['parameters']['steps_forward']*results[0]['parameters']['timestep'], 
                color='k', linestyle='--', label='Velocity Reversal')
    plt.xlabel('Time (fs)')
    plt.ylabel('X Position (Å)')
    plt.title('Sample Trajectories')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('timestep_study.png')
    
    return results

def temperature_study():
    """Study the effect of temperature on reversibility."""
    print("\n=== TEMPERATURE STUDY ===")
    
    # Parameters to keep constant
    size = 3  # 3x3x3 unit cells (108 atoms)
    timestep = 5.0
    steps_forward = 10000  # Reduced for speed
    
    # Temperatures to test (in K)
    temperatures = [50, 100, 150, 200, 300]
    
    results = []
    for temp in temperatures:
        result = run_simulation(
            size, temp, timestep, steps_forward,
            f"temperature_{temp}"
        )
        results.append(result)
    
    # Plot results
    plt.figure(figsize=(12, 10))
    
    # Plot 1: Mean position difference vs temperature
    plt.subplot(221)
    plt.plot(temperatures, [r['results']['mean_difference'] for r in results], 'o-')
    plt.xlabel('Temperature (K)')
    plt.ylabel('Mean Position Difference (Å)')
    plt.title('Effect of Temperature on Reversibility')
    
    # Plot 2: Max position difference vs temperature
    plt.subplot(222)
    plt.plot(temperatures, [r['results']['max_difference'] for r in results], 'o-')
    plt.xlabel('Temperature (K)')
    plt.ylabel('Max Position Difference (Å)')
    plt.title('Effect of Temperature on Maximum Error')
    
    # Plot 3: Runtime vs temperature
    plt.subplot(223)
    plt.plot(temperatures, [r['results']['runtime'] for r in results], 'o-')
    plt.xlabel('Temperature (K)')
    plt.ylabel('Runtime (s)')
    plt.title('Effect of Temperature on Simulation Runtime')
    
    # Plot 4: Sample trajectory for lowest and highest temperature
    plt.subplot(224)
    plt.plot(
        results[0]['trajectory']['time'], 
        results[0]['trajectory']['x'], 
        'b-', 
        label=f"Temp={temperatures[0]} K"
    )
    plt.plot(
        results[-1]['trajectory']['time'], 
        results[-1]['trajectory']['x'], 
        'r-', 
        label=f"Temp={temperatures[-1]} K"
    )
    plt.axvline(x=results[0]['parameters']['steps_forward']*results[0]['parameters']['timestep'], 
                color='k', linestyle='--', label='Velocity Reversal')
    plt.xlabel('Time (fs)')
    plt.ylabel('X Position (Å)')
    plt.title('Sample Trajectories')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('temperature_study.png')
    
    return results

if __name__ == "__main__":
    # Create output directory for results
    os.makedirs("parameter_study_results", exist_ok=True)
    
    # Run studies
    timestep_results = timestep_study()
    temperature_results = temperature_study()
    
    # Save all results
    np.savez("parameter_study_results/timestep_study.npz", results=timestep_results)
    np.savez("parameter_study_results/temperature_study.npz", results=temperature_results)
    
    print("\nParameter studies complete. Results saved.")
    print("Plots saved as timestep_study.png and temperature_study.png") 