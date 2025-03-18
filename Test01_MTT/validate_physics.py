import numpy as np
import matplotlib.pyplot as plt
from ase import Atoms
from ase.lattice.cubic import FaceCenteredCubic
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.md.verlet import VelocityVerlet
from ase import units

def energy_conservation_test(steps=10000, timestep=5.0, size=3, temperature=120):
    """
    Test energy conservation in the simulation to validate physics.
    
    Parameters:
    -----------
    steps : int
        Number of simulation steps
    timestep : float
        Simulation timestep in femtoseconds
    size : int
        Size of FCC lattice (size x size x size)
    temperature : float
        Temperature in Kelvin
    """
    # Create lattice
    argon_atoms = FaceCenteredCubic(
        size=(size, size, size),
        symbol='Ar',
        pbc=True
    )
    
    num_atoms = len(argon_atoms)
    box_size = argon_atoms.cell[0, 0]
    
    print(f"Running energy conservation test for {steps} steps...")
    print(f"System: {num_atoms} atoms, box size: {box_size:.2f} Å, temperature: {temperature} K")
    
    # Convert timestep to ASE units
    timestep_ase = timestep * units.fs
    
    # Initialize velocities
    MaxwellBoltzmannDistribution(argon_atoms, temperature_K=temperature)
    
    # Set up calculator
    from ase.calculators.lj import LennardJones
    argon_atoms.calc = LennardJones(epsilon=0.0104, sigma=3.4, rc=10.0)
    
    # Run dynamics
    dyn = VelocityVerlet(argon_atoms, timestep_ase)
    
    # Store energy values
    times = []
    potential_energies = []
    kinetic_energies = []
    total_energies = []
    
    for i in range(steps):
        dyn.run(1)
        
        # Calculate energies
        potential_energy = argon_atoms.get_potential_energy()
        kinetic_energy = argon_atoms.get_kinetic_energy()
        total_energy = potential_energy + kinetic_energy
        
        # Store values
        times.append(i * timestep)
        potential_energies.append(potential_energy)
        kinetic_energies.append(kinetic_energy)
        total_energies.append(total_energy)
        
        if i % (steps // 10) == 0:
            print(f"Step {i}/{steps}, Total Energy: {total_energy:.6f} eV")
    
    # Convert to numpy arrays
    times = np.array(times)
    potential_energies = np.array(potential_energies)
    kinetic_energies = np.array(kinetic_energies)
    total_energies = np.array(total_energies)
    
    # Calculate energy conservation statistics
    mean_energy = np.mean(total_energies)
    energy_fluctuation = np.std(total_energies)
    relative_fluctuation = energy_fluctuation / abs(mean_energy)
    
    print("\nEnergy Conservation Results:")
    print(f"Mean Total Energy: {mean_energy:.6f} eV")
    print(f"Standard Deviation: {energy_fluctuation:.6f} eV")
    print(f"Relative Fluctuation: {relative_fluctuation*100:.6f}%")
    
    # Plot results
    plt.figure(figsize=(12, 8))
    
    # Plot energies over time
    plt.subplot(211)
    plt.plot(times, potential_energies, 'r-', label='Potential Energy')
    plt.plot(times, kinetic_energies, 'g-', label='Kinetic Energy')
    plt.plot(times, total_energies, 'b-', label='Total Energy')
    plt.xlabel('Time (fs)')
    plt.ylabel('Energy (eV)')
    plt.title('Energy Components vs Time')
    plt.legend()
    
    # Plot total energy fluctuation
    plt.subplot(212)
    plt.plot(times, total_energies - mean_energy, 'b-')
    plt.axhline(y=0, color='k', linestyle='--')
    plt.axhline(y=energy_fluctuation, color='r', linestyle='--', 
                label=f'Std Dev: {energy_fluctuation:.4f} eV')
    plt.axhline(y=-energy_fluctuation, color='r', linestyle='--')
    plt.xlabel('Time (fs)')
    plt.ylabel('Energy Fluctuation (eV)')
    plt.title(f'Total Energy Fluctuation (Relative: {relative_fluctuation*100:.4f}%)')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('energy_conservation.png')
    
    return {
        'mean_energy': mean_energy,
        'energy_fluctuation': energy_fluctuation,
        'relative_fluctuation': relative_fluctuation
    }

def momentum_conservation_test(steps=10000, timestep=5.0, size=3, temperature=120):
    """Test momentum conservation in the simulation."""
    # Create lattice
    argon_atoms = FaceCenteredCubic(
        size=(size, size, size),
        symbol='Ar',
        pbc=True
    )
    
    num_atoms = len(argon_atoms)
    box_size = argon_atoms.cell[0, 0]
    
    print(f"Running momentum conservation test for {steps} steps...")
    print(f"System: {num_atoms} atoms, box size: {box_size:.2f} Å, temperature: {temperature} K")
    
    # Convert timestep to ASE units
    timestep_ase = timestep * units.fs
    
    # Initialize velocities
    MaxwellBoltzmannDistribution(argon_atoms, temperature_K=temperature)
    
    # Set total momentum to zero
    momenta = argon_atoms.get_momenta()
    total_momentum = np.sum(momenta, axis=0)
    momenta -= total_momentum / len(argon_atoms)
    argon_atoms.set_momenta(momenta)
    
    # Set up calculator
    from ase.calculators.lj import LennardJones
    argon_atoms.calc = LennardJones(epsilon=0.0104, sigma=3.4, rc=10.0)
    
    # Run dynamics
    dyn = VelocityVerlet(argon_atoms, timestep_ase)
    
    # Store momentum values
    times = []
    momentum_x = []
    momentum_y = []
    momentum_z = []
    momentum_mag = []
    
    for i in range(steps):
        dyn.run(1)
        
        # Calculate momentum
        mom = np.sum(argon_atoms.get_momenta(), axis=0)
        mom_magnitude = np.linalg.norm(mom)
        
        # Store values
        times.append(i * timestep)
        momentum_x.append(mom[0])
        momentum_y.append(mom[1])
        momentum_z.append(mom[2])
        momentum_mag.append(mom_magnitude)
        
        if i % (steps // 10) == 0:
            print(f"Step {i}/{steps}, Total Momentum Magnitude: {mom_magnitude:.6f}")
    
    # Convert to numpy arrays
    times = np.array(times)
    momentum_x = np.array(momentum_x)
    momentum_y = np.array(momentum_y)
    momentum_z = np.array(momentum_z)
    momentum_mag = np.array(momentum_mag)
    
    # Calculate momentum conservation statistics
    mean_momentum = np.mean(momentum_mag)
    max_momentum = np.max(momentum_mag)
    
    print("\nMomentum Conservation Results:")
    print(f"Mean Total Momentum: {mean_momentum:.6f}")
    print(f"Maximum Total Momentum: {max_momentum:.6f}")
    
    # Plot results
    plt.figure(figsize=(12, 8))
    
    # Plot momentum components
    plt.subplot(211)
    plt.plot(times, momentum_x, 'r-', label='Momentum X')
    plt.plot(times, momentum_y, 'g-', label='Momentum Y')
    plt.plot(times, momentum_z, 'b-', label='Momentum Z')
    plt.xlabel('Time (fs)')
    plt.ylabel('Momentum')
    plt.title('Momentum Components vs Time')
    plt.legend()
    
    # Plot momentum magnitude
    plt.subplot(212)
    plt.plot(times, momentum_mag, 'k-')
    plt.xlabel('Time (fs)')
    plt.ylabel('Momentum Magnitude')
    plt.title('Total Momentum Magnitude vs Time')
    
    plt.tight_layout()
    plt.savefig('momentum_conservation.png')
    
    return {
        'mean_momentum': mean_momentum,
        'max_momentum': max_momentum
    }

def test_time_symmetry(steps=1000, timestep=5.0, size=2, temperature=120):
    """
    Test the time symmetry property of the velocity Verlet integrator.
    
    For a time-symmetric integrator, running forward for N steps, reversing velocities,
    and running forward for N more steps should return to the initial state.
    """
    # Create lattice (using smaller size for faster testing)
    argon_atoms = FaceCenteredCubic(
        size=(size, size, size),
        symbol='Ar',
        pbc=True
    )
    
    num_atoms = len(argon_atoms)
    box_size = argon_atoms.cell[0, 0]
    
    print(f"Testing time symmetry of velocity Verlet integrator...")
    print(f"System: {num_atoms} atoms, box size: {box_size:.2f} Å, temperature: {temperature} K")
    
    # Convert timestep to ASE units
    timestep_ase = timestep * units.fs
    
    # Initialize velocities
    MaxwellBoltzmannDistribution(argon_atoms, temperature_K=temperature)
    
    # Save initial state
    initial_positions = argon_atoms.get_positions().copy()
    initial_velocities = argon_atoms.get_velocities().copy()
    
    # Set up calculator
    from ase.calculators.lj import LennardJones
    argon_atoms.calc = LennardJones(epsilon=0.0104, sigma=3.4, rc=10.0)
    
    # Run forward for N steps
    dyn_forward = VelocityVerlet(argon_atoms, timestep_ase)
    for i in range(steps):
        dyn_forward.run(1)
        if i % (steps // 5) == 0:
            print(f"Forward step {i}/{steps}")
    
    # Save midpoint state
    midpoint_positions = argon_atoms.get_positions().copy()
    midpoint_velocities = argon_atoms.get_velocities().copy()
    
    # Reverse velocities
    argon_atoms.set_velocities(-midpoint_velocities)
    
    # Run forward again for N steps (equivalent to backward)
    dyn_backward = VelocityVerlet(argon_atoms, timestep_ase)
    for i in range(steps):
        dyn_backward.run(1)
        if i % (steps // 5) == 0:
            print(f"Backward step {i}/{steps}")
    
    # Get final state
    final_positions = argon_atoms.get_positions().copy()
    final_velocities = argon_atoms.get_velocities().copy()
    
    # Calculate differences
    position_diff = np.linalg.norm(final_positions - initial_positions, axis=1)
    mean_pos_diff = np.mean(position_diff)
    max_pos_diff = np.max(position_diff)
    
    velocity_diff = np.linalg.norm(final_velocities + initial_velocities, axis=1)
    mean_vel_diff = np.mean(velocity_diff)
    max_vel_diff = np.max(velocity_diff)
    
    print("\nTime Symmetry Test Results:")
    print(f"Position difference - Mean: {mean_pos_diff:.6f} Å, Max: {max_pos_diff:.6f} Å")
    print(f"Velocity difference - Mean: {mean_vel_diff:.6f} Å/fs, Max: {max_vel_diff:.6f} Å/fs")
    
    # Plot positions
    plt.figure(figsize=(12, 10))
    
    # Plot first atom trajectory through x-coordinate
    plt.subplot(221)
    plt.plot([0, steps*timestep], [initial_positions[0, 0], midpoint_positions[0, 0]], 'b-', label='Forward')
    plt.plot([steps*timestep, 2*steps*timestep], [midpoint_positions[0, 0], final_positions[0, 0]], 'r-', label='Backward')
    plt.axhline(y=initial_positions[0, 0], color='g', linestyle='--', label='Initial Position')
    plt.xlabel('Time (fs)')
    plt.ylabel('X Position (Å)')
    plt.title('X-coordinate of First Atom')
    plt.legend()
    
    # Plot position differences
    plt.subplot(222)
    plt.bar(range(num_atoms), position_diff)
    plt.axhline(y=mean_pos_diff, color='r', linestyle='--', label=f'Mean: {mean_pos_diff:.6f} Å')
    plt.xlabel('Atom Index')
    plt.ylabel('Position Difference (Å)')
    plt.title('Final vs Initial Position Difference')
    plt.legend()
    
    # Plot velocity differences
    plt.subplot(223)
    plt.bar(range(num_atoms), velocity_diff)
    plt.axhline(y=mean_vel_diff, color='r', linestyle='--', label=f'Mean: {mean_vel_diff:.6f} Å/fs')
    plt.xlabel('Atom Index')
    plt.ylabel('Velocity Difference (Å/fs)')
    plt.title('Final vs Initial Velocity Difference')
    plt.legend()
    
    # Plot scatter of initial vs final positions (first component)
    plt.subplot(224)
    plt.scatter(initial_positions[:, 0], final_positions[:, 0])
    min_val = min(np.min(initial_positions[:, 0]), np.min(final_positions[:, 0]))
    max_val = max(np.max(initial_positions[:, 0]), np.max(final_positions[:, 0]))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Match')
    plt.xlabel('Initial X Position (Å)')
    plt.ylabel('Final X Position (Å)')
    plt.title('Initial vs Final X Positions')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('time_symmetry_test.png')
    
    return {
        'mean_position_difference': mean_pos_diff,
        'max_position_difference': max_pos_diff,
        'mean_velocity_difference': mean_vel_diff,
        'max_velocity_difference': max_vel_diff
    }

if __name__ == "__main__":
    # Run all validation tests
    print("=== VALIDATING PHYSICS OF SIMULATION ===\n")
    
    # Test energy conservation
    energy_results = energy_conservation_test(steps=5000)
    
    # Test momentum conservation
    momentum_results = momentum_conservation_test(steps=5000)
    
    # Test time symmetry
    symmetry_results = test_time_symmetry(steps=2000)
    
    # Save results
    np.savez("validation_results.npz", 
             energy=energy_results,
             momentum=momentum_results,
             symmetry=symmetry_results)
    
    print("\nValidation complete. Results saved to validation_results.npz")
    print("Plots saved as energy_conservation.png, momentum_conservation.png, and time_symmetry_test.png") 