# Molecular Time Travel

This project tests the reversibility of molecular dynamics simulations. It simulates 100 argon atoms for 1 nanosecond, then reverses their velocities and runs the simulation backward to see if the initial state can be reconstructed.

## Requirements

- Python 3.6+
- NumPy
- Matplotlib
- ASE (Atomic Simulation Environment)

## Installation

1. Clone this repository
2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Running the Simulation

This repository contains several scripts:

### Main Simulation

```
python molecular_time_travel.py
```

The main simulation:
1. Sets up 100 argon atoms in a box
2. Runs a forward simulation for 1 ns
3. Reverses all velocities
4. Runs a backward simulation for 1 ns
5. Compares the final positions with the initial positions
6. Generates plots showing the deviation and trajectory

### In-Depth Analysis

```
python analyze_results.py
```

This script provides advanced analysis of the simulation results:
- Histogram of position differences
- 3D scatter plot of initial vs final positions
- Position difference by atom index
- Analysis of error vs initial velocity

### Parameter Studies

```
python parameter_study.py
```

This script studies how different parameters affect the reversibility:
- Effect of timestep size (1-20 fs)
- Effect of temperature (50-300 K)

For each parameter variation, it measures:
- Mean position difference
- Maximum position difference
- Runtime
- Sample trajectories

### Physics Validation

```
python validate_physics.py
```

This script validates the physical correctness of the simulation by testing:
- Energy conservation: Checks if total energy remains constant
- Momentum conservation: Verifies total momentum is preserved
- Time symmetry: Tests the time-reversal symmetry of the integrator

The validation tests verify that our simulation correctly implements the underlying physics principles necessary for time reversibility.

## Output

The scripts will generate:
- Terminal output showing progress and final statistics
- PNG files with visualizations of the results
- NPZ files containing raw data for further analysis

## Theory

In a perfect Newtonian system with exact numerical integration, reversing all particle velocities after some time should cause the system to exactly retrace its trajectory back to its starting point. In practice, numerical errors, finite timestep issues, and floating-point precision can lead to deviations.

This simulation tests how well the time-reversibility principle holds in a computational molecular dynamics setting. 