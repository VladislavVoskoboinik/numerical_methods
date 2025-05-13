# Numerical Schemes for 2D Wave Equation

This project implements two numerical schemes for solving the two-dimensional wave equation in a rectangular domain:

1. **Cross Scheme** (explicit, conditionally stable)
2. **Evolutionary Factorization Scheme** (implicit, unconditionally stable for σ ≥ 0.25)

## Problem Statement

We solve the following initial-boundary value problem:

```
∂²u/∂t² = a²Δu + f(x,y,t),  x ∈ (0, lx), y ∈ (0, ly), t ∈ (0, T]
u|t=0 = φ(x, y)
∂u/∂t|t=0 = ψ(x, y)
```

With appropriate boundary conditions on the rectangular domain.

## Files

The project contains the following files:

- `wave_equation_2d.py`: Implementation of the Cross scheme and Factorized scheme for the example problem from the text
- `self_study_problem.py`: Solution to the self-study problem with both schemes
- `requirements.txt`: List of required Python packages
- `setup_venv.bat`: Script to create a virtual environment on Windows
- `setup_venv.sh`: Script to create a virtual environment on Linux/macOS
- `activate_venv.bat`: Script to activate the virtual environment on Windows
- `activate_venv.sh`: Script to activate the virtual environment on Linux/macOS

## Installation

### Setting up a Virtual Environment

#### On Windows:
```
setup_venv.bat
```

#### On Linux/macOS:
```
chmod +x setup_venv.sh
./setup_venv.sh
```

After creating the virtual environment, you need to activate it and install the requirements:

#### On Windows:
```
venv\Scripts\activate
pip install -r requirements.txt
```

#### On Linux/macOS:
```
source venv/bin/activate
pip install -r requirements.txt
```

### Manual Installation
If you prefer to install the dependencies globally or in an existing environment:
```
pip install -r requirements.txt
```

## Usage

### Activating the Environment

If you've already set up the virtual environment but need to activate it:

#### On Windows:
```
activate_venv.bat
```
Or directly:
```
venv\Scripts\activate
```

#### On Linux/macOS:
```
chmod +x activate_venv.sh
./activate_venv.sh
```
Or directly:
```
source venv/bin/activate
```

### Running the Simulations

To run the example from the text:

```
python wave_equation_2d.py
```

To run the self-study problem:

```
python self_study_problem.py
```

The scripts will produce plots of the numerical solutions at different time steps, comparing them with analytical solutions.

## Theoretical Background

### Cross Scheme

The Cross scheme is an explicit finite difference method for the wave equation. It uses a stencil of 5 points (center, east, west, north, south) on the current time layer and the center point from the previous time layer to compute the solution at the next time step.

The scheme has stability condition:
```
a·τ·√(1/hₓ² + 1/h_y²) ≤ 1
```

### Factorized Scheme

The Factorized scheme is an implicit method derived from the weighted scheme by approximation of the operator with a product of one-dimensional operators. It results in solving a sequence of one-dimensional problems.

The scheme is unconditionally stable for σ ≥ 0.25, where σ is the weight parameter.

## Implementation Details

For both schemes:
1. The grid is uniformly discretized in space and time
2. Initial conditions are approximated with second-order accuracy
3. A tridiagonal solver is used for the implicit scheme
4. Boundary conditions are applied at each time step

## Results

The results are saved as PNG files showing:
- Numerical solution
- Analytical solution (when available)
- Error between numerical and analytical solutions

Error metrics are also printed to the console for comparison between the schemes. 