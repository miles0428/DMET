# DMET

## Overview
DMET (Density Matrix Embedding Theory) is a quantum embedding method for solving strongly correlated electron systems. This repository provides an implementation of DMET for the Hubbard model.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/DMET-Hubbard.git
   cd DMET-Hubbard
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Install the package:
   ```bash
   python setup.py install
   ```

## Usage

### Running Tests
To ensure everything is working correctly, run the test suite:
```bash
python -m unittest discover tests
```

## Core Classes and Functions in the DMET Package

### DMET.DMET
- **Class `DMET`**: The main class for performing Density Matrix Embedding Theory (DMET) calculations.
  - **Attributes**:
    - `problem_formulation`: The problem formulation object containing the system's Hamiltonian and other properties.
    - `fragments`: A list of fragments for the embedding calculations.
    - `problem_solver`: The solver used to solve the fragment problems.
    - `verbose`: A flag for detailed output.
  - **Methods**:
    - `run()`: Executes the DMET self-consistent loop to compute the ground state energy.

### DMET.ProblemFormulation
- **Class `ProblemFormulation`**: Represents the problem formulation for the DMET calculation.
  - **Attributes**:
    - `one_body_problem_formulation`: The one-body Hamiltonian formulation.
    - `many_body_problem_formulation`: The many-body Hamiltonian formulation.

### DMET.ProblemSolver.ClassicalEigenSolver
- **Class `EigenSolver`**: A classical eigenvalue solver for solving the fragment problems.
  - **Methods**:
    - `solve(fragment_hamiltonian)`: Solves the given fragment Hamiltonian and returns the energy and reduced density matrices.

### DMET.ProblemFormulation.Hubbard
- **Class `OneBodyHubbardFormulation`**: Represents the one-body formulation of the Hubbard model.
  - **Attributes**:
    - `L`: The number of lattice sites.
    - `t`: The hopping parameter.
    - `number_of_electrons`: The number of electrons in the system.

- **Class `ManyBodyHubbardFormulation`**: Represents the many-body formulation of the Hubbard model.
  - **Attributes**:
    - `L`: The number of lattice sites.
    - `t`: The hopping parameter.
    - `U`: The on-site interaction strength.

## Contributing
Feel free to submit issues or pull requests to improve the project.

## License
This project is licensed under the Apache License 2.0.