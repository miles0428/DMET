## Module Physics Problem

### Class ProblemFormulation

#### class OneBodyHubbardFormulation
- **Attributes**:
  - `L`: The number of lattice sites.
  - `t`: The hopping parameter.
  - `number_of_electrons`: The number of electrons in the system.
  - `H`: The one-body Hamiltonian.
- **Methods**:
  - `get_hamiltonian() -> FermionOperator`: Constructs the one-body Hamiltonian.
  - `get_analytic_solution(number_of_electrons) -> Tuple[float, np.ndarray]`: Computes the ground state energy and wavefunction.
  - `get_density_matrix() -> np.ndarray`: Returns the one-body reduced density matrix.

#### class ManyBodyHubbardFormulation
- **Attributes**:
  - `L`: The number of lattice sites.
  - `t`: The hopping parameter.
  - `U`: The on-site interaction strength.
  - `H`: The many-body Hamiltonian.
  - `onebody_terms`: One-body terms of the Hamiltonian.
  - `twobody_terms`: Two-body terms of the Hamiltonian.
- **Methods**:
  - `get_hamiltonian() -> Tuple[FermionOperator, np.ndarray, np.ndarray]`: Constructs the many-body Hamiltonian and its terms.

## Module DMET

### class DMET
- **Attributes**:
  - `one_body_problem_formulation`: The one-body problem formulation.
  - `many_body_problem_formulation`: The many-body problem formulation.
  - `fragments`: A list of fragments for embedding calculations.
  - `problem_solver`: The solver used to solve fragment problems.
  - `onebodyrdm`: The one-body reduced density matrix.
- **Methods**:
  - `get_projectors() -> List[np.ndarray]`: Generates projector matrices for each fragment.
  - `get_multiplier_hamiltonian(mu: float, fragment: np.ndarray) -> FermionOperator`: Constructs the Hamiltonian with a chemical potential term for a fragment.
  - `get_fragment_hamiltonians() -> List[FragmentHamiltonian]`: Constructs effective Hamiltonians for all fragments.
  - `get_fragment_energy(fragment_hamiltonian, onebody_rdm, twobody_rdm, fragment_length) -> float`: Calculates the energy of a fragment Hamiltonian.
  - `solve_fragment(fragment_hamiltonian, multiplier_hamiltonian, number_of_orbitals, fragment_length) -> Tuple[float, np.ndarray, np.ndarray]`: Solves the fragment Hamiltonian.
  - `singleshot(mu: float) -> Tuple[float, float]`: Performs a single-shot DMET calculation.
  - `run(mu0: float = None, mu1: float = None) -> float`: Performs a self-consistent DMET calculation to find the chemical potential.

### class FragmentHamiltonian
- **Attributes**:
  - `onebody_terms`: The one-body integrals.
  - `twobody_terms`: The two-body integrals.
  - `H`: The embedded Hamiltonian.
- **Methods**:
  - `__init__(onebody_terms: np.ndarray, twobody_terms: np.ndarray)`: Initializes the embedded Hamiltonian.

## Module ProblemSolver

### class ProblemSolver
- **Methods**:
  - `solve(hamiltonian: FermionOperator, number_of_orbitals: int) -> Tuple[float, np.ndarray, np.ndarray]`: Solves the given Hamiltonian and returns the energy and reduced density matrices.

### class EigenSolver
- **Methods**:
  - `solve(fragment_hamiltonian) -> Tuple[float, np.ndarray, np.ndarray]`: Solves the fragment Hamiltonian and returns the energy and RDMs.

