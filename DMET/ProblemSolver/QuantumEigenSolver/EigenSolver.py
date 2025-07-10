import cudaq
import numpy as np
from openfermion.transforms import jordan_wigner
from openfermion import FermionOperator
from itertools import combinations
from DMET.ProblemSolver import ProblemSolver

class EigenSolver(ProblemSolver):
    def __init__(self):
        super().__init__()

    def solve(self, hamiltonian: FermionOperator, number_of_orbitals: int,
              number_of_electrons: int, **kwargs):
        if not isinstance(hamiltonian, FermionOperator):
            raise TypeError("Hamiltonian must be a FermionOperator")

        # Step 1: Jordan-Wigner transform to QubitOperator then CUDA-Q PauliSum
        qubit_op = jordan_wigner(hamiltonian)
        cudaq_ham = cudaq.PauliSum()
        for term, coeff in qubit_op.terms.items():
            if term == ():  # Identity
                cudaq_ham += coeff
            else:
                pauli_str = ' '.join(f"{p[1]}{p[0]}" for p in term)
                cudaq_ham += coeff * cudaq.spin.pauli_string(pauli_str)

        # Step 2: Define particle-number-conserving ASWAP ansatz
        def make_ansatz(n_qubits, number_of_electrons=None):
            if number_of_electrons > n_qubits:
                raise ValueError("Number of electrons cannot exceed number of orbitals")
            if number_of_electrons <= 0:
                raise ValueError("Number of electrons cannot be negative or zero")
            param_count = n_qubits * (n_qubits - 1) // 2
            kernel, params = cudaq.make_kernel(*([float] * param_count))
            q = kernel.qalloc(n_qubits)

            # conservation of particle number
            for i in range(number_of_electrons):
                kernel.x(q[i])

            p_idx = 0
            for i, j in combinations(range(n_qubits), 2):
                kernel.cx(q[i], q[j])
                kernel.ry(params[p_idx], q[j])
                kernel.cx(q[i], q[j])
                p_idx += 1

            return kernel, params

        kernel, params = make_ansatz(number_of_orbitals, number_of_electrons)

        # Step 3: Run VQE
        result = cudaq.vqe(kernel, cudaq_ham, optimizer="COBYLA")
        energy = result.expectation_value
        psi = result.optimal_state  # shape: (2**n,)
        opt_params = result.optimal_parameters

        # Step 4: Compute 1-RDM
        one_rdm = np.zeros((number_of_orbitals, number_of_orbitals), dtype=np.complex128)
        for p in range(number_of_orbitals):
            for q in range(number_of_orbitals):
                op = jordan_wigner(FermionOperator(f"{p}^ {q}"))
                pauli_op = cudaq.PauliSum()
                for term, coeff in op.terms.items():
                    if term == (): pauli_op += coeff
                    else:
                        pauli_str = ' '.join(f"{pauli[1]}{pauli[0]}" for pauli in term)
                        pauli_op += coeff * cudaq.spin.pauli_string(pauli_str)
                val = cudaq.observe(kernel, pauli_op, opt_params).expectation_value
                one_rdm[p, q] = val

        # Step 5: Compute 2-RDM
        two_rdm = np.zeros((number_of_orbitals, number_of_orbitals,
                            number_of_orbitals, number_of_orbitals), dtype=np.complex128)
        for p in range(number_of_orbitals):
            for q in range(number_of_orbitals):
                for r in range(number_of_orbitals):
                    for s in range(number_of_orbitals):
                        op = jordan_wigner(FermionOperator(f"{p}^ {q}^ {s} {r}"))
                        pauli_op = cudaq.PauliSum()
                        for term, coeff in op.terms.items():
                            if term == (): pauli_op += coeff
                            else:
                                pauli_str = ' '.join(f"{pauli[1]}{pauli[0]}" for pauli in term)
                                pauli_op += coeff * cudaq.spin.pauli_string(pauli_str)
                        val = cudaq.observe(kernel, pauli_op, opt_params).expectation_value
                        two_rdm[p, q, r, s] = val

        return energy, one_rdm, two_rdm


if __name__ == "__main__":
    # Example usage
    H = FermionOperator("0^ 1", 1.0) + FermionOperator("1^ 0", 1.0)
    solver = EigenSolver()
    energy, rdm1, rdm2 = solver.solve(H, number_of_orbitals=4, number_of_electrons=2, occupied_orbitals=[1, 2])
    print("Energy:", energy)
    print("1-RDM:\n", rdm1)
    print("2-RDM shape:", rdm2.shape)
