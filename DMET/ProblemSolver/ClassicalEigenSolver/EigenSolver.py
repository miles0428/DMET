from DMET.ProblemSolver import ProblemSolver
from openfermion import FermionOperator
from openfermion.linalg import get_ground_state, get_sparse_operator
from scipy.sparse import csc_array
import numpy as np
from numpy import ndarray


class EigenSolver(ProblemSolver):
    def __init__(self):
        super().__init__()

    def solve(self, hamiltonian, number_of_orbitals, **kwargs) -> tuple[float, ndarray, ndarray]:
        """
        Solve the problem defined by the Hamiltonian.

        Args:
            hamiltonian (FermionOperator): The Hamiltonian to solve.
            number_of_orbitals (int): The number of spin-orbitals in the system.
            **kwargs: Additional arguments for the solver.

        Returns:
            tuple: (energy, one-body density matrix, two-body density matrix)
        """
        hamiltonian_sparse = self._transform_hamiltonian_to_matrix(hamiltonian, number_of_orbitals)
        energy, eigenstate = get_ground_state(hamiltonian_sparse, **kwargs)
        psi = eigenstate.ravel()  # Convert sparse vector to dense

        one_rdm = self._get_one_body_density_matrix(psi, number_of_orbitals)
        two_rdm = self._get_two_body_density_matrix(psi, number_of_orbitals)
        return energy, one_rdm, two_rdm

    def _transform_hamiltonian_to_matrix(self, hamiltonian, number_of_orbitals) -> csc_array:
        """
        Convert a FermionOperator Hamiltonian into sparse matrix form.

        Args:
            hamiltonian (FermionOperator): The fermionic Hamiltonian.
            number_of_orbitals (int): Total number of spin-orbitals.

        Returns:
            csc_array: Sparse matrix representation of the Hamiltonian.
        """
        return get_sparse_operator(hamiltonian, n_qubits=number_of_orbitals)

    def _get_one_body_density_matrix(self, psi: np.ndarray, number_of_orbitals: int) -> ndarray:
        """
        Compute the one-body density matrix from the eigenstate.

        Args:
            psi (ndarray): Dense ground-state vector.
            number_of_orbitals (int): Number of spin-orbitals.

        Returns:
            ndarray: 1-RDM of shape (n, n)
        """
        rdm1 = np.zeros((number_of_orbitals, number_of_orbitals), dtype=np.complex128)
        for p in range(number_of_orbitals):
            for q in range(number_of_orbitals):
                op = FermionOperator(f"{p}^ {q}")
                mat = get_sparse_operator(op, n_qubits=number_of_orbitals)
                rdm1[p, q] = np.vdot(psi, mat @ psi)
        return rdm1

    def _get_two_body_density_matrix(self, psi: np.ndarray, number_of_orbitals: int) -> ndarray:
        """
        Compute the two-body density matrix from the eigenstate.

        Args:
            psi (ndarray): Dense ground-state vector.
            number_of_orbitals (int): Number of spin-orbitals.

        Returns:
            ndarray: 2-RDM of shape (n, n, n, n)
        """
        rdm2 = np.zeros((number_of_orbitals, number_of_orbitals,
                         number_of_orbitals, number_of_orbitals), dtype=np.complex128)
        for p in range(number_of_orbitals):
            for q in range(number_of_orbitals):
                for r in range(number_of_orbitals):
                    for s in range(number_of_orbitals):
                        op = FermionOperator(f"{p}^ {q}^ {s} {r}")
                        mat = get_sparse_operator(op, n_qubits=number_of_orbitals)
                        rdm2[p, q, r, s] = np.vdot(psi, mat @ psi)
        return rdm2


if __name__ == "__main__":
    eigen_solver = EigenSolver()
    # Test with a 4-orbital 2-body operator (adjust as needed)
    hamiltonian = FermionOperator('0^ 1^ 2 3', 1.0)
    sparse_ham = eigen_solver._transform_hamiltonian_to_matrix(hamiltonian, 4)
    print("Sparse Hamiltonian shape:", sparse_ham.shape)

    energy, rdm1, rdm2 = eigen_solver.solve(hamiltonian, 4)
    print("Ground state energy:", energy)
    print("1-RDM:\n", rdm1)
    print("2-RDM shape:", rdm2.shape)
