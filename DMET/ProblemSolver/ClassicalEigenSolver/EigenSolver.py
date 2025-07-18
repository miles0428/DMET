from DMET.ProblemSolver import ProblemSolver
from openfermion import FermionOperator
from openfermion.linalg import get_ground_state, get_sparse_operator, get_number_preserving_sparse_operator
from scipy.sparse import csc_array
import numpy as np
from numpy import ndarray


class EigenSolver(ProblemSolver):
    def __init__(self):
        super().__init__()

    def solve(self, hamiltonian, number_of_orbitals, number_of_electrons = None, **kwargs) -> tuple[float, ndarray, ndarray]:
        """
        Solve the problem defined by the Hamiltonian.

        Args:
            hamiltonian (FermionOperator): The Hamiltonian to solve.
            number_of_orbitals (int): The number of spin-orbitals in the system.
            **kwargs: Additional arguments for the solver.

        Returns:
            tuple: (energy, one-body density matrix, two-body density matrix)

        Main Concept:
            Solves the eigenvalue problem for the Hamiltonian to find the ground state energy and density matrices.

        Math Detail:
            The eigenvalue problem is solved as:
                H \psi = E \psi
            The density matrices are computed from the ground state wavefunction \psi.
        """
        # print(hamiltonian)
        if not isinstance(hamiltonian, FermionOperator):
            raise TypeError("The Hamiltonian must be an instance of FermionOperator.")

        hamiltonian_sparse = self._transform_hamiltonian_to_matrix(hamiltonian, number_of_orbitals, number_of_electrons)
        energy, eigenstate = get_ground_state(hamiltonian_sparse, **kwargs)
        psi = eigenstate.ravel()  # Convert sparse vector to dense
        one_rdm = self._get_one_body_density_matrix(psi, number_of_orbitals, number_of_electrons)
        two_rdm = self._get_two_body_density_matrix(psi, number_of_orbitals, number_of_electrons)
        return energy, one_rdm, two_rdm

    def _transform_hamiltonian_to_matrix(self, hamiltonian, number_of_orbitals, number_of_electrons=None) -> csc_array:
        """
        Convert a FermionOperator Hamiltonian into sparse matrix form.

        Args:
            hamiltonian (FermionOperator): The fermionic Hamiltonian.
            number_of_orbitals (int): Total number of spin-orbitals.

        Returns:
            csc_array: Sparse matrix representation of the Hamiltonian.

        Main Concept:
            Transforms the second-quantized Hamiltonian into a sparse matrix representation for numerical calculations.
        """
        if number_of_electrons is not None:
            return get_number_preserving_sparse_operator(hamiltonian, number_of_orbitals, number_of_electrons)
        else:
            return get_sparse_operator(hamiltonian, n_qubits=number_of_orbitals)
    
    def _get_one_body_density_matrix(self, psi: np.ndarray, number_of_orbitals: int, number_of_electrons =None) -> ndarray:
        """
        Compute the one-body density matrix from the eigenstate.

        Args:
            psi (ndarray): Dense ground-state vector.
            number_of_orbitals (int): Number of spin-orbitals.

        Returns:
            ndarray: 1-RDM of shape (n, n)

        Main Concept:
            The one-body density matrix is computed as:
                \gamma_{pq} = <\psi| c_p^\dagger c_q |\psi>
            where c_p^\dagger and c_q are creation and annihilation operators.
        """
        rdm1 = np.zeros((number_of_orbitals, number_of_orbitals), dtype=np.complex128)
        for p in range(number_of_orbitals):
            for q in range(number_of_orbitals):
                op = FermionOperator(f"{p}^ {q}")
                mat = self._transform_hamiltonian_to_matrix(op, number_of_orbitals, number_of_electrons)
                value = np.vdot(psi, mat @ psi)
                rdm1[p, q] = value
        return rdm1

    def _get_two_body_density_matrix(self, psi: np.ndarray, number_of_orbitals: int, number_of_electrons=None) -> ndarray:
        """
        Compute the two-body density matrix from the eigenstate.

        Args:
            psi (ndarray): Dense ground-state vector.
            number_of_orbitals (int): Number of spin-orbitals.

        Returns:
            ndarray: 2-RDM of shape (n, n, n, n)

        Main Concept:
            The two-body density matrix is computed as:
                \Gamma_{pqrs} = <\psi| c_p^\dagger c_q^\dagger c_s c_r |\psi>
            where c_p^\dagger, c_q^\dagger, c_s, and c_r are creation and annihilation operators.
        """
        rdm2 = np.zeros((number_of_orbitals, number_of_orbitals,
                         number_of_orbitals, number_of_orbitals), dtype=np.complex128)
        
        for k in range(number_of_orbitals):
            for l in range(number_of_orbitals):
                for m in range(number_of_orbitals):
                    for n in range(number_of_orbitals):
                        op = FermionOperator(f"{k}^ {m}^ {n} {l}")
                        mat = self._transform_hamiltonian_to_matrix(op, number_of_orbitals, number_of_electrons)
                        rdm2[k,l,m,n] = np.vdot(psi, mat @ psi)
        return rdm2


if __name__ == "__main__":
    # Example usage
    import openfermion
    import openfermionpyscf
    from openfermion.transforms import jordan_wigner, get_fermion_operator
    import matplotlib.pyplot as plt
    from scipy.optimize import minimize
    import numpy as np
    import pickle
    eigen_solver = EigenSolver()
    # Test with a 4-orbital 2-body operator (adjust as needed)
    # H = FermionOperator("0^ 1", 1.0) + FermionOperator("1^ 0", 1.0) + FermionOperator("0^ 1 2 3^", -.5) + FermionOperator("1^ 0 3 2^", -.5)
    geometry = [('H', (0, 0, 0.0)),
                ('H', (0, 0, 0.74))]
    basis = 'sto3g'
    multiplicity = 1
    charge = 0
    molecule = openfermionpyscf.run_pyscf(
        openfermion.MolecularData(geometry, basis, multiplicity, charge))
    molecular_hamiltonian = molecule.get_molecular_hamiltonian()
    H = get_fermion_operator(molecular_hamiltonian)
    solver = EigenSolver()
    print('aaaa')
    energy, rdm1, rdm2 = solver.solve(H, number_of_orbitals=4, number_of_electrons=2)
    print("Energy:", energy)
    print("1-RDM:\n", rdm1)
    print("2-RDM shape:", rdm2)
    with open("array.pkl", "wb") as f:
        pickle.dump(rdm2, f)

