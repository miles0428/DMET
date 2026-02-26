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
        """Solve the ground state of a given Hamiltonian and compute the 1-RDM and 2-RDM.
        Args:
            hamiltonian (FermionOperator): The Hamiltonian to solve.
            number_of_orbitals (int): The number of orbitals in the system.
            number_of_electrons (int, optional): The number of electrons in the system.
        Returns:
            energy (float): The ground state energy.
            one_rdm (ndarray): The 1-RDM of the ground state.
            two_rdm (ndarray): The 2-RDM of the ground state.
        """
        if not isinstance(hamiltonian, FermionOperator):
            raise TypeError("The Hamiltonian must be an instance of FermionOperator.")
        
        if number_of_electrons == 0:
            energy = 0 
            one_rdm = np.zeros((number_of_orbitals, number_of_orbitals))
            two_rdm = np.zeros((number_of_orbitals, number_of_orbitals,
                                number_of_orbitals, number_of_orbitals))
            return energy, one_rdm, two_rdm
        
        elif number_of_electrons == number_of_orbitals:  
            hamiltonian_sparse = self._transform_hamiltonian_to_matrix(hamiltonian, number_of_orbitals, number_of_electrons)
            energy = hamiltonian_sparse[0, 0].real 
            one_rdm = np.eye(number_of_orbitals)
            two_rdm = np.zeros((number_of_orbitals, number_of_orbitals,
                                number_of_orbitals, number_of_orbitals))
            for p in range(number_of_orbitals):
                for q in range(number_of_orbitals):
                    for r in range(number_of_orbitals):
                        for s in range(number_of_orbitals):
                            if p == q and r == s:
                                two_rdm[p, q, r, s] = 1.0 
            return energy, one_rdm, two_rdm
        
        hamiltonian_sparse = self._transform_hamiltonian_to_matrix(hamiltonian, number_of_orbitals, number_of_electrons)
        energy, eigenstate = get_ground_state(hamiltonian_sparse, **kwargs)
        psi = eigenstate.ravel()
        
        # 測量優化演算法計算 RDM 的時間
        one_rdm, two_rdm = self._get_rdms_tensorized(psi, number_of_orbitals, number_of_electrons)
        
        return energy, one_rdm, two_rdm

    def _transform_hamiltonian_to_matrix(self, hamiltonian, number_of_orbitals, number_of_electrons=None) -> csc_array:
        if number_of_electrons is not None:
            return get_number_preserving_sparse_operator(hamiltonian, number_of_orbitals, number_of_electrons)
        else:
            return get_sparse_operator(hamiltonian, n_qubits=number_of_orbitals)
    
    def _get_rdms_tensorized(self, psi: np.ndarray, number_of_orbitals: int, number_of_electrons=None):
        """Compute the 1-RDM and 2-RDM using a tensorized approach to avoid O(N^4) loops.
            This method constructs an intermediate tensor of single excitations and then uses einsum to compute the 2-RDM efficiently.
            Args:
                psi (ndarray): The ground state wavefunction as a vector.
                number_of_orbitals (int): The number of orbitals in the system.
                number_of_electrons (int, optional): The number of electrons in the system.
            Returns:
                one_rdm (ndarray): The computed 1-RDM.
                two_rdm (ndarray): The computed 2-RDM.
        """
        dim = len(psi)
        Phi_tensor = np.zeros((number_of_orbitals, number_of_orbitals, dim), dtype=np.complex128)
        rdm1 = np.zeros((number_of_orbitals, number_of_orbitals), dtype=np.complex128)
        
        # 建立單體激發中間態 (只需執行 N^2 次)
        for i in range(number_of_orbitals):
            for j in range(number_of_orbitals):
                op = FermionOperator(f"{i}^ {j}")
                mat = self._transform_hamiltonian_to_matrix(op, number_of_orbitals, number_of_electrons)
                
                phi_ij = mat @ psi
                Phi_tensor[i, j, :] = phi_ij
                rdm1[i, j] = np.vdot(psi, phi_ij)

        # 核心加速：使用 einsum 一次性求出 2-RDM，消滅 O(N^4) 迴圈
        term1 = np.einsum('lkx, mnx -> klmn', Phi_tensor.conj(), Phi_tensor)
        delta_lm = np.eye(number_of_orbitals)
        term2 = np.einsum('lm, kn -> klmn', delta_lm, rdm1)
        
        rdm2 = term1 - term2
        return rdm1, rdm2


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

