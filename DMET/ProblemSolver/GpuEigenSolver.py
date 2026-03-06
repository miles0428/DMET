try:
    from DMET.ProblemSolver.ProblemSolver import ProblemSolver
except Exception:
    # Fallback minimal base class if package import conflicts exist (for isolated smoke tests)
    from abc import ABC, abstractmethod
    class ProblemSolver(ABC):
        def __init__(self):
            pass
        @abstractmethod
        def solve(self, *args, **kwargs):
            pass

from openfermion import FermionOperator
from openfermion.linalg import get_sparse_operator, get_number_preserving_sparse_operator
import numpy as np
from numpy import ndarray

# GPU libs
import cupy as cp
import cupyx.scipy.sparse as csp
import cupyx.scipy.sparse.linalg as cslx


class EigenSolver(ProblemSolver):
    def __init__(self):
        super().__init__()

    def solve(self, hamiltonian: FermionOperator, number_of_orbitals: int, number_of_electrons: int = None, **kwargs) -> tuple[float, ndarray, ndarray]:
        """Solve the ground state of a given Hamiltonian and compute the 1-RDM and 2-RDM on GPU.

        Args:
            hamiltonian (FermionOperator): The Hamiltonian to solve.
            number_of_orbitals (int): The number of orbitals in the system.
            number_of_electrons (int, optional): The number of electrons in the system.

        Returns:
            energy (float): The ground state energy.
            one_rdm (ndarray): The 1-RDM of the ground state (CPU numpy array).
            two_rdm (ndarray): The 2-RDM of the ground state (CPU numpy array).
        """
        if not isinstance(hamiltonian, FermionOperator):
            raise TypeError("The Hamiltonian must be an instance of FermionOperator.")

        # trivial cases
        if number_of_electrons == 0:
            energy = 0.0
            one_rdm = np.zeros((number_of_orbitals, number_of_orbitals), dtype=float)
            two_rdm = np.zeros((number_of_orbitals, number_of_orbitals, number_of_orbitals, number_of_orbitals), dtype=float)
            return energy, one_rdm, two_rdm

        if number_of_electrons == number_of_orbitals:
            # fully filled: trivial occupancy
            energy = 0.0
            one_rdm = np.eye(number_of_orbitals, dtype=float)
            two_rdm = np.zeros((number_of_orbitals, number_of_orbitals, number_of_orbitals, number_of_orbitals), dtype=float)
            for p in range(number_of_orbitals):
                for q in range(number_of_orbitals):
                    for r in range(number_of_orbitals):
                        for s in range(number_of_orbitals):
                            if p == q and r == s:
                                two_rdm[p, q, r, s] = 1.0
            return energy, one_rdm, two_rdm

        # build GPU sparse matrix from OpenFermion operator
        sparse_gpu = self._transform_hamiltonian_to_matrix(hamiltonian, number_of_orbitals, number_of_electrons)

        # compute ground state on GPU
        energy, psi_gpu = get_ground_state_gpu(sparse_gpu, **kwargs)

        # compute RDMs on GPU
        one_rdm_gpu, two_rdm_gpu = self._get_rdms_tensorized(psi_gpu, number_of_orbitals, number_of_electrons)

        # transfer back to CPU numpy
        one_rdm = cp.asnumpy(one_rdm_gpu).real
        two_rdm = cp.asnumpy(two_rdm_gpu).real

        return float(energy), one_rdm, two_rdm

    def _transform_hamiltonian_to_matrix_cpu(self, hamiltonian, number_of_orbitals, number_of_electrons=None):
        """Get scipy sparse matrix representation (CPU)."""
        if number_of_electrons is not None:
            return get_number_preserving_sparse_operator(hamiltonian, number_of_orbitals, number_of_electrons)
        else:
            return get_sparse_operator(hamiltonian, n_qubits=number_of_orbitals)

    def _transform_hamiltonian_to_matrix(self, hamiltonian, number_of_orbitals, number_of_electrons=None):
        """Convert scipy sparse (CPU) to cupy sparse (GPU)."""
        cpu_sparse = self._transform_hamiltonian_to_matrix_cpu(hamiltonian, number_of_orbitals, number_of_electrons)
        # cupyx accepts scipy.sparse matrices in constructor
        try:
            gpu_sparse = csp.csr_matrix(cpu_sparse)
        except Exception:
            # fallback: convert to coo and construct manually
            coo = cpu_sparse.tocoo()
            data = cp.asarray(coo.data)
            row = cp.asarray(coo.row)
            col = cp.asarray(coo.col)
            gpu_sparse = csp.coo_matrix((data, (row, col)), shape=coo.shape).tocsr()
        return gpu_sparse

    def _get_rdms_tensorized(self, psi_gpu: cp.ndarray, number_of_orbitals: int, number_of_electrons=None):
        """Compute 1-RDM and approximate 2-RDM on GPU using tensorized single excitation contractions.

        Note: this constructs phi_ij = a_i^\dagger a_j |psi> for all i,j, which is O(N^2 * dim) in memory/time.
        Suitable for small number_of_orbitals only.
        """
        dim = psi_gpu.size
        # allocate
        Phi_tensor = cp.zeros((number_of_orbitals, number_of_orbitals, dim), dtype=cp.complex128)
        rdm1 = cp.zeros((number_of_orbitals, number_of_orbitals), dtype=cp.complex128)

        # build single-excitation action matrices and apply to psi
        # NOTE: constructing full operator for each a_i^\dagger a_j is expensive; we use OpenFermion->sparse conversion per-op
        for i in range(number_of_orbitals):
            for j in range(number_of_orbitals):
                # build fermionic one-body operator a_i^\dagger a_j
                op = FermionOperator(f"{i}^ {j}")
                mat_gpu = self._transform_hamiltonian_to_matrix(op, number_of_orbitals, number_of_electrons)
                # sparse mat vector multiply on GPU
                phi_ij = mat_gpu.dot(psi_gpu)
                Phi_tensor[i, j, :] = phi_ij
                rdm1[i, j] = cp.vdot(psi_gpu, phi_ij)

        # term1 = sum_x Phi_tensor^*[:,:,x] * Phi_tensor[:,:,x]  (einsum contraction)
        term1 = cp.einsum('lkx, mnx -> klmn', Phi_tensor.conj(), Phi_tensor)
        delta_lm = cp.eye(number_of_orbitals, dtype=cp.complex128)
        term2 = cp.einsum('lm, kn -> klmn', delta_lm, rdm1)

        rdm2 = term1 - term2
        return rdm1, rdm2


def get_ground_state_gpu(sparse_operator_gpu, initial_guess=None, tol=1e-8):
    """Compute lowest eigenvalue and eigenstate using GPU sparse eigsh (cupyx).

    Fallbacks to dense cp.linalg.eigh if matrix is small.
    """
    n = sparse_operator_gpu.shape[0]
    # for very small n, convert to dense and use eigh on GPU
    if n <= 64:
        try:
            dense = sparse_operator_gpu.toarray()
            vals, vecs = cp.linalg.eigh(dense)
            idx = int(cp.argmin(vals))
            return float(vals[idx].real), vecs[:, idx]
        except Exception:
            # final fallback: move to CPU
            import scipy.sparse as sp
            dense_cpu = sp.csr_matrix(sparse_operator_gpu.get()).toarray()
            w, v = np.linalg.eigh(dense_cpu)
            idx = int(np.argmin(w))
            return float(w[idx].real), cp.asarray(v[:, idx])

    # use eigsh on GPU
    try:
        values, vectors = cslx.eigsh(sparse_operator_gpu, k=1, v0=initial_guess, which='SA', tol=tol, maxiter=int(1e6))
        order = cp.argsort(values)
        values = values[order]
        vectors = vectors[:, order]
        eigval = float(values[0].real)
        eigvec = vectors[:, 0]
        return eigval, eigvec
    except Exception as e:
        raise RuntimeError(f"GPU eigsh failed: {e}")


if __name__ == "__main__":
    # minimal smoke test using a tiny FermionOperator (2-orbital hopping)
    from openfermion.transforms import get_fermion_operator
    from openfermion.utils import normal_ordered

    # build simple 2-site hopping Hamiltonian
    H = FermionOperator('0^ 1',  -1.0) + FermionOperator('1^ 0', -1.0)
    solver = EigenSolver()
    e, r1, r2 = solver.solve(H, number_of_orbitals=2, number_of_electrons=1)
    print('smoke test energy:', e)
    print('1-RDM:', r1)
    print('2-RDM shape:', r2.shape)
