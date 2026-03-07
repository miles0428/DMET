import unittest
from openfermion import FermionOperator
from openfermion.linalg import get_ground_state, get_number_preserving_sparse_operator
from DMET.ProblemSolver.ClassicalEigenSolver.EigenSolver import EigenSolver as ClassicalEigenSolver
from DMET.ProblemSolver.QuantumEigenSolver.EigenSolver import EigenSolver as QuantumEigenSolver
try:
    from DMET.ProblemSolver.ClassicalEigenSolver.GpuEigenSolver import EigenSolver as GpuEigenSolver
    HAS_GPU = True
except ImportError:
    GpuEigenSolver = None
    HAS_GPU = False
import numpy as np


class TestEigenSolver(unittest.TestCase):
    def setUp(self):
        self.t = 1.0
        self.hamiltonian = FermionOperator()

        # Hopping terms (spin up)
        self.hamiltonian += FermionOperator('0^ 2', -self.t)  # a_0_up^† a_1_up
        self.hamiltonian += FermionOperator('2^ 0', -self.t)  # a_1_up^† a_0_up

        # Hopping terms (spin down)
        self.hamiltonian += FermionOperator('1^ 3', -self.t)  # a_0_down^† a_1_down
        self.hamiltonian += FermionOperator('3^ 1', -self.t)  # a_1_down^† a_0_down

        self.number_of_orbitals = 4
        self.number_of_electrons = 2

    def _compute_expected_rdms(self):
        """Compute expected 1-RDM and 2-RDM via exact diagonalization."""
        sparse_hamiltonian = get_number_preserving_sparse_operator(
            self.hamiltonian, self.number_of_orbitals, self.number_of_electrons
        )
        _, psi = get_ground_state(sparse_hamiltonian)
        psi = psi.ravel()

        expected_onerdm = np.zeros((self.number_of_orbitals, self.number_of_orbitals), dtype=np.complex128)
        expected_twordm = np.zeros(
            (self.number_of_orbitals, self.number_of_orbitals, self.number_of_orbitals, self.number_of_orbitals),
            dtype=np.complex128,
        )

        for p in range(self.number_of_orbitals):
            for q in range(self.number_of_orbitals):
                mat_pq = get_number_preserving_sparse_operator(
                    FermionOperator(f"{p}^ {q}"), self.number_of_orbitals, self.number_of_electrons
                )
                expected_onerdm[p, q] = np.vdot(psi, mat_pq @ psi)
                for r in range(self.number_of_orbitals):
                    for s in range(self.number_of_orbitals):
                        mat_rs = get_number_preserving_sparse_operator(
                            FermionOperator(f"{r}^ {s}"), self.number_of_orbitals, self.number_of_electrons
                        )
                        term1 = np.vdot(psi, (mat_pq @ mat_rs) @ psi)
                        delta_qr = 1.0 if q == r else 0.0
                        mat_ps = get_number_preserving_sparse_operator(
                            FermionOperator(f"{p}^ {s}"), self.number_of_orbitals, self.number_of_electrons
                        )
                        term2 = delta_qr * np.vdot(psi, mat_ps @ psi)
                        expected_twordm[p, q, r, s] = term1 - term2

        return expected_onerdm, expected_twordm

    def test_classical_eigen_solver(self):
        """Test ClassicalEigenSolver (CPU-based)."""
        eigen_solver = ClassicalEigenSolver()
        energy, onerdm, twordm = eigen_solver.solve(
            self.hamiltonian,
            self.number_of_orbitals,
            number_of_electrons=self.number_of_electrons,
        )

        expected_onerdm, expected_twordm = self._compute_expected_rdms()

        self.assertTrue(np.allclose(onerdm, expected_onerdm), "Classical: 1-RDM does not match.")
        self.assertTrue(np.allclose(twordm, expected_twordm), "Classical: 2-RDM does not match.")

    def test_gpu_eigen_solver(self):
        """Test GpuEigenSolver (GPU-based)."""
        try:
            import cupy
        except ImportError:
            self.skipTest("CuPy not available, skipping GPU test")

        eigen_solver = GpuEigenSolver()
        energy, onerdm, twordm = eigen_solver.solve(
            self.hamiltonian,
            self.number_of_orbitals,
            number_of_electrons=self.number_of_electrons,
        )

        expected_onerdm, expected_twordm = self._compute_expected_rdms()

        self.assertTrue(np.allclose(onerdm, expected_onerdm), "GPU: 1-RDM does not match.")
        self.assertTrue(np.allclose(twordm, expected_twordm), "GPU: 2-RDM does not match.")

    def test_quantum_eigen_solver(self):
        """Test QuantumEigenSolver (VQE-based)."""
        try:
            import cudaq
        except ImportError:
            self.skipTest("CUDA-Q not available, skipping quantum test")

        eigen_solver = QuantumEigenSolver(depth=1, mode='classical')
        energy, onerdm, twordm = eigen_solver.solve(
            self.hamiltonian,
            self.number_of_orbitals,
            number_of_electrons=self.number_of_electrons,
        )

        expected_onerdm, expected_twordm = self._compute_expected_rdms()

        # Quantum solver has tolerance; use larger atol
        self.assertTrue(np.allclose(onerdm, expected_onerdm, atol=1e-1), "Quantum: 1-RDM does not match.")
        self.assertTrue(np.allclose(twordm, expected_twordm, atol=1e-1), "Quantum: 2-RDM does not match.")


if __name__ == "__main__":
    unittest.main()
