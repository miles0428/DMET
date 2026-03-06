import unittest
from openfermion import FermionOperator
from openfermion.linalg import get_ground_state, get_number_preserving_sparse_operator
from DMET.ProblemSolver.ClassicalEigenSolver.EigenSolver import EigenSolver
import numpy as np

class TestEigenSolver(unittest.TestCase):
    def setUp(self):
        self.eigen_solver = EigenSolver()
        self.t = 1.0
        self.hamiltonian = FermionOperator()

        # Hopping terms (spin up)
        self.hamiltonian += FermionOperator('0^ 2', -self.t)  # a_0_up^† a_1_up
        self.hamiltonian += FermionOperator('2^ 0', -self.t)  # a_1_up^† a_0_up

        # Hopping terms (spin down)
        self.hamiltonian += FermionOperator('1^ 3', -self.t)  # a_0_down^† a_1_down
        self.hamiltonian += FermionOperator('3^ 1', -self.t)  # a_1_down^† a_0_down

    def test_eigen_solver(self):
        number_of_orbitals = 4
        number_of_electrons = 2
        energy, onerdm, twordm = self.eigen_solver.solve(
            self.hamiltonian,
            number_of_orbitals,
            number_of_electrons=number_of_electrons,
        )

        sparse_hamiltonian = get_number_preserving_sparse_operator(
            self.hamiltonian, number_of_orbitals, number_of_electrons
        )
        _, psi = get_ground_state(sparse_hamiltonian)
        psi = psi.ravel()

        expected_onerdm = np.zeros((number_of_orbitals, number_of_orbitals), dtype=np.complex128)
        expected_twordm = np.zeros(
            (number_of_orbitals, number_of_orbitals, number_of_orbitals, number_of_orbitals),
            dtype=np.complex128,
        )

        for p in range(number_of_orbitals):
            for q in range(number_of_orbitals):
                mat_pq = get_number_preserving_sparse_operator(
                    FermionOperator(f"{p}^ {q}"), number_of_orbitals, number_of_electrons
                )
                expected_onerdm[p, q] = np.vdot(psi, mat_pq @ psi)
                for r in range(number_of_orbitals):
                    for s in range(number_of_orbitals):
                        mat_rs = get_number_preserving_sparse_operator(
                            FermionOperator(f"{r}^ {s}"), number_of_orbitals, number_of_electrons
                        )
                        term1 = np.vdot(psi, (mat_pq @ mat_rs) @ psi)
                        delta_qr = 1.0 if q == r else 0.0
                        mat_ps = get_number_preserving_sparse_operator(
                            FermionOperator(f"{p}^ {s}"), number_of_orbitals, number_of_electrons
                        )
                        term2 = delta_qr * np.vdot(psi, mat_ps @ psi)
                        expected_twordm[p, q, r, s] = term1 - term2

        self.assertTrue(np.allclose(onerdm, expected_onerdm), "1-RDM does not match the expected value.")
        self.assertTrue(np.allclose(twordm, expected_twordm), "2-RDM does not match the expected value.")

if __name__ == "__main__":
    unittest.main()
