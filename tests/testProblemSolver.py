import unittest
from openfermion import FermionOperator
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
        energy, onerdm, twordm = self.eigen_solver.solve(self.hamiltonian, number_of_orbitals)

        # Expected 1-RDM
        exact_onerdm = np.array([
            [0.5, 0.0, 0.5, 0.0],
            [0.0, 0.5, 0.0, 0.5],
            [0.5, 0.0, 0.5, 0.0],
            [0.0, 0.5, 0.0, 0.5]
        ])

        # Expected 2-RDM
        exact_twordm = np.zeros((4, 4, 4, 4), dtype=np.complex128)
        for p in range(4):
            for q in range(4):
                for r in range(4):
                    for s in range(4):
                        exact_twordm[p, q, r, s] = (
                            exact_onerdm[p, r] * exact_onerdm[q, s] - exact_onerdm[p, s] * exact_onerdm[q, r]
                        )

        # Assertions
        self.assertTrue(np.allclose(onerdm, exact_onerdm), "1-RDM does not match the expected value.")
        self.assertTrue(np.allclose(twordm, exact_twordm), "2-RDM does not match the expected value.")

if __name__ == "__main__":
    unittest.main()


