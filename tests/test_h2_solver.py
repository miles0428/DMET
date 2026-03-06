import unittest

import numpy as np
from openfermion import FermionOperator, MolecularData
from openfermion.linalg import get_ground_state, get_number_preserving_sparse_operator
from openfermion.transforms import get_fermion_operator
from openfermionpyscf import run_pyscf

from DMET.ProblemSolver.ClassicalEigenSolver.EigenSolver import EigenSolver


def _expectation(state, operator):
    return state.conj().T @ (operator @ state)


def _compute_expected_rdms(psi, n_orbitals, n_electrons):
    expected_one = np.zeros((n_orbitals, n_orbitals), dtype=np.complex128)
    expected_two = np.zeros((n_orbitals, n_orbitals, n_orbitals, n_orbitals), dtype=np.complex128)
    for p in range(n_orbitals):
        for q in range(n_orbitals):
            mat_pq = get_number_preserving_sparse_operator(
                FermionOperator(f"{p}^ {q}"), n_orbitals, n_electrons)
            expected_one[p, q] = _expectation(psi, mat_pq)
            for r in range(n_orbitals):
                for s in range(n_orbitals):
                    mat_rs = get_number_preserving_sparse_operator(
                        FermionOperator(f"{r}^ {s}"), n_orbitals, n_electrons)
                    term1 = _expectation(psi, mat_pq @ mat_rs)
                    delta_qr = 1.0 if q == r else 0.0
                    mat_ps = get_number_preserving_sparse_operator(
                        FermionOperator(f"{p}^ {s}"), n_orbitals, n_electrons)
                    term2 = delta_qr * _expectation(psi, mat_ps)
                    expected_two[p, q, r, s] = term1 - term2
    return expected_one, expected_two


class TestH2SolverValues(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        geometry = [('H', (0.0, 0.0, 0.0)), ('H', (0.0, 0.0, 0.7414))]
        basis = 'sto-3g'
        multiplicity = 1
        charge = 0
        molecule = MolecularData(geometry, basis, multiplicity, charge)
        molecule = run_pyscf(molecule, run_scf=1, run_fci=1)

        fermion_hamiltonian = get_fermion_operator(molecule.get_molecular_hamiltonian())
        solver = EigenSolver()
        cls.energy, cls.onerdm, cls.twordm = solver.solve(
            fermion_hamiltonian,
            molecule.n_qubits,
            number_of_electrons=molecule.n_electrons,
        )

        sparse_hamiltonian = get_number_preserving_sparse_operator(
            fermion_hamiltonian, molecule.n_qubits, molecule.n_electrons)
        _, ground = get_ground_state(sparse_hamiltonian)
        psi = ground.ravel()

        cls.expected_energy = molecule.fci_energy
        (cls.expected_onerdm, cls.expected_twordm) = _compute_expected_rdms(
            psi, molecule.n_qubits, molecule.n_electrons)

    def test_energy_matches_fci(self):
        self.assertAlmostEqual(self.energy, self.expected_energy, places=9)

    def test_rdms_match_reference(self):
        self.assertTrue(np.allclose(self.onerdm, self.expected_onerdm, atol=1e-12))
        self.assertTrue(np.allclose(self.twordm, self.expected_twordm, atol=1e-12))
