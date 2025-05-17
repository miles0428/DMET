from openfermion import FermionOperator
import numpy as np
from ..ProblemFormulation import OneBodyProblemFormulation, ManyBodyProblemFormulation

class OneBodyHubbardFormulation(OneBodyProblemFormulation):
    def __init__(self, L, t, number_of_electrons):
        super().__init__()
        self.L = L
        self.t = t
        self._wavefunction = None
        self.H = self.get_hamiltonian()
        self.number_of_electrons = number_of_electrons

    def get_hamiltonian(self):
        H = FermionOperator()
        for i in range(self.L):
            j = (i + 1) % self.L
            for spin in (0, 1):
                p = 2 * i + spin
                q = 2 * j + spin
                H += -self.t * FermionOperator(f"{p}^ {q}", 1.0)
                H += -self.t * FermionOperator(f"{q}^ {p}", 1.0)
        return H

    def get_analytic_solution(self, number_of_electrons):
        assert number_of_electrons <= 2 * self.L
        k_vals = np.array([2 * np.pi * n / self.L for n in range(self.L)])
        epsilon = -2 * self.t * np.cos(k_vals)
        all_energies = np.concatenate([epsilon, epsilon])
        all_k = np.concatenate([k_vals, k_vals])
        all_spin = np.array([0]*self.L + [1]*self.L)
        sorted_indices = np.argsort(all_energies)
        occ_indices = sorted_indices[:number_of_electrons]

        e_ground = np.sum(all_energies[occ_indices])

        wavefunction = np.zeros((2*self.L, number_of_electrons), dtype=complex)
        x = np.arange(self.L)
        for i, idx in enumerate(occ_indices):
            k = all_k[idx]
            spin = all_spin[idx]
            orbital = np.exp(1j * k * x) / np.sqrt(self.L)
            spinor = np.array([1, 0]) if spin == 0 else np.array([0, 1])
            chi = np.kron(orbital, spinor)
            wavefunction[:, i] = chi
        self._wavefunction = wavefunction
        return e_ground, wavefunction

    def get_density_matrix(self):
        if self._wavefunction is None:
            _, self._wavefunction = self.get_analytic_solution(self.number_of_electrons)
        return np.dot(self._wavefunction, self._wavefunction.conjugate().T)

class ManyBodyHubbardFormulation(ManyBodyProblemFormulation):
    def __init__(self, L, t, U):
        super().__init__()
        self.L = L
        self.t = t
        self.U = U
        self.H,self.onebody_terms,self.twobody_terms = self.get_hamiltonian()

    def get_hamiltonian(self):
        H = FermionOperator()
        onebody_terms = np.zeros((2*self.L, 2*self.L))
        twobody_terms = np.zeros((2*self.L, 2*self.L, 2*self.L, 2*self.L))
        for i in range(self.L):
            j = (i + 1) % self.L
            for spin in (0, 1):
                p = 2 * i + spin
                q = 2 * j + spin
                H += -self.t * FermionOperator(f"{p}^ {q}", 1.0)
                H += -self.t * FermionOperator(f"{q}^ {p}", 1.0)
                onebody_terms[p, q] = -self.t
                onebody_terms[q, p] = -self.t
        for i in range(self.L):
            p_up = 2 * i
            p_down = 2 * i + 1
            H += self.U * FermionOperator(f"{p_up}^ {p_up} {p_down}^ {p_down}", 1.0)
            twobody_terms[p_up, p_up, p_down, p_down] = self.U
        return H, onebody_terms, twobody_terms



