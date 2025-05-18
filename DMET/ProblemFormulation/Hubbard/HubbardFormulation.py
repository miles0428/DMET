from openfermion import FermionOperator
import numpy as np
from ..ProblemFormulation import OneBodyProblemFormulation, ManyBodyProblemFormulation

class OneBodyHubbardFormulation(OneBodyProblemFormulation):
    def __init__(self, L, t, number_of_electrons):
        """
        Initialize the one-body Hubbard formulation.

        Args:
            L (int): The number of lattice sites.
            t (float): The hopping parameter.
            number_of_electrons (int): The number of electrons in the system.

        Attributes:
            L (int): Number of lattice sites.
            t (float): Hopping parameter.
            number_of_electrons (int): Number of electrons in the system.
            H (FermionOperator): The one-body Hamiltonian.
            _wavefunction (np.ndarray): The wavefunction of the system.
        """
        super().__init__()
        self.L = L
        self.t = t
        self._wavefunction = None
        self.H = self.get_hamiltonian()
        self.number_of_electrons = number_of_electrons

    def get_hamiltonian(self):
        """
        Construct the one-body Hamiltonian for the Hubbard model.

        Returns:
            FermionOperator: The one-body Hamiltonian.

        Main Concept:
            The Hamiltonian is constructed as:
                H = -t * \sum_{<i,j>,\sigma} (c_i^\dagger c_j + c_j^\dagger c_i)
            where <i,j> denotes nearest neighbors and \sigma is the spin index.
        """
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
        """
        Compute the ground state energy and wavefunction analytically.

        Args:
            number_of_electrons (int): The number of electrons in the system.

        Returns:
            Tuple[float, np.ndarray]:
                - Ground state energy (float).
                - Wavefunction (np.ndarray).

        Main Concept:
            The ground state energy is computed by filling the lowest-energy orbitals.

        Math Detail:
            The energy levels are given by:
                \epsilon_k = -2t * cos(k)
            The wavefunction is constructed as:
                \psi(x) = exp(i * k * x) / sqrt(L)
        """
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
        """
        Compute the one-body reduced density matrix.

        Returns:
            np.ndarray: The one-body reduced density matrix.

        Main Concept:
            The density matrix is computed as:
                \gamma = \psi \psi^\dagger
            where \psi is the wavefunction.
        """
        if self._wavefunction is None:
            _, self._wavefunction = self.get_analytic_solution(self.number_of_electrons)
        return np.dot(self._wavefunction, self._wavefunction.conjugate().T)

class ManyBodyHubbardFormulation(ManyBodyProblemFormulation):
    def __init__(self, L, t, U):
        """
        Initialize the many-body Hubbard formulation.

        Args:
            L (int): The number of lattice sites.
            t (float): The hopping parameter.
            U (float): The on-site interaction strength.

        Attributes:
            L (int): Number of lattice sites.
            t (float): Hopping parameter.
            U (float): On-site interaction strength.
            H (FermionOperator): The many-body Hamiltonian.
            onebody_terms (np.ndarray): One-body terms of the Hamiltonian.
            twobody_terms (np.ndarray): Two-body terms of the Hamiltonian.
        """
        super().__init__()
        self.L = L
        self.t = t
        self.U = U
        self.H,self.onebody_terms,self.twobody_terms = self.get_hamiltonian()

    def get_hamiltonian(self):
        r"""
        Construct the many-body Hamiltonian for the Hubbard model.

        Returns:
            Tuple[FermionOperator, np.ndarray, np.ndarray]:
                - The many-body Hamiltonian (FermionOperator).
                - One-body terms (np.ndarray).
                - Two-body terms (np.ndarray).

        Main Concept:
            The Hamiltonian is constructed as:
                H = -t * \sum_{<i,j>,\sigma} (c_i^\dagger c_j + c_j^\dagger c_i)
                    + U * \sum_i n_{i,\uparrow} n_{i,\downarrow}
            where <i,j> denotes nearest neighbors and \sigma is the spin index.

        Math Detail:
            The one-body terms are stored in a matrix:
                h_{ij} = -t if i and j are neighbors, 0 otherwise.
            The two-body terms are stored in a tensor:
                g_{ijkl} = U if i = j = k = l, 0 otherwise.
        """
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



