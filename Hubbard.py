from openfermion import FermionOperator
import numpy as np

class OneBodyHubbard:
    """
    Class to represent the one-body Hubbard model Hamiltonian (non-interacting limit) on a 1D lattice.

    Attributes:
        L (int): Number of lattice sites.
        t (float): Nearest-neighbor hopping amplitude.
        H (FermionOperator): The tight-binding Hamiltonian operator in second quantization.
    """
    def __init__(self, L, t):
        """
        Initialize the one-body Hubbard model parameters and Hamiltonian.

        Args:
            L (int): Number of lattice sites.
            t (float): Hopping parameter.
        """
        self.L = L
        self.t = t
        self.H = self.getHamiltonian()
    
    def getHamiltonian(self):
        """
        Construct the non-interacting Hubbard (tight-binding) Hamiltonian under periodic boundary conditions.

        The operator has the form:
            H = -t \sum_{<i,j>, \sigma} (a^\dagger_{i\sigma} a_{j\sigma} + h.c.)

        Each lattice site i has two spin-orbitals: p = 2*i (spin-up), 2*i+1 (spin-down).

        Returns:
            FermionOperator: The second-quantized Hamiltonian.
        """
        H = FermionOperator()

        for i in range(self.L):
            j = (i + 1) % self.L  # periodic boundary
            for spin in (0, 1):  # 0 = up, 1 = down
                p = 2 * i + spin
                q = 2 * j + spin
                H += -self.t * FermionOperator(f"{p}^ {q}", 1.0)
                H += -self.t * FermionOperator(f"{q}^ {p}", 1.0)

        return H

    def getAnalyticSolution(self, number_of_electrons):
        """
        Compute the analytic ground state of the non-interacting Hubbard model as a Slater determinant.

        Single-particle spin-orbitals are plane-wave states \psi_{k,\sigma}(x) = exp(i k x)/sqrt(L) ⊗ |\sigma>.

        The 2L spin-orbitals (L momenta × 2 spins) are sorted by their single-particle energies
        ε_k = -2t cos(k), and the lowest `number_of_electrons` states are occupied.

        Args:
            number_of_electrons (int): Total number of electrons (must be ≤ 2L).

        Returns:
            E_ground (float): Ground state energy = sum of occupied ε_k.
            wavefunction (ndarray): Array of shape (2L, number_of_electrons), whose columns are the
                occupied spin-orbital wavefunctions in the real-space × spin basis.
        """
        assert number_of_electrons <= 2 * self.L, "Number of electrons exceeds available spin-orbitals."
        # Allowed momenta
        k_vals = np.array([2 * np.pi * n / self.L for n in range(self.L)])
        # Energies for each momentum
        epsilon = -2 * self.t * np.cos(k_vals)

        # Build arrays for both spins
        all_energies = np.concatenate([epsilon, epsilon])  # energies for spin-up and spin-down
        all_k = np.concatenate([k_vals, k_vals])         # momentum values
        all_spin = np.array([0]*self.L + [1]*self.L)     # 0: up, 1: down

        # Sort spin-orbitals by energy and select the lowest
        sorted_indices = np.argsort(all_energies)
        occ_indices = sorted_indices[:number_of_electrons]

        # Ground state energy
        E_ground = np.sum(all_energies[occ_indices])

        # Build the Slater determinant matrix
        wavefunction = np.zeros((2*self.L, number_of_electrons), dtype=complex)
        x = np.arange(self.L)
        for i, idx in enumerate(occ_indices):
            k = all_k[idx]
            spin = all_spin[idx]
            orbital = np.exp(1j * k * x) / np.sqrt(self.L)
            spinor = np.array([1, 0]) if spin == 0 else np.array([0, 1])
            chi = np.kron(orbital, spinor)  # shape (2L,)
            wavefunction[:, i] = chi

        return E_ground, wavefunction

    def getDensityMatrix(self, number_of_electrons):
        """
        Compute the one-particle density matrix in the spin-orbital basis.

        The density matrix ρ is defined as:
            ρ = Φ Φ†,
        where Φ is the (2L × N) Slater determinant matrix of occupied spin-orbitals.

        Args:
            number_of_electrons (int): Total number of electrons.

        Returns:
            density_matrix (ndarray): Array of shape (2L, 2L) representing the spin-orbital density matrix.
        """
        _, wavefunction = self.getAnalyticSolution(number_of_electrons)
        density_matrix = np.dot(wavefunction, wavefunction.conjugate().T)
        return density_matrix

class MultiBodyHubbard:
    """
    Class to represent the multi-body Hubbard model Hamiltonian (interacting limit) on a 1D lattice.

    Attributes:
        L (int): Number of lattice sites.
        t (float): Nearest-neighbor hopping amplitude.
        U (float): On-site interaction strength.
        H (FermionOperator): The Hubbard Hamiltonian operator in second quantization.
    """
    def __init__(self, L, t, U):
        """
        Initialize the multi-body Hubbard model parameters and Hamiltonian.

        Args:
            L (int): Number of lattice sites.
            t (float): Hopping parameter.
            U (float): On-site interaction strength.
        """
        self.L = L
        self.t = t
        self.U = U
        self.H = self.getHamiltonian()
    
    def getHamiltonian(self):
        r"""
        Construct the interacting Hubbard Hamiltonian under periodic boundary conditions.

        The operator has the form:
            H = -t \sum_{<i,j>, \sigma} (a^\dagger_{i\sigma} a_{j\sigma} + h.c.) + U \sum_{i} n_{i\uparrow} n_{i\downarrow}

        Each lattice site i has two spin-orbitals: p = 2*i (spin-up), 2*i+1 (spin-down).

        Returns:
            FermionOperator: The second-quantized Hamiltonian.
        """ 
        H = FermionOperator()

        # Hopping terms
        for i in range(self.L):
            j = (i + 1) % self.L  # periodic boundary
            for spin in (0, 1):  # 0 = up, 1 = down
                p = 2 * i + spin
                q = 2 * j + spin
                H += -self.t * FermionOperator(f"{p}^ {q}", 1.0)
                H += -self.t * FermionOperator(f"{q}^ {p}", 1.0)

        # Interaction terms
        for i in range(self.L):
            p_up = 2 * i
            p_down = 2 * i + 1
            H += self.U * FermionOperator(f"{p_up}^ {p_up} {p_down}^ {p_down}", 1.0)
  
# Example usage
def main():
    L = 5
    t = 1.0
    number_of_electrons = 4

    hubbard_model = OneBodyHubbard(L, t)
    E_ground, wavefunction = hubbard_model.getAnalyticSolution(number_of_electrons)
    density_matrix = hubbard_model.getDensityMatrix(number_of_electrons)

    print("Ground state energy:", E_ground)
    print("Wavefunction shape:", wavefunction.shape)
    print("Density matrix:", density_matrix.trace())

if __name__ == "__main__":
    main()
