from openfermion import FermionOperator
import numpy as np
from ..ProblemFormulation import OneBodyProblemFormulation, ManyBodyProblemFormulation

class OneBodySSHHFormulation(OneBodyProblemFormulation):
    def __init__(self, N_cells, t1, t2, number_of_electrons, PBC, alpha=0):
        """
        Initialize the one-body Hubbard formulation.

        Args:
            L (int): The number of lattice sites.
            t1 (float): Hopping parameter for even number of sites. (0->1 in same lattice)
            t2 (float): Hopping parameter for odd number of sites. (1->2 in different lattice)
            number_of_electrons (int): The number of electrons in the system.

        Attributes:
            L (int): Number of lattice sites.
            t1 (float): Hopping parameter for even number of sites. (0->1 in same lattice)
            t2 (float): Hopping parameter for odd number of sites. (1->2 in different lattice)
            number_of_electrons (int): Number of electrons in the system.
            H (FermionOperator): The one-body Hamiltonian.
            _wavefunction (np.ndarray): The wavefunction of the system.
        """
        super().__init__()
        self.N_cells = N_cells
        self.t1 = t1
        self.t2 = t2
        self.PBC = PBC
        self._wavefunction = None
        self.H = self.get_hamiltonian()
        self.number_of_electrons = number_of_electrons
        self.alpha=alpha

    def get_hamiltonian(self):
        """
        Construct the one-body Hamiltonian for the SSH-Hubbard model.

        Returns:
            FermionOperator: The one-body Hamiltonian.

        Main Concept:
            The Hamiltonian is constructed as:
                H = - \sum_{<i,j>,\sigma} (t1 * c_i^\dagger c_j + t2 * c_j^\dagger c_i)
            where <i,j> denotes nearest neighbors and \sigma is the spin index.
        """

        """
    One-body Hamiltonian with flat index ordering:
    [site0_up, site0_dn, site1_up, site1_dn, ..., siteL-1_up, siteL-1_dn]
    """
        L = 2 * self.N_cells  # total sites
        dim = 2 * L  # total orbitals (sites * spins)

        H = np.zeros((dim, dim), dtype=complex)

        for i in range(L):

            if self.PBC == True:
                j = (i + 1) % L
            else:
                j = (i + 1) 
                if j >= L:
                    break 

            t = self.t1 if i % 2 == 0 else self.t2

            # spin up (spin=0)
            p_up = 2 * i + 0
            q_up = 2 * j + 0
            H[p_up, q_up] = -t
            H[q_up, p_up] = -t

            # spin down (spin=1)
            p_dn = 2 * i + 1
            q_dn = 2 * j + 1
            H[p_dn, q_dn] = -t
            H[q_dn, p_dn] = -t

        return H
    
    def get_slater_strong_interaction(self, number_of_electrons):
        """
        Construct a Slater determinant wavefunction for a strong-interaction limit
        by filling orbitals in an 'even-then-odd' order from the one-body Hamiltonian.
    
        Args:
            number_of_electrons (int): Number of electrons to occupy.
    
        Returns:
            e_ground (float): Sum of selected one-body energies (approximate).
            wavefunction (np.ndarray): Slater determinant wavefunction of shape (4*L, number_of_electrons).
        """
        H = self.get_hamiltonian()
    
        # Diagonalize one-body Hamiltonian
        eigenvalues, eigenvectors = np.linalg.eigh(H)
    
        # Sort eigenvalues
        idx = np.argsort(eigenvalues)
    
        # Create 'even-then-odd' filling order
        even_idx = idx[::2]
        odd_idx = idx[1::2]
        filling_order = np.concatenate([even_idx, odd_idx])
    
        # Select the first number_of_electrons orbitals in this order
        selected_idx = filling_order[:number_of_electrons]
    
        # Approximate ground state energy
        e_ground = np.sum(eigenvalues[selected_idx])
    
        # Construct wavefunction
        wavefunction = eigenvectors[:, selected_idx]
    
        self._wavefunction = wavefunction
        return e_ground, wavefunction


    def get_slater_weak_interaction(self, number_of_electrons):
        """
        Diagonalize the one-body Hamiltonian and construct the Slater determinant wavefunction.

        Args:
            number_of_electrons (int): Number of electrons to occupy lowest energy orbitals.

        Returns:
            e_ground (float): Total one-body ground state energy (sum of lowest eigenvalues).
            wavefunction (np.ndarray): Slater determinant wavefunction of shape (4*L, number_of_electrons).
        """
        H = self.get_hamiltonian()

        # 對 one-body Hamiltonian 對角化
        eigenvalues, eigenvectors = np.linalg.eigh(H)

        # 取能量最低的 number_of_electrons 個軌域
        idx = np.argsort(eigenvalues)
        e_ground = np.sum(eigenvalues[idx[:number_of_electrons]])

        # 對應的波函數 (列是 basis, 行是 occupied orbital)
        wavefunction = eigenvectors[:, idx[:number_of_electrons]]

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
        _, self._wavefunction_weak = self.get_slater_weak_interaction(self.number_of_electrons)
        _, self._wavefunction_strong = self.get_slater_strong_interaction(self.number_of_electrons)
        
        factor_weak = np.sqrt(1 - self.alpha)
        factor_strong = np.sqrt(self.alpha)
        rdm_weak = (factor_weak**2) * np.dot(self._wavefunction_weak, self._wavefunction_weak.conjugate().T).real.round(10)
        rdm_strong = (factor_strong **2) * np.dot(self._wavefunction_strong, self._wavefunction_strong.conjugate().T).real.round(10)
        # overlap matrix
        S = np.dot(self._wavefunction_weak.conjugate().T, self._wavefunction_strong)

        # determinant and inverse
        detS = np.linalg.det(S)
        invS = np.linalg.inv(S)

        # cross term
        rdm_cross = factor_weak * factor_strong * (
            detS * np.dot(self._wavefunction_strong, np.dot(invS, self._wavefunction_weak.conjugate().T))
            + np.conjugate(detS) * np.dot(self._wavefunction_weak, np.dot(invS.conjugate().T, self._wavefunction_strong.conjugate().T))
        )

        rdm_cross = rdm_cross.real.round(10)
        return rdm_weak + rdm_strong + rdm_cross

class ManyBodySSHHFormulation(ManyBodyProblemFormulation):
    def __init__(self, N_cells, t1, t2, U, PBC):
        """
        Initialize the many-body SSH-Hubbard formulation.

        Args:
            N_cells (int): Number of unit cells (each cell has 2 sites).
            t1 (float): Hopping parameter for even bonds (intra-cell hopping).
            t2 (float): Hopping parameter for odd bonds (inter-cell hopping).
            U (float): On-site Coulomb interaction.

        Attributes:
            L (int): Total number of sites = 2 * N_cells.
            dim (int): Total number of orbitals (including spin).
            H (FermionOperator): Many-body Hamiltonian.
            onebody_terms (np.ndarray): One-body hopping matrix.
            twobody_terms (np.ndarray): Two-body interaction tensor.
        """
        super().__init__()
        self.N_cells = N_cells
        self.L = 2 * N_cells  # total number of sites
        self.t1 = t1
        self.t2 = t2
        self.U = U
        self.PBC = PBC
        self.H, self.onebody_terms, self.twobody_terms = self.get_hamiltonian()

    def get_hamiltonian(self):
        """
        Construct the many-body Hamiltonian for SSH-Hubbard model.

        Returns:
            Tuple[FermionOperator, np.ndarray, np.ndarray]:
                - H: FermionOperator representing the Hamiltonian.
                - onebody_terms: (2L x 2L) hopping matrix.
                - twobody_terms: (2L x 2L x 2L x 2L) interaction tensor.
        """
        dim = 2 * self.L  # include spin
        H = FermionOperator()
        onebody_terms = np.zeros((dim, dim), dtype=float)
        twobody_terms = np.zeros((dim, dim, dim, dim), dtype=float)

        # Hopping terms
        for i in range(self.L):
            if self.PBC == True:
                j = (i + 1) % self.L
            else:
                j = (i + 1) 
                if j >= self.L:
                    break 
            t = self.t1 if i % 2 == 0 else self.t2
            for spin in (0, 1):
                p = 2 * i + spin
                q = 2 * j + spin
                H += -t * FermionOperator(f"{p}^ {q}") + -t * FermionOperator(f"{q}^ {p}")
                onebody_terms[p, q] = -t
                onebody_terms[q, p] = -t

        # On-site U terms
        for i in range(self.L):
            p = 2 * i     # spin up
            q = 2 * i + 1 # spin down
            H += self.U * FermionOperator(f"{p}^ {q}^ {q} {p}")
            twobody_terms[p, p, q, q] = self.U

        return H, onebody_terms, twobody_terms

if __name__=="__main__":
    # --- 系統參數 ---
    N_cells = 2         # 四個格點
    t1 = 1.0            # SSH 模型 t1 hopping
    t2 = 0.5            # SSH 模型 t2 hopping
    U = 2.0             # On-site interaction
    number_of_electrons = 4

    # --- One-body problem ---
    onebody = OneBodySSHHFormulation(N_cells=N_cells, t1=t1, t2=t2, number_of_electrons=number_of_electrons, PBC = True)
    H = onebody.get_hamiltonian()
    print("H:", H.real)
    e_ground, wavefunction = onebody.get_slater(number_of_electrons)
    density_matrix = onebody.get_density_matrix()

    print("\n=== One-body SSH Hamiltonian ===")
    print("One-body Ground Energy:", e_ground)
    print("Density Matrix:\n", density_matrix)

    # --- Many-body problem ---
    manybody = ManyBodySSHHFormulation(N_cells=N_cells, t1=t1, t2=t2, U=U, PBC = True)
    H_manybody = manybody.H

    print("\n=== Many-body SSH-Hubbard Hamiltonian (terms count) ===")
    print("Number of terms in FermionOperator:", len(H_manybody.terms))

    # --- 驗證 onebody_terms vs H ---
    print("\nOne-body matrix (from Many-body Hamiltonian):")
    print(manybody.onebody_terms)

    # --- 驗證兩體項 ---
    nonzero_twobody = np.nonzero(manybody.twobody_terms)
    print("\nNon-zero twobody terms indices:", list(zip(*nonzero_twobody)))

    
        