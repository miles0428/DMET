from openfermion import FermionOperator
import numpy as np
try:
    from .SSH_HubbardFormulation import OneBodySSHHFormulation, ManyBodySSHHFormulation
except ImportError:
    from DMET.ProblemFormulation.SSHH.SSH_HubbardFormulation import OneBodySSHHFormulation, ManyBodySSHHFormulation

class OneBodySSHHFormulation_HF(OneBodySSHHFormulation):
    def __init__(self, N_cells, t1, t2, U, number_of_electrons, PBC=True, alpha=0, tol=1e-8, max_iter=300, **kwargs):
        super().__init__(N_cells, t1, t2, number_of_electrons, PBC, alpha)
        self.U = U
        self.tol = tol
        self.max_iter = max_iter
        self.density_matrix = None

    def run_hf(self, alpha = 0.1):
        L = 2 * self.N_cells
        dim = 2 * L
        H0 = self.get_hamiltonian()

        # Initialize density matrix (non-interacting solution)
        eigenvals, eigenvecs = np.linalg.eigh(H0)
        idx = np.argsort(eigenvals)[:self.number_of_electrons]
        # D = np.dot(eigenvecs[:, idx], eigenvecs[:, idx].conj().T)
        D = super().get_density_matrix()

        for iteration in range(self.max_iter):
            # Build mean-field Hamiltonian
            H_mf = H0.copy()
            for i in range(L):
                p_up = 2 * i
                p_dn = 2 * i + 1
                n_dn = D[p_dn, p_dn].real
                n_up = D[p_up, p_up].real
                H_mf[p_up, p_up] += self.U * n_dn
                H_mf[p_dn, p_dn] += self.U * n_up

            # Diagonalize mean-field Hamiltonian
            e_vals, e_vecs = np.linalg.eigh(H_mf)
            idx = np.argsort(e_vals)[:self.number_of_electrons]
            D_new = np.dot(e_vecs[:, idx], e_vecs[:, idx].conj().T)

            # Check convergence
            delta = np.linalg.norm(D_new - D)
            D = D_new* alpha + D * (1 - alpha)
            if delta < self.tol:
                break
        else:
            print("Warning: HF did not converge after max iterations")

        # Store results
        self._wavefunction = e_vecs[:, idx]
        self.density_matrix = D

        # HF energy
        e_hf = np.sum(e_vals[idx]) - 0.5 * self.U * sum(D[2*i, 2*i].real * D[2*i+1, 2*i+1].real for i in range(L))
        return e_hf

    def get_density_matrix(self, hf_update_rate=0.1):
        """
        Compute or retrieve the HF one-body density matrix for DMET.
        If HF has not been run yet, run it automatically.
        """
        if self.density_matrix is None:
            self.run_hf(alpha=hf_update_rate)
        return self.density_matrix

    def get_slater(self, hf_update_rate=0.1):
        """
        Return the HF Slater determinant orbitals.
        Automatically runs HF if needed.
        """
        if self._wavefunction is None:
            self.run_hf(alpha=hf_update_rate)
        return self._wavefunction


class ManyBodySSHHFormulation_HF(ManyBodySSHHFormulation):
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
        super().__init__(N_cells=N_cells, t1=t1, t2=t2, U=U, PBC=PBC)


if __name__=="__main__":
    # --- 系統參數 ---
    N_cells = 10         # 四個格點
    t1 = 1.0            # SSH 模型 t1 hopping
    t2 = 0.5            # SSH 模型 t2 hopping
    U = 1             # On-site interaction
    number_of_electrons = 4

    # --- One-body problem ---
    onebody = OneBodySSHHFormulation_HF(N_cells=N_cells, t1=t1, t2=t2, number_of_electrons=number_of_electrons, PBC = True, U=U)
    H = onebody.get_hamiltonian()
    print("H:", H.real)
    wavefunction = onebody.get_slater()
    density_matrix = onebody.get_density_matrix()

    print("\n=== One-body SSH Hamiltonian ===")
    print("Density Matrix:\n", density_matrix)

    # --- Many-body problem ---
    manybody = ManyBodySSHHFormulation_HF(N_cells=N_cells, t1=t1, t2=t2, U=U, PBC = True)
    H_manybody = manybody.H

    print("\n=== Many-body SSH-Hubbard Hamiltonian (terms count) ===")
    print("Number of terms in FermionOperator:", len(H_manybody.terms))

    # --- 驗證 onebody_terms vs H ---
    print("\nOne-body matrix (from Many-body Hamiltonian):")
    print(manybody.onebody_terms)

    # --- 驗證兩體項 ---
    nonzero_twobody = np.nonzero(manybody.twobody_terms)
    print("\nNon-zero twobody terms indices:", list(zip(*nonzero_twobody)))

    
        