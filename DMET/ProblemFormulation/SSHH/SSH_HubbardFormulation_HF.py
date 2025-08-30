from openfermion import FermionOperator
import numpy as np
from ..ProblemFormulation import OneBodyProblemFormulation, ManyBodyProblemFormulation

class OneBodySSHHFormulation_HF(OneBodyProblemFormulation):
    def __init__(self, N_cells, t1, t2, U, number_of_electrons, PBC):
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
        self.U = U
        self.PBC = PBC
        self._wavefunction = None
        self.H = self.get_hamiltonian()
        self.number_of_electrons = number_of_electrons

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

    def get_slater(self, number_of_electrons):
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


    def get_density_matrix(self, max_cycle: int = 200, conv_tol: float = 1e-12,
                       hartree_shift: bool = True, enforce_pm: bool = False):
        """
        UHF mean-field 1-RDM for lattice Hubbard in spin-orbital layout:
        [site0_up, site0_dn, site1_up, site1_dn, ...].
        - Optional Hartree (n - 1/2) shift for better half-filling stability.
        - Stronger SCF stabilization (+ Newton refine).
        - Optional paramagnetic enforcement by averaging alpha/beta.

        Returns
        -------
        gamma : (2L, 2L) real 1-RDM
        """
        try:
            from pyscf import gto, scf
            import numpy as np
        except Exception:
            # fallback: Slater (U=0) projector
            if self._wavefunction is None:
                _, self._wavefunction = self.get_slater(self.number_of_electrons)
            return (self._wavefunction @ self._wavefunction.conjugate().T).real.round(10)

        U = float(getattr(self, "U", 0.0))
        H_so = self.get_hamiltonian()
        dim = H_so.shape[0]
        assert dim % 2 == 0, "Spin-orbital Hamiltonian dimension must be even."
        nao = dim // 2

        # Split into alpha/beta one-body blocks (space-orbital basis)
        h1e_a = H_so[0::2, 0::2].real.copy()
        h1e_b = H_so[1::2, 1::2].real.copy()
        S = np.eye(nao)

        ne = int(self.number_of_electrons)
        n_alpha = (ne + 1) // 2
        n_beta  = ne // 2
        spin_mult = n_alpha - n_beta  # 2S

        mol = gto.Mole()
        mol.atom = [["H", (i, 0.0, 0.0)] for i in range(nao)]  # dummy chain
        mol.basis = "sto-3g"
        mol.charge = 0
        mol.spin = spin_mult
        mol.nelectron = ne
        mol.build(parse_arg=False, dump_input=False, verbose=0)

        mf = scf.UHF(mol)
        mf.get_ovlp = (lambda *args, **kwargs: S)
        mf.get_hcore = (lambda *args, **kwargs: (h1e_a, h1e_b))

        # Mean-field decoupling with optional (n - 1/2) shift
        def get_veff(_mol=None, dm=None, *args, **kwargs):
            zero = np.zeros_like(h1e_a)
            if U == 0.0 or dm is None:
                return (zero, zero)
            dm_a, dm_b = dm
            n_a = np.clip(np.diag(dm_a), 0.0, 1.0)
            n_b = np.clip(np.diag(dm_b), 0.0, 1.0)
            if hartree_shift:
                Va = np.diag(U * (n_b - 0.5))
                Vb = np.diag(U * (n_a - 0.5))
            else:
                Va = np.diag(U * n_b)
                Vb = np.diag(U * n_a)
            return (Va, Vb)

        mf.get_veff = get_veff
        mf._eri = None
        mf.energy_nuc = lambda *a, **k: 0.0
        mf.max_cycle = max_cycle
        mf.conv_tol = conv_tol
        mf.conv_tol_grad = 1e-8
        mf.diis = scf.diis.DIIS()
        mf.diis_space = 12
        mf.level_shift = 0.2   # small positive shift helps metals
        mf.damp = 0.2

        # Build a better initial dm (U=0 eigenvectors)
        ea, Ca = np.linalg.eigh(h1e_a)
        eb, Cb = np.linalg.eigh(h1e_b)
        Ca_occ = Ca[:, np.argsort(ea)[:n_alpha]]
        Cb_occ = Cb[:, np.argsort(eb)[:n_beta]]
        dm_a0 = Ca_occ @ Ca_occ.T
        dm_b0 = Cb_occ @ Cb_occ.T

        # First run with DIIS/level shift
        mf.kernel(dm0=(dm_a0, dm_b0))

        # Newton refinement (often reduces spin contamination & idempotency error)
        try:
            mf = scf.newton(mf)
            mf.level_shift = 0.0
            mf.damp = 0.0
            mf.kernel()
        except Exception:
            pass

        dm_a, dm_b = mf.make_rdm1()

        # (Optional) enforce paramagnetic solution by averaging α/β
        if enforce_pm:
            dm_mean = 0.5 * (dm_a + dm_b)
            dm_a = dm_mean.copy()
            dm_b = dm_mean.copy()

        gamma = np.zeros((dim, dim), dtype=float)
        gamma[0::2, 0::2] = dm_a
        gamma[1::2, 1::2] = dm_b

        # Quick sanity report (prints once; comment out in production)
        try:
            # Hermiticity & trace
            herm_err = np.linalg.norm(gamma - gamma.T, ord='fro')
            tr = np.trace(gamma)
            # Idempotency check for Slater-type (UHF is single-determinant): ||γ^2 - γ||
            idem_err = np.linalg.norm(gamma @ gamma - gamma, ord='fro')
            # Occupation bounds
            w = np.linalg.eigvalsh(gamma)
            print(f"[RDM check] trace={tr:.8f} (target {ne}), "
                f"Hermiticity|γ-γ^T|={herm_err:.2e}, "
                f"idempotency|γ²-γ|={idem_err:.2e}, "
                f"eig in [{w.min():.4f}, {w.max():.4f}]")
        except Exception:
            pass

        return gamma.round(10)



class ManyBodySSHHFormulation_HF(ManyBodyProblemFormulation):
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

    
        