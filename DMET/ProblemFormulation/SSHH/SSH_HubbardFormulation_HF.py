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


    def get_density_matrix(self, max_cycle: int = 50, conv_tol: float = 1e-10):
        """
        Compute the one-body reduced density matrix using PySCF UHF.
        - Includes on-site Hubbard U via mean-field decoupling:
          V^alpha = diag(U * n_beta), V^beta = diag(U * n_alpha)
        - Keeps spin-orbital layout:
          [site0_up, site0_dn, site1_up, site1_dn, ..., siteL-1_up, siteL-1_dn]

        Args:
            U (float): on-site Hubbard interaction strength. U=0 -> non-interacting.
            max_cycle (int): maximum SCF iterations.
            conv_tol (float): SCF energy convergence tolerance.

        Returns:
            np.ndarray: (2*L) x (2*L) spin-orbital 1-RDM (real, rounded to 1e-10).
        """
        try:
            from pyscf import gto, scf
            import numpy as np
            print("有裝pyscf")
        except Exception:
            print("沒裝pyscf")
            # 後備：若沒裝 PySCF，就用原本的 Slater projector
            if self._wavefunction is None:
                _, self._wavefunction = self.get_slater(self.number_of_electrons)
            return np.dot(self._wavefunction, self._wavefunction.conjugate().T).real.round(10)
        U = self.U
        # 一體哈密頓量（spin-orbital，大小 2L x 2L）
        H_so = self.get_hamiltonian()
        dim = H_so.shape[0]
        assert dim % 2 == 0, "Spin-orbital Hamiltonian dimension must be even."
        nao = dim // 2  # 空間軌域數（每個空間軌域對應一個 site）

        # 拆成 alpha / beta 的一體項（空間軌域基底）
        h1e_a = H_so[0::2, 0::2].real.copy()
        h1e_b = H_so[1::2, 1::2].real.copy()

        # 單位重疊（晶格模型）
        S = np.eye(nao)

        # 電子數與自旋分佈
        ne = int(self.number_of_electrons)
        n_alpha = (ne + 1) // 2
        n_beta  = ne // 2
        spin_mult = n_alpha - n_beta  # 2S

        # 建立虛擬分子（覆寫積分，避免用到實體 AO 積分）
        mol = gto.Mole()
        mol.atom = [["H", (i, 0.0, 0.0)] for i in range(nao)]  # 只為了維度
        mol.basis = "sto-3g"
        mol.charge = 0
        mol.spin = spin_mult
        mol.nelectron = ne
        mol.build(parse_arg=False, dump_input=False, verbose=0)

        mf = scf.UHF(mol)
        mf.get_ovlp = (lambda *args, **kwargs: S)
        mf.get_hcore = (lambda *args, **kwargs: (h1e_a, h1e_b))

        # 覆寫兩體有效勢：
        #   U=0 -> 回傳 0；U>0 -> 依據當前 dm 的對角元組 V^alpha/beta
        def get_veff(_mol=None, dm=None, *args, **kwargs):
            if U == 0.0 or dm is None:
                zero_a = np.zeros_like(h1e_a)
                zero_b = np.zeros_like(h1e_b)
                return (zero_a, zero_b)
            dm_a, dm_b = dm
            # 站點佔據數（空間軌域對角元）
            n_a = np.clip(np.diag(dm_a), 0.0, 1.0)
            n_b = np.clip(np.diag(dm_b), 0.0, 1.0)
            Va = np.diag(U * n_b)  # alpha 受 beta 影響
            Vb = np.diag(U * n_a)  # beta 受 alpha 影響
            return (Va, Vb)

        mf.get_veff = get_veff
        mf._eri = None                  # 不用雙電子積分
        mf.energy_nuc = lambda *a, **k: 0.0  # 模型核能=0
        mf.max_cycle = max_cycle
        mf.conv_tol = conv_tol
        mf.diis = scf.diis.DIIS()       # 幫助收斂
        mf.level_shift = 0.0
        mf.damp = 0.0

        # 初值：非交互（U=0）的 hcore 本徵向量填佔
        # ——這可加速收斂
        e_a, C_a = np.linalg.eigh(h1e_a)
        e_b, C_b = np.linalg.eigh(h1e_b)
        occ_a_idx = np.argsort(e_a)[:n_alpha]
        occ_b_idx = np.argsort(e_b)[:n_beta]
        Ca_occ = C_a[:, occ_a_idx]
        Cb_occ = C_b[:, occ_b_idx]
        dm_a0 = Ca_occ @ Ca_occ.T
        dm_b0 = Cb_occ @ Cb_occ.T

        # 以自訂初值啟動 SCF（避免預設去呼叫二電子積分）
        mf.kernel(dm0=(dm_a0, dm_b0))

        # 取回 1-RDM（空間軌域）
        dm_a, dm_b = mf.make_rdm1()

        # 組回 spin-orbital 形式
        gamma = np.zeros((dim, dim), dtype=float)
        gamma[0::2, 0::2] = dm_a
        gamma[1::2, 1::2] = dm_b
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

    
        