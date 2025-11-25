from openfermion import FermionOperator
import numpy as np
from ..ProblemFormulation import OneBodyProblemFormulation, ManyBodyProblemFormulation
# from DMET.ProblemFormulation.ProblemFormulation import OneBodyProblemFormulation, ManyBodyProblemFormulation
import itertools
class OneBodySSHHFormulation(OneBodyProblemFormulation):
    def __init__(self, N_cells, t1, t2 , U, number_of_electrons, PBC):
        """
        Initialize the one-body SSH-Hubbard formulation with Brute Force Search capability.

        Args:
            N_cells (int): Number of unit cells (each cell has 2 sites).
            t1 (float): Hopping parameter for even bonds (intra-cell hopping).
            t2 (float): Hopping parameter for odd bonds (inter-cell hopping).
            U (float): On-site Coulomb interaction.
            number_of_electrons (int): Number of electrons in the system.
            PBC (bool): Whether to use periodic boundary conditions.

        Attributes:
            N_cells (int): Number of unit cells.
            t1 (float): Hopping parameter for even bonds.
            t2 (float): Hopping parameter for odd bonds.
            U (float): On-site Coulomb interaction.
            PBC (bool): Periodic boundary conditions flag.
            number_of_electrons (int): Number of electrons.
            H (np.ndarray): One-body Hamiltonian matrix.
            eigenvalues (np.ndarray): Eigenvalues of the Hamiltonian (cached).
            eigenvectors (np.ndarray): Eigenvectors of the Hamiltonian (cached).
            interaction_matrix (np.ndarray): Precomputed interaction matrix for brute force search.
        """
        super().__init__()
        self.N_cells = N_cells
        self.t1 = t1
        self.t2 = t2
        self.PBC = PBC
        self.U = U
        self.number_of_electrons = number_of_electrons
        
        self._wavefunction = None
        self.H = self.get_hamiltonian()
        
        # 用於緩存對角化結果與交互作用矩陣
        self.eigenvalues = None
        self.eigenvectors = None
        self.interaction_matrix = None 

    def get_hamiltonian(self):
        """
        Construct the one-body Hamiltonian for the SSH-Hubbard model.

        Returns:
            np.ndarray: One-body Hamiltonian matrix of shape (2*L, 2*L) where L = 2*N_cells.
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

    def _calculate_pair_interaction(self, vec_i, vec_j):
        """
        Calculate the interaction energy between two orbitals (U term).

        Args:
            vec_i (np.ndarray): Eigenvector for orbital i.
            vec_j (np.ndarray): Eigenvector for orbital j.

        Returns:
            float: Pair interaction energy between the two orbitals.
        """
        L = self.N_cells * 2
        U_energy_pair = 0.0
        
        # 預先計算密度分布
        rho_i = np.abs(vec_i)**2
        rho_j = np.abs(vec_j)**2
        
        for k in range(L):
            p_up = 2 * k
            p_dn = 2 * k + 1
            
            # U * (n_i_up * n_j_dn + n_i_dn * n_j_up)
            # 這裡計算的是當兩個電子分別佔據軌域 i 和 j 時，在同一個格點上相遇(double occupancy)的能量
            term = self.U * (rho_i[p_up] * rho_j[p_dn] + rho_i[p_dn] * rho_j[p_up])
            U_energy_pair += term
            
        return U_energy_pair

    def _precompute_interaction_matrix(self):
        """
        Precompute the interaction matrix for all orbital pairs and store in self.interaction_matrix.

        The interaction matrix is symmetric and stores the pairwise interaction energies
        between all eigenvector orbitals for use in brute force search.
        """
        dim = len(self.eigenvalues)
        self.interaction_matrix = np.zeros((dim, dim))
        
        print("Pre-computing interaction matrix for Brute Force...")
        for i in range(dim):
            for j in range(i + 1, dim):
                val = self._calculate_pair_interaction(self.eigenvectors[:, i], self.eigenvectors[:, j])
                self.interaction_matrix[i, j] = val
                self.interaction_matrix[j, i] = val # 對稱矩陣
        print("Pre-computation done.")

    def get_slater(self, number_of_electrons):
        """
        Brute-force search for the ground state Slater determinant.
        
        Args:
            number_of_electrons (int): Number of electrons.

        Returns:
            tuple: A tuple containing:
                - e_ground (float): Total ground state energy.
                - wavefunction (np.ndarray): Slater determinant wavefunction of shape (dim, number_of_electrons).
        """
        
        self.number_of_electrons = number_of_electrons

        # 1. 如果還沒對角化，先做
        if self.eigenvalues is None:
            self.eigenvalues, self.eigenvectors = np.linalg.eigh(self.H)
        if number_of_electrons == 0: 
            # 如果沒有電子，能量為 0，密度矩陣為零矩陣
            self._wavefunction = np.zeros((len(self.eigenvalues), 0))
            return 0.0, self._wavefunction
        
        # 2. 如果還沒計算交互作用矩陣，先算 (這就是存下 energy pair 的步驟)
        if self.interaction_matrix is None:
            self._precompute_interaction_matrix()
            
        dim = len(self.eigenvalues)
        all_indices = range(dim)
        
        min_total_energy = float('inf')
        best_combination = None
        
        # 3. 遍歷所有組合 (C^N_m)
        # itertools.combinations 會產生排序好的 index tuple，例如 (0, 1, 2)
        for indices in itertools.combinations(all_indices, number_of_electrons):
            
            # --- One-body Energy ---
            # 直接加總特徵值
            e_onebody = sum(self.eigenvalues[i] for i in indices)
            
            # --- Two-body Energy ---
            # 查表加總
            e_interaction = 0.0
            
            # 這裡把 tuple 轉 list 方便操作，雖然 tuple 也可以 iterate
            current_indices = list(indices)
            
            # 雙重迴圈查表 (只查上三角即可，因為我們 matrix 是存兩倍或者對稱的)
            # 這裡的邏輯：總能量 = Sum(E_onebody) + Sum_{i<j} (Pair_Interaction)
            for i in range(len(current_indices)):
                for j in range(i + 1, len(current_indices)):
                    idx_a = current_indices[i]
                    idx_b = current_indices[j]
                    e_interaction += self.interaction_matrix[idx_a, idx_b]
            
            total_energy = e_onebody + e_interaction
            
            # 更新最小值
            if total_energy < min_total_energy:
                min_total_energy = total_energy
                best_combination = current_indices

        # 4. 構建最佳波函數與結果
        # best_combination 是一組 index，例如 [0, 2, 3, 5]
        # 我們從 eigenvectors 中取出這些行，堆疊成 (dim, m) 矩陣
        wavefunction = np.column_stack([self.eigenvectors[:, i] for i in best_combination])
        
        # 更新 class 屬性
        self._wavefunction = wavefunction
        e_ground = min_total_energy
        
        print(f"Brute Force found best indices: {best_combination}")
        
        return e_ground, wavefunction

    def next(self, number_of_electrons):
        """
        Update electron number (for consistency with original interface).

        Args:
            number_of_electrons (int): New number of electrons to set.
        """
        self.number_of_electrons = number_of_electrons
        self._wavefunction = None  # 重置波函數
        
    def get_density_matrix(self):
        """
        Compute the one-body reduced density matrix.

        Returns:
            np.ndarray: The one-body reduced density matrix γ = ψψ†.
        """
        # 確保有波函數
        if self._wavefunction is None:
             self.get_slater(self.number_of_electrons)
             
        # γ = ψ ψ†
        rdm = np.dot(self._wavefunction, self._wavefunction.conjugate().T)
        return rdm

class ManyBodySSHHFormulation(ManyBodyProblemFormulation):
    def __init__(self, N_cells, t1, t2, U, PBC):
        """
        Initialize the many-body SSH-Hubbard formulation.

        Args:
            N_cells (int): Number of unit cells (each cell has 2 sites).
            t1 (float): Hopping parameter for even bonds (intra-cell hopping).
            t2 (float): Hopping parameter for odd bonds (inter-cell hopping).
            U (float): On-site Coulomb interaction.
            PBC (bool): Whether to use periodic boundary conditions.

        Attributes:
            N_cells (int): Number of unit cells.
            L (int): Total number of sites = 2 * N_cells.
            t1 (float): Hopping parameter for even bonds.
            t2 (float): Hopping parameter for odd bonds.
            U (float): On-site Coulomb interaction.
            PBC (bool): Periodic boundary conditions flag.
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
            tuple: A tuple containing:
                - H (FermionOperator): FermionOperator representing the Hamiltonian.
                - onebody_terms (np.ndarray): One-body hopping matrix of shape (2L, 2L).
                - twobody_terms (np.ndarray): Two-body interaction tensor of shape (2L, 2L, 2L, 2L).
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
    N_cells = 6      # 四個格點
    t1 = 1.0            # SSH 模型 t1 hopping
    t2 = 0.5            # SSH 模型 t2 hopping
    U = 10          # On-site interaction
    number_of_electrons = 12

    # --- One-body problem ---
    onebody = OneBodySSHHFormulation(N_cells=N_cells, t1=t1, t2=t2, U = U,number_of_electrons=number_of_electrons, PBC = True)
    H = onebody.get_hamiltonian()
    print("H:", H.real)
    e_ground, wavefunction = onebody.get_slater(number_of_electrons)
    density_matrix = onebody.get_density_matrix()

    print("\n=== One-body SSH Hamiltonian ===")
    print("One-body Ground Energy:", e_ground)
    print("Density Matrix:\n", density_matrix.round(3))

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
    onebody.next(number_of_electrons + 2)
    e_ground, wavefunction = onebody.get_slater(onebody.number_of_electrons)
    print("Next 1-body ground energy with", onebody.number_of_electrons, "electrons:", e_ground)    
    print("density matrix:\n", onebody.get_density_matrix().round(3))
    
    

    
        