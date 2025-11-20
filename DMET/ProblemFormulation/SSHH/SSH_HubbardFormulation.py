from openfermion import FermionOperator
import numpy as np
from ..ProblemFormulation import OneBodyProblemFormulation, ManyBodyProblemFormulation
# from DMET.ProblemFormulation.ProblemFormulation import OneBodyProblemFormulation, ManyBodyProblemFormulation
class OneBodySSHHFormulation(OneBodyProblemFormulation):
    def __init__(self, N_cells, t1, t2 , U, number_of_electrons, PBC):
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
        self.U = U
        if not hasattr(self, 'pointer_high_odd'):
            self.pointer_high_odd = 1
        if not hasattr(self, 'pointer_high_even'):
            self.pointer_high_even = 0
        if not hasattr(self, 'pointer_low'):
            self.pointer_low = 0

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
        if not hasattr(self, 'eigenvalues') or not hasattr(self, 'eigenvectors'):
            self.eigenvalues, self.eigenvectors = np.linalg.eigh(H)
            # print("eigenvectors", self.eigenvectors)
        
        if not hasattr(self, 'idx'):
            self.idx = np.argsort(self.eigenvalues)
        
        # print(self._wavefunction)
        if not hasattr(self, '_wavefunction_stack'):
            print("Initializing wavefunction storage.")
            self._wavefunction_stack = []
        
        if not hasattr(self, 'chosen_indices'):
            self.chosen_indices = []
            self.non_chosen_indices = list(self.idx)
        print("non_chosen_indices at start:", self.non_chosen_indices)
        
        if not hasattr(self, 'additional_energies'):
            self.additional_energies = []
        if number_of_electrons == 0 :
            return 0.0, np.zeros((self.eigenvectors.shape[0],0))
        print("number_of_electrons:", number_of_electrons)
        print("len of wavefunction_stack before adding:", len(self._wavefunction_stack))
        for _ in range(number_of_electrons - len(self._wavefunction_stack) if number_of_electrons > len(self._wavefunction_stack) else 0):
            if len(self._wavefunction_stack) == 0:
                chosen_index = self.idx[0]
                print("Chosen index:", chosen_index)
                self._wavefunction_stack.append(self.eigenvectors[:, chosen_index])
                # print(np.array(self._wavefunction_stack).shape)
                self.chosen_indices.append(0)
                self.non_chosen_indices.remove(0)
                self.pointer_low = min(self.non_chosen_indices)
                self.pointer_high_even += 2
                self.pointer_high_odd += 2
                self.additional_energies.append(self.eigenvalues[chosen_index])
                # print("Chosen index:", chosen_index)
                # print("non_chosen_indices:", self.non_chosen_indices)
            else:
                # calculate U-interaction energy from the probability of chosen orbitals
                wavefunction_high_even = self.eigenvectors[:, self.idx[self.pointer_high_even]] if self.pointer_high_even < len(self.idx) else None
                wavefunction_high_odd = self.eigenvectors[:, self.idx[self.pointer_high_odd]] if self.pointer_high_odd < len(self.idx) else None
                wavefunction_low = self.eigenvectors[:, self.idx[self.pointer_low]]
                U_energy_even = self.calculate_U_from_wavefunction(wavefunction_high_even) if wavefunction_high_even is not None else float('inf')
                U_energy_odd = self.calculate_U_from_wavefunction(wavefunction_high_odd) if wavefunction_high_odd is not None else float('inf')
                U_energy_low = self.calculate_U_from_wavefunction(wavefunction_low)
                Us = [U_energy_even, U_energy_odd, U_energy_low]
                total_energy_even = self.eigenvalues[self.idx[self.pointer_high_even]] + U_energy_even if wavefunction_high_even is not None else float('inf')
                total_energy_odd = self.eigenvalues[self.idx[self.pointer_high_odd]] + U_energy_odd if wavefunction_high_odd is not None else float('inf')
                total_energy_low = self.eigenvalues[self.idx[self.pointer_low]]+ U_energy_low if wavefunction_low is not None else float('inf')
                # find the minimum energy
                energies = [total_energy_even, total_energy_odd, total_energy_low]
                # print("--------------------")
                # print("Energies:", energies)
                # print("U energies:", Us)

                indices = [self.pointer_high_even, self.pointer_high_odd, self.pointer_low]
                min_energy_index = np.argmin(energies)
                chosen_index = indices[min_energy_index]
                self._wavefunction_stack.append(self.eigenvectors[:, self.idx[chosen_index]])
                # print(self.pointer_high_even, self.pointer_high_odd, self.pointer_low)
                # print("Chosen index:", chosen_index)
                # print("non_chosen_indices:", self.non_chosen_indices)
                # print("choices indices:", self.chosen_indices)
                self.chosen_indices.append(chosen_index)
                self.non_chosen_indices.remove(chosen_index)
                # update pointers
                if chosen_index == self.pointer_low:
                    if self.non_chosen_indices:
                        self.pointer_low = min(self.non_chosen_indices)
                if chosen_index == self.pointer_high_even or chosen_index == self.pointer_high_odd:
                    self.pointer_high_even += 2
                    self.pointer_high_odd += 2
                self.additional_energies.append(energies[min_energy_index])
     
                
        # print(np.array(self._wavefunction_stack[:number_of_electrons]).shape)
        # print(self.eigenvectors.shape)
        wavefunction = np.column_stack(self._wavefunction_stack[:number_of_electrons])
        e_ground = sum(self.additional_energies[:number_of_electrons])
        self._wavefunction = wavefunction
        
        
        return e_ground, wavefunction

    def calculate_U_from_wavefunction(self, wavefunction):
        """
        Calculate the effective U interaction energy from the wavefunction which is chosen and also target wavefunction.

        Args:
            wavefunction (np.ndarray): eigenvectors of the target
        Returns:
            float: The effective U interaction energy.
        """
        L = self.N_cells * 2
        U_energy = 0.0
        for wave_const in self._wavefunction_stack:
            # w = [site0_up, site0_dn, site1_up, site1_dn, ..., siteL-1_up, siteL-1_dn]
            # wavefunction = [site0_up, site0_dn, site1_up, site1_dn, ..., siteL-1_up, siteL-1_dn]
            # wave_const and wavefunction are not the same
            # Calculate double occupancy contribution
            wave_const = wave_const / np.linalg.norm(wave_const).reshape(-1)
            wavefunction = wavefunction / np.linalg.norm(wavefunction).reshape(-1)
            # print("wave_const:", wave_const)
            # print("wavefunction:", wavefunction)
            for i in range(L):
                p_up = 2 * i
                p_dn = 2 * i + 1
                # Probability of finding electron at same site with opposite spins at the same time
                p_up_coeff = wave_const[p_up]
                p_dn_coeff = wave_const[p_dn]
                p_up_wf_coeff = wavefunction[p_up]
                p_dn_wf_coeff = wavefunction[p_dn]
                n_up = round(np.abs(p_up_coeff)**2 ,3)
                n_dn = round(np.abs(p_dn_coeff)**2,3)
                n_up_wf = round(np.abs(p_up_wf_coeff)**2,3)
                n_dn_wf = np.abs(p_dn_wf_coeff)**2
                # because if double occupancy happens, both spin-up and spin-down must be present
                # print(f"Site {i}: n_up={n_up}, n_dn={n_dn}, n_up_wf={n_up_wf}, n_dn_wf={n_dn_wf}")
                U_energy += self.U *(n_up * n_dn_wf + n_dn * n_up_wf)
        return U_energy
            

    def next(self,number_of_electrons):
        """
        Iterator to next number of electrons.

        Args:
            number_of_electrons (int): Number of electrons to occupy lowest energy orbitals.
        Returns:
            None
        """
        self.number_of_electrons = number_of_electrons
        
    def get_density_matrix(self):
        """
        Compute the one-body reduced density matrix.

        Returns:
            np.ndarray: The one-body reduced density matrix.

        Main Concept:
            The density matrix is computed as:
                γ = ψ ψ†
            where ψ is the wavefunction.
        """
        # Get wavefunctions
        _, self._wavefunction_weak = self.get_slater(self.number_of_electrons)
        # Diagonal terms
        rdm_weak = np.dot(self._wavefunction_weak, self._wavefunction_weak.conjugate().T)
        # Final 1-RDM
        return rdm_weak 

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
    N_cells = 2      # 四個格點
    t1 = 1.0            # SSH 模型 t1 hopping
    t2 = 0.5            # SSH 模型 t2 hopping
    U = 4          # On-site interaction
    number_of_electrons = 6

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
    
    

    
        