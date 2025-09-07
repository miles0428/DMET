from __future__ import annotations
from typing import Any, List, Tuple, Dict, Union, Optional
import scipy.sparse as sp
from ..ProblemFormulation.ProblemFormulation import ProblemFormulation
import numpy as np
from openfermion import FermionOperator
from ..ProblemSolver import ProblemSolver
from tqdm import tqdm
from datetime import datetime
from filelock import FileLock

def with_default_kwargs(defaults: Dict[str, Any]) -> Any:
    def decorator(func: Any) -> Any:
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            for key, value in defaults.items():
                kwargs.setdefault(key, value)
            return func(*args, **kwargs)
        return wrapper
    return decorator

class DMET:
    @with_default_kwargs({'bath_threshold': 1e-5, 'verbose': False, 'PBC': False, "process_mode": "default", 'number_of_workers': 1 })
    def __init__(self, problem_formulation: ProblemFormulation, fragments: List[np.ndarray], problem_solver: ProblemSolver, **kwargs: Any) -> None:
        """
        Initialize the DMET class with a problem formulation, fragments, and a solver.

        Args:
            problem_formulation (ProblemFormulation): The problem formulation containing one-body and many-body terms.
            fragments (List[np.ndarray]): A list of arrays, where each array contains indices of a fragment.
            problem_solver (ProblemSolver): An instance of a solver class to solve the fragment Hamiltonians.
            **kwargs: Additional optional parameters, such as 'bath_threshold', 'verbose', and 'number_of_bath_orbitals'.

        Raises:
            ValueError: If not all lattice sites are included in the fragments.

        Main Concept:
            This initializes the DMET object, setting up the problem formulation, fragments, and solver.

        Math Detail:
            The one-body reduced density matrix (RDM) is initialized from the one-body problem formulation.
        """
        self.one_body_problem_formulation = problem_formulation.one_body_problem_formulation
        self.many_body_problem_formulation = problem_formulation.many_body_problem_formulation
        self.fragments = fragments  # list of fragment indices
        self.problem_solver = problem_solver  # instance of ProblemSolver
        self.onebodyrdm = self.one_body_problem_formulation.get_density_matrix()
        self.kwargs = kwargs
        # verbose mode
        # verbose level, supports True/False and also 0/1/2
        # if verbose is True, set it to 1, otherwise set it to 0
        # This allows for flexible verbosity control in the code.
        if isinstance(self.kwargs.get('verbose', False), bool):
            self.kwargs['verbose'] = 1 if self.kwargs['verbose'] else 0
        self._block_print = lambda msg: print(msg)
        # Check if all lattice sites are in the fragment
        if not self.is_all_lattice_sites_in_fragment(fragments):
            raise ValueError("Not all lattice sites are included in the fragment.")
        # END: __init__
    def get_projectors(self, reorder_idxs: Optional[np.ndarray] = None) -> Tuple[List[np.ndarray], List[float]]:
        """
        Generate the projection operators for each fragment.

        Args:
            reorder_idxs (Union[np.ndarray, None], optional): Reordering indices for the orbitals. Defaults to None.

        Returns:
            List[np.ndarray]: A list of projector matrices, one for each fragment.

        Main Concept:
            Projector matrices are used to embed fragments into the full system.

        Math Detail:
            Each projector is computed as:
                P = get_projector_matrix(onebody_rdm, fragment, bath_threshold)
            where `onebody_rdm` is the one-body reduced density matrix, and `bath_threshold` is a cutoff value.
        """
        from ._helpers.Projector import get_projector_matrix
        projectors = []
        number_of_electrons_in_fragments = []
        if reorder_idxs is None:
            from ._helpers.Projector import get_projector_reorder_idxs
            reorder_idxs = get_projector_reorder_idxs(self.fragments, self.onebodyrdm.shape[0])
        for i,fragment in enumerate(self.fragments):
            reorder_idx = reorder_idxs[i] 
            projector, number_of_electrons = get_projector_matrix(self.onebodyrdm, fragment, reorder_idx, self.kwargs['bath_threshold'])
            projectors.append(projector)
            number_of_electrons_in_fragments.append(number_of_electrons)
        return projectors, number_of_electrons_in_fragments
    
    def get_multiplier_hamiltonian(self, mu: float, fragment: np.ndarray) -> FermionOperator:
        """
        Construct the Hamiltonian with a chemical potential term for a fragment.

        Args:
            mu (float): The chemical potential.
            fragment (np.ndarray): The indices of the fragment.

        Returns:
            FermionOperator: The Hamiltonian with the chemical potential term added.

        Main Concept:
            Adds a chemical potential term to the Hamiltonian for a specific fragment.

        Math Detail:
            The Hamiltonian is constructed as:
                H = \sum_{i \in fragment} mu * c_i^\dagger c_i
            where c_i^\dagger and c_i are creation and annihilation operators.
        """
        hamiltonian = FermionOperator()
        for i in range(len(fragment)):
            hamiltonian += FermionOperator(f"{i}^ {i}", mu)
        # print(f"Fragment {fragment}: Chemical potential Hamiltonian: {hamiltonian}")
        return hamiltonian
    
    def get_fragment_hamiltonians(self) -> List['FragmentHamiltonian']:
        """
        Construct the effective Hamiltonians for all fragments.

        Returns:
            List[FragmentHamiltonian]: A list of embedded Hamiltonians for each fragment.

        Main Concept:
            Embeds the one-body and two-body terms into the fragment space using projectors.

        Math Detail:
            The embedded one-body and two-body terms are computed as:
                H_onebody = P^\dagger H_onebody P
                H_twobody = \sum_{pqrs} P^\dagger_p P^\dagger_q H_twobody_{pqrs} P_r P_s
            where P is the projector matrix.
        """
        from ._helpers.Projector import get_projector_reorder_idxs
        reorder_idxs = get_projector_reorder_idxs(self.fragments, self.onebodyrdm.shape[0])
        projectors, num_electrons = self.get_projectors(reorder_idxs=reorder_idxs)
        onebody_terms = self.many_body_problem_formulation.onebody_terms
        twobody_terms = self.many_body_problem_formulation.twobody_terms
        embedded_hamiltonians = []
        print("Calculating fragment Hamiltonians...")
        for i,projector in tqdm(enumerate(projectors), total=len(projectors)):
            projector_conjugate = np.conjugate(projector.T)
            
            # check if the projector is unitary
            if not np.allclose(np.eye(projector.shape[1]), projector_conjugate @ projector, atol=1e-5):
                info =f'''
                The projector for fragment {i} is not unitary.
                This can happen if the projector is not constructed correctly or if the one-body density matrix is not properly defined.
                Please check the one-body density matrix and the projector construction.
                If the issue persists, consider adjusting the bath threshold or the fragment definition.
                If the problem continues, it may indicate a deeper issue with the problem formulation or the solver.
                The projector matrix is:
                {projector}
                The conjugate projector matrix is:
                {projector_conjugate}
                The product of the projector and its conjugate is:
                {projector @ projector_conjugate}
                '''
                raise ValueError(info)
            
            reorder_idx = reorder_idxs[i]
            # print(f"Fragment {i}: Reorder indices: {reorder_idx}")
            fragment_length = len(self.fragments[i])
            reorder_onebody_terms = onebody_terms[np.ix_(reorder_idx, reorder_idx)]
            frag_idx = reorder_idx[:fragment_length]
            embedded_twobody_terms = twobody_terms[np.ix_(
                frag_idx, frag_idx, frag_idx, frag_idx
            )]
            embedded_onebody_terms = projector_conjugate @ reorder_onebody_terms @ projector
            embedded_hamiltonian = FragmentHamiltonian(embedded_onebody_terms, embedded_twobody_terms,number_of_electrons= num_electrons[i])
            embedded_hamiltonians.append(embedded_hamiltonian)
            if self.kwargs['PBC']:
                break
        print("----------------------")
        return embedded_hamiltonians
            
    def get_fragment_energy(self, fragment_hamiltonian: 'FragmentHamiltonian', onebody_rdm: np.ndarray, twobody_rdm: np.ndarray, fragment_length: int) -> float:
        """
        Calculate the energy of a fragment Hamiltonian.

        Args:
            fragment_hamiltonian (FragmentHamiltonian): The embedded Hamiltonian for the fragment.
            onebody_rdm (np.ndarray): The one-body reduced density matrix.
            twobody_rdm (np.ndarray): The two-body reduced density matrix.
            fragment_length (int): The number of orbitals in the fragment.

        Returns:
            float: The energy of the fragment.

        Math:
            E = Tr(h * \gamma) +  \sum H_{ijkl} * \Gamma_{ijkl} 
        """
        idx = list(range(fragment_length))
        idx_bath = list(range(fragment_length, onebody_rdm.shape[0]))
        h = fragment_hamiltonian.onebody_terms[np.ix_(idx, idx)]
        g = fragment_hamiltonian.twobody_terms[np.ix_(idx, idx, idx, idx)]
        h_fragment_bath = fragment_hamiltonian.onebody_terms[np.ix_(idx, idx_bath)]
        gamma = onebody_rdm[np.ix_(idx, idx)]
        gamma_fragment_bath = onebody_rdm[np.ix_(idx, idx_bath)]
        Gamma = twobody_rdm[np.ix_(idx, idx, idx, idx)]
        energy = (
            np.einsum("ij,ij->", h, gamma) 
            +np.einsum("klmn,klmn->", g, Gamma)
            + np.einsum("ij,ij->", h_fragment_bath, gamma_fragment_bath)
        )
        return energy

    def solve_fragment(self, fragment_hamiltonian: 'FragmentHamiltonian', multiplier_hamiltonian: FermionOperator, number_of_orbitals: Optional[int] = None, fragment_length: Optional[int] = None) -> Tuple[float, np.ndarray, np.ndarray]:
        """
        Solve the fragment Hamiltonian with the multiplier Hamiltonian.

        Args:
            fragment_hamiltonian (FragmentHamiltonian): The embedded Hamiltonian for the fragment.
            multiplier_hamiltonian (FermionOperator): The Hamiltonian with the chemical potential term.
            number_of_orbitals (Union[int, None], optional): The number of orbitals in the fragment. Defaults to None.
            fragment_length (Union[int, None], optional): The length of the fragment. Defaults to None.

        Returns:
            Tuple[float, np.ndarray, np.ndarray]:
                - Fragment energy (float)
                - One-body reduced density matrix (np.ndarray)
                - Two-body reduced density matrix (np.ndarray)

        Main Concept:
            Solves the fragment Hamiltonian to obtain the energy and RDMs.

        Math Detail:
            The total Hamiltonian is:
                H_total = H_fragment + H_multiplier
            The solver computes the ground state energy and RDMs for H_total.
        """
        embedded_hamiltonian = fragment_hamiltonian.H + multiplier_hamiltonian
        # fragment_length = len(fragment_hamiltonian.onebody_terms)
        energy, onebody_rdm, twobody_rdm= self.problem_solver.solve(embedded_hamiltonian, number_of_orbitals=number_of_orbitals, number_of_electrons=fragment_hamiltonian.number_of_electrons)
        # print(onebody_rdm.round(5).real)
        fragment_energy = self.get_fragment_energy(fragment_hamiltonian, onebody_rdm, twobody_rdm, fragment_length)
        print(f"Fragment energy: {fragment_energy}")
        return fragment_energy, onebody_rdm, twobody_rdm

    def singleshot(self, mu: float) -> Tuple[float, float]:
        """
        Perform a single-shot DMET calculation.

        Args:
            mu (float): The chemical potential.

        Returns:
            Tuple[float, float]:
                - Total energy (float)
                - Number of electrons (float)

        Main Concept:
            Computes the total energy and electron count for a given chemical potential.

        Math Detail:
            The total energy is the sum of fragment energies:
                E_total = \sum_{fragments} E_fragment
            The number of electrons is computed as:
                N_electrons = \sum_{fragments} Tr(RDM_onebody_fragment)
        """
        self.shot += 1
        if self.kwargs['verbose'] >= 1:
            print(f"Shot {self.shot}: mu = {mu}")
        if self.kwargs["PBC"]:
            energy, number_of_electrons =  self.singleshot_PBC(mu)
        elif self.kwargs["process_mode"] == "multiprocess":
            # energy, number_of_electrons = self.singleshot_multiprocess(mu)
            energy, number_of_electrons = self.singleshot_joblib(mu)
        elif self.kwargs["process_mode"] == "threading" or self.kwargs["process_mode"] == "default":
            energy, number_of_electrons = self.singleshot_threading(mu)
        if self.kwargs['verbose'] >= 1:
            print(f"Total energy: {energy:.5f}, Number of electrons: {number_of_electrons:.5f}")
            print("---------------------")
        self.total_energies.append(energy)
        self.write_shot_total_energy_to_hdf5(energy)
        self.write_shot_total_electrons_to_hdf5(number_of_electrons)
        return energy, number_of_electrons
    
    def singleshot_PBC(self, mu: float) -> Tuple[float, float]:
        """
        Perform a single-shot DMET calculation with periodic boundary conditions (PBC).

        Args:
            mu (float): The chemical potential.

        Returns:
            Tuple[float, float]:
                - Total energy (float)
                - Number of electrons (float)

        Main Concept:
            Similar to `singleshot`, but assumes periodic boundary conditions.

        Math Detail:
            The total energy and electron count are computed as in `singleshot`, but with PBC considerations.
        """
        fragment = self.fragments[0]
        fragmentHamiltonian = self.fragment_hamiltonians[0]
        fragmentEnergy, ne = self.get_fragment_energy_and_number_of_electrons(fragment, fragmentHamiltonian, mu)
        total_energy = fragmentEnergy * len(self.fragments)
        number_of_electrons = ne * len(self.fragments)
        return total_energy, number_of_electrons
    
    def singleshot_threading(self, mu: float) -> Tuple[float, float]:
        """
        Perform a single-shot DMET calculation using threading for parallel processing.

        Args:
            mu (float): The chemical potential.

        Returns:
            Tuple[float, float]:
                - Total energy (float)
                - Number of electrons (float)

        Main Concept:
            Similar to `singleshot`, but uses threading to compute fragment energies in parallel.

        Math Detail:
            The total energy and electron count are computed as in `singleshot`, but with parallel processing.
        """
        from concurrent.futures import ThreadPoolExecutor
        total_energy = 0.0
        number_of_electrons = 0
        with ThreadPoolExecutor(max_workers=self.kwargs['number_of_workers']) as executor:
            futures = [executor.submit(self.get_fragment_energy_and_number_of_electrons, fragment, fragmentHamiltonian, mu) for fragment, fragmentHamiltonian in zip(self.fragments, self.fragment_hamiltonians)]
            for future in futures:
                fragmentEnergy, ne = future.result()
                total_energy += fragmentEnergy
                number_of_electrons += ne
        return total_energy, number_of_electrons
    
    def singleshot_joblib(self, mu: float) -> Tuple[float, float]:
        from joblib import Parallel, delayed
        results = Parallel(n_jobs=self.kwargs['number_of_workers'])(
            delayed(self.get_fragment_energy_and_number_of_electrons)(frag, ham, mu)
            for frag, ham in zip(self.fragments, self.fragment_hamiltonians)
        )
        total_energy = sum(e for e, _ in results)
        number_of_electrons = sum(ne for _, ne in results)
        print(f"Total energy from joblib: {total_energy}, Number of electrons from joblib: {number_of_electrons}")
        return total_energy, number_of_electrons

    def init_hdf5_file(self, filename_prefix: str = "fragment_results") -> None:
        """
        Initialize an HDF5 file for storing fragment calculation results.

        Parameters
        ----------
        filename_prefix : str, optional
            Prefix for the file name (default: 'fragment_results').

        Notes
        -----
        Automatically generates a timestamp and saves the path to self.hdf5_path.
        A new file is created each time run() is called.
        """
        import h5py
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.hdf5_path = f"{filename_prefix}_{timestamp}.h5"
        self.lock_path = self.hdf5_path + ".lock"
        # Create an empty HDF5 file and add a creation timestamp attribute
        with h5py.File(self.hdf5_path, "w") as f:
            f.attrs["created"] = timestamp
        print(f"[DMET] HDF5 file initialized: {self.hdf5_path}")

    def write_fragment_to_hdf5(self, fragment: np.ndarray, mu: float, fragment_energy: float, ne: float, onebody_rdm: np.ndarray, twobody_rdm: np.ndarray) -> None:
        """
        Write the results of a fragment calculation to the HDF5 file.
        Parameters
        ----------
        fragment : np.ndarray
            The indices of the fragment.
        mu : float
            The chemical potential for the fragment.
        fragment_energy : float
            The energy of the fragment.
        ne : float                  
            The number of electrons in the fragment.
        onebody_rdm : np.ndarray
            The one-body reduced density matrix for the fragment.   
        twobody_rdm : np.ndarray
            The two-body reduced density matrix for the fragment.
        Notes
        -----
        This function appends the fragment data to the HDF5 file under the current shot group
        
        HDF5 Structure:
        Root
        └─ shot_{shot_number}
            └─ fragment_{fragment_index}
                ├─ mu : float                                       
                ├─ fragment_energy : float
                ├─ ne : float
                ├─ fragment : np.ndarray
                ├─ onebody_rdm_sparse : sparse matrix (COO format)
                |   ├─ data : np.ndarray (1D array)
                |   ├─ row : np.ndarray (1D array)
                |   ├─ col : np.ndarray (1D array)
                |   └─ shape : np.ndarray (2D array)
                └─ twobody_rdm_sparse : sparse matrix (COO format)
                    ├─ data : np.ndarray (1D array)
                    ├─ row : np.ndarray (1D array)
                    ├─ col : np.ndarray (1D array)
                    └─ shape : np.ndarray (2D array)
        Notes
                  
        """
        import h5py
        import scipy.sparse as sp
        lock = FileLock(self.lock_path)
        with lock:
            with h5py.File(self.hdf5_path, "a") as f:
                shot_group = f.require_group(f"shot_{self.shot}")
                frag_group = shot_group.create_group(f"fragment_{str(fragment)}")
                frag_group.create_dataset("mu", data=mu)
                frag_group.create_dataset("fragment_energy", data=fragment_energy)
                frag_group.create_dataset("ne", data=ne)
                frag_group.create_dataset("fragment", data=fragment)
                # onebody_rdm 轉 coo
                ob_rdm_coo = sp.coo_array(onebody_rdm)
                ob_grp = frag_group.create_group("onebody_rdm_sparse")
                ob_grp.create_dataset("data", data=ob_rdm_coo.data, compression="gzip")
                ob_grp.create_dataset("row", data=ob_rdm_coo.row, compression="gzip")
                ob_grp.create_dataset("col", data=ob_rdm_coo.col, compression="gzip")
                ob_grp.create_dataset("shape", data=ob_rdm_coo.shape)
                # twobody_rdm 轉 coo
                tb_rdm_coo = sp.coo_array(twobody_rdm)
                tb_grp = frag_group.create_group("twobody_rdm_sparse")
                tb_grp.create_dataset("data", data=tb_rdm_coo.data, compression="gzip")
                tb_grp.create_dataset("row", data=tb_rdm_coo.row, compression="gzip")
                tb_grp.create_dataset("col", data=tb_rdm_coo.col, compression="gzip")
                tb_grp.create_dataset("shape", data=tb_rdm_coo.shape)
                tb_grp.attrs["original_shape"] = twobody_rdm.shape  # 存原始 4D shape

    def write_shot_total_energy_to_hdf5(self, total_energy: float) -> None:
        """
        Write the total energy of the current shot to the HDF5 file.
        Parameters
        ----------
        total_energy : float
            The total energy calculated for the current shot.
        Notes
        -----
        This function appends the total energy to the HDF5 file under the current shot group.

        HDF5 Structure:
        Root
        └─ shot_{shot_number}
            └─ total_energy : float
        """
        import h5py
        lock = FileLock(self.lock_path)
        with lock:
            with h5py.File(self.hdf5_path, "a") as f:
                shot_group = f.require_group(f"shot_{self.shot}")
                shot_group.create_dataset("total_energy", data=total_energy)

    def write_shot_total_electrons_to_hdf5(self, total_electrons: float) -> None:
        """
        Write the total number of electrons for the current shot to the HDF5 file.
        Parameters
        ----------
        total_electrons : float
            The total number of electrons calculated for the current shot.
        Notes
        -----
        This function appends the total number of electrons to the HDF5 file under the current shot group.
        HDF5 Structure:
        Root
        └─ shot_{shot_number}
            └─ total_electrons : float
        """
        import h5py
        lock = FileLock(self.lock_path)
        with lock:
            with h5py.File(self.hdf5_path, "a") as f:
                shot_group = f.require_group(f"shot_{self.shot}")
                shot_group.create_dataset("total_electrons", data=total_electrons)

    def get_fragment_energy_and_number_of_electrons(self, fragment: np.ndarray, fragmentHamiltonian: 'FragmentHamiltonian', mu: float) -> Tuple[float, float]:
        """
        Calculate fragment energy and electron number, and write results to HDF5.

        Parameters
        ----------
        fragment : array-like
            Fragment index array.
        fragmentHamiltonian : FragmentHamiltonian
            Fragment Hamiltonian object.
        mu : float
            Chemical potential.

        Returns
        -------
        fragment_energy : float
            Fragment energy.
        ne : float
            Number of electrons in the fragment.

        Notes
        -----
        Elements in the RDMs smaller than the threshold (default 1e-5, set by rdm_threshold) are set to zero.
        """
        import h5py
        number_of_orbitals = max(fragmentHamiltonian.onebody_terms.shape[0], fragmentHamiltonian.twobody_terms.shape[0])
        if self.kwargs['verbose'] >= 1:
            print(f"[DMET] Calculating energy for fragment {fragment} with chemical potential {mu}, number of orbitals {number_of_orbitals}")
        multiplier_hamiltonian = self.get_multiplier_hamiltonian(mu, fragment)
        fragment_energy, onebody_rdm, twobody_rdm = self.solve_fragment(fragmentHamiltonian, multiplier_hamiltonian, number_of_orbitals=number_of_orbitals,fragment_length=len(fragment))
        ne = np.trace(onebody_rdm[np.ix_(range(len(fragment)), range(len(fragment)))])
        if self.kwargs['verbose'] >= 2:
            print(f"[DMET] Fragment {fragment}: Energy = {fragment_energy:.5f}, Number of orbitals: {number_of_orbitals}, Number of electrons = {ne:.5f}")
        import scipy.sparse as sp
        threshold = self.kwargs.get('rdm_threshold', 1e-5)
        onebody_rdm[np.abs(onebody_rdm) < threshold] = 0
        twobody_rdm[np.abs(twobody_rdm) < threshold] = 0
        self.write_fragment_to_hdf5(fragment, mu, fragment_energy, ne, onebody_rdm, twobody_rdm)
        return fragment_energy, ne

    def run(self, mu0: Optional[float] = None, mu1: Optional[float] = None, singleshot: bool = False, filenameprefix: str = 'filename') -> float:
        """
        Perform a self-consistent DMET calculation to find the chemical potential.
        All fragment results are written directly to HDF5, legacy list-based storage is removed.

        Args:
            mu0 (float, optional): Initial guess for the chemical potential (default: None).
            mu1 (float, optional): Second guess for the chemical potential (default: None).
            singleshot (bool, optional): If True, performs a single-shot calculation and returns immediately (default: False).
            filenameprefix (str, optional): Prefix for the HDF5 file name (default: 'filename').

        Returns:
            float: The total energy from the DMET calculation.

        Notes:
            - Automatically generates a timestamp for the HDF5 file.
            - The chemical potential is found using a root-finding method on the objective function.
        """
        self.init_hdf5_file(filename_prefix=filenameprefix)

        from scipy.optimize import newton
        from scipy.optimize import minimize_scalar
        from scipy.optimize import root_scalar

        mu0 = np.random.rand()[0] if mu0 is None else mu0
        mu1 = -np.random.rand()[0] if mu1 is None else mu1
        if mu0 == mu1:
            print("mu0 and mu1 are the same. Adjusting mu1.")
            mu1 = mu0 + 0.1
        if mu0 < mu1:
            mu0, mu1 = mu1, mu0
        self.total_energies = []
        self.shot = 0
        self.fragment_hamiltonians = self.get_fragment_hamiltonians()
        self.objective_buffer = {}
        objective_value = self.objective(mu0)
        if abs(objective_value) < 1e-4:
            msg = (
                f"\n{'='*40}\n"
                f"[DMET] Converged to chemical potential:\n"
                f"    {mu0}\n"
                f"[DMET] Final total energy: {self.total_energies[-1]:.8f}\n"
                f"{'='*40}\n"
            )
            print(msg)
            return self.total_energies[-1]
        if singleshot:
            return self.total_energies[-1]
        objective_value1 = self.objective(mu1)
        for _ in range(10):
            if objective_value * objective_value1 > 0:
                if abs(objective_value) < abs(objective_value1):
                    mu0, mu1 = mu0 + (mu0-mu1) , mu0
                    objective_value1 = objective_value
                    objective_value = self.objective(mu0)
                else:
                    mu0, mu1 = mu1, mu1 - (mu0-mu1)
                    objective_value = objective_value1
                    objective_value1 = self.objective(mu1)
            else:
                break
        if objective_value * objective_value1 > 0:
            msg = (
                f"\n{'='*40}\n"
                f"[DMET] Objective values at mu0 and mu1 are both positive or negative:\n"
                f"    mu0 = {mu0:.6f}, value = {objective_value:.6f}\n"
                f"    mu1 = {mu1:.6f}, value = {objective_value1:.6f}\n"
                f"  Adjusting mu0 and mu1.\n"
                f"{'='*40}\n"
            )
            print(msg)
            result = newton(self.objective,x0=mu0, tol=1e-4, maxiter=1000, x1=mu1)
        elif objective_value * objective_value1 <= 0:
            print("571行開始自洽")
            result = root_scalar(self.objective, bracket=[mu0, mu1], method='brentq', xtol=1e-4, rtol=1e-4, maxiter=1000)
        if result is not None:
            msg = (
                f"\n{'='*40}\n"
                f"[DMET] Converged to chemical potential:\n"
                f"    {result}\n"
                f"[DMET] Final total energy: {self.total_energies[-1]:.8f}\n"
                f"{'='*40}\n"
            )
            print(msg)
        else:
            msg = (
                f"\n{'='*40}\n"
                f"[DMET] Failed to converge to a solution.\n"
                f"{'='*40}\n"
            )
            print(msg)
        return self.total_energies[-1]

    def objective(self, mu: float) -> float:
        """
        Calculate the objective function for the DMET calculation.

        Args:
            mu (float): The chemical potential.

        Returns:
            float: The difference between the DMET and one-body electron counts.

        Main Concept:
            Measures the deviation of the DMET electron count from the target electron count.

        Math Detail:
            The objective is computed as:
                objective = N_electrons_DMET(mu) - N_electrons_onebody
        """
        mu = mu  # Ensure mu is an integer
        if mu in self.objective_buffer:
            return self.objective_buffer[mu]
        _, number_of_electrons_dmet = self.singleshot(mu)
        number_of_electrons_one_body = np.real(np.trace(self.onebodyrdm))
        number_of_electrons_dmet = np.real(number_of_electrons_dmet)
        self.objective_buffer[mu] = number_of_electrons_dmet - number_of_electrons_one_body
        return number_of_electrons_dmet - number_of_electrons_one_body
    
    def is_all_lattice_sites_in_fragment(self, fragments: Union[List[np.ndarray], np.ndarray]) -> bool:
        """
        Check if all lattice sites are included in the provided fragments.

        Args:
            fragments (np.ndarray): List or array of fragment indices.

        Returns:
            bool: True if all lattice sites are covered by the fragments, False otherwise.

        Concept:
            Ensures that every site in the system is assigned to at least one fragment.

        Details:
            Flattens the fragments and checks if every site index from 0 to N-1 is present in the union.
        """
        all_lattice_sites = np.arange(self.onebodyrdm.shape[0])
        # Flatten fragments into a single array
        fragment = np.concatenate(fragments)
        return np.all(np.isin(all_lattice_sites, fragment))

class FragmentHamiltonian:
    def __init__(self, onebody_terms: np.ndarray, twobody_terms: np.ndarray, number_of_electrons: Optional[int] = None) -> None:
        """
        Initialize a fragment Hamiltonian with one-body and two-body integrals.

        Args:
            onebody_terms (np.ndarray): One-body integrals for the fragment.
            twobody_terms (np.ndarray): Two-body integrals for the fragment.
            number_of_electrons (int): Number of electrons in the fragment (required).

        Raises:
            AssertionError: If number_of_electrons is not provided.

        Concept:
            Sets up the embedded Hamiltonian for fragment calculations, ensuring shapes are compatible.

        Details:
            Pads onebody_terms or twobody_terms as needed so their shapes match for further calculations.
        """
        from ._helpers.BuildHamiltonian import build_hamiltonian_from_one_two_body
        self.onebody_terms = onebody_terms
        self.twobody_terms = twobody_terms
        self.number_of_electrons = number_of_electrons
        assert number_of_electrons is not None, "Number of electrons must be specified for the fragment Hamiltonian."
        self.H = build_hamiltonian_from_one_two_body(onebody_terms, twobody_terms)
        # Ensure onebody_terms and twobody_terms have compatible shapes
        if self.onebody_terms.shape[0] < self.twobody_terms.shape[0]:
            new_shape = (self.twobody_terms.shape[0], self.twobody_terms.shape[0])
            self.onebody_terms = np.pad(self.onebody_terms, ((0, new_shape[0] - self.onebody_terms.shape[0]), (0, new_shape[1] - self.onebody_terms.shape[1])), mode='constant')
        elif self.onebody_terms.shape[0] > self.twobody_terms.shape[0]:
            new_shape = (self.onebody_terms.shape[0], self.onebody_terms.shape[0], self.onebody_terms.shape[0], self.onebody_terms.shape[0])
            self.twobody_terms = np.pad(self.twobody_terms, ((0, new_shape[0] - self.twobody_terms.shape[0]), (0, new_shape[1] - self.twobody_terms.shape[1]), (0, new_shape[2] - self.twobody_terms.shape[2]), (0, new_shape[3] - self.twobody_terms.shape[3])), mode='constant')

    def __str__(self) -> str:
        return self.H.__str__()



