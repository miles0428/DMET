from __future__ import annotations
from ..ProblemFormulation.ProblemFormulation import ProblemFormulation
import numpy as np
from typing import List,Union
from openfermion import FermionOperator
from ..ProblemSolver import ProblemSolver
from tqdm import tqdm

def with_default_kwargs(defaults):
    def decorator(func):
        def wrapper(*args,**kwargs):
            for key, value in defaults.items():
                kwargs.setdefault(key, value)
            return func(*args,**kwargs)
        return wrapper
    return decorator

class DMET:
    @with_default_kwargs({'bath_threshold': 1e-5, 'verbose': False, 'number_of_bath_orbitals': None, 'PBC': False})
    def __init__(self, problem_formulation: ProblemFormulation, fragments: List[np.ndarray], problem_solver: ProblemSolver, **kwargs):
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
        # Check if all lattice sites are in the fragment
        if not self.is_all_lattice_sites_in_fragment(fragments):
            raise ValueError("Not all lattice sites are included in the fragment.")
     
    def get_projectors(self, reorder_idxs: Union[np.ndarray, None] = None):
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
    
    def get_multiplier_hamiltonian(self, mu: float, fragment: np.ndarray):
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
    
    def get_fragment_hamiltonians(self):
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
                print(projector @ projector_conjugate)
                raise ValueError("Projector is not unitary.")
            
            reorder_idx = reorder_idxs[i]
            # print(f"Fragment {i}: Reorder indices: {reorder_idx}")
            fragment_length = len(self.fragments[i])
            reorder_onebody_terms = onebody_terms[np.ix_(reorder_idx, reorder_idx)]
            frag_idx = reorder_idx[:fragment_length]
            embedded_twobody_terms = twobody_terms[np.ix_(
                frag_idx, frag_idx, frag_idx, frag_idx
            )]
            # reorder_twobody_terms = twobody_terms[np.ix_(reorder_idx, reorder_idx, reorder_idx, reorder_idx)]
            embedded_onebody_terms = projector_conjugate @ reorder_onebody_terms @ projector
            # embedded_twobody_terms = reorder_twobody_terms[np.ix_(np.arange(fragment_length), np.arange(fragment_length), np.arange(fragment_length), np.arange(fragment_length))]
            embedded_hamiltonian = FragmentHamiltonian(embedded_onebody_terms, embedded_twobody_terms,number_of_electrons= num_electrons[i])
            embedded_hamiltonians.append(embedded_hamiltonian)
            if self.kwargs['PBC']:
                break
        print("----------------------")
        return embedded_hamiltonians
            
    def get_fragment_energy(self, fragment_hamiltonian: FragmentHamiltonian, onebody_rdm: np.ndarray, twobody_rdm: np.ndarray, fragment_length: int) -> float:
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

    def solve_fragment(self, fragment_hamiltonian: FragmentHamiltonian, multiplier_hamiltonian: FermionOperator, number_of_orbitals: Union[int, None] = None, fragment_length: Union[int, None] = None):
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
        return fragment_energy, onebody_rdm, twobody_rdm

    def singleshot(self, mu: float):
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
        total_energy = 0.0
        number_of_electrons = 0
        self.shot += 1
        if self.kwargs['verbose']:
            print(f"Shot {self.shot}: mu = {mu}")
            
        for i,fragmentHamiltonian in enumerate(self.fragment_hamiltonians):
            fragment = self.fragments[i]
            number_of_orbitals = max(fragmentHamiltonian.onebody_terms.shape[0], fragmentHamiltonian.twobody_terms.shape[0])
            multiplier_hamiltonian = self.get_multiplier_hamiltonian(mu, fragment)
            fragment_energy, onebody_rdm, twobody_rdm = self.solve_fragment(fragmentHamiltonian, multiplier_hamiltonian, number_of_orbitals=number_of_orbitals,fragment_length=len(fragment))
            total_energy += fragment_energy
            # calculate the number of electrons in the fragment
            ne = np.trace(onebody_rdm[np.ix_(range(len(fragment)), range(len(fragment)))])
            number_of_electrons += ne
            if self.kwargs['verbose']:
                print(f"Fragment {i}: Energy = {fragment_energy:.5f}, Number of orbitals: {number_of_orbitals}, Number of electrons = {ne:.5f}")
            if self.kwargs['PBC']:
                total_energy*= len(self.fragments)
                number_of_electrons*= len(self.fragments)
                break
        self.total_energies.append(total_energy)
        if self.kwargs['verbose']:
            print(f"Total energy: {total_energy:.5f}, Number of electrons: {number_of_electrons:.5f}")
            print("---------------------")
        return total_energy, number_of_electrons

    def run(self, mu0: float = None, mu1: float = None):
        """
        Perform a self-consistent DMET calculation to find the chemical potential.

        Args:
            mu0 (float, optional): Initial guess for the chemical potential. Defaults to None.
            mu1 (float, optional): Second guess for the chemical potential. Defaults to None.

        Returns:
            float: The total energy after convergence.

        Main Concept:
            Uses a root-finding algorithm to adjust the chemical potential until the electron count matches.

        Math Detail:
            The objective function is:
                objective(mu) = N_electrons_DMET(mu) - N_electrons_onebody
            The chemical potential is adjusted to minimize this objective.
        """
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
        objective_value1 = self.objective(mu1)
        for _ in range(10):
            if objective_value * objective_value1 > 0:
                # check which one is closer to zero
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
            print(f"Objective values at mu0 and mu1 are both positive or negative: {objective_value}, {objective_value1}. Adjusting mu0 and mu1.")
            result = newton(self.objective,x0=mu0, tol=1e-4, maxiter=1000, x1=mu1)
        elif objective_value * objective_value1 < 0:
            # result = minimize_scalar(self.objective, bracket=(mu0, mu1), method='brent', xtol= 1e-3)
            result = root_scalar(self.objective, bracket=[mu0, mu1], method='brentq', xtol=1e-5, rtol=1e-4, maxiter=1000)
            
        if result is not None:
            print(f"Converged to chemical potential: \n{result}")   
            print(f"Total energy: {self.total_energies[-1]}")
        else:
            print("Failed to converge to a solution.")
        return self.total_energies[-1]

    def objective(self, mu: float):
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
    
    def is_all_lattice_sites_in_fragment(self, fragments: np.ndarray):
        """
        Check if all lattice sites are included in the fragment.

        Args:
            fragments (np.ndarray): The indices of the fragments.

        Returns:
            bool: True if all lattice sites are in the fragment, False otherwise.

        Main Concept:
            Validates that the fragment includes all necessary lattice sites.

        Math Detail:
            The check is performed using:
                all_lattice_sites = np.arange(self.onebodyrdm.shape[0])
                return np.all(np.isin(all_lattice_sites, fragment))
        """
        all_lattice_sites = np.arange(self.onebodyrdm.shape[0])
        # flatten the fragments
        fragment = np.concatenate(fragments)
        return np.all(np.isin(all_lattice_sites,fragment))
    
class FragmentHamiltonian:
    def __init__(self, onebody_terms: np.ndarray, twobody_terms: np.ndarray, number_of_electrons: int = None):
        """
        Initialize the embedded Hamiltonian with one-body and two-body integrals.

        Args:
            onebody_terms (np.ndarray): The one-body integrals.
            twobody_terms (np.ndarray): The two-body integrals.

        Output:
            None

        Main Concept:
            Sets up the embedded Hamiltonian used for fragment calculations.

        Math Detail:
            The Hamiltonian is built from the provided one-body and two-body terms using:
                H = build_hamiltonian_from_one_two_body(onebody_terms, twobody_terms)
        """
        from ._helpers.BuildHamiltonian import build_hamiltonian_from_one_two_body
        self.onebody_terms = onebody_terms
        self.twobody_terms = twobody_terms
        self.number_of_electrons = number_of_electrons
        assert number_of_electrons != None, "Number of electrons must be specified for the fragment Hamiltonian." 
        self.H = build_hamiltonian_from_one_two_body(onebody_terms, twobody_terms)
        # make sure onebody_terms and twobody_terms have the same shape
        if self.onebody_terms.shape[0] < self.twobody_terms.shape[0]:
            new_shape = (self.twobody_terms.shape[0], self.twobody_terms.shape[0])
            self.onebody_terms = np.pad(self.onebody_terms, ((0, new_shape[0] - self.onebody_terms.shape[0]), (0, new_shape[1] - self.onebody_terms.shape[1])), mode='constant')
        elif self.onebody_terms.shape[0] > self.twobody_terms.shape[0]:
            new_shape = (self.onebody_terms.shape[0], self.onebody_terms.shape[0], self.onebody_terms.shape[0], self.onebody_terms.shape[0])
            self.twobody_terms = np.pad(self.twobody_terms, ((0, new_shape[0] - self.twobody_terms.shape[0]), (0, new_shape[1] - self.twobody_terms.shape[1]), (0, new_shape[2] - self.twobody_terms.shape[2]), (0, new_shape[3] - self.twobody_terms.shape[3])), mode='constant')

    def __str__(self) -> str:
        return self.H.__str__()



