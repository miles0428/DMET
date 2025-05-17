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
    @with_default_kwargs({'bath_threshold': 1e-5, 'verbose': False})
    def __init__(self, problem_formulation: ProblemFormulation, fragments: List[np.ndarray], problem_solver: ProblemSolver, **kwargs):
        """
        Initialize the DMET class with a problem formulation, fragments, and a solver.

        Args:
            problem_formulation (ProblemFormulation): The problem formulation containing one-body and many-body terms.
            fragments (List[np.ndarray]): A list of arrays, where each array contains indices of a fragment.
            problem_solver: An instance of a solver class to solve the fragment Hamiltonians.
            **kwargs: Additional optional parameters, such as 'bath_threshold'.

        Output:
            None

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
        
    def get_projectors(self):
        """
        Generate the projection operators for each fragment.

        Args:
            None

        Output:
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
        for fragment in self.fragments:
            projector = get_projector_matrix(self.onebodyrdm, fragment, self.kwargs['bath_threshold'])
            projectors.append(projector)
        return projectors
    
    def get_multiplier_hamiltonian(self, mu: float, fragment: np.ndarray):
        """
        Construct the Hamiltonian with a chemical potential term for a fragment.

        Args:
            mu (float): The chemical potential.
            fragment (np.ndarray): The indices of the fragment.

        Output:
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
        return hamiltonian
        
    def get_fragment_hamiltonians(self):
        """
        Construct the effective Hamiltonians for all fragments.

        Args:
            None

        Output:
            List[FragmentHamiltonian]: A list of embedded Hamiltonians for each fragment.

        Main Concept:
            Embeds the one-body and two-body terms into the fragment space using projectors.

        Math Detail:
            The embedded one-body and two-body terms are computed as:
                H_onebody = P^\dagger H_onebody P
                H_twobody = \sum_{pqrs} P^\dagger_p P^\dagger_q H_twobody_{pqrs} P_r P_s
            where P is the projector matrix.
        """
        projectors = self.get_projectors()
        onebody_terms = self.many_body_problem_formulation.onebody_terms
        twobody_terms = self.many_body_problem_formulation.twobody_terms
        embedded_hamiltonians = []
        print("Calculating fragment Hamiltonians...")
        fragment_idx = 0
        for projector in tqdm(projectors):
            projector_conjugate = projector.conjugate().T
            # Add one-body terms
            embedded_onebody_terms = projector_conjugate @ onebody_terms @ projector
            # embedded_twobody_terms= np.einsum( 'pi,qj,pqrs,rk,sl->ijkl', 
            #                                   projector_conjugate, 
            #                                   projector_conjugate, 
            #                                   twobody_terms, 
            #                                   projector, 
            #                                   projector,
            #                                   optimize='greedy')
            # twobody terms no need to be projected
            embedded_twobody_terms = twobody_terms[np.ix_(self.fragments[fragment_idx], self.fragments[fragment_idx], self.fragments[fragment_idx], self.fragments[fragment_idx])]
            embedded_hamiltonian = FragmentHamiltonian(embedded_onebody_terms, embedded_twobody_terms)
            embedded_hamiltonians.append(embedded_hamiltonian)
            fragment_idx += 1
        return embedded_hamiltonians
            
    def get_fragment_energy(self, fragment_hamiltonian: FragmentHamiltonian, onebody_rdm: np.ndarray, twobody_rdm: np.ndarray):
        """
        Calculate the energy of a fragment Hamiltonian.

        Args:
            fragment_hamiltonian (FragmentHamiltonian): The embedded Hamiltonian for the fragment.
            onebody_rdm (np.ndarray): The one-body reduced density matrix.
            twobody_rdm (np.ndarray): The two-body reduced density matrix.

        Output:
            float: The energy of the fragment.

        Main Concept:
            Computes the energy contribution of a fragment using its Hamiltonian and RDMs.

        Math Detail:
            The energy is calculated as:
                E = Tr(H_onebody * RDM_onebody) + 0.5 * \sum_{ijkl} H_twobody_{ijkl} * RDM_twobody_{ijkl}
        """
        # Calculate the energy of the fragment Hamiltonian
        fragment_energy = np.einsum('ij,ji->', onebody_rdm, fragment_hamiltonian.onebody_terms) + \
                          0.5 * np.einsum('ijkl,ijkl->', twobody_rdm, fragment_hamiltonian.twobody_terms)
        return fragment_energy

    def solve_fragment(self, fragment_hamiltonian: FragmentHamiltonian, multiplier_hamiltonian: FermionOperator, number_of_orbitals: Union[int, None] = None):
        """
        Solve the fragment Hamiltonian with the multiplier Hamiltonian.

        Args:
            fragment_hamiltonian (FragmentHamiltonian): The embedded Hamiltonian for the fragment.
            multiplier_hamiltonian (FermionOperator): The Hamiltonian with the chemical potential term.

        Output:
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
        energy, onebody_rdm, twobody_rdm= self.problem_solver.solve(embedded_hamiltonian, number_of_orbitals=number_of_orbitals)
        fragment_energy = self.get_fragment_energy(fragment_hamiltonian, onebody_rdm, twobody_rdm)
        return fragment_energy, onebody_rdm, twobody_rdm

    def singleshot(self, mu: float):
        """
        Perform a single-shot DMET calculation.

        Args:
            mu (float): The chemical potential.

        Output:
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
            number_of_orbitals = fragmentHamiltonian.onebody_terms.shape[0]
            multiplier_hamiltonian = self.get_multiplier_hamiltonian(mu, fragment)
            fragment_energy, onebody_rdm, twobody_rdm = self.solve_fragment(fragmentHamiltonian, multiplier_hamiltonian, number_of_orbitals=number_of_orbitals)
            total_energy += fragment_energy
            # calculate the number of electrons in the fragment
            number_of_electrons += np.trace(onebody_rdm[np.ix_(range(len(fragment)), range(len(fragment)))])
            if self.kwargs['verbose']:
                print(f"Fragment {i}: Energy = {fragment_energy}, Number of electrons = {number_of_electrons}")

        self.total_energies.append(total_energy)
        if self.kwargs['verbose']:
            print(f"Total energy: {total_energy}")
            print("---------------------")
        return total_energy, number_of_electrons

    def run(self):
        """
        Perform a self-consistent DMET calculation to find the chemical potential.

        Args:
            None

        Output:
            float: The total energy after convergence.

        Main Concept:
            Uses a root-finding algorithm to adjust the chemical potential until the electron count matches.

        Math Detail:
            The objective function is:
                objective(mu) = N_electrons_DMET(mu) - N_electrons_onebody
            The chemical potential is adjusted to minimize this objective.
        """
        from scipy.optimize import newton
        mu = 0.0
        self.total_energies = []
        self.shot = 0
        # use gradient descent to find the chemical potential
        self.fragment_hamiltonians = self.get_fragment_hamiltonians()
        result = newton(func=self.objective, x0=mu, tol=1e-5, maxiter=100,disp=True)
        if result is not None:
            print(f"Converged to chemical potential: {result}")
            print(f"Total energy: {self.total_energies[-1]}")
        else:
            print("Failed to converge to a solution.")
        return self.total_energies[-1]

    def objective(self, mu: float):
        """
        Calculate the objective function for the DMET calculation.

        Args:
            mu (float): The chemical potential.

        Output:
            float: The difference between the DMET and one-body electron counts.

        Main Concept:
            Measures the deviation of the DMET electron count from the target electron count.

        Math Detail:
            The objective is computed as:
                objective = N_electrons_DMET(mu) - N_electrons_onebody
        """
        _, number_of_electrons_dmet = self.singleshot(mu)
        number_of_electrons_one_body = np.trace(self.onebodyrdm)
        return number_of_electrons_dmet - number_of_electrons_one_body
    
    def is_all_lattice_sites_in_fragment(self, fragments: np.ndarray):
        """
        Check if all lattice sites are included in the fragment.

        Args:
            fragment (np.ndarray): The indices of the fragment.

        Output:
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
    def __init__(self, onebody_terms: np.ndarray, twobody_terms: np.ndarray):
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
        self.H = build_hamiltonian_from_one_two_body(onebody_terms, twobody_terms)
        
    def __str__(self) -> str:
        return self.H.__str__()
    


