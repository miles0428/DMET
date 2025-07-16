from __future__ import annotations
from abc import ABC, abstractmethod
import typing

class ProblemFormulation(ABC):
    def __init__(self):
        """
        Initialize the problem formulation with one-body and many-body formulations.

        Args:
            one_body_problem_formulation (OneBodyProblemFormulation): The one-body problem formulation.
            many_body_problem_formulation (ManyBodyProblemFormulation): The many-body problem formulation.

        Attributes:
            one_body_problem_formulation (OneBodyProblemFormulation): Stores the one-body problem formulation.
            many_body_problem_formulation (ManyBodyProblemFormulation): Stores the many-body problem formulation.
        """
        self.one_body_problem_formulation: typing.Optional[OneBodyProblemFormulation] 
        self.many_body_problem_formulation: typing.Optional[ManyBodyProblemFormulation] 

 
class OneBodyProblemFormulation(ABC):
    """Abstract base class for one-body problem formulations.
    This class defines the interface for one-body problem formulations in quantum mechanics.
    It includes methods to get the Hamiltonian and the density matrix.
    Attributes:
        H (FermionOperator): The Hamiltonian operator for the one-body problem. (Optional)
        density_matrix (ndarray): The density matrix for the one-body problem. (Optional)
    Methods:
        get_hamiltonian(): Returns the Hamiltonian operator for the one-body problem. (Optional)
        get_density_matrix(): Returns the density matrix for the one-body problem. (Must Required)
    Main Concept:
        The one-body problem formulation is a simplified model in quantum mechanics that describes the behavior of a single particle in a potential field.
        It is often used as a starting point for more complex many-body problems.
        
    """
    def __init__(self):
        """
        Initialize the one-body problem formulation.
        Attributes:
            H (FermionOperator): The Hamiltonian operator for the one-body problem.
            density_matrix (ndarray): The density matrix for the one-body problem.
        """
        self.H = None
        self.density_matrix = None

    @abstractmethod
    def get_hamiltonian(self):
        pass

    @abstractmethod
    def get_density_matrix(self):
        pass

class ManyBodyProblemFormulation(ABC):
    """Abstract base class for many-body problem formulations.
    This class defines the interface for many-body problem formulations in quantum mechanics.
    It includes methods to get the Hamiltonian and the reduced density matrices.
    Attributes:
        H (FermionOperator): The Hamiltonian operator for the many-body problem.(Optional)
        onebody_terms (ndarray): The one-body terms of the Hamiltonian. (Must Required)
        twobody_terms (ndarray): The two-body terms of the Hamiltonian. (Must Required)
    Methods:
        get_hamiltonian(): Returns the Hamiltonian operator for the many-body problem. (Optional)
        get_onebody_terms(): Returns the one-body terms of the Hamiltonian. (Optional)
        get_twobody_terms(): Returns the two-body terms of the Hamiltonian. (Optional)
    Main Concept:
        The many-body problem formulation describes the behavior of multiple interacting particles in a quantum system.
        It is a fundamental aspect of quantum mechanics and is essential for understanding complex systems such as solids, liquids, and gases.
    """
    def __init__(self):
        self.H = None
        self.onebody_terms = None
        self.twobody_terms = None

    @abstractmethod
    def get_hamiltonian(self):
        pass
