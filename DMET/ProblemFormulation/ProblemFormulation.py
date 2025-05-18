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
    def __init__(self):
        self.H = None
        self.density_matrix = None

    @abstractmethod
    def get_hamiltonian(self):
        pass

    @abstractmethod
    def get_density_matrix(self):
        pass

class ManyBodyProblemFormulation(ABC):
    def __init__(self):
        self.H = None
        self.onebody_terms = None
        self.twobody_terms = None

    @abstractmethod
    def get_hamiltonian(self):
        pass
