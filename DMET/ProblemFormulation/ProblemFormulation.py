from __future__ import annotations
from abc import ABC, abstractmethod
import typing

class ProblemFormulation(ABC):
    def __init__(self):
        self.one_body_problem_formulation: typing.Optional[OneBodyProblemFormulation] = None
        self.many_body_problem_formulation: typing.Optional[ManyBodyProblemFormulation] = None

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
