from __future__ import annotations
from abc import ABC, abstractmethod
import typing
from openfermion import FermionOperator
from numpy import ndarray
from openfermion.linalg import get_sparse_operator
from ProblemFormulation.ProblemFormulation import OneBodyProblemFormulation, ManyBodyProblemFormulation

class ProblemSolver(ABC):
    def __init__(self):
        pass
    
    @abstractmethod
    def solve(self, hamiltonian : FermionOperator, number_of_orbitals :int, **kwargs) -> tuple[float, ndarray, ndarray]:
        """
        Solve the problem defined by the Hamiltonian.
        Args:
            hamiltonian (FermionOperator): The Hamiltonian to solve.
            **kwargs: Additional arguments for the solver.
        Returns:
            tuple[float, ndarray]: The energy and the one-body density matrix and the two-body density matrix.
        """
        pass
