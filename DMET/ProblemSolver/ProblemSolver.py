from __future__ import annotations
from abc import ABC, abstractmethod
import typing
from openfermion import FermionOperator
from numpy import ndarray
from openfermion.linalg import get_sparse_operator

class ProblemSolver(ABC):
    def __init__(self):
        pass
    
    @abstractmethod
    def solve(self, hamiltonian : FermionOperator, number_of_orbitals :int, **kwargs) -> tuple[float, ndarray, ndarray]:
        """
        Solve the given Hamiltonian and return the energy and reduced density matrices.

        Args:
            hamiltonian (FermionOperator): The Hamiltonian to solve.
            number_of_orbitals (int): The number of orbitals in the system.

        Returns:
            Tuple[float, np.ndarray, np.ndarray]:
                - Ground state energy (float).
                - One-body reduced density matrix (np.ndarray).
                - Two-body reduced density matrix (np.ndarray).

        Main Concept:
            Solves the eigenvalue problem for the Hamiltonian to find the ground state energy and density matrices.

        Math Detail:
            The eigenvalue problem is solved as:
                H \psi = E \psi
            The density matrices are computed from the ground state wavefunction \psi.
        """
        pass
