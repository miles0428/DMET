from DMET.ProblemFormulation.Hubbard import OneBodyHubbardFormulation, ManyBodyHubbardFormulation
from DMET.ProblemFormulation.ProblemFormulation import ProblemFormulation
from DMET.DMET import DMET
from DMET.ProblemSolver.ClassicalEigenSolver import EigenSolver
import numpy as np
one_body_hubbard = OneBodyHubbardFormulation(L=6, t=1.0, number_of_electrons=6)
many_body_hubbard = ManyBodyHubbardFormulation(L=6, t=1.0, U=2)
hubbard_problem = ProblemFormulation()
hubbard_problem.one_body_problem_formulation = one_body_hubbard
hubbard_problem.many_body_problem_formulation = many_body_hubbard
# fragments = [[0, 1], [2, 3],[4,5],[6,7]]
# fragments = [[0, 1,2], [3,4,5],[6,7]]
# fragments = [[0, 1,2,3], [4,5,6,7]]
# fragments = [list(range(8))]
fragments = [[ i+4*j for  i in range(4)]for j in range(3)]
problemsolver = EigenSolver()
DMET_instance = DMET(hubbard_problem,fragments=fragments,problem_solver=problemsolver,verbose=True)
energy =  DMET_instance.run(1,-1)
# use classical solver to get the energy
# many_H = many_body_hubbard.H
# # change many_H to a sparse matrix
# from openfermion import get_sparse_operator
# many_H_sparse = get_sparse_operator(many_H, n_qubits=8)
# from scipy.sparse.linalg import eigsh
# values, vectors = eigsh(
#     many_H_sparse, k=100, v0=None, which='SA', maxiter=1e7
# )

# order = np.argsort(values)
# values = values[order]
# vectors = vectors[:, order]
# print(f"DMET energy: {energy}")
# print(f"Exact energy: {values}")