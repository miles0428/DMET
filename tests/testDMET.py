from DMET.ProblemFormulation.Hubbard import OneBodyHubbardFormulation, ManyBodyHubbardFormulation
from DMET.ProblemFormulation.ProblemFormulation import ProblemFormulation
from DMET.DMET import DMET
from DMET.ProblemSolver.ClassicalEigenSolver import EigenSolver
one_body_hubbard = OneBodyHubbardFormulation(L=4, t=1.0, number_of_electrons=3)
many_body_hubbard = ManyBodyHubbardFormulation(L=4, t=1.0, U=2.0)
hubbard_problem = ProblemFormulation()
hubbard_problem.one_body_problem_formulation = one_body_hubbard
hubbard_problem.many_body_problem_formulation = many_body_hubbard
fragments = [[0, 1], [2, 3],[4,5],[6,7]]
problemsolver = EigenSolver()

DMET_instance = DMET(hubbard_problem,fragments=fragments,problem_solver=problemsolver,verbose=True)
energy =  DMET_instance.run()
