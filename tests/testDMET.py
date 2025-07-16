from DMET.ProblemFormulation.Hubbard import OneBodyHubbardFormulation, ManyBodyHubbardFormulation
from DMET.ProblemFormulation.ProblemFormulation import ProblemFormulation
from DMET.DMET import DMET
from DMET.ProblemSolver.ClassicalEigenSolver import EigenSolver
from DMET.ProblemSolver.QuantumEigenSolver import EigenSolver as QuantumEigenSolver
import numpy as np
L = 120
t= 1.0
U = 8.0
number_of_electrons = 100
fragment_size = 6  # Each fragment will have 2 orbitals
fragments = [[ i+fragment_size*j for  i in range(fragment_size)]for j in range(L//fragment_size*2)]  # Create L/2 fragments, each with 2 orbitals


one_body_hubbard = OneBodyHubbardFormulation(L=L, t=t, number_of_electrons=number_of_electrons)
many_body_hubbard = ManyBodyHubbardFormulation(L=L, t=t, U=U)
hubbard_problem = ProblemFormulation()
hubbard_problem.one_body_problem_formulation = one_body_hubbard
hubbard_problem.many_body_problem_formulation = many_body_hubbard
problemsolver = EigenSolver()

DMET_instance = DMET(hubbard_problem,fragments=fragments,problem_solver=problemsolver,verbose=True,PBC=True, process_mode="threading", number_of_workers=1)
energy =  DMET_instance.run(0,-2)
