from DMET.ProblemFormulation.Hubbard import OneBodyHubbardFormulation, ManyBodyHubbardFormulation
from DMET.ProblemFormulation.ProblemFormulation import ProblemFormulation
from DMET.DMET import DMET
from DMET.ProblemSolver.ClassicalEigenSolver import EigenSolver
import numpy as np
import matplotlib.pyplot as plt

L= 10
number_of_fragments = L//2
number_of_orbitals_in_total = 2*L
each_fragment_size = number_of_orbitals_in_total//number_of_fragments


one_body_hubbard = OneBodyHubbardFormulation(L=L, t=1, number_of_electrons=10)
many_body_hubbard = ManyBodyHubbardFormulation(L=L, t=1, U=2.0)
hubbard_problem = ProblemFormulation()
hubbard_problem.one_body_problem_formulation = one_body_hubbard
hubbard_problem.many_body_problem_formulation = many_body_hubbard
fragments= []


for i in range(number_of_fragments):
    fragment = np.arange(i*each_fragment_size,(i+1)*each_fragment_size)
    fragments.append(fragment)


print(fragments)
    
problemsolver = EigenSolver()

DMET_instance = DMET(hubbard_problem,fragments=fragments,problem_solver=problemsolver,verbose=True)
projectors = DMET_instance.get_projectors()

# for i in range(len(projectors)):
#     print(f"projector {i}:")
#     # use hopmap to plot the projector
#     plt.imshow(projectors[i].real, cmap='hot', interpolation='nearest')
#     plt.colorbar()
#     plt.title(f"Projector {i}")

    
fhs=DMET_instance.get_fragment_hamiltonians()
for i in range(len(fhs)):
    print(f"fragment hamiltonian {i}:")
    # show one body terms
    one_body_terms = fhs[i].onebody_terms.real.round(4)
    print("one body terms:")
    print(one_body_terms)
    break



    