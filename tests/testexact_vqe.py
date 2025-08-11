from DMET.ProblemFormulation.Hubbard import OneBodyHubbardFormulation, ManyBodyHubbardFormulation
from DMET.ProblemFormulation.ProblemFormulation import ProblemFormulation
from DMET.DMET import DMET
from DMET.ProblemSolver.QuantumEigenSolver import EigenSolver as QuantumEigenSolver
from DMET.ProblemSolver.ClassicalEigenSolver import EigenSolver as ClassicalEigenSolver
import numpy as np
import matplotlib.pyplot as plt
import cudaq

cudaq.set_target('nvidia', option='mqpu')
# --- 固定參數 ---

L = 120
t = 1.0
fragment_size = 4
fragments = [[i + fragment_size * j for i in range(fragment_size)] for j in range(L // fragment_size * 2)]
scan_points = list(range(2, L+1, 8))
U = 8

# --- Quantum Solver ---
#q_solver = QuantumEigenSolver()
q_solver = QuantumEigenSolver (async_observe = True, depth=3, mode='cudaq-vqe') #depth up #itermax up
x_q, y_q = [], []

for number_of_electrons in scan_points:   
    one_body_hubbard = OneBodyHubbardFormulation(L=L, t=t, number_of_electrons=number_of_electrons)
    many_body_hubbard = ManyBodyHubbardFormulation(L=L, t=t, U=U)

    hubbard_problem = ProblemFormulation()
    hubbard_problem.one_body_problem_formulation = one_body_hubbard
    hubbard_problem.many_body_problem_formulation = many_body_hubbard

    DMET_instance = DMET(hubbard_problem, fragments=fragments, problem_solver = q_solver, verbose=False, PBC=True)
    #DMET_instance = DMET(hubbard_problem, fragments=fragments, problem_solver = q_solver, verbose=True, PBC=True, process_mode="threading", number_of_workers=1)
    energy = DMET_instance.run(0, -2, filenameprefix = f'Hubbard_ASWAP+CZ_U-{U}_L-{L}_frag-{fragment_size}_ne{number_of_electrons}_Quantum')
    x_q.append(number_of_electrons / L)
    y_q.append(energy / L)

# --- Classical Solver ---
c_solver = ClassicalEigenSolver()
x_c, y_c = [], []

for number_of_electrons in scan_points:   
    one_body_hubbard = OneBodyHubbardFormulation(L=L, t=t, number_of_electrons=number_of_electrons)
    many_body_hubbard = ManyBodyHubbardFormulation(L=L, t=t, U=U)

    hubbard_problem = ProblemFormulation()
    hubbard_problem.one_body_problem_formulation = one_body_hubbard
    hubbard_problem.many_body_problem_formulation = many_body_hubbard

    DMET_instance = DMET(hubbard_problem, fragments=fragments, problem_solver = c_solver, verbose=False, PBC=True)
    energy = DMET_instance.run(0, -2, filenameprefix = f'Hubbard_ASWAP+CZ_U-{U}_L-{L}_frag-{fragment_size}_ne{number_of_electrons}_Classical')
    x_c.append(number_of_electrons / L)
    y_c.append(energy / L)

# --- 畫圖 ---

plt.plot(x_c, y_c, marker='o', label='Classical EigenSolver')
plt.plot(x_q, y_q, marker='s', label='Quantum EigenSolver(cudaq-vqe)')
plt.xlabel('Site occupancy ⟨n⟩')
plt.ylabel('Energy per site')
plt.title(f'Hubbard model (fragment size=4)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("Hubbard model(fragment size=4)ASWAP+CZ.png")
plt.show()
