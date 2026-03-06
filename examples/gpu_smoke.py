"""Small GPU smoke test runner for GpuEigenSolver
Usage (on a machine with CUDA & cupy installed):

$ git fetch origin SSHHCenter && git checkout -t origin/SSHHCenter
$ python3 -m venv venv && source venv/bin/activate
$ pip install -e .
# install cupy matching CUDA (example for CUDA 11.8):
# pip install cupy-cuda118
$ python examples/gpu_smoke.py
"""

from DMET.ProblemSolver import GpuEigenSolver
from openfermion import FermionOperator


def main():
    if GpuEigenSolver is None:
        print('GpuEigenSolver not available (cupy not installed).')
        return

    solver = GpuEigenSolver()
    # tiny 2-orbital hopping Hamiltonian (spinless minimal example)
    H = FermionOperator('0^ 1', -1.0) + FermionOperator('1^ 0', -1.0)

    print('Running GPU smoke test...')
    e, r1, r2 = solver.solve(H, number_of_orbitals=2, number_of_electrons=1)
    print('Energy:', e)
    print('1-RDM:', r1)
    print('2-RDM shape:', r2.shape)


if __name__ == '__main__':
    main()
