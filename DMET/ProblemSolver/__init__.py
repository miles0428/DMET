from .ProblemSolver import ProblemSolver
from .ClassicalEigenSolver.EigenSolver import EigenSolver
from .QuantumEigenSolver.EigenSolver import EigenSolver
# GPU solver (optional)
try:
    from .ClassicalEigenSolver.GpuEigenSolver import EigenSolver as GpuEigenSolver
except Exception:
    # If GPU solver cannot be imported (no cupy), ignore
    GpuEigenSolver = None
