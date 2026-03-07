from .EigenSolver import EigenSolver

try:
    from .GpuEigenSolver import EigenSolver as GpuEigenSolver
except Exception:
    GpuEigenSolver = None
