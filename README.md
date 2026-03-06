# DMET

## Overview
DMET (Density Matrix Embedding Theory) is a quantum embedding method for solving strongly correlated electron systems. This repository provides an implementation of DMET for the Hubbard model.

## Installation

### Install directly from GitHub (no local clone needed)

- **CPU-only install (default)**
  ```bash
  pip install git+https://github.com/miles0428/DMET.git
  ```
- **GPU-enabled install (requires `DMET_ENABLE_NV_GPU=1` and proper CUDA/CuPy setup)**
  ```bash
  DMET_ENABLE_NV_GPU=1 pip install git+https://github.com/miles0428/DMET.git
  ```

Setting `DMET_ENABLE_NV_GPU=1` before installation ensures that the optional `cudaq` / `cudaq_solvers` dependencies are pulled in. Leave the variable unset on machines without CUDA tooling to avoid pulling GPU packages.

### Clone and install locally (recommended when developing or tweaking the code)

```bash
git clone https://github.com/miles0428/DMET.git
cd DMET
```

- **CPU-only install (default for local source)**
  ```bash
  pip install .
  ```
- **GPU-enabled install (requires `DMET_ENABLE_NV_GPU=1` and proper CUDA/CuPy setup)**
  ```bash
  DMET_ENABLE_NV_GPU=1 pip install .
  ```

## Getting Started

To get started, please refer to the Jupyter notebooks in the `examples/` directory:
- `ProblemFormulation.ipynb`: Introduction to problem formulation and Hamiltonian construction.
- `ProblemSolver.ipynb`: How to solve quantum many-body problems and use solvers in DMET.
- `DMET_Hubbard.ipynb`: Full DMET workflow for the Hubbard model.

These notebooks provide step-by-step instructions and code examples for using DMET and its modules.

## Contributing
Feel free to submit issues or pull requests to improve the project.

## Author
This repository is maintained by Yu-Cheng Chung and Ssu-Yi Chen.

## License
This project is licensed under the Apache License 2.0.