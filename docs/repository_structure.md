# DMET Repository Structure

這份文件專注在 `DMET/` 這個科學 codebase 本身的資料與模組佈局，方便與資料庫無關的 AI 或工程師快速了解 DMET 包內部。

## 1. DMET/ 根目錄概覽

| 路徑 | 說明 |
| --- | --- |
| `DMET/DMET/` | Python 套件，實作 DMET 流程、helpers、formulation 與 solver。 |
| `docs/` | 和科學專案相關的說明文件（本文件就放在這裡）。 |
| `examples/` | Jupyter notebook 與 GPU smoke test script，適用於 sanity check 或教學。 |
| `tests/` | PyTest 單元測試，包含 `testDMET.py`, `testProblemSolver.py`, `test_h2_solver.py` 等。 |
| `README.md`, `LICENSE`, `setup.py`, `modulestructure.md`, `figure4.png` | 針對發行、說明、圖片、module drawing 等維度的專案 metadata。 |

## 2. `DMET/DMET/` 套件細節

- `DMET.py`: DMET 算法的控制中心，負責處理 fragments、embedding space、與問題 solver 的交互。
- `_helpers/`: Hilbert-space 與 projector 等輔助工具（例如 `BuildHamiltonian.py`, `Projector.py`）。
- `ProblemFormulation/`: 抽象 base class 以及多種 Hamiltonian 實作
  - `ProblemFormulation.py` 定義 `OneBodyProblemFormulation` 與 `ManyBodyProblemFormulation`。
  - `Hubbard/`, `SSHH/`, `SSHHCenterImpurity/` 提供不同 lattice、impurity 與 Hartree-Fock 版本的 Hamiltonian。
- `ProblemSolver/`: solver stack
  - `ClassicalEigenSolver/`: classical eigenvalue 求解器與 GPU-enhanced 版本。
  - `QuantumEigenSolver/`: 量子（VQE-like）求解器。
  - `ProblemSolver.py`: 將 formulation 和 solver 串起來的 orchestrator。
- `__init__.py`: 對外暴露 `DMET`, `ProblemSolver`, `ProblemFormulation` 等模組的 package API。

## 3. Repo tree 快照（來源：`tree DMET`）

- `DMET/DMET/`: 包含 `DMET.py`、`_helpers/`、`ProblemFormulation/`、`ProblemSolver/`。
- `DMET/docs/`: 架構說明文件（即本檔）。
- `DMET/examples/`: `DMET_Hubbard.ipynb`、`ProblemFormulation.ipynb`、`ProblemSolver.ipynb`、`gpu_smoke.py`。
- `DMET/tests/`: PyTest test suite 与 `__pycache__`。
- 其他頂層檔案：`README.md`、`LICENSE`、`setup.py`、`moduleStructure.md`、`figure4.png`。

## 4. HF experiment runner

- `DMET/experiments/fragment_center_impurity_gpu.py` 是 HF 專案的 runner；它會讀 `DMET/experiments/center_impurity_jobs.json`、對每個 job 的 `(t1,t2)` 及 parity 連續 sweep electron 數，並把 HF+DMET 能量寫到 `DMET/fortest/gpu/...`。
- JSON job list (`center_impurity_jobs.json`) 內含 `N_cells`、`frag_size`、所有 `(t1,t2)` 組合、HF 設定、`scan_radius` 等參數；如果需要新增組合，只要在該檔新增 entry。
- Runner 仍可用 `--frag-sizes` etc 直接 sweep（fallback），但建議以 JSON job list 集中管理實驗。

## 5. 使用提示

- 如果想延伸新的 formulation/solver，就把 code 加進 `ProblemFormulation/` 或 `ProblemSolver/`，並在 `DMET.dm` 中註冊。
- 實驗腳本可以呼叫 DMET package API（例如 `DMET/DMET.py` + `ProblemFormulation`），但建議把實驗程式與 log 放在 `experiments/` 以外的專案根目錄，避免污染 package。
- 文件、圖、notes 等放 `docs/`，以便獨立管理並供其他 repo（例如 `DMET/docs/`）引用。
