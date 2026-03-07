#!/usr/bin/env python3
"""
正式部署測試：N=6, N=7
- Fragment layout: 6,4,4
- 電子數遍歷範圍
- 兩組 (t1, t2): (0.5, 1.5), (1.5, 0.5)
- U=10
- Single-shot
- CPU solver
"""

import sys
sys.path.insert(0, '/home/ubuntu/.openclaw/workspace-coding/DMET')

from DMET.DMET import DMET
from DMET.ProblemFormulation.ProblemFormulation import ProblemFormulation
from DMET.ProblemFormulation.SSHHCenterImpurity.SSHHCenterImpurityFormulation_HF import (
    ManyBodyCenterImpuritySSHHFormulation_HF,
    OneBodyCenterImpuritySSHHFormulation_HF,
)
from DMET.ProblemSolver.ClassicalEigenSolver import EigenSolver as ClassicalEigenSolver
from experiments.fragment_center_impurity_gpu import make_fragments
import numpy as np
import traceback
from datetime import datetime
from pathlib import Path

# === 參數配置 ===
test_configs = [
    # N=6
    {"N_cells": 6, "t1": 0.5, "t2": 1.5, "name": "N6_t05_t15"},
    {"N_cells": 6, "t1": 1.5, "t2": 0.5, "name": "N6_t15_t05"},
    # N=7
    {"N_cells": 7, "t1": 0.5, "t2": 1.5, "name": "N7_t05_t15"},
    {"N_cells": 7, "t1": 1.5, "t2": 0.5, "name": "N7_t15_t05"},
]

U = 10.0
layout = {"center": 6, "edge": 4, "other": 4}
base_dir = Path("DMET/fortest/gpu/cpu正式測試")
base_dir.mkdir(parents=True, exist_ok=True)

# 電子數範圍：從 0 到 2*L (full)
# N=6: L=13, orbitals=26, range: 0-26
# N=7: L=15, orbitals=30, range: 0-30

solver = ClassicalEigenSolver()
print(f"[INFO] Using ClassicalEigenSolver (CPU)")
print(f"[INFO] U={U}, layout={layout}")
print(f"[START] {datetime.now()}\n")

results = []

for cfg in test_configs:
    N_cells = cfg["N_cells"]
    t1, t2 = cfg["t1"], cfg["t2"]
    name = cfg["name"]
    L = 2 * N_cells + 1
    total_orbitals = 2 * L
    ne_range = range(0, total_orbitals + 1)  # 0 to 2L
    
    print(f"\n{'#'*70}")
    print(f"# {name}: N_cells={N_cells}, L={L}, orbitals={total_orbitals}")
    print(f"# t1={t1}, t2={t2}, U={U}")
    print(f"# Electron range: {ne_range.start} to {ne_range.stop-1}")
    print(f"{'#'*70}")
    
    # Fragments
    fragments = make_fragments(L, frag_size=4, layout=layout)
    print(f"Fragments: {len(fragments)}")
    for i, f in enumerate(fragments):
        print(f"  F{i}: {list(f)}")
    
    # 電子數遍歷
    for Ne in ne_range:
        h5_path = base_dir / f"{name}_Ne{Ne}.h5"
        log_path = base_dir / f"{name}_Ne{Ne}.log"
        
        try:
            # Problem
            one_body = OneBodyCenterImpuritySSHHFormulation_HF(
                N_cells=N_cells, t1=t1, t2=t2, U=U,
                number_of_electrons=Ne, alpha=0.1, tol=1e-8, max_iter=1000,
            )
            many_body = ManyBodyCenterImpuritySSHHFormulation_HF(
                N_cells=N_cells, t1=t1, t2=t2, U=U,
            )
            problem = ProblemFormulation()
            problem.one_body_problem_formulation = one_body
            problem.many_body_problem_formulation = many_body
            
            # HF
            hf_energy = one_body.run_hf(alpha=0.1)
            dm = one_body.get_density_matrix()
            
            # DMET
            dmet = DMET(problem, fragments=fragments, problem_solver=solver, verbose=0, PBC=False)
            dmet_energy = dmet.run(mu0=1.0, mu1=-1.0, singleshot=False, filenameprefix=str(h5_path.with_suffix('')))
            
            # 記錄結果
            results.append({
                "config": name, "Ne": Ne, "t1": t1, "t2": t2, "U": U,
                "hf_energy": hf_energy, "dmet_energy": dmet_energy.real,
                "status": "OK", "h5": str(h5_path)
            })
            print(f"  Ne={Ne:2d}: HF={hf_energy:10.6f}, DMET={dmet_energy.real:10.6f} ✓")
            
        except Exception as e:
            # 記錄錯誤
            error_msg = traceback.format_exc()
            with open(log_path, 'w') as f:
                f.write(f"Error at {name} Ne={Ne}\n")
                f.write(error_msg)
            results.append({
                "config": name, "Ne": Ne, "t1": t1, "t2": t2, "U": U,
                "hf_energy": None, "dmet_energy": None,
                "status": "FAIL", "log": str(log_path), "error": str(e)
            })
            print(f"  Ne={Ne:2d}: FAILED - {str(e)[:50]} ✗")

print(f"\n{'='*70}")
print(f"[DONE] {datetime.now()}")
print(f"{'='*70}")

# 總結
print("\n=== 結果總結 ===")
ok_count = sum(1 for r in results if r["status"] == "OK")
fail_count = sum(1 for r in results if r["status"] == "FAIL")
print(f"OK: {ok_count}, Failed: {fail_count}")

# 寫入結果 CSV
csv_path = base_dir / "results_summary.csv"
with open(csv_path, 'w') as f:
    f.write("config,Ne,t1,t2,U,hf_energy,dmet_energy,status\n")
    for r in results:
        f.write(f"{r['config']},{r['Ne']},{r['t1']},{r['t2']},{r['U']},{r.get('hf_energy','')},{r.get('dmet_energy','')},{r['status']}\n")
print(f"Summary: {csv_path}")
