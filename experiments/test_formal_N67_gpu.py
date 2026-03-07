#!/usr/bin/env python3
"""
正式部署測試：N=6, N=7
- Fragment layout: 6,4,4
- 電子數遍歷範圍：0 到 2L
- 兩組 (t1, t2): (0.5, 1.5), (1.5, 0.5)
- U=10
- Self-consistent (singleshot=False)
- GPU solver (with CPU fallback)
"""

import sys
sys.path.insert(0, '/home/ssuyichen/DMET')

import numpy as np
import traceback
from datetime import datetime
from pathlib import Path

# === Inline make_fragments ===
def make_fragments(L, frag_size, layout=None):
    if layout is None:
        layout = {"center": 6, "edge": 4, "other": 4}
    return build_orbital_layout_fragments(L, layout)

def build_orbital_layout_fragments(L, layout):
    total_orbitals = 2 * L
    center_count = min(total_orbitals, max(0, layout.get("center", 0)))
    edge_count = max(0, layout.get("edge", 0))
    other_count = max(0, layout.get("other", 0))
    
    center_start = max(0, (total_orbitals - center_count) // 2) if center_count > 0 else 0
    center = list(range(center_start, center_start + center_count)) if center_count > 0 else []
    center_end = center_start + len(center)
    
    left = list(range(0, center_start))
    right = list(range(center_end, total_orbitals))
    
    fragments = []
    if center:
        fragments.append(center)
    
    def take_left(n):
        if n <= 0 or not left: return []
        t = left[-n:] if n <= len(left) else left[:]
        del left[-len(t):]
        return t
    
    def take_right(n):
        if n <= 0 or not right: return []
        t = right[:n] if n <= len(right) else right[:]
        del right[:len(t):]
        return t
    
    last_left, last_right = None, None
    
    if edge_count > 0:
        le = take_left(edge_count)
        if le:
            fragments.append(le)
            last_left = len(fragments) - 1
        re = take_right(edge_count)
        if re:
            fragments.append(re)
            last_right = len(fragments) - 1
    
    if other_count > 0:
        while left or right:
            if left:
                c = take_left(other_count)
                if c:
                    if len(c) < other_count and last_left is not None:
                        fragments[last_left].extend(c)
                    else:
                        fragments.append(c)
                        last_left = len(fragments) - 1
            if right:
                c = take_right(other_count)
                if c:
                    if len(c) < other_count and last_right is not None:
                        fragments[last_right].extend(c)
                    else:
                        fragments.append(c)
                        last_right = len(fragments) - 1
    
    if other_count == 0 and (left or right):
        if left:
            if last_left is not None:
                fragments[last_left].extend(left)
            elif fragments:
                fragments[0].extend(left)
            else:
                fragments.append(left)
        if right:
            if last_right is not None:
                fragments[last_right].extend(right)
            elif fragments:
                fragments[0].extend(right)
            else:
                fragments.append(right)
    
    return [np.array(f, dtype=int) for f in fragments]

# === Main ===
from DMET.DMET import DMET
from DMET.ProblemFormulation.ProblemFormulation import ProblemFormulation
from DMET.ProblemFormulation.SSHHCenterImpurity.SSHHCenterImpurityFormulation_HF import (
    ManyBodyCenterImpuritySSHHFormulation_HF,
    OneBodyCenterImpuritySSHHFormulation_HF,
)
from DMET.ProblemSolver.ClassicalEigenSolver import EigenSolver as ClassicalEigenSolver, GpuEigenSolver

# === 參數配置 ===
test_configs = [
    {"N_cells": 6, "t1": 0.5, "t2": 1.5, "name": "N6_t05_t15"},
    {"N_cells": 6, "t1": 1.5, "t2": 0.5, "name": "N6_t15_t05"},
    {"N_cells": 7, "t1": 0.5, "t2": 1.5, "name": "N7_t05_t15"},
    {"N_cells": 7, "t1": 1.5, "t2": 0.5, "name": "N7_t15_t05"},
]

U = 10.0
layout = {"center": 6, "edge": 4, "other": 4}
base_dir = Path("/home/ssuyichen/DMET/fortest/gpu_gpu正式測試")
base_dir.mkdir(parents=True, exist_ok=True)

# === 選擇 Solver ===
solver_cls = GpuEigenSolver or ClassicalEigenSolver
solver_name = "GPU" if GpuEigenSolver else "CPU-fallback"
solver = solver_cls()

print(f"[INFO] Using solver: {solver_name}")
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
    print(f"# t1={t1}, t2={t2}, U={U}, solver={solver_name}")
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
            
            results.append({
                "config": name, "Ne": Ne, "t1": t1, "t2": t2, "U": U,
                "solver": solver_name,
                "hf_energy": hf_energy, "dmet_energy": dmet_energy.real,
                "status": "OK", "h5": str(h5_path)
            })
            print(f"  Ne={Ne:2d}: HF={hf_energy:10.6f}, DMET={dmet_energy.real:10.6f} OK")
            
        except Exception as e:
            error_msg = traceback.format_exc()
            with open(log_path, 'w') as f:
                f.write(f"Error at {name} Ne={Ne}\n")
                f.write(error_msg)
            results.append({
                "config": name, "Ne": Ne, "t1": t1, "t2": t2, "U": U,
                "solver": solver_name,
                "hf_energy": None, "dmet_energy": None,
                "status": "FAIL", "log": str(log_path), "error": str(e)
            })
            print(f"  Ne={Ne:2d}: FAIL - {str(e)[:50]}")

print(f"\n{'='*70}")
print(f"[DONE] {datetime.now()}")
print(f"{'='*70}")

ok_count = sum(1 for r in results if r["status"] == "OK")
fail_count = sum(1 for r in results if r["status"] == "FAIL")
print(f"\n=== Summary ===")
print(f"OK: {ok_count}, Failed: {fail_count}")

csv_path = base_dir / "results_summary.csv"
with open(csv_path, 'w') as f:
    f.write("config,Ne,t1,t2,U,solver,hf_energy,dmet_energy,status\n")
    for r in results:
        f.write(f"{r['config']},{r['Ne']},{r['t1']},{r['t2']},{r['U']},{r.get('solver','')},{r.get('hf_energy','')},{r.get('dmet_energy','')},{r['status']}\n")
print(f"CSV: {csv_path}")
