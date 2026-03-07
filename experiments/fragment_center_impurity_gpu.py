from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path
from typing import Iterable, Optional

import json
import numpy as np

from DMET.DMET import DMET
from DMET.ProblemFormulation.ProblemFormulation import ProblemFormulation
from DMET.ProblemFormulation.SSHHCenterImpurity.SSHHCenterImpurityFormulation_HF import (
    ManyBodyCenterImpuritySSHHFormulation_HF,
    OneBodyCenterImpuritySSHHFormulation_HF,
)
from DMET.ProblemSolver.ClassicalEigenSolver import EigenSolver as ClassicalEigenSolver, GpuEigenSolver


def map_parameters_from_fragment_size(frag_size: int) -> dict:
    t1 = 0.4 + 0.05 * frag_size
    t2 = 1.0 + 0.08 * frag_size
    U = 3.5 + 0.25 * frag_size
    N_cells = max(2, 2 + frag_size // 2)
    alpha = 1 + 0.1 * frag_size
    return dict(N_cells=N_cells, t1=t1, t2=t2, U=U, alpha=alpha)


def make_fragments(L: int, frag_size: int, layout: Optional[dict[str, int] | str] = None) -> list[list[int]]:
    layout_conf: dict[str, int] | None = None
    if isinstance(layout, dict):
        layout_conf = layout
    elif isinstance(layout, str) and layout == "edge4-center6":
        layout_conf = {"center": 6, "edge": 4, "other": 4}

    if layout_conf and any(layout_conf.get(key, 0) > 0 for key in ("center", "edge", "other")):
        return build_orbital_layout_fragments(L, layout_conf)

    # fallback (simple even splitting) when no layout is provided
    center = L // 2
    fragments = [[center]]
    bath_sites = [i for i in range(L) if i != center]
    width = frag_size if frag_size > 0 else 1
    for offset in range(0, len(bath_sites), width):
        fragments.append(bath_sites[offset:offset + width])
    orbital_fragments = []
    for frag in fragments:
        orb = []
        for site in frag:
            orb.append(2 * site)
            orb.append(2 * site + 1)
        orbital_fragments.append(np.array(orb, dtype=int))
    return orbital_fragments


def build_orbital_layout_fragments(L: int, layout: dict[str, int]) -> list[list[int]]:
    """Orbital-space layout with exactly three keys: center -> edge -> other.

    Inputs are *spin-orbital counts* (can be odd). Cutting is done purely on the linear orbital indices
    [0, 1, ..., 2*L-1], NOT via sites.

    Semantics:
      - center: one centered fragment of size `center`
      - edge: one layer adjacent to center (at most one left fragment + one right fragment), each of size `edge`
      - other: all remaining fragments outward, chunked repeatedly with size `other`

    Leftover orbitals on a side that don't fill a full chunk are merged into the nearest fragment on that side
    (i.e., the most recently created fragment from that side).
    """

    total_orbitals = 2 * L

    def clamp_count(value: int) -> int:
        return max(0, int(value))

    center_count = min(total_orbitals, clamp_count(layout.get("center", 0)))
    edge_count = clamp_count(layout.get("edge", 0))
    other_count = clamp_count(layout.get("other", 0))

    # 1) Center fragment: place `center_count` orbitals as centered as possible.
    center_start = 0
    center: list[int] = []
    if center_count > 0:
        center_start = max(0, (total_orbitals - center_count) // 2)
        center_end = min(total_orbitals, center_start + center_count)
        center = list(range(center_start, center_end))
    center_end = center_start + len(center)

    # Remaining orbitals split by side.
    left = list(range(0, center_start))
    right = list(range(center_end, total_orbitals))

    fragments: list[list[int]] = []
    if center:
        fragments.append(center)

    def take_left_near_center(count: int) -> list[int]:
        if count <= 0 or not left:
            return []
        take = left[-count:] if count <= len(left) else left[:]
        del left[-len(take):]
        return take

    def take_right_near_center(count: int) -> list[int]:
        if count <= 0 or not right:
            return []
        take = right[:count] if count <= len(right) else right[:]
        del right[:len(take)]
        return take

    # Track nearest fragments per side for leftover merging.
    last_left_idx: int | None = None
    last_right_idx: int | None = None

    # 2) Edge layer: take once on each side.
    if edge_count > 0:
        le = take_left_near_center(edge_count)
        if le:
            fragments.append(le)
            last_left_idx = len(fragments) - 1
        re = take_right_near_center(edge_count)
        if re:
            fragments.append(re)
            last_right_idx = len(fragments) - 1

    # 3) Other layer: repeat outward.
    if other_count > 0:
        while left or right:
            if left:
                chunk = take_left_near_center(other_count)
                if chunk:
                    if len(chunk) < other_count and last_left_idx is not None:
                        fragments[last_left_idx].extend(chunk)
                    else:
                        fragments.append(chunk)
                        last_left_idx = len(fragments) - 1
            if right:
                chunk = take_right_near_center(other_count)
                if chunk:
                    if len(chunk) < other_count and last_right_idx is not None:
                        fragments[last_right_idx].extend(chunk)
                    else:
                        fragments.append(chunk)
                        last_right_idx = len(fragments) - 1

    # If other_count==0, just merge all remaining into nearest available fragments.
    if other_count == 0 and (left or right):
        if left:
            if last_left_idx is not None:
                fragments[last_left_idx].extend(left)
            elif fragments:
                fragments[0].extend(left)  # fall back to center
            else:
                fragments.append(left)
            left = []
        if right:
            if last_right_idx is not None:
                fragments[last_right_idx].extend(right)
            elif fragments:
                fragments[0].extend(right)
            else:
                fragments.append(right)
            right = []

    # Determinism/readability: sort within each fragment.
    orbital_fragments: list[list[int]] = []
    for frag in fragments:
        orbital_fragments.append(np.array(sorted(frag), dtype=int))
    return orbital_fragments


def make_dirs_and_prefix(base_dir: Path, config_name: str, frag_size: int, params: dict, solver: str, n_elec: int) -> Path:
    suffix = f"frag-{frag_size}_t1-{params['t1']:.2f}_t2-{params['t2']:.2f}_U-{params['U']:.2f}"
    full_dir = base_dir / solver / config_name / suffix
    full_dir.mkdir(parents=True, exist_ok=True)
    return full_dir / f"Ne_{n_elec}"


def load_job_config(path: Path) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def run_fragment_center_impurity(
    base_dir: Path,
    config_name: str,
    frag_size: int,
    scan_radius: int,
    hf_alpha: float,
    hf_tol: float,
    hf_max_iter: int,
    job_label: Optional[str] = None,
    base_params: Optional[dict] = None,
    fragment_layout: Optional[str] = None,
) -> None:
    if base_params is None:
        params = map_parameters_from_fragment_size(frag_size)
    else:
        params = dict(base_params)
        params.setdefault("alpha", 1.0)
    N_cells = int(params["N_cells"])
    L = 2 * N_cells + 1
    fragments = make_fragments(L, frag_size, fragment_layout)

    solver_cls = GpuEigenSolver or ClassicalEigenSolver
    solver_name = "classical-gpu" if GpuEigenSolver else "classical-cpu"
    if solver_name == "classical-cpu":
        print("[WARNING] GPU solver not available; falling back to ClassicalEigenSolver (CPU).")
    q_solver = solver_cls()
    one_body = OneBodyCenterImpuritySSHHFormulation_HF(
        N_cells=N_cells,
        t1=params["t1"],
        t2=params["t2"],
        U=params["U"],
        number_of_electrons=0,
        alpha=hf_alpha,
        tol=hf_tol,
        max_iter=hf_max_iter,
    )
    many_body = ManyBodyCenterImpuritySSHHFormulation_HF(
        N_cells=N_cells,
        t1=params["t1"],
        t2=params["t2"],
        U=params["U"],
    )

    problem = ProblemFormulation()
    problem.one_body_problem_formulation = one_body
    problem.many_body_problem_formulation = many_body

    dmet = DMET(problem, fragments=fragments, problem_solver=q_solver, verbose=2, PBC=False)

    base_fill = L
    scan_points = list(range(base_fill - scan_radius, base_fill + scan_radius + 1))
    job_prefix = job_label or f"frag{frag_size}"

    for n_elec in scan_points:
        one_body.next(number_of_electrons=n_elec)
        hf_energy = one_body.run_hf(alpha=hf_alpha)
        # DEBUG: check density matrix
        dm = one_body.get_density_matrix()
        print(f"[DEBUG] Ne={n_elec}, DM trace={np.trace(dm).real}, max={np.max(np.abs(dm))}")
        filenameprefix = make_dirs_and_prefix(base_dir, config_name, frag_size, params, solver_name, n_elec)
        try:
            energy = dmet.run(mu0=1.0, mu1=-1.0, singleshot=True, filenameprefix=str(filenameprefix))
            print(f"[{job_prefix}] frag={frag_size}, Ne={n_elec}, HF={hf_energy:.6f}, DMET={energy:.6f}")
        except Exception as exc:
            print(f"[{job_prefix}] Failed frag={frag_size}, Ne={n_elec}: {exc}")


def parse_args() -> ArgumentParser:
    parser = ArgumentParser(description="Fragment-controlled center-impurity GPU DMET sweep.")
    parser.add_argument("--base-dir", type=Path, default=Path("DMET/fortest/gpu"), help="輸出資料夾根目錄")
    parser.add_argument("--config-name", type=str, default="fragment_center_impurity", help="用來命名實驗集合")
    parser.add_argument("--frag-sizes", type=int, nargs="+", default=[2, 3, 4], help="要跑的 fragment size list")
    parser.add_argument("--scan-radius", type=int, default=3, help="從半填充開始前後 sweep 的 electron 個數")
    parser.add_argument("--hf-alpha", type=float, default=0.1, help="HF density matrix mixing rate")
    parser.add_argument("--hf-tol", type=float, default=1e-8, help="HF 收斂 tolerance")
    parser.add_argument("--hf-max-iter", type=int, default=1000, help="HF 最大迭代次數")
    parser.add_argument("--job-config", type=Path, default=Path("DMET/experiments/center_impurity_jobs.json"), help="JSON file defining HF experiments")
    return parser


def main() -> None:
    parser = parse_args()
    args = parser.parse_args()

    jobs = load_job_config(args.job_config) if args.job_config and args.job_config.exists() else None

    if jobs:
        for job_name, job in jobs.items():
            frag_size = job.get("frag_size", args.frag_sizes[0])
            scan_radius = job.get("scan_radius", args.scan_radius)
            hf_conf = job.get("hf", {})
            hf_alpha = hf_conf.get("alpha", args.hf_alpha)
            hf_tol = hf_conf.get("tol", args.hf_tol)
            hf_max_iter = hf_conf.get("max_iter", args.hf_max_iter)
            t1t2_pairs = job.get("t1t2_pairs", [])
            if not t1t2_pairs:
                print(f"Job {job_name} has no (t1,t2) pairs; skipping.")
                continue
            for t1, t2 in t1t2_pairs:
                base_params = {
                    "N_cells": job["N_cells"],
                    "t1": t1,
                    "t2": t2,
                    "U": job.get("U", 4),
                    "alpha": job.get("alpha", 1),
                }
                config_label = f"{args.config_name}/{job_name}/t{t1}_t{t2}"
                run_fragment_center_impurity(
                    args.base_dir,
                    config_label,
                    frag_size,
                    scan_radius,
                    hf_alpha,
                    hf_tol,
                    hf_max_iter,
                    job_label=job_name,
                    base_params=base_params,
                    fragment_layout=job.get("fragment_layout"),
                )
    else:
        for frag_size in args.frag_sizes:
            run_fragment_center_impurity(
                args.base_dir,
                args.config_name,
                frag_size,
                args.scan_radius,
                args.hf_alpha,
                args.hf_tol,
                args.hf_max_iter,
                fragment_layout=None,
            )


if __name__ == "__main__":
    main()
