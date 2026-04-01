from __future__ import annotations

from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from reaxkit.engine.ams.adapter import AMSAdapter


def _save_dataframe(df: pd.DataFrame, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        df.to_excel(out.with_suffix(".xlsx"), index=False)
    except Exception:
        df.to_csv(out.with_suffix(".txt"), sep="\t", index=False)


def _save_ndarray(arr: np.ndarray, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    a = np.asarray(arr)
    with open(out.with_suffix(".txt"), "w", encoding="utf-8") as fh:
        fh.write(f"shape={a.shape}\n")
        fh.write(np.array2string(a, threshold=2000))
        fh.write("\n")


def _save_any(value: Any, out: Path) -> None:
    if isinstance(value, pd.DataFrame):
        _save_dataframe(value, out)
        return
    if isinstance(value, np.ndarray):
        _save_ndarray(value, out)
        return
    if isinstance(value, (list, tuple)):
        with open(out.with_suffix(".txt"), "w", encoding="utf-8") as fh:
            fh.write(f"type={type(value).__name__} len={len(value)}\n")
            for i, item in enumerate(value):
                fh.write(f"\n[{i}] type={type(item).__name__}\n")
                if isinstance(item, np.ndarray):
                    fh.write(f"shape={item.shape}\n")
                    fh.write(np.array2string(item, threshold=2000))
                    fh.write("\n")
                else:
                    fh.write(f"{item}\n")
        return
    with open(out.with_suffix(".txt"), "w", encoding="utf-8") as fh:
        fh.write(repr(value))
        fh.write("\n")


def _save_dataclass_artifacts(obj: Any, out_dir: Path, name: str, *, skip_fields: set[str] | None = None) -> None:
    assert is_dataclass(obj)
    obj_dir = out_dir / name
    obj_dir.mkdir(parents=True, exist_ok=True)
    skip = skip_fields or set()
    for f in fields(obj):
        if f.name in skip:
            continue
        value = getattr(obj, f.name)
        _save_any(value, obj_dir / f.name)


def _save_selected_frames_as_excel(frames: Any, out_dir: Path, base_name: str) -> None:
    if not isinstance(frames, (list, tuple)) or len(frames) == 0:
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    selected: list[tuple[str, int]] = [("frame_0000", 0)]
    if len(frames) > 1:
        selected.append(("frame_0001", 1))
    selected.append((f"frame_{len(frames)-1:04d}", len(frames) - 1))

    seen: set[int] = set()
    for label, idx in selected:
        if idx in seen:
            continue
        seen.add(idx)
        arr = np.asarray(frames[idx])
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        if arr.ndim != 2:
            continue
        df = pd.DataFrame(arr)
        _save_dataframe(df, out_dir / f"{base_name}_{label}")


def _real_kf_path() -> Path:
    repo_root = Path(__file__).resolve().parents[3]
    return repo_root / "full_sim_examples" / "ZMO_Orio_AMS_KF" / "reaxout.kf"


def test_ams_adapter_loaders_export_artifacts():
    kf_path = _real_kf_path()
    if not kf_path.exists():
        pytest.skip(f"AMS test input file not found: {kf_path}")

    try:
        from scm.plams.tools.kftools import KFFile  # noqa: F401
    except Exception:
        pytest.skip("AMS backend unavailable: scm.plams.tools.kftools is not installed.")

    repo_root = Path(__file__).resolve().parents[3]
    out_dir = repo_root / "tests" / "artifacts" / "ams"
    out_dir.mkdir(parents=True, exist_ok=True)
    adapter = AMSAdapter()
    args = {"input": str(kf_path)}

    loaded_kf = adapter.load_kf(args)
    _save_any(np.asarray(loaded_kf["General%Step numbers"]), out_dir / "load_kf" / "step_numbers")

    trajectory = adapter.load_trajectory(args)
    _save_dataclass_artifacts(trajectory, out_dir, "trajectory")

    connectivity = adapter.load_connectivity(args)
    _save_dataclass_artifacts(connectivity, out_dir, "connectivity", skip_fields={"connectivity", "bond_orders"})
    _save_selected_frames_as_excel(connectivity.connectivity, out_dir / "connectivity", "connectivity")
    _save_selected_frames_as_excel(connectivity.bond_orders, out_dir / "connectivity", "bond_orders")

    connectivity_trajectory = adapter.load_connectivity_trajectory(args)
    _save_dataclass_artifacts(connectivity_trajectory.connectivity, out_dir, "connectivity_trajectory_connectivity")
    _save_dataclass_artifacts(connectivity_trajectory.trajectory, out_dir, "connectivity_trajectory_trajectory")

    simulation = adapter.load_simulation(args)
    _save_dataclass_artifacts(simulation, out_dir, "simulation")

    partial_energy = adapter.load_partial_energy(args)
    _save_dataclass_artifacts(partial_energy, out_dir, "partial_energy")

    charges = adapter.load_charges(args)
    _save_dataclass_artifacts(charges, out_dir, "charges")

    kinematics = adapter.load_atomic_kinematics(args)
    _save_dataclass_artifacts(kinematics, out_dir, "atomic_kinematics")

    atom_strain_energy = adapter.load_atom_strain_energy(args)
    _save_dataclass_artifacts(atom_strain_energy, out_dir, "atom_strain_energy")

    molecular_analysis = adapter.load_molecular_analysis(args)
    _save_dataclass_artifacts(molecular_analysis, out_dir, "molecular_analysis")

    stress = adapter.load_stress(args)
    _save_dataclass_artifacts(stress, out_dir, "stress")

    expected_dirs = [
        "load_kf",
        "trajectory",
        "connectivity",
        "connectivity_trajectory_connectivity",
        "connectivity_trajectory_trajectory",
        "simulation",
        "partial_energy",
        "charges",
        "atomic_kinematics",
        "atom_strain_energy",
        "molecular_analysis",
        "stress",
    ]
    for name in expected_dirs:
        assert (out_dir / name).exists(), f"Missing artifact output directory: {name}"
