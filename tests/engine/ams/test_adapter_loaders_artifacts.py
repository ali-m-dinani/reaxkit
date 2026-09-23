from __future__ import annotations

from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from reaxkit.domain.data_models import (
    ConnectivityData,
    ConnectivityTrajectoryData,
    ElectrostaticsData,
    TrajectoryData,
)
from reaxkit.engine.ams.adapter import AMSAdapter


class _FakeKF:
    def __init__(self):
        self.coordinate_reads: list[int] = []
        self.frames = {
            1: [0.0, 0.0, 0.0, 2.0, 0.0, 0.0],
            2: [1.0, 0.0, 0.0, 3.0, 0.0, 0.0],
            3: [2.0, 0.0, 0.0, 4.0, 0.0, 0.0],
        }

    def read_section(self, section: str):
        if section == "History":
            return {"Coords(1)": self.frames[1]}
        raise KeyError(section)

    def read(self, section: str, variable: str):
        if (section, variable) == ("MDHistory", "nEntries"):
            return 3
        if (section, variable) == ("Molecule", "AtomSymbols"):
            return "C O"
        if section == "History":
            match = variable.startswith("Coords(") and variable.endswith(")")
            if match:
                frame_no = int(variable.removeprefix("Coords(").removesuffix(")"))
                self.coordinate_reads.append(frame_no)
                return self.frames[frame_no]
        raise KeyError(f"{section}%{variable}")

    def __getitem__(self, key: str):
        if key == "General%Step numbers":
            return np.asarray([10, 20, 30], dtype=int)
        raise KeyError(key)


class _FakeKFWithoutFrameCount(_FakeKF):
    def read(self, section: str, variable: str):
        if variable == "nEntries":
            raise KeyError(f"{section}%{variable}")
        return super().read(section, variable)


class _FakeConnectivityKF:
    def __init__(self):
        self.n_frames = 3
        self.connectivity_reads: list[tuple[str, int]] = []

    def read_section(self, section: str):
        raise AssertionError(f"read_section({section!r}) should not be used for direct RKF connectivity")

    def read(self, section: str, variable: str):
        if (section, variable) == ("MDHistory", "nEntries"):
            return self.n_frames
        if (section, variable) == ("Molecule", "AtomSymbols"):
            return "C O"
        if section == "History":
            frame_no = None
            for prefix in (
                "Bonds.Index",
                "Bonds.Atoms",
                "Bonds.Orders",
            ):
                if variable.startswith(f"{prefix}(") and variable.endswith(")"):
                    frame_no = int(variable.removeprefix(f"{prefix}(").removesuffix(")"))
                    self.connectivity_reads.append((prefix, frame_no))
                    if not 1 <= frame_no <= self.n_frames:
                        raise KeyError(f"{section}%{variable}")
                    if prefix == "Bonds.Index":
                        return [1, 2, 3]
                    if prefix == "Bonds.Atoms":
                        return [2, 1]
                    if prefix == "Bonds.Orders":
                        return [0.75 + 0.01 * frame_no, 0.75 + 0.01 * frame_no]
        raise KeyError(f"{section}%{variable}")

    def __getitem__(self, key: str):
        if key == "General%Step numbers":
            return np.asarray([10, 20, 30], dtype=int)
        raise KeyError(key)


class _FakeStandaloneReaxoutKF:
    """Standalone ReaxFF KF layout keyed by the stored MD step number."""

    def __init__(self):
        self.steps = np.asarray([0, 1000, 2000], dtype=int)
        self.history_reads: list[str] = []
        self.section_reads: list[str] = []

    def read_section(self, section: str):
        self.section_reads.append(section)
        raise AssertionError("Selected KF streaming must not load a complete section")

    def read(self, section: str, variable: str):
        if (section, variable) == ("General", "Step numbers"):
            return self.steps
        if (section, variable) == ("Molecule", "AtomSymbols"):
            return "Al N"
        if section == "History":
            self.history_reads.append(variable)
            for frame_index, step in enumerate(self.steps.tolist()):
                if variable == f"Coordinates {step}":
                    offset = float(frame_index)
                    return [offset, 0.0, 0.0, 2.0 + offset, 0.0, 0.0]
                if variable == f"Atomic charges {step}":
                    return [1.5 + 0.1 * frame_index, -1.5 - 0.1 * frame_index]
                if variable == f"Unit cell axes {step}":
                    length = 10.0 + frame_index
                    return [length, 0.0, 0.0, 0.0, length, 0.0, 0.0, 0.0, 20.0]
                if variable == f"Unit cell angles {step}":
                    return [90.0, 90.0, 90.0]
        raise KeyError(f"{section}%{variable}")

    def __getitem__(self, key: str):
        if key == "General%Step numbers":
            return self.steps
        raise KeyError(key)


class _FakeStandaloneIndexedAtomNamesKF(_FakeStandaloneReaxoutKF):
    """Standalone layout with species stored only in step-indexed History keys."""

    def read(self, section: str, variable: str):
        if (section, variable) == ("Molecule", "AtomSymbols"):
            raise KeyError(f"{section}%{variable}")
        if section == "History":
            for step in self.steps.tolist():
                if variable == f"Atom names {step}":
                    self.history_reads.append(variable)
                    return "AlN "
        return super().read(section, variable)


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


def test_ams_load_trajectory_prefers_direct_coords_over_single_history_section_frame(monkeypatch):
    adapter = AMSAdapter()
    monkeypatch.setattr(adapter, "load_kf", lambda _args: _FakeKF())

    trajectory = adapter.load_trajectory({"input": "fake.rkf"})

    assert trajectory.positions.shape == (3, 2, 3)
    assert trajectory.elements == ["C", "O"]
    assert trajectory.iterations.tolist() == [10, 20, 30]
    assert trajectory.positions[2, 1, 0] == pytest.approx(4.0 * 0.529177210903)


def test_ams_adapter_streams_history_as_one_frame_payloads(monkeypatch):
    adapter = AMSAdapter()
    fake = _FakeKF()
    monkeypatch.setattr(adapter, "load_kf", lambda _args: fake)

    frames = list(
        adapter.stream(
            TrajectoryData,
            {"input": "fake.rkf", "progress": False},
        )
    )

    assert fake.coordinate_reads == [1, 2, 3]
    assert [frame.source_frame_indices.tolist() for frame in frames] == [[0], [1], [2]]
    assert [frame.iterations.tolist() for frame in frames] == [[10], [20], [30]]
    assert all(frame.positions.shape == (1, 2, 3) for frame in frames)


def test_ams_streams_selected_standalone_kf_electrostatics_without_section_load(monkeypatch):
    adapter = AMSAdapter()
    fake = _FakeStandaloneReaxoutKF()
    monkeypatch.setattr(adapter, "load_kf", lambda _args: fake)

    frames = list(
        adapter.stream(
            ElectrostaticsData,
            {
                "input": "reaxout.kf",
                "_frame_indices": [2, 0],
                "progress": False,
            },
        )
    )

    assert fake.section_reads == []
    assert [frame.trajectory.source_frame_indices.tolist() for frame in frames] == [[2], [0]]
    assert [frame.trajectory.iterations.tolist() for frame in frames] == [[2000], [0]]
    assert frames[0].charges.charges.tolist() == [[1.7, -1.7]]
    assert frames[1].charges.charges.tolist() == [[1.5, -1.5]]
    assert frames[0].trajectory.positions[0, 1, 0] == pytest.approx(4.0)
    assert frames[0].trajectory.simulation.cell_lengths[0].tolist() == pytest.approx([12.0, 12.0, 20.0])
    assert "Coordinates 1000" not in fake.history_reads
    assert "Atomic charges 1000" not in fake.history_reads


def test_ams_streams_step_indexed_atom_names_without_molecule_section(monkeypatch):
    adapter = AMSAdapter()
    fake = _FakeStandaloneIndexedAtomNamesKF()
    monkeypatch.setattr(adapter, "load_kf", lambda _args: fake)

    frames = list(
        adapter.stream(
            ElectrostaticsData,
            {
                "input": "reaxout.kf",
                "_frame_indices": [2, 0],
                "progress": False,
            },
        )
    )

    assert fake.section_reads == []
    assert frames[0].trajectory.elements == ["Al", "N"]
    assert frames[1].trajectory.elements == ["Al", "N"]
    assert "Atom names 2000" in fake.history_reads
    assert "Atom names 1000" not in fake.history_reads


def test_ams_selected_standalone_trajectory_does_not_read_atomic_charges(monkeypatch):
    adapter = AMSAdapter()
    fake = _FakeStandaloneReaxoutKF()
    monkeypatch.setattr(adapter, "load_kf", lambda _args: fake)

    frames = list(
        adapter.stream(
            TrajectoryData,
            {"input": "reaxout.kf", "_frame_indices": [1], "progress": False},
        )
    )

    assert len(frames) == 1
    assert frames[0].source_frame_indices.tolist() == [1]
    assert frames[0].iterations.tolist() == [1000]
    assert not any(name.startswith("Atomic charges") for name in fake.history_reads)


def test_ams_loads_only_selected_standalone_kf_charges(monkeypatch):
    adapter = AMSAdapter()
    fake = _FakeStandaloneReaxoutKF()
    monkeypatch.setattr(adapter, "load_kf", lambda _args: fake)

    charges = adapter.load_charges({"input": "reaxout.kf", "_frame_indices": [2, 0]})

    assert fake.section_reads == []
    assert charges.charges.tolist() == [[1.7, -1.7], [1.5, -1.5]]
    assert charges.iterations.tolist() == [2000, 0]
    assert charges.metadata["source_frame_indices"] == [2, 0]


def test_ams_load_connectivity_reads_direct_rkf_frames_without_history_section(monkeypatch):
    adapter = AMSAdapter()
    monkeypatch.setattr(adapter, "load_kf", lambda _args: _FakeConnectivityKF())
    progress: list[tuple[str, int, int, str | None]] = []

    connectivity = adapter.load_connectivity(
        {"input": "fake.rkf"},
        reporter=lambda stage, current, total, message=None: progress.append((stage, current, total, message)),
    )

    assert len(connectivity.connectivity) == 3
    assert connectivity.connectivity[0].tolist() == [[2], [1]]
    assert np.allclose(connectivity.bond_orders[2], [[0.78], [0.78]])
    assert connectivity.sum_bond_orders.shape == (3, 2)
    assert connectivity.elements == ["C", "O"]
    assert connectivity.iterations.tolist() == [10, 20, 30]
    assert progress == [
        ("load", 1, 3, "Reading AMS RKF connectivity"),
        ("load", 2, 3, "Reading AMS RKF connectivity"),
        ("load", 3, 3, "Reading AMS RKF connectivity"),
    ]


def test_ams_load_trajectory_reads_only_selected_rkf_frames(monkeypatch):
    adapter = AMSAdapter()
    fake = _FakeKF()
    monkeypatch.setattr(adapter, "load_kf", lambda _args: fake)

    trajectory = adapter.load(
        TrajectoryData,
        {"input": "fake.rkf", "frames": [2, 0]},
    )

    assert fake.coordinate_reads == [3, 1]
    assert trajectory.positions.shape == (2, 2, 3)
    assert trajectory.iterations.tolist() == [30, 10]
    assert trajectory.source_frame_indices.tolist() == [2, 0]


def test_ams_selective_trajectory_does_not_scan_coords_when_nentries_is_missing(monkeypatch):
    adapter = AMSAdapter()
    fake = _FakeKFWithoutFrameCount()
    monkeypatch.setattr(adapter, "load_kf", lambda _args: fake)

    trajectory = adapter.load(
        TrajectoryData,
        {"input": "fake.rkf", "frames": [2]},
    )

    assert fake.coordinate_reads == [3]
    assert trajectory.source_frame_indices.tolist() == [2]


def test_ams_load_connectivity_reads_only_selected_rkf_frames(monkeypatch):
    adapter = AMSAdapter()
    fake = _FakeConnectivityKF()
    monkeypatch.setattr(adapter, "load_kf", lambda _args: fake)

    connectivity = adapter.load(
        ConnectivityData,
        {"input": "fake.rkf", "frame_indices": [2, 0]},
    )

    assert fake.connectivity_reads == [
        ("Bonds.Index", 3),
        ("Bonds.Atoms", 3),
        ("Bonds.Orders", 3),
        ("Bonds.Index", 1),
        ("Bonds.Atoms", 1),
        ("Bonds.Orders", 1),
    ]
    assert len(connectivity.connectivity) == 2
    assert connectivity.iterations.tolist() == [30, 10]
    assert connectivity.source_frame_indices.tolist() == [2, 0]


def test_ams_resolve_kf_path_prefers_explicit_input_over_rewritten_run_dir(tmp_path):
    explicit = tmp_path / "30_173_ams.rkf"
    explicit.write_text("fake", encoding="utf-8")
    raw_dir = tmp_path / "reaxkit_workspace" / "data" / "raw" / "run_x"
    raw_dir.mkdir(parents=True)

    resolved = AMSAdapter._resolve_kf_path(
        {
            "input": str(explicit),
            "run_dir": str(raw_dir),
        }
    )

    assert resolved == explicit


def test_ams_required_input_files_prefers_explicit_kf_input_name():
    adapter = AMSAdapter()

    names = adapter.required_input_files(
        ConnectivityTrajectoryData,
        {"input": "30_1073_ams.rkf"},
    )

    assert names == ("30_1073_ams.rkf",)
    assert adapter.required_input_files(
        ElectrostaticsData,
        {"input": "reaxout.kf"},
    ) == ("reaxout.kf",)


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
