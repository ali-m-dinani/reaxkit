"""LAMMPS engine adapter (scaffold)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from reaxkit.core.engine_registry import register_engine
from reaxkit.domain.data_models import ConnectivityData, ConnectivityTrajectoryData, SimulationData, TrajectoryData
from reaxkit.engine.base import EngineAdapter
from reaxkit.engine.common.xyz_generator import write_xyz_trajectory
from reaxkit.engine.lammps.dump_handler import LAMMPSDumpHandler
from reaxkit.engine.lammps.lammps_log_handler import LAMMPSLogHandler


@register_engine("lammps")
class LAMMPSAdapter(EngineAdapter):
    """Adapter scaffold for LAMMPS trajectory formats."""

    def detect(self, path: str | Path) -> float:
        p = Path(path)
        if p.is_dir():
            if (p / "log.lammps").exists():
                return 0.8
            if any("dump" in child.name.lower() for child in p.iterdir() if child.is_file()):
                return 0.8
        if p.is_file():
            if "dump" in p.name.lower():
                return 0.8
            if (p.parent / "log.lammps").exists():
                return 0.8
            if p.name.lower() == "log.lammps":
                return 0.8
        if "dump" in p.name.lower():
            return 0.8
        return 0.0

    def required_input_files(self, data_type, args: dict) -> tuple[str, ...] | None:
        _ = args
        mapping: dict[object, tuple[str, ...]] = {
            TrajectoryData: ("dump.xyz", "dump.lammpstrj", "lammpstrj"),
            SimulationData: ("log.lammps", "dump.xyz"),
            ConnectivityData: ("dump.xyz", "dump.lammpstrj", "lammpstrj"),
            ConnectivityTrajectoryData: ("dump.xyz", "dump.lammpstrj", "lammpstrj", "log.lammps"),
        }
        return mapping.get(data_type)

    @staticmethod
    def _resolve_run_path(args: dict, *keys: str, default_name: str) -> Path:
        for key in keys:
            raw = args.get(key)
            if raw:
                p = Path(raw)
                if p.is_dir():
                    return p / default_name
                return p

        run_dir = args.get("run_dir")
        if run_dir:
            return Path(run_dir) / default_name

        input_path = args.get("input")
        if input_path:
            p = Path(input_path)
            if p.is_dir():
                return p / default_name
            return p

        return Path(default_name)

    @staticmethod
    def _first_dump_in_dir(folder: Path) -> Path | None:
        if not folder.exists() or not folder.is_dir():
            return None
        candidates = sorted(
            [child for child in folder.iterdir() if child.is_file() and "dump" in child.name.lower()],
            key=lambda p: p.name.lower(),
        )
        return candidates[0] if candidates else None

    def _resolve_dump_path(self, args: dict) -> Path:
        initial = self._resolve_run_path(args, "dump", "dump_file", "trajectory", default_name="dump.xyz")
        if initial.exists():
            return initial
        if initial.is_dir():
            found = self._first_dump_in_dir(initial)
            if found is not None:
                return found
        parent = initial.parent
        found = self._first_dump_in_dir(parent)
        if found is not None:
            return found
        return initial

    def _resolve_log_path(self, args: dict) -> Path:
        initial = self._resolve_run_path(args, "log", "log_file", "simulation", default_name="log.lammps")
        if initial.exists():
            return initial
        if initial.is_dir():
            candidate = initial / "log.lammps"
            if candidate.exists():
                return candidate
        candidate = initial.parent / "log.lammps"
        if candidate.exists():
            return candidate
        return initial

    @staticmethod
    def _column(data: dict[str, np.ndarray], *names: str) -> np.ndarray | None:
        for name in names:
            if name in data:
                return np.asarray(data[name])
        return None

    def load_trajectory(self, args: dict, reporter=None) -> TrajectoryData:
        dump_path = self._resolve_dump_path(args)
        handler = LAMMPSDumpHandler(dump_path, reporter=reporter)
        sim_df = handler.dataframe()
        n_frames = handler.n_frames()
        if n_frames <= 0:
            raise RuntimeError(f"No trajectory frames found in LAMMPS dump file: {dump_path}")

        frames = [handler.frame(i) for i in range(n_frames)]
        n_atoms = max(int(np.asarray(fr["coords"]).shape[0]) for fr in frames) if frames else 0

        positions = np.full((n_frames, n_atoms, 3), np.nan, dtype=float)
        atom_labels = np.full((n_frames, n_atoms), "", dtype=object)
        element_map: dict[int, str] = {}

        for fi, frame in enumerate(frames):
            coords = np.asarray(frame.get("coords"), dtype=float)
            atom_types = list(frame.get("atom_types") or [])
            frame_atoms = min(coords.shape[0], n_atoms)
            if frame_atoms <= 0:
                continue
            positions[fi, :frame_atoms, :] = coords[:frame_atoms, :]
            for ai in range(frame_atoms):
                label = str(atom_types[ai]) if ai < len(atom_types) else ""
                atom_labels[fi, ai] = label
                if label and (ai + 1) not in element_map:
                    element_map[ai + 1] = label

        atom_ids = list(range(1, n_atoms + 1))
        elements = [element_map.get(aid, "X") for aid in atom_ids]
        iterations = (
            sim_df["iter"].to_numpy(dtype=int)
            if "iter" in sim_df.columns
            else np.arange(n_frames, dtype=int)
        )

        cell_lengths = None
        if {"xlo", "xhi", "ylo", "yhi", "zlo", "zhi"}.issubset(sim_df.columns):
            x = pd.to_numeric(sim_df["xhi"], errors="coerce") - pd.to_numeric(sim_df["xlo"], errors="coerce")
            y = pd.to_numeric(sim_df["yhi"], errors="coerce") - pd.to_numeric(sim_df["ylo"], errors="coerce")
            z = pd.to_numeric(sim_df["zhi"], errors="coerce") - pd.to_numeric(sim_df["zlo"], errors="coerce")
            cell_lengths = np.column_stack([x, y, z]).astype(float, copy=False)
        cell_angles = np.full((n_frames, 3), 90.0, dtype=float) if cell_lengths is not None else None
        num_of_atoms = (
            sim_df["num_of_atoms"].to_numpy(dtype=int)
            if "num_of_atoms" in sim_df.columns
            else None
        )

        simulation = SimulationData(
            atom_ids=atom_ids,
            iterations=iterations,
            elements=elements,
            num_of_atoms=num_of_atoms,
            cell_lengths=cell_lengths,
            cell_angles=cell_angles,
        )

        return TrajectoryData(
            positions=positions,
            elements=elements,
            atom_ids=atom_ids,
            simulation=simulation,
            iterations=iterations,
            atom_labels=atom_labels,
        )

    def load_simulation(self, args: dict, reporter=None) -> SimulationData:
        _ = reporter
        log_path = self._resolve_log_path(args)
        payload = LAMMPSLogHandler(log_path).read()
        thermo = payload.get("thermo")
        if not isinstance(thermo, dict) or not thermo:
            raise RuntimeError(f"No thermodynamic table found in LAMMPS log file: {log_path}")

        iterations = self._column(thermo, "Step")
        if iterations is None:
            first = next(iter(thermo.values()))
            n_rows = int(np.asarray(first).size)
            iterations = np.arange(n_rows, dtype=int)
        else:
            iterations = np.asarray(iterations, dtype=int).ravel()

        n_atoms = 0
        dump_candidate = self._resolve_dump_path(args)
        if dump_candidate.exists():
            try:
                n_atoms = int(LAMMPSDumpHandler(dump_candidate).metadata().get("n_atoms") or 0)
            except Exception:
                n_atoms = 0
        atoms_col = self._column(thermo, "Atoms", "NumAtoms", "NAtoms")
        if n_atoms <= 0 and atoms_col is not None and np.asarray(atoms_col).size > 0:
            n_atoms = int(np.asarray(atoms_col).ravel()[0])

        atom_ids = list(range(1, n_atoms + 1))

        return SimulationData(
            atom_ids=atom_ids,
            iterations=iterations,
            time=self._column(thermo, "Time", "time"),
            potential_energy=self._column(thermo, "PotEng", "E_pair", "TotEng"),
            volume=self._column(thermo, "Volume"),
            temperature=self._column(thermo, "Temp", "Temperature"),
            pressure=self._column(thermo, "Press", "Pressure"),
            density=self._column(thermo, "Density"),
            elapsed_time=self._column(thermo, "Elapsed", "CPU", "ElapsedTime"),
        )

    def load_connectivity(self, args: dict, reporter=None) -> ConnectivityData:
        _ = (args, reporter)
        return ConnectivityData(connectivity=np.empty((0, 0), dtype=int))

    def load_connectivity_trajectory(self, args: dict, reporter=None) -> ConnectivityTrajectoryData:
        return ConnectivityTrajectoryData(
            connectivity=self.load_connectivity(args, reporter=reporter),
            trajectory=self.load_trajectory(args, reporter=reporter),
        )

    def write_trajectory(self, data: TrajectoryData, out_path: str | Path, args: dict | None = None):
        args = args or {}
        return write_xyz_trajectory(data, out_path, precision=int(args.get("precision", 6)))
