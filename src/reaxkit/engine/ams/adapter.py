"""Adapt AMS KF/RKF content into ReaxKit canonical data models.

This adapter implements AMS-specific extraction from ``History`` and molecule
sections and maps those arrays/tables into engine-agnostic domain models.
It focuses on data loading/writing concerns and delegates analysis behavior
to downstream tasks and workflows.

**Usage context**

- Engine resolution: Registered under ``"ams"`` for adapter auto-detection.
- Data ingestion: Produces trajectory/connectivity/simulation and related models.
- Export workflow: Writes canonical trajectory data to XYZ output.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import re

import numpy as np
import pandas as pd

from reaxkit.core.platform.engine_resolver import register_engine
from reaxkit.domain.data_models import (
    AtomicKinematicsData,
    AtomStrainEnergyData,
    AtomTemperatureData,
    ChargeData,
    ConnectivityData,
    ConnectivityTrajectoryData,
    MolecularAnalysisData,
    PartialEnergyData,
    SimulationData,
    StressData,
    TrajectoryData,
)
from reaxkit.engine.base import EngineAdapter
from reaxkit.engine.ams.rkf_handler import RKFHandler
from reaxkit.engine.common.generators.xyz_generator import write_xyz_trajectory


BOHR_TO_ANG = 0.529177210903


@register_engine("ams")
class AMSAdapter(EngineAdapter):
    """Adapter scaffold for AMS KF/RKF-based loading."""

    HANDLER_VERSION = "2"

    _AMS_ENERGY_COMPONENTS: tuple[str, ...] = (
        "E_pot",  # 1: Total potential
        "E_over_under_coordination",  # 2: Over-/Undercoordination
        "Ebond",  # 3: Bond
        "Econj",  # 4: Conjugation
        "E_angle_conj",  # 5: Angle conjugation
        "Efield",  # 6: Electric field
        "Ehbo",  # 7: Hydrogen bond
        "Elp",  # 8: Lone pair
        "Ecoul",  # 9: Coulomb
        "E_penalty_double_bond",  # 10: Penalty dbl bond
        "Etors",  # 11: Torsion angle
        "Eval",  # 12: Valence angle
        "Evdw",  # 13: VdWaals
        "Echarge",  # 14: Charge
        "E_kin",  # 15: Kinetic
        "E_tot",  # 16: Total
        "E_thermostat",  # 17: Thermostat
    )

    def detect(self, path: str | Path) -> float:
        """Score whether a path appears to be an AMS run input/output location.

        Parameters
        ----------
        path : str | Path
            Candidate file or directory path.

        Returns
        -------
        float
            Detection score in ``[0.0, 1.0]`` where higher means more likely AMS.

        Examples
        --------
        ```python
        score = AMSAdapter().detect("reaxout.kf")
        ```
        """
        p = Path(path)
        if p.is_file() and p.suffix.lower() in {".rkf", ".kf"}:
            # score is 0.90 not 0.95 to be able to handle AMS Standalone
            # runs where both xmolout, fort.7, etc. files are generated
            # along with .kf files
            return 0.90
        if p.is_dir() and (any(p.glob("*.rkf")) or any(p.glob("*.kf"))):
            return 0.9
        return 0.0

    @staticmethod
    def _emit_load_timing(
        args: dict,
        *,
        handler: str,
        source_path: Path | str | None,
        seconds: float,
    ) -> None:
        """Dispatch load timing metrics to an optional callback."""
        cb = args.get("_load_timing_callback")
        if not callable(cb):
            return
        source = None
        source_full = None
        if source_path is not None:
            p = Path(source_path)
            source = p.name
            source_full = str(p)
        cb(handler=str(handler), source=source, source_path=source_full, seconds=float(seconds))

    @staticmethod
    def _resolve_kf_path(args: dict) -> Path:
        """Resolve the effective AMS KF/RKF path from adapter arguments."""
        def _as_candidate(raw) -> Path | None:
            if not raw:
                return None
            p = Path(raw)
            if p.is_dir():
                for name in ("reaxout.kf", "reaxout.rkf"):
                    candidate = p / name
                    if candidate.exists():
                        return candidate
                globbed = sorted([*p.glob("*.kf"), *p.glob("*.rkf")])
                if globbed:
                    return globbed[0]
                return p / "reaxout.kf"
            return p

        for key in ("rkf", "kf", "ams_kf", "ams_rkf"):
            candidate = _as_candidate(args.get(key))
            if candidate is not None:
                return candidate

        input_candidate = _as_candidate(args.get("input"))
        if input_candidate is not None and str(args.get("input")) != ".":
            return input_candidate

        run_dir_candidate = _as_candidate(args.get("run_dir"))
        if run_dir_candidate is not None:
            return run_dir_candidate

        input_candidate = _as_candidate(args.get("input"))
        if input_candidate is not None:
            return input_candidate

        return Path("reaxout.kf")

    def _build_rkf_handler(self, args: dict) -> RKFHandler:
        """Build an ``RKFHandler`` configured for the resolved KF/RKF path."""
        kf_path = self._resolve_kf_path(args)
        return RKFHandler(
            kf_path,
            timing_callback=lambda source_path, seconds: self._emit_load_timing(
                args,
                handler="KFFile",
                source_path=source_path,
                seconds=seconds,
            ),
        )

    @classmethod
    def clear_runtime_cache(cls) -> None:
        """Clear AMS runtime caches used by RKF handler instances.

        Parameters
        ----------
        None

        Returns
        -------
        None

        Examples
        --------
        ```python
        AMSAdapter.clear_runtime_cache()
        ```
        """
        RKFHandler.clear_runtime_cache()

    def required_input_files(self, data_type, args: dict) -> tuple[str, ...] | None:
        """Return expected AMS file names for supported domain data types.

        Parameters
        ----------
        data_type : type
            Requested canonical data-model class.
        args : dict
            Adapter runtime arguments (unused for this decision).

        Returns
        -------
        tuple[str, ...] | None
            Candidate AMS file names, or ``None`` when not constrained.

        Examples
        --------
        ```python
        files = adapter.required_input_files(TrajectoryData, {})
        ```
        """
        if data_type in {
            AtomicKinematicsData,
            AtomStrainEnergyData,
            AtomTemperatureData,
            TrajectoryData,
            ConnectivityData,
            ConnectivityTrajectoryData,
            SimulationData,
            PartialEnergyData,
            ChargeData,
            StressData,
        }:
            explicit_input = args.get("input") if isinstance(args, dict) else None
            if explicit_input and str(explicit_input) != ".":
                input_path = Path(str(explicit_input))
                if input_path.suffix.lower() in {".kf", ".rkf"}:
                    return (input_path.name,)
            return ("reaxout.kf", "reaxout.rkf")
        return None

    def load_kf(self, args: dict):
        """Return a cached AMS ``KFFile`` handle for downstream loaders.

        Parameters
        ----------
        args : dict
            Adapter arguments containing optional KF/RKF path selectors.

        Returns
        -------
        Any
            AMS ``KFFile`` object from ``RKFHandler``.

        Examples
        --------
        ```python
        kf = adapter.load_kf({"rkf": "reaxout.rkf"})
        ```
        """
        return self._build_rkf_handler(args).kf()

    @staticmethod
    def _history_entries(history_data: dict, prefix: str, *, dtype=float) -> list[np.ndarray]:
        """Collect and optionally index-sort ``History`` arrays by key prefix."""
        rows: list[tuple[int | None, np.ndarray]] = []
        for key, values in history_data.items():
            if not str(key).startswith(prefix):
                continue
            idx = None
            match = re.search(r"(?:\((\d+)\)|\s+(\d+))\s*$", str(key))
            if match:
                idx = int(match.group(1) or match.group(2))
            rows.append((idx, np.asarray(values, dtype=dtype).ravel()))
        if rows and all(idx is not None for idx, _ in rows):
            rows.sort(key=lambda x: int(x[0]))
        return [arr for _, arr in rows]

    @staticmethod
    def _coordinate_entries(history_data: dict, *, dtype=float) -> list[np.ndarray]:
        """Collect AMS coordinate frames, preferring RKF ``Coords(n)`` keys."""
        coords_entries = AMSAdapter._history_entries(history_data, "Coords", dtype=dtype)
        if coords_entries:
            return coords_entries
        return AMSAdapter._history_entries(history_data, "Coordinates", dtype=dtype)

    @staticmethod
    def _read_kf_variable(kf, section: str, variable: str):
        """Read one KF variable using both PLAMS access styles."""
        try:
            return kf.read(section, variable)
        except Exception:
            pass
        try:
            return kf[f"{section}%{variable}"]
        except Exception:
            return None

    @staticmethod
    def _read_history_frame_variable(kf, prefix: str, frame_no: int):
        """Read a frame-indexed History variable without loading the whole section."""
        for variable in (f"{prefix}({frame_no})", f"{prefix} {frame_no}"):
            raw = AMSAdapter._read_kf_variable(kf, "History", variable)
            if raw is not None:
                return raw
        return None

    @staticmethod
    def _history_frame_count(kf) -> int:
        """Return RKF trajectory frame count using TRACT's MDHistory fallback."""
        for section, variable in (("MDHistory", "nEntries"), ("History", "nEntries")):
            raw = AMSAdapter._read_kf_variable(kf, section, variable)
            if raw is not None:
                try:
                    return int(np.asarray(raw).ravel()[0])
                except Exception:
                    pass

        n = 0
        while True:
            probe = AMSAdapter._read_kf_variable(kf, "History", f"Coords({n + 1})")
            if probe is None:
                probe = AMSAdapter._read_kf_variable(kf, "History", f"Coordinates({n + 1})")
            if probe is None:
                break
            n += 1
        return n

    @staticmethod
    def _direct_coordinate_entries(kf, *, dtype=float, reporter=None) -> list[np.ndarray]:
        """Read RKF coordinate frames directly as ``History/Coords(n)``."""
        n_frames = AMSAdapter._history_frame_count(kf)
        entries: list[np.ndarray] = []
        for frame_no in range(1, n_frames + 1):
            raw = AMSAdapter._read_kf_variable(kf, "History", f"Coords({frame_no})")
            if raw is None:
                raw = AMSAdapter._read_kf_variable(kf, "History", f"Coordinates({frame_no})")
            if raw is None:
                continue
            entries.append(np.asarray(raw, dtype=dtype).ravel())
            if callable(reporter):
                reporter("load", frame_no, n_frames, "Reading AMS RKF coordinates")
        return entries

    @staticmethod
    def _direct_connectivity_entries(kf, *, reporter=None) -> tuple[dict[str, list[np.ndarray]], int]:
        """Read AMS connectivity series frame-by-frame from ``History`` keys."""
        n_frames = AMSAdapter._history_frame_count(kf)
        series: dict[str, list[np.ndarray]] = {
            "neighbors": [],
            "bond_orders": [],
            "num_neighb": [],
            "total_bo": [],
            "lone_pairs": [],
        }

        first_bond_index = AMSAdapter._read_history_frame_variable(kf, "Bonds.Index", 1)
        first_bond_atoms = AMSAdapter._read_history_frame_variable(kf, "Bonds.Atoms", 1)
        first_bond_orders = AMSAdapter._read_history_frame_variable(kf, "Bonds.Orders", 1)
        if first_bond_index is not None and first_bond_atoms is not None and first_bond_orders is not None:
            for frame_no in range(1, n_frames + 1):
                if frame_no == 1:
                    bi_raw = first_bond_index
                    ba_raw = first_bond_atoms
                    bo_raw = first_bond_orders
                else:
                    bi_raw = AMSAdapter._read_history_frame_variable(kf, "Bonds.Index", frame_no)
                    ba_raw = AMSAdapter._read_history_frame_variable(kf, "Bonds.Atoms", frame_no)
                    bo_raw = AMSAdapter._read_history_frame_variable(kf, "Bonds.Orders", frame_no)
                if bi_raw is None or ba_raw is None or bo_raw is None:
                    series["neighbors"].append(np.asarray([], dtype=int))
                    series["bond_orders"].append(np.asarray([], dtype=float))
                    series["num_neighb"].append(np.asarray([], dtype=int))
                    series["total_bo"].append(np.asarray([], dtype=float))
                    series["lone_pairs"].append(np.asarray([], dtype=float))
                    if callable(reporter):
                        reporter("load", frame_no, n_frames, "Reading AMS RKF connectivity")
                    continue

                bi = np.asarray(bi_raw, dtype=int).ravel()
                ba = np.asarray(ba_raw, dtype=int).ravel()
                bo = np.asarray(bo_raw, dtype=float).ravel()
                if bi.size >= 2:
                    starts = np.maximum(bi[:-1] - 1, 0)
                    ends = np.maximum(bi[1:] - 1, starts)
                    counts = np.maximum(ends - starts, 0).astype(int, copy=False)
                    total_bo = np.zeros((counts.size,), dtype=float)
                    for ai, (start, end) in enumerate(zip(starts, ends)):
                        start_i = min(int(start), bo.size)
                        end_i = min(int(end), bo.size)
                        if end_i > start_i:
                            total_bo[ai] = float(np.nansum(bo[start_i:end_i]))
                else:
                    counts = np.asarray([], dtype=int)
                    total_bo = np.asarray([], dtype=float)

                series["neighbors"].append(ba)
                series["bond_orders"].append(bo)
                series["num_neighb"].append(counts)
                series["total_bo"].append(total_bo)
                series["lone_pairs"].append(np.full((counts.size,), np.nan, dtype=float))
                if callable(reporter):
                    reporter("load", frame_no, n_frames, "Reading AMS RKF connectivity")
            return series, n_frames

        specs = (
            ("neighbors", "ConnTab neighbors", int),
            ("bond_orders", "ConnTab bond order", float),
            ("num_neighb", "ConnTab num neighb", int),
            ("total_bo", "Total bond orders", float),
            ("lone_pairs", "Lone pairs", float),
        )
        found_any = False
        for frame_no in range(1, n_frames + 1):
            frame_found = False
            for key, prefix, dtype in specs:
                raw = AMSAdapter._read_history_frame_variable(kf, prefix, frame_no)
                if raw is None:
                    series[key].append(np.asarray([], dtype=dtype))
                    continue
                arr = np.asarray(raw, dtype=dtype).ravel()
                series[key].append(arr)
                frame_found = True
                found_any = True
            if frame_found and callable(reporter):
                reporter("load", frame_no, n_frames, "Reading AMS RKF connectivity")

            # If the RKF lacks direct ConnTab frame keys, avoid probing every
            # frame before falling back to the section-based legacy reader.
            if frame_no == 1 and not frame_found:
                return {key: [] for key in series}, 0

        return (series if found_any else {key: [] for key in series}, n_frames if found_any else 0)

    @classmethod
    def _minimal_simulation_context(
        cls,
        kf,
        *,
        n_frames: int,
        n_atoms: int,
        elements: list[str],
        iterations: np.ndarray,
    ) -> SimulationData | None:
        """Build lightweight simulation context without loading full History."""
        fixed_axes = cls._molecule_lattice_vectors(kf)
        if fixed_axes is None:
            return None
        cell_lengths = np.tile(np.linalg.norm(fixed_axes, axis=1), (n_frames, 1))
        cell_angles = np.tile(cls._cell_angles_from_axes(fixed_axes), (n_frames, 1))
        return SimulationData(
            atom_ids=list(range(1, n_atoms + 1)),
            iterations=iterations,
            elements=elements,
            cell_lengths=cell_lengths,
            cell_angles=cell_angles,
        )

    @staticmethod
    def _molecule_lattice_vectors(kf) -> np.ndarray | None:
        """Return fixed molecule lattice vectors in angstrom when present."""
        if kf is None:
            return None
        raw = None
        try:
            raw = kf.read("Molecule", "LatticeVectors")
        except Exception:
            raw = None
        if raw is None:
            try:
                raw = kf["Molecule%LatticeVectors"]
            except Exception:
                raw = None
        if raw is None:
            return None
        arr = np.asarray(raw, dtype=float).ravel()
        if arr.size < 9:
            return None
        axes = arr[:9].reshape(3, 3) * BOHR_TO_ANG
        return axes if np.isfinite(axes).all() else None

    @staticmethod
    def _cell_angles_from_axes(axes: np.ndarray) -> np.ndarray:
        """Return alpha, beta, gamma angles in degrees from row-major axes."""
        a, b, c = [np.asarray(v, dtype=float) for v in np.asarray(axes, dtype=float).reshape(3, 3)]

        def _angle(u: np.ndarray, v: np.ndarray) -> float:
            denom = float(np.linalg.norm(u) * np.linalg.norm(v))
            if denom <= 0.0:
                return np.nan
            cosang = float(np.dot(u, v) / denom)
            return float(np.degrees(np.arccos(np.clip(cosang, -1.0, 1.0))))

        return np.asarray([_angle(b, c), _angle(a, c), _angle(a, b)], dtype=float)

    @staticmethod
    def _parse_atom_names(raw_atom_names) -> list[str]:
        """Parse element symbols from AMS atom-name payload formats."""
        names_arr = np.asarray(raw_atom_names, dtype=object).ravel()
        if names_arr.size == 0:
            return []
        if names_arr.size == 1:
            text = str(names_arr[0])
        else:
            text = " ".join(str(v) for v in names_arr.tolist())
        # Accept both whitespace-separated and concatenated symbol streams (e.g. "ZnZnMg...").
        symbols = re.findall(r"[A-Z][a-z]?", text)
        return [str(s) for s in symbols]

    @staticmethod
    def _extract_atom_names_raw(kf, history_data: dict):
        """Fetch raw atom-name data from History keys or direct KF lookup."""
        raw_atom_names = history_data.get("Atom names")
        if raw_atom_names is None:
            for key, value in history_data.items():
                key_s = str(key).strip()
                if key_s == "Atom names" or key_s.lower().startswith("atom names"):
                    raw_atom_names = value
                    break
        if raw_atom_names is None and kf is not None:
            try:
                raw_atom_names = kf["History%Atom names"]
            except Exception:
                raw_atom_names = None
        if raw_atom_names is None and kf is not None:
            try:
                raw_atom_names = kf.read("Molecule", "AtomSymbols")
            except Exception:
                raw_atom_names = None
        if raw_atom_names is None and kf is not None:
            try:
                raw_atom_names = kf["Molecule%AtomSymbols"]
            except Exception:
                raw_atom_names = None
        return raw_atom_names

    @staticmethod
    def _elements_from_history(kf, history_data: dict, n_atoms: int) -> list[str]:
        """Return per-atom element symbols padded/truncated to ``n_atoms``."""
        raw_atom_names = AMSAdapter._extract_atom_names_raw(kf, history_data)
        if raw_atom_names is None:
            return ["X"] * n_atoms
        parsed = AMSAdapter._parse_atom_names(raw_atom_names)
        elements = parsed[:n_atoms]
        if len(elements) < n_atoms:
            elements.extend(["X"] * (n_atoms - len(elements)))
        return elements

    @staticmethod
    def _atom_name_count(kf, history_data: dict) -> int:
        """Return parsed atom count from AMS atom-name metadata."""
        raw_atom_names = AMSAdapter._extract_atom_names_raw(kf, history_data)
        if raw_atom_names is None:
            return 0
        return len(AMSAdapter._parse_atom_names(raw_atom_names))

    @staticmethod
    def _iterations_from_general(kf, n_frames: int) -> np.ndarray:
        """Derive iteration numbers from ``General%Step numbers`` with fallback."""
        try:
            step_numbers = np.asarray(kf["General%Step numbers"], dtype=int).ravel()
        except Exception:
            step_numbers = np.empty((0,), dtype=int)
        if n_frames <= 0:
            if step_numbers.size > 0:
                return step_numbers
            return np.empty((0,), dtype=int)
        if step_numbers.size >= n_frames:
            return step_numbers[:n_frames]
        if step_numbers.size > 0:
            out = np.arange(n_frames, dtype=int)
            out[: step_numbers.size] = step_numbers
            return out
        return np.arange(n_frames, dtype=int)

    @staticmethod
    def _flat_xyz_to_df(
        flat: np.ndarray,
        *,
        value_cols: tuple[str, str, str],
        n_atoms: int | None = None,
    ) -> pd.DataFrame:
        """Convert flattened XYZ triples into an ``atom_id``-indexed DataFrame."""
        arr = np.asarray(flat, dtype=float).ravel()
        if arr.size == 0:
            return pd.DataFrame(columns=["atom_id", *value_cols])
        if arr.size % 3 != 0:
            n_triplets = arr.size // 3
            arr = arr[: n_triplets * 3]
        xyz = arr.reshape(-1, 3)
        if n_atoms is not None:
            if xyz.shape[0] < n_atoms:
                pad = np.full((n_atoms - xyz.shape[0], 3), np.nan, dtype=float)
                xyz = np.vstack([xyz, pad])
            elif xyz.shape[0] > n_atoms:
                xyz = xyz[:n_atoms, :]
        df = pd.DataFrame(xyz, columns=list(value_cols))
        df.insert(0, "atom_id", np.arange(1, len(df) + 1, dtype=int))
        return df

    @staticmethod
    def _to_1d_array(values, *, dtype=float) -> np.ndarray:
        """Normalize values to a flattened NumPy array with target dtype."""
        return np.asarray(values, dtype=dtype).ravel()

    @staticmethod
    def _scalar_or_first(values, default="") -> str:
        """Return scalar text value or first element from an array-like input."""
        arr = np.asarray(values, dtype=object).ravel()
        if arr.size == 0:
            return str(default)
        return str(arr[0])

    @staticmethod
    def _formula_atom_count(formula: str) -> int:
        """Count atoms represented by a chemical formula string."""
        pairs = re.findall(r"([A-Z][a-z]*)(\d*)", str(formula))
        if not pairs:
            return 0
        total = 0
        for _, count_s in pairs:
            total += int(count_s) if count_s else 1
        return total

    @staticmethod
    def _formula_mass(formula: str) -> float:
        """Compute formula mass using ASE atomic masses when available."""
        pairs = re.findall(r"([A-Z][a-z]*)(\d*)", str(formula))
        if not pairs:
            return np.nan
        try:
            from ase.data import atomic_masses, atomic_numbers  # type: ignore
        except Exception:
            return np.nan
        mass = 0.0
        for symbol, count_s in pairs:
            z = atomic_numbers.get(symbol)
            if z is None:
                return np.nan
            count = int(count_s) if count_s else 1
            mass += float(atomic_masses[int(z)]) * count
        return mass

    @staticmethod
    def _read_first_section(kf, names: tuple[str, ...]) -> dict:
        """Return first non-empty KF section from a list of candidate names."""
        for name in names:
            try:
                data = kf.read_section(name)
            except Exception:
                continue
            if isinstance(data, dict) and data:
                return data
        return {}

    @staticmethod
    def _read_molecule_sections(kf) -> list[dict]:
        """Read base and numbered molecule sections from AMS KF content."""
        sections: list[dict] = []
        for base_name in ("Molecule", "Molecules"):
            try:
                data = kf.read_section(base_name)
            except Exception:
                data = {}
            if isinstance(data, dict) and data:
                sections.append(data)

        misses_in_a_row = 0
        for i in range(1, 256):
            sec_name = f"Molecules #{i}"
            try:
                data = kf.read_section(sec_name)
            except Exception:
                data = {}
            if isinstance(data, dict) and data:
                sections.append(data)
                misses_in_a_row = 0
            else:
                misses_in_a_row += 1
                if misses_in_a_row >= 3:
                    break
        return sections

    @staticmethod
    def _stack_1d_per_frame(
        entries: list[np.ndarray],
        *,
        n_frames: int,
        n_atoms: int,
        fill_value: float,
    ) -> np.ndarray:
        """Stack per-frame 1D arrays into a fixed ``(n_frames, n_atoms)`` matrix."""
        out = np.full((n_frames, n_atoms), fill_value, dtype=float)
        for fi in range(min(n_frames, len(entries))):
            arr = np.asarray(entries[fi], dtype=float).ravel()
            n = min(n_atoms, arr.size)
            if n > 0:
                out[fi, :n] = arr[:n]
        return out

    @staticmethod
    def _stack_6c_per_frame(
        entries: list[np.ndarray],
        *,
        n_frames: int,
        n_atoms: int,
    ) -> np.ndarray:
        """Stack flattened six-component per-atom tensors across frames."""
        out = np.full((n_frames, n_atoms, 6), np.nan, dtype=float)
        for fi in range(min(n_frames, len(entries))):
            flat = np.asarray(entries[fi], dtype=float).ravel()
            usable = (flat.size // 6) * 6
            if usable <= 0:
                continue
            frame = flat[:usable].reshape(-1, 6)
            n = min(n_atoms, frame.shape[0])
            if n > 0:
                out[fi, :n, :] = frame[:n, :]
        return out

    @staticmethod
    def _reshape_conntab_entries(
        entries: list[np.ndarray],
        *,
        n_frames: int,
        n_atoms: int,
        dtype,
        fill_value: float | int,
    ) -> list[np.ndarray]:
        """Reshape flattened connectivity entries into per-frame 2D arrays."""
        frames: list[np.ndarray] = []
        for fi in range(n_frames):
            arr = np.asarray(entries[fi], dtype=dtype).ravel() if fi < len(entries) else np.asarray([], dtype=dtype)
            if n_atoms <= 0:
                frames.append(np.empty((0, 0), dtype=dtype))
                continue
            if arr.size == 0:
                frames.append(np.empty((n_atoms, 0), dtype=dtype))
                continue
            if arr.size % n_atoms == 0:
                frames.append(arr.reshape(n_atoms, -1))
                continue
            padded = np.full((n_atoms,), fill_value, dtype=dtype)
            n = min(n_atoms, arr.size)
            padded[:n] = arr[:n]
            frames.append(padded.reshape(n_atoms, 1))
        return frames

    @staticmethod
    def _reshape_conntab_by_num_neighb(
        *,
        flat_entries: list[np.ndarray],
        num_neighb_entries: list[np.ndarray],
        n_frames: int,
        n_atoms: int,
        dtype,
        fill_value: float | int,
    ) -> list[np.ndarray]:
        """Reshape connectivity arrays using per-atom neighbor-count vectors."""
        frames: list[np.ndarray] = []
        for fi in range(n_frames):
            nnei_raw = np.asarray(num_neighb_entries[fi], dtype=int).ravel() if fi < len(num_neighb_entries) else np.zeros((0,), dtype=int)
            if nnei_raw.size < n_atoms:
                nnei = np.zeros((n_atoms,), dtype=int)
                nnei[: nnei_raw.size] = np.maximum(nnei_raw, 0)
            else:
                nnei = np.maximum(nnei_raw[:n_atoms], 0)

            max_nei = int(np.max(nnei)) if nnei.size > 0 else 0
            frame = np.full((n_atoms, max_nei), fill_value, dtype=dtype)
            flat = np.asarray(flat_entries[fi], dtype=dtype).ravel() if fi < len(flat_entries) else np.asarray([], dtype=dtype)

            cursor = 0
            for ai in range(n_atoms):
                k = int(nnei[ai])
                if k <= 0 or max_nei <= 0:
                    continue
                end = min(cursor + k, flat.size)
                take = end - cursor
                if take > 0:
                    frame[ai, :take] = flat[cursor:end]
                cursor += k
            frames.append(frame)
        return frames

    def load_trajectory(self, args: dict, reporter=None) -> TrajectoryData:
        """Load canonical trajectory data from AMS ``History%Coordinates``.

        Parameters
        ----------
        args : dict
            Adapter options including optional KF/RKF path hints.
        reporter : callable | None, optional
            Progress callback ``(phase, done, total, message)``.

        Returns
        -------
        TrajectoryData
            Frame-major positions, inferred elements, atom IDs, and iterations.

        Examples
        --------
        ```python
        traj = AMSAdapter().load_trajectory({"rkf": "reaxout.rkf"})
        ```
        """
        kf = self.load_kf(args)
        direct_coord_entries = self._direct_coordinate_entries(kf, dtype=float, reporter=reporter)
        if direct_coord_entries:
            history_data = {}
            coord_entries = direct_coord_entries
        else:
            history_data = kf.read_section("History")
            coord_entries = self._coordinate_entries(history_data, dtype=float)
        frames: list[np.ndarray] = []
        for flat in coord_entries:
            if flat.size % 3 != 0:
                continue
            coords = flat.reshape(-1, 3) * BOHR_TO_ANG
            frames.append(coords)

        if not frames:
            raise RuntimeError("No 'Coords*' or 'Coordinates*' entries were found in AMS History section.")

        n_frames = len(frames)
        n_atoms = max(coords.shape[0] for coords in frames)
        positions = np.full((n_frames, n_atoms, 3), np.nan, dtype=float)
        for fi, coords in enumerate(frames):
            n = coords.shape[0]
            positions[fi, :n, :] = coords
            if callable(reporter) and not direct_coord_entries:
                reporter("load", fi + 1, n_frames, "Reading AMS coordinates from History")

        elements = self._elements_from_history(kf, history_data, n_atoms)
        iterations = self._iterations_from_general(kf, n_frames)
        atom_ids = list(range(1, n_atoms + 1))
        atom_labels = np.tile(np.asarray(elements, dtype=object), (n_frames, 1))
        simulation = self._minimal_simulation_context(
            kf,
            n_frames=n_frames,
            n_atoms=n_atoms,
            elements=elements,
            iterations=iterations,
        )

        return TrajectoryData(
            positions=positions,
            elements=elements,
            atom_ids=atom_ids,
            atom_labels=atom_labels,
            iterations=iterations,
            simulation=simulation,
        )

    def load_connectivity(self, args: dict, reporter=None) -> ConnectivityData:
        """Load connectivity and bond-order arrays from AMS history records.

        Parameters
        ----------
        args : dict
            Adapter options including optional KF/RKF path hints.
        reporter : callable | None, optional
            Progress callback ``(phase, done, total, message)``.

        Returns
        -------
        ConnectivityData
            Canonical connectivity payload with per-frame neighbor and bond data.

        Examples
        --------
        ```python
        conn = AMSAdapter().load_connectivity({"kf": "reaxout.kf"})
        ```
        """
        kf = self.load_kf(args)
        direct_entries, direct_n_frames = self._direct_connectivity_entries(kf, reporter=reporter)
        if direct_n_frames > 0:
            history_data = {}
            neighbors_entries = direct_entries["neighbors"]
            bond_order_entries = direct_entries["bond_orders"]
            num_neighb_entries = direct_entries["num_neighb"]
            total_bo_entries = direct_entries["total_bo"]
            lone_pairs_entries = direct_entries["lone_pairs"]
        else:
            if callable(reporter):
                reporter("load", 0, 0, "Reading AMS connectivity History section")
            history_data = kf.read_section("History")

            neighbors_entries = self._history_entries(history_data, "ConnTab neighbors", dtype=int)
            bond_order_entries = self._history_entries(history_data, "ConnTab bond order", dtype=float)
            num_neighb_entries = self._history_entries(history_data, "ConnTab num neighb", dtype=int)
            total_bo_entries = self._history_entries(history_data, "Total bond orders", dtype=float)
            lone_pairs_entries = self._history_entries(history_data, "Lone pairs", dtype=float)

        n_frames = max(
            len(neighbors_entries),
            len(bond_order_entries),
            len(num_neighb_entries),
            len(total_bo_entries),
            len(lone_pairs_entries),
        )
        if n_frames == 0:
            raise RuntimeError("No connectivity entries were found in AMS History section.")

        if num_neighb_entries:
            n_atoms = max(arr.size for arr in num_neighb_entries)
        elif total_bo_entries:
            n_atoms = max(arr.size for arr in total_bo_entries)
        elif lone_pairs_entries:
            n_atoms = max(arr.size for arr in lone_pairs_entries)
        else:
            coords_entries = self._coordinate_entries(history_data, dtype=float)
            n_atoms = 0
            for arr in coords_entries:
                if arr.size % 3 == 0:
                    n_atoms = max(n_atoms, int(arr.size // 3))
            if n_atoms == 0:
                n_atoms = self._atom_name_count(kf, history_data)

        elements = self._elements_from_history(kf, history_data, n_atoms)
        atom_ids = list(range(1, n_atoms + 1))
        iterations = self._iterations_from_general(kf, n_frames)

        if num_neighb_entries:
            connectivity = self._reshape_conntab_by_num_neighb(
                flat_entries=neighbors_entries,
                num_neighb_entries=num_neighb_entries,
                n_frames=n_frames,
                n_atoms=n_atoms,
                dtype=int,
                fill_value=0,
            )
            bond_orders = self._reshape_conntab_by_num_neighb(
                flat_entries=bond_order_entries,
                num_neighb_entries=num_neighb_entries,
                n_frames=n_frames,
                n_atoms=n_atoms,
                dtype=float,
                fill_value=0.0,
            )
        else:
            connectivity = self._reshape_conntab_entries(
                neighbors_entries,
                n_frames=n_frames,
                n_atoms=n_atoms,
                dtype=int,
                fill_value=0,
            )
            bond_orders = self._reshape_conntab_entries(
                bond_order_entries,
                n_frames=n_frames,
                n_atoms=n_atoms,
                dtype=float,
                fill_value=0.0,
            )
        sum_bond_orders = self._stack_1d_per_frame(total_bo_entries, n_frames=n_frames, n_atoms=n_atoms, fill_value=np.nan)
        num_lone_pairs = self._stack_1d_per_frame(lone_pairs_entries, n_frames=n_frames, n_atoms=n_atoms, fill_value=np.nan)

        if callable(reporter) and direct_n_frames <= 0:
            for frame_no in range(1, n_frames + 1):
                reporter("load", frame_no, n_frames, "Reading AMS connectivity from History")

        return ConnectivityData(
            connectivity=connectivity,
            bond_orders=bond_orders,
            sum_bond_orders=sum_bond_orders,
            num_lone_pairs=num_lone_pairs,
            atom_ids=atom_ids,
            elements=elements,
            iterations=iterations,
        )

    def load_simulation(self, args: dict, reporter=None) -> SimulationData:
        """Load simulation-level scalar series from AMS History sections.

        Parameters
        ----------
        args : dict
            Adapter options including optional KF/RKF path hints.
        reporter : callable | None, optional
            Progress callback ``(phase, done, total, message)``.

        Returns
        -------
        SimulationData
            Iterations, energies, thermodynamic series, and cell geometry.

        Examples
        --------
        ```python
        sim = AMSAdapter().load_simulation({"rkf": "reaxout.rkf"})
        ```
        """
        kf = self.load_kf(args)
        history_data = kf.read_section("History")

        energies_entries = self._history_entries(history_data, "Energies", dtype=float)
        scalars_entries = self._history_entries(history_data, "Scalars", dtype=float)
        mol_index_entries = self._history_entries(history_data, "ConnTab mol index", dtype=float)
        cell_axes_entries = self._history_entries(history_data, "Unit cell axes", dtype=float)
        cell_angle_entries = self._history_entries(history_data, "Unit cell angles", dtype=float)
        coords_entries = self._coordinate_entries(history_data, dtype=float)
        direct_n_frames = self._history_frame_count(kf)

        n_frames = max(
            len(energies_entries),
            len(scalars_entries),
            len(mol_index_entries),
            len(cell_axes_entries),
            len(cell_angle_entries),
            len(coords_entries),
            direct_n_frames,
        )
        if n_frames == 0:
            raise RuntimeError("No simulation entries were found in AMS History section.")

        n_atoms = 0
        if mol_index_entries:
            n_atoms = max(arr.size for arr in mol_index_entries)
        if n_atoms == 0:
            for arr in coords_entries:
                if arr.size % 3 == 0:
                    n_atoms = max(n_atoms, int(arr.size // 3))
        if n_atoms == 0 and direct_n_frames > 0:
            first_coords = self._read_kf_variable(kf, "History", "Coords(1)")
            if first_coords is None:
                first_coords = self._read_kf_variable(kf, "History", "Coordinates(1)")
            if first_coords is not None:
                flat = np.asarray(first_coords, dtype=float).ravel()
                if flat.size % 3 == 0:
                    n_atoms = int(flat.size // 3)
        if n_atoms == 0:
            n_atoms = self._atom_name_count(kf, history_data)

        atom_ids = list(range(1, n_atoms + 1))
        elements = self._elements_from_history(kf, history_data, n_atoms)
        iterations = self._iterations_from_general(kf, n_frames)

        potential_energy = np.full((n_frames,), np.nan, dtype=float)
        for fi in range(min(n_frames, len(energies_entries))):
            arr = np.asarray(energies_entries[fi], dtype=float).ravel()
            if arr.size > 0:
                potential_energy[fi] = arr[0]

        temperature = np.full((n_frames,), np.nan, dtype=float)
        pressure = np.full((n_frames,), np.nan, dtype=float)
        density = np.full((n_frames,), np.nan, dtype=float)
        for fi in range(min(n_frames, len(scalars_entries))):
            arr = np.asarray(scalars_entries[fi], dtype=float).ravel()
            if arr.size > 1:
                temperature[fi] = arr[1]
            if arr.size > 7:
                pressure[fi] = arr[7]
            if arr.size > 0:
                density[fi] = arr[0]

        molecule_nums = self._stack_1d_per_frame(mol_index_entries, n_frames=n_frames, n_atoms=n_atoms, fill_value=np.nan)

        cell_lengths = np.full((n_frames, 3), np.nan, dtype=float)
        cell_axes_by_frame: list[np.ndarray | None] = [None] * n_frames
        for fi in range(min(n_frames, len(cell_axes_entries))):
            arr = np.asarray(cell_axes_entries[fi], dtype=float).ravel()
            if arr.size >= 9:
                axes = arr[:9].reshape(3, 3) * BOHR_TO_ANG
                cell_axes_by_frame[fi] = axes
                cell_lengths[fi] = np.linalg.norm(axes, axis=1)
            elif arr.size >= 3:
                cell_lengths[fi] = arr[:3] * BOHR_TO_ANG
        if not np.isfinite(cell_lengths).any():
            fixed_axes = self._molecule_lattice_vectors(kf)
            if fixed_axes is not None:
                cell_lengths[:, :] = np.linalg.norm(fixed_axes, axis=1)
        else:
            fixed_axes = None

        cell_angles = np.full((n_frames, 3), np.nan, dtype=float)
        for fi in range(min(n_frames, len(cell_angle_entries))):
            arr = np.asarray(cell_angle_entries[fi], dtype=float).ravel()
            if arr.size >= 3:
                cell_angles[fi] = arr[:3]
        for fi, axes in enumerate(cell_axes_by_frame):
            if axes is not None and not np.isfinite(cell_angles[fi]).all():
                cell_angles[fi] = self._cell_angles_from_axes(axes)
        if not np.isfinite(cell_angles).any() and fixed_axes is not None:
            cell_angles[:, :] = self._cell_angles_from_axes(fixed_axes)

        if callable(reporter):
            reporter("load", n_frames, n_frames, "Reading AMS simulation series from History")

        return SimulationData(
            atom_ids=atom_ids,
            iterations=iterations,
            elements=elements,
            potential_energy=potential_energy,
            temperature=temperature,
            pressure=pressure,
            density=density,
            molecule_nums=molecule_nums,
            cell_lengths=cell_lengths,
            cell_angles=cell_angles,
        )

    def load_partial_energy(self, args: dict, reporter=None) -> PartialEnergyData:
        """Load per-frame AMS partial-energy components from History.

        Parameters
        ----------
        args : dict
            Adapter options including optional KF/RKF path hints.
        reporter : callable | None, optional
            Progress callback ``(phase, done, total, message)``.

        Returns
        -------
        PartialEnergyData
            Matrix of component energies and associated iteration axis.

        Examples
        --------
        ```python
        pe = AMSAdapter().load_partial_energy({"rkf": "reaxout.rkf"})
        ```
        """
        kf = self.load_kf(args)
        history_data = kf.read_section("History")
        energies_entries = self._history_entries(history_data, "Energies", dtype=float)
        if not energies_entries:
            raise RuntimeError("No 'Energies*' entries were found in AMS History section.")

        n_frames = len(energies_entries)
        n_components = len(self._AMS_ENERGY_COMPONENTS)
        values = np.full((n_frames, n_components), np.nan, dtype=float)
        for fi, arr in enumerate(energies_entries):
            flat = np.asarray(arr, dtype=float).ravel()
            n = min(n_components, flat.size)
            if n > 0:
                values[fi, :n] = flat[:n]

        iterations = self._iterations_from_general(kf, n_frames)
        if callable(reporter):
            reporter("load", n_frames, n_frames, "Reading AMS partial energies from History")

        return PartialEnergyData(
            iterations=iterations,
            components=self._AMS_ENERGY_COMPONENTS,
            values=values,
            metadata={
                "source": "History%Energies",
                "component_order_1_based": list(range(1, n_components + 1)),
            },
        )

    def load_charges(self, args: dict, reporter=None) -> ChargeData:
        """Load per-atom charges and framewise total charge from AMS History.

        Parameters
        ----------
        args : dict
            Adapter options including optional KF/RKF path hints.
        reporter : callable | None, optional
            Progress callback ``(phase, done, total, message)``.

        Returns
        -------
        ChargeData
            Charge matrix, summed charge series, and iteration indices.

        Examples
        --------
        ```python
        charges = AMSAdapter().load_charges({"kf": "reaxout.kf"})
        ```
        """
        kf = self.load_kf(args)
        history_data = kf.read_section("History")
        charge_entries = self._history_entries(history_data, "Atomic charges", dtype=float)
        if not charge_entries:
            raise RuntimeError("No 'Atomic charges*' entries were found in AMS History section.")

        n_frames = len(charge_entries)
        n_atoms = max(arr.size for arr in charge_entries)
        charges = np.full((n_frames, n_atoms), np.nan, dtype=float)
        for fi, arr in enumerate(charge_entries):
            flat = np.asarray(arr, dtype=float).ravel()
            n = min(n_atoms, flat.size)
            if n > 0:
                charges[fi, :n] = flat[:n]

        iterations = self._iterations_from_general(kf, n_frames)
        total_charge = np.nansum(charges, axis=1)
        if callable(reporter):
            reporter("load", n_frames, n_frames, "Reading AMS atomic charges from History")

        return ChargeData(
            charges=charges,
            total_charge=total_charge,
            iterations=iterations,
            metadata={"source": "History%Atomic charges"},
        )

    def load_atomic_kinematics(self, args: dict, reporter=None) -> AtomicKinematicsData:
        """Load latest-frame coordinate/velocity/acceleration tables from AMS.

        Parameters
        ----------
        args : dict
            Adapter options including optional KF/RKF path hints.
        reporter : callable | None, optional
            Progress callback ``(phase, done, total, message)``.

        Returns
        -------
        AtomicKinematicsData
            Snapshot data tables for coordinates, velocities, and accelerations.

        Examples
        --------
        ```python
        kin = AMSAdapter().load_atomic_kinematics({"rkf": "reaxout.rkf"})
        ```
        """
        kf = self.load_kf(args)
        history_data = kf.read_section("History")

        vel_entries = self._history_entries(history_data, "Velocities", dtype=float)
        acc_entries = self._history_entries(history_data, "Acceleration", dtype=float)
        coord_entries = self._history_entries(history_data, "Coordinates", dtype=float)

        if not vel_entries:
            raise RuntimeError("No 'Velocities*' entries were found in AMS History section.")

        vel_flat = vel_entries[-1]
        n_atoms = int(np.asarray(vel_flat, dtype=float).size // 3)

        coords_df = (
            self._flat_xyz_to_df(coord_entries[-1], value_cols=("x", "y", "z"), n_atoms=n_atoms)
            if coord_entries
            else self._flat_xyz_to_df(np.asarray([], dtype=float), value_cols=("x", "y", "z"), n_atoms=n_atoms)
        )
        velocities_df = self._flat_xyz_to_df(vel_flat, value_cols=("vx", "vy", "vz"), n_atoms=n_atoms)
        accelerations_df = (
            self._flat_xyz_to_df(acc_entries[-1], value_cols=("ax", "ay", "az"), n_atoms=n_atoms)
            if acc_entries
            else self._flat_xyz_to_df(np.asarray([], dtype=float), value_cols=("ax", "ay", "az"), n_atoms=n_atoms)
        )

        if callable(reporter):
            reporter("load", 1, 1, "Reading AMS atomic kinematics from History")

        return AtomicKinematicsData(
            coordinates=coords_df,
            velocities=velocities_df,
            accelerations=accelerations_df,
            previous_accelerations=pd.DataFrame(columns=["atom_id", "ax", "ay", "az"]),
            metadata={
                "source_coordinates": "History%Coordinates (last frame)",
                "source_velocities": "History%Velocities (last entry)",
                "source_accelerations": "History%Acceleration (last entry)",
            },
        )

    def load_atom_temperature(self, args: dict, reporter=None) -> AtomTemperatureData:
        """Load per-atom temperature series from AMS ``History`` entries.

        Parameters
        ----------
        args : dict
            Adapter options including optional KF/RKF path hints.
        reporter : callable | None, optional
            Progress callback ``(phase, done, total, message)``.

        Returns
        -------
        AtomTemperatureData
            Temperature matrix indexed by frame and atom.

        Examples
        --------
        ```python
        t = AMSAdapter().load_atom_temperature({"rkf": "reaxout.rkf"})
        ```
        """
        kf = self.load_kf(args)
        history_data = kf.read_section("History")
        atom_temp_entries = self._history_entries(history_data, "Atom temperature", dtype=float)
        if not atom_temp_entries:
            raise RuntimeError("No 'Atom temperature*' entries were found in AMS History section.")

        n_frames = len(atom_temp_entries)
        n_atoms = max(np.asarray(arr, dtype=float).size for arr in atom_temp_entries)
        temperatures = self._stack_1d_per_frame(
            atom_temp_entries,
            n_frames=n_frames,
            n_atoms=n_atoms,
            fill_value=np.nan,
        )
        iterations = self._iterations_from_general(kf, n_frames)
        if callable(reporter):
            reporter("load", n_frames, n_frames, "Reading AMS atom temperatures from History")
        return AtomTemperatureData(
            iterations=iterations,
            temperatures=temperatures,
            metadata={"source": "History%Atom temperature"},
        )

    def load_atom_strain_energy(self, args: dict, reporter=None) -> AtomStrainEnergyData:
        """Load per-atom strain-energy series from AMS ``History`` entries.

        Parameters
        ----------
        args : dict
            Adapter options including optional KF/RKF path hints.
        reporter : callable | None, optional
            Progress callback ``(phase, done, total, message)``.

        Returns
        -------
        AtomStrainEnergyData
            Strain-energy matrix indexed by frame and atom.

        Examples
        --------
        ```python
        se = AMSAdapter().load_atom_strain_energy({"kf": "reaxout.kf"})
        ```
        """
        kf = self.load_kf(args)
        history_data = kf.read_section("History")
        strain_entries = self._history_entries(history_data, "Strain energy", dtype=float)
        if not strain_entries:
            raise RuntimeError("No 'Strain energy*' entries were found in AMS History section.")

        n_frames = len(strain_entries)
        n_atoms = max(np.asarray(arr, dtype=float).size for arr in strain_entries)
        strain_energy = self._stack_1d_per_frame(
            strain_entries,
            n_frames=n_frames,
            n_atoms=n_atoms,
            fill_value=np.nan,
        )
        iterations = self._iterations_from_general(kf, n_frames)
        if callable(reporter):
            reporter("load", n_frames, n_frames, "Reading AMS atom strain energy from History")
        return AtomStrainEnergyData(
            iterations=iterations,
            strain_energy=strain_energy,
            metadata={"source": "History%Strain energy"},
        )

    def load_molecular_analysis(self, args: dict, reporter=None) -> MolecularAnalysisData:
        """Load molecule counts/species tables from AMS molecule sections.

        Parameters
        ----------
        args : dict
            Adapter options including optional KF/RKF path hints.
        reporter : callable | None, optional
            Progress callback ``(phase, done, total, message)``.

        Returns
        -------
        MolecularAnalysisData
            Iteration-indexed totals and molecular species frequency table.

        Examples
        --------
        ```python
        mol = AMSAdapter().load_molecular_analysis({"rkf": "reaxout.rkf"})
        ```
        """
        kf = self.load_kf(args)
        mol_sections = self._read_molecule_sections(kf)
        if not mol_sections:
            raise RuntimeError("No Molecule/Molecules section was found in AMS KF file.")

        iter_values = self._iterations_from_general(kf, 0)
        iter_lookup = {int(i): int(v) for i, v in enumerate(iter_values.tolist())}

        rows_species: list[dict] = []
        total_molecules_by_iter: dict[int, float] = defaultdict(float)
        total_atoms_by_iter: dict[int, float] = defaultdict(float)
        seen_iter_indices: set[int] = set()

        for sec_i, mol_data in enumerate(mol_sections, start=1):
            names_by_id: dict[int, str] = {}
            section_counts_by_iter: dict[int, float] = {}
            section_atoms_by_mol_iter: dict[tuple[int, int], list[int]] = {}
            for key, value in mol_data.items():
                m = re.match(r"^Molecule name\s+(\d+)$", str(key))
                if not m:
                    continue
                mol_id = int(m.group(1))
                names_by_id[mol_id] = self._scalar_or_first(value, default=f"Molecule_{sec_i}_{mol_id}")

            if not names_by_id:
                names_by_id[1] = f"Molecule_{sec_i}_1"

            for key, value in mol_data.items():
                m_num = re.match(r"^Num molecules\s+(\d+)$", str(key))
                if m_num:
                    iter_idx = int(m_num.group(1))
                    seen_iter_indices.add(iter_idx)
                    counts = self._to_1d_array(value, dtype=float)
                    freq = float(counts[0]) if counts.size > 0 else 0.0
                    if np.isnan(freq):
                        freq = 0.0
                    section_counts_by_iter[iter_idx] = freq
                    total_molecules_by_iter[iter_idx] += freq
                    continue

                m_atoms = re.match(r"^Atoms\s+(\d+)\s+(\d+)$", str(key))
                if m_atoms:
                    mol_id = int(m_atoms.group(1))
                    iter_idx = int(m_atoms.group(2))
                    seen_iter_indices.add(iter_idx)
                    atom_ids = self._to_1d_array(value, dtype=int)
                    section_atoms_by_mol_iter[(mol_id, iter_idx)] = [int(v) for v in atom_ids.tolist()]
                    total_atoms_by_iter[iter_idx] += float(atom_ids.size)

            section_iter_indices: set[int] = set(section_counts_by_iter.keys())
            section_iter_indices.update(iter_idx for _, iter_idx in section_atoms_by_mol_iter.keys())
            for iter_idx in sorted(section_iter_indices):
                iter_value = iter_lookup.get(iter_idx, iter_idx)
                freq = float(section_counts_by_iter.get(iter_idx, 0.0))
                for mol_id, mol_name in names_by_id.items():
                    rows_species.append(
                        {
                            "iter": int(iter_value),
                            "molecular_formula": str(mol_name),
                            "freq": freq,
                            "molecular_mass": np.nan,
                        }
                    )

        sorted_iter_indices = sorted(seen_iter_indices)
        if not sorted_iter_indices:
            sorted_iter_indices = list(range(int(iter_values.size)))

        rows_totals: list[dict] = []
        for pos, iter_idx in enumerate(sorted_iter_indices, start=1):
            iter_value = iter_lookup.get(iter_idx, iter_idx)
            rows_totals.append(
                {
                    "iter": int(iter_value),
                    "total_molecules": float(total_molecules_by_iter.get(iter_idx, 0.0)),
                    "total_atoms": float(total_atoms_by_iter.get(iter_idx, 0.0)),
                    "total_molecular_mass": np.nan,
                }
            )
            if callable(reporter):
                reporter("load", pos, len(sorted_iter_indices), "Reading AMS molecular analysis")

        species_df = pd.DataFrame(
            rows_species,
            columns=["iter", "molecular_formula", "freq", "molecular_mass"],
        )
        totals_df = pd.DataFrame(rows_totals, columns=["iter", "total_molecules", "total_atoms", "total_molecular_mass"])
        if not species_df.empty:
            species_df = species_df.sort_values(["iter", "molecular_formula"], kind="stable").reset_index(drop=True)
        if not totals_df.empty:
            totals_df = totals_df.sort_values(["iter"], kind="stable").reset_index(drop=True)

        return MolecularAnalysisData(
            iterations=totals_df["iter"].to_numpy(dtype=int) if not totals_df.empty else np.empty((0,), dtype=int),
            totals=totals_df,
            molecular_species=species_df,
        )

    def load_stress(self, args: dict, reporter=None) -> StressData:
        """Load per-atom stress tensor series and derived stress variants.

        Parameters
        ----------
        args : dict
            Adapter options including optional KF/RKF path hints.
        reporter : callable | None, optional
            Progress callback ``(phase, done, total, message)``.

        Returns
        -------
        StressData
            Stress tensor arrays and optional averaged/isotropic series.

        Examples
        --------
        ```python
        stress = AMSAdapter().load_stress({"rkf": "reaxout.rkf"})
        ```
        """
        kf = self.load_kf(args)
        history_data = kf.read_section("History")
        atomic_entries = self._history_entries(history_data, "Atomic stress", dtype=float)
        avg_entries = self._history_entries(history_data, "Average atomic stress", dtype=float)
        avg_iso_entries = self._history_entries(history_data, "Avg iso atomic stress", dtype=float)
        iso_entries = self._history_entries(history_data, "Iso atomic stress", dtype=float)
        loavg_entries = self._history_entries(history_data, "LoAvg atomic stress", dtype=float)
        loavg_iso_entries = self._history_entries(history_data, "LoAvg iso atom stress", dtype=float)

        n_frames = max(
            len(atomic_entries),
            len(avg_entries),
            len(avg_iso_entries),
            len(iso_entries),
            len(loavg_entries),
            len(loavg_iso_entries),
        )
        if n_frames == 0:
            raise RuntimeError("No 'Atomic stress*' entries were found in AMS History section.")

        n_atoms = 0
        for arr in atomic_entries + avg_entries + loavg_entries:
            n_atoms = max(n_atoms, int(np.asarray(arr, dtype=float).size // 6))
        for arr in iso_entries + avg_iso_entries + loavg_iso_entries:
            n_atoms = max(n_atoms, int(np.asarray(arr, dtype=float).size))

        values = self._stack_6c_per_frame(atomic_entries, n_frames=n_frames, n_atoms=n_atoms)
        average_values = self._stack_6c_per_frame(avg_entries, n_frames=n_frames, n_atoms=n_atoms) if avg_entries else None
        loavg_values = self._stack_6c_per_frame(loavg_entries, n_frames=n_frames, n_atoms=n_atoms) if loavg_entries else None
        iso_values = self._stack_1d_per_frame(iso_entries, n_frames=n_frames, n_atoms=n_atoms, fill_value=np.nan) if iso_entries else None
        avg_iso_values = (
            self._stack_1d_per_frame(avg_iso_entries, n_frames=n_frames, n_atoms=n_atoms, fill_value=np.nan)
            if avg_iso_entries
            else None
        )
        loavg_iso_values = (
            self._stack_1d_per_frame(loavg_iso_entries, n_frames=n_frames, n_atoms=n_atoms, fill_value=np.nan)
            if loavg_iso_entries
            else None
        )
        for fi in range(n_frames):
            if callable(reporter):
                reporter("load", fi + 1, n_frames, "Reading AMS stress series from History")

        iterations = self._iterations_from_general(kf, n_frames)
        return StressData(
            iterations=iterations,
            components=("xx", "yy", "zz", "yx", "zx", "zy"),
            values=values,
            average_values=average_values,
            iso_values=iso_values,
            avg_iso_values=avg_iso_values,
            loavg_values=loavg_values,
            loavg_iso_values=loavg_iso_values,
            metadata={
                "source": "History",
                "series": [
                    "Atomic stress",
                    "Average atomic stress",
                    "Avg iso atomic stress",
                    "Iso atomic stress",
                    "LoAvg atomic stress",
                    "LoAvg iso atom stress",
                ],
            },
        )

    def load_connectivity_trajectory(self, args: dict, reporter=None) -> ConnectivityTrajectoryData:
        """Load connectivity and trajectory together as a composite payload.

        Parameters
        ----------
        args : dict
            Adapter options including optional KF/RKF path hints.
        reporter : callable | None, optional
            Progress callback passed through to both loaders.

        Returns
        -------
        ConnectivityTrajectoryData
            Combined connectivity and trajectory canonical models.

        Examples
        --------
        ```python
        combo = AMSAdapter().load_connectivity_trajectory({"kf": "reaxout.kf"})
        ```
        """
        _ = reporter
        return ConnectivityTrajectoryData(
            connectivity=self.load_connectivity(args, reporter=reporter),
            trajectory=self.load_trajectory(args, reporter=reporter),
        )

    def write_trajectory(self, data: TrajectoryData, out_path: str | Path, args: dict | None = None):
        """Write canonical trajectory data to an XYZ trajectory file.

        Parameters
        ----------
        data : TrajectoryData
            Canonical trajectory payload to serialize.
        out_path : str | Path
            Destination file path for XYZ output.
        args : dict | None, optional
            Optional writer settings. Supports ``precision`` (default ``6``).

        Returns
        -------
        Any
            Return value from ``write_xyz_trajectory``.

        Examples
        --------
        ```python
        adapter.write_trajectory(traj, "traj.xyz", {"precision": 8})
        ```
        """
        args = args or {}
        return write_xyz_trajectory(data, out_path, precision=int(args.get("precision", 6)))
