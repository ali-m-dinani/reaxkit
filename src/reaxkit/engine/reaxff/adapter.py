"""ReaxFF engine adapter."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from reaxkit.core.engine_registry import register_engine
from reaxkit.domain.data_models import (
    ChargeData,
    ConnectivityData,
    ElectricFieldData,
    ForceFieldData,
    MolecularAnalysisData,
    SimulationData,
    TrajectoryData,
)
from reaxkit.engine.base import EngineAdapter
from reaxkit.engine.reaxff.io.ffield_handler import FFieldHandler
from reaxkit.engine.reaxff.io.fort7_handler import Fort7Handler
from reaxkit.engine.reaxff.io.fort78_handler import Fort78Handler
from reaxkit.engine.reaxff.io.molfra_handler import MolFraHandler
from reaxkit.engine.reaxff.io.summary_handler import SummaryHandler
from reaxkit.engine.reaxff.io.xmolout_handler import XmoloutHandler


class _SparseFrame:
    """Lightweight sparse frame wrapper compatible with analysis loaders."""

    def __init__(self, n_atoms: int, pairs: dict[tuple[int, int], float]):
        self.n_atoms = int(n_atoms)
        if pairs:
            ij = np.asarray(list(pairs.keys()), dtype=int)
            self._rows = ij[:, 0]
            self._cols = ij[:, 1]
            self._vals = np.asarray(list(pairs.values()), dtype=float)
        else:
            self._rows = np.empty((0,), dtype=int)
            self._cols = np.empty((0,), dtype=int)
            self._vals = np.empty((0,), dtype=float)

    @property
    def shape(self) -> tuple[int, int]:
        return (self.n_atoms, self.n_atoms)

    def toarray(self) -> np.ndarray:
        mat = np.zeros((self.n_atoms, self.n_atoms), dtype=float)
        if self._vals.size:
            mat[self._rows, self._cols] = self._vals
        return mat

    def todense(self) -> np.ndarray:
        return self.toarray()

    def sum(self, axis=None):
        if axis == 1:
            out = np.zeros((self.n_atoms,), dtype=float)
            if self._vals.size:
                np.add.at(out, self._rows, self._vals)
            return out
        return self.toarray().sum(axis=axis)


def _merge_simulation_data(
    base: SimulationData | None,
    extra: SimulationData | None,
) -> SimulationData | None:
    if base is None:
        return extra
    if extra is None:
        return base
    return SimulationData(
        atom_ids=base.atom_ids if base.atom_ids is not None else extra.atom_ids,
        iterations=base.iterations if base.iterations is not None else extra.iterations,
        time=base.time if base.time is not None else extra.time,
        elements=base.elements if base.elements is not None else extra.elements,
        num_of_atoms=base.num_of_atoms if base.num_of_atoms is not None else extra.num_of_atoms,
        potential_energy=base.potential_energy if base.potential_energy is not None else extra.potential_energy,
        V=base.V if base.V is not None else extra.V,
        T=base.T if base.T is not None else extra.T,
        P=base.P if base.P is not None else extra.P,
        D=base.D if base.D is not None else extra.D,
        elap_time=base.elap_time if base.elap_time is not None else extra.elap_time,
        atom_type_nums=base.atom_type_nums if base.atom_type_nums is not None else extra.atom_type_nums,
        molecule_nums=base.molecule_nums if base.molecule_nums is not None else extra.molecule_nums,
        cell_lengths=base.cell_lengths if base.cell_lengths is not None else extra.cell_lengths,
        cell_angles=base.cell_angles if base.cell_angles is not None else extra.cell_angles,
    )


def _fort7_per_atom_arrays(
    frames_df: list[pd.DataFrame],
    atom_ids: list[int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_frames = len(frames_df)
    n_atoms = len(atom_ids)
    atom_to_idx = {int(a): i for i, a in enumerate(atom_ids)}
    atom_type_nums = np.zeros((n_frames, n_atoms), dtype=int)
    molecule_nums = np.zeros((n_frames, n_atoms), dtype=int)
    num_lone_pairs = np.zeros((n_frames, n_atoms), dtype=float)

    for fi, fr in enumerate(frames_df):
        for _, row in fr.iterrows():
            aid = int(row["atom_num"]) if "atom_num" in fr.columns else None
            if aid is None:
                continue
            j = atom_to_idx.get(aid)
            if j is None:
                continue
            if "atom_type_num" in fr.columns and pd.notna(row["atom_type_num"]):
                atom_type_nums[fi, j] = int(row["atom_type_num"])
            if "molecule_num" in fr.columns and pd.notna(row["molecule_num"]):
                molecule_nums[fi, j] = int(row["molecule_num"])
            if "num_LPs" in fr.columns and pd.notna(row["num_LPs"]):
                num_lone_pairs[fi, j] = float(row["num_LPs"])

    return atom_type_nums, molecule_nums, num_lone_pairs


def _trajectory_from_xmolout_handler(handler: XmoloutHandler) -> TrajectoryData:
    """Normalize an ``XmoloutHandler`` into ``TrajectoryData``."""
    n_frames = handler.n_frames()
    frames = [handler.frame(i) for i in range(n_frames)]
    positions = np.stack([f["coords"] for f in frames], axis=0)
    elements = list(frames[0]["atom_types"]) if frames else []
    atom_ids = list(range(1, len(elements) + 1))

    df = handler.dataframe()
    iterations = df["iter"].to_numpy() if "iter" in df.columns else np.arange(n_frames)
    num_of_atoms = df["num_of_atoms"].to_numpy(dtype=int) if "num_of_atoms" in df.columns else None
    potential_energy = df["E_pot"].to_numpy(dtype=float) if "E_pot" in df.columns else None
    cell_lengths = df[["a", "b", "c"]].to_numpy() if {"a", "b", "c"}.issubset(df.columns) else None
    if {"alpha", "beta", "gamma"}.issubset(df.columns):
        cell_angles = df[["alpha", "beta", "gamma"]].to_numpy()
    else:
        cell_angles = np.full((n_frames, 3), 90.0)

    return TrajectoryData(
        positions=positions,
        elements=elements,
        atom_ids=atom_ids,
        simulation=SimulationData(
            atom_ids=atom_ids,
            iterations=iterations,
            time=None,
            elements=elements,
            num_of_atoms=num_of_atoms,
            potential_energy=potential_energy,
            cell_lengths=cell_lengths,
            cell_angles=cell_angles,
        ),
        iterations=iterations,
    )


def _connectivity_from_fort7_handler(handler: Fort7Handler, reporter=None) -> ConnectivityData:
    """Normalize a ``Fort7Handler`` into ``ConnectivityData``."""
    sim_df = handler.dataframe()
    frames_df = [handler.frame(i) for i in range(handler.n_frames())]

    if not frames_df:
        return ConnectivityData(
            connectivity=[],
            bond_orders=[],
            sum_bond_orders=np.empty((0, 0), dtype=float),
            metadata={"source": "fort7", "simulation_name": handler.metadata().get("simulation_name", "")},
        )

    first = frames_df[0]
    if "atom_num" in first.columns:
        atom_ids = first["atom_num"].astype(int).to_numpy()
    else:
        atom_ids = np.arange(1, len(first) + 1, dtype=int)
    n_atoms = len(atom_ids)
    atom_to_idx = {int(a): i for i, a in enumerate(atom_ids)}
    atom_type_nums, molecule_nums, num_lone_pairs = _fort7_per_atom_arrays(frames_df, atom_ids.tolist())

    bo_frames: list[_SparseFrame] = []
    connectivity_frames: list[_SparseFrame] = []
    sum_rows = []
    n_frames = len(frames_df)
    for step_i, fr in enumerate(frames_df, start=1):
        bo_pairs: dict[tuple[int, int], float] = {}
        conn_pairs: dict[tuple[int, int], float] = {}
        bo_cols = [c for c in fr.columns if str(c).startswith("BO")]
        for row_idx, row in fr.iterrows():
            src_atom = int(row["atom_num"]) if "atom_num" in fr.columns else int(row_idx) + 1
            src_i = atom_to_idx.get(src_atom)
            if src_i is None:
                continue
            for bo_col in bo_cols:
                slot = bo_col[2:]
                cnn_col = f"atom_cnn{slot}"
                if cnn_col not in fr.columns:
                    continue
                dst_atom = int(row[cnn_col]) if pd.notna(row[cnn_col]) else 0
                if dst_atom <= 0:
                    continue
                dst_i = atom_to_idx.get(dst_atom)
                if dst_i is None:
                    continue
                conn_pairs[(src_i, dst_i)] = 1.0
                bo = float(row[bo_col]) if pd.notna(row[bo_col]) else 0.0
                if bo <= 0.0:
                    continue
                key = (src_i, dst_i)
                bo_pairs[key] = max(bo_pairs.get(key, 0.0), bo)

        bo_fr = _SparseFrame(n_atoms, bo_pairs)
        bo_frames.append(bo_fr)
        connectivity_frames.append(_SparseFrame(n_atoms, conn_pairs))
        sum_rows.append(np.asarray(bo_fr.sum(axis=1), dtype=float))
        if reporter:
            reporter("load", step_i, n_frames, "Building connectivity matrices from fort.7")

    sum_arr = np.vstack(sum_rows) if sum_rows else np.empty((0, 0), dtype=float)
    iterations = sim_df["iter"].to_numpy(dtype=int) if "iter" in sim_df.columns else np.arange(len(bo_frames), dtype=int)
    meta = handler.metadata()
    return ConnectivityData(
        connectivity=connectivity_frames,
        bond_orders=bo_frames,
        sum_bond_orders=sum_arr,
        num_lone_pairs=num_lone_pairs,
        num_of_bonds=sim_df["num_of_bonds"].to_numpy(dtype=int) if "num_of_bonds" in sim_df.columns else None,
        total_bond_order=sim_df["total_BO"].to_numpy(dtype=float) if "total_BO" in sim_df.columns else None,
        total_lone_pairs=sim_df["total_LP"].to_numpy(dtype=float) if "total_LP" in sim_df.columns else None,
        total_bond_order_uncorrected=(
            sim_df["total_BO_uncorrected"].to_numpy(dtype=float) if "total_BO_uncorrected" in sim_df.columns else None
        ),
        atom_ids=atom_ids,
        simulation=SimulationData(
            atom_ids=atom_ids,
            iterations=iterations,
            atom_type_nums=atom_type_nums,
            molecule_nums=molecule_nums,
        ),
        iterations=iterations,
        metadata={"source": "fort7", "simulation_name": meta.get("simulation_name", ""), "bond_orders_format": "sparse_frame_list"},
    )


def _force_field_from_ffield_handler(handler: FFieldHandler) -> ForceFieldData:
    """Normalize an ``FFieldHandler`` into ``ForceFieldData``."""
    sections = handler.sections
    meta = handler.metadata()

    return ForceFieldData(
        general_parameters=sections.get(FFieldHandler.SECTION_GENERAL, pd.DataFrame()).copy(),
        atom_parameters=sections.get(FFieldHandler.SECTION_ATOM, pd.DataFrame()).copy(),
        bond_parameters=sections.get(FFieldHandler.SECTION_BOND, pd.DataFrame()).copy(),
        off_diagonal_parameters=sections.get(FFieldHandler.SECTION_OFF_DIAGONAL, pd.DataFrame()).copy(),
        angle_parameters=sections.get(FFieldHandler.SECTION_ANGLE, pd.DataFrame()).copy(),
        torsion_parameters=sections.get(FFieldHandler.SECTION_TORSION, pd.DataFrame()).copy(),
        hydrogen_bond_parameters=sections.get(FFieldHandler.SECTION_HBOND, pd.DataFrame()).copy(),
        source="reaxff/ffield",
        metadata=dict(meta),
    )


def _charges_from_fort7_handler(
    handler: Fort7Handler,
    *,
    simulation: SimulationData | None = None,
    reporter=None,
) -> ChargeData:
    """Normalize fort.7 partial charges into ``ChargeData``."""
    sim_df = handler.dataframe()
    frames_df = [handler.frame(i) for i in range(handler.n_frames())]
    if not frames_df:
        return ChargeData(
            charges=np.empty((0, 0), dtype=float),
            simulation=simulation,
            iterations=np.empty((0,), dtype=int),
            metadata={"source": "fort7"},
        )

    if simulation is not None and simulation.atom_ids:
        atom_ids = [int(a) for a in simulation.atom_ids]
    else:
        first = frames_df[0]
        if "atom_num" in first.columns:
            atom_ids = first["atom_num"].astype(int).tolist()
        else:
            atom_ids = list(range(1, len(first) + 1))
        simulation = SimulationData(atom_ids=atom_ids)

    atom_to_idx = {int(a): i for i, a in enumerate(atom_ids)}
    atom_type_nums, molecule_nums, _ = _fort7_per_atom_arrays(frames_df, atom_ids)
    simulation = _merge_simulation_data(
        simulation,
        SimulationData(
            atom_ids=atom_ids,
            iterations=(sim_df["iter"].to_numpy(dtype=int) if "iter" in sim_df.columns else None),
            atom_type_nums=atom_type_nums,
            molecule_nums=molecule_nums,
            num_of_atoms=(sim_df["num_of_atoms"].to_numpy(dtype=int) if "num_of_atoms" in sim_df.columns else None),
        ),
    )
    n_atoms = len(atom_ids)
    rows: list[np.ndarray] = []
    n_frames = len(frames_df)
    for step_i, fr in enumerate(frames_df, start=1):
        q = np.zeros((n_atoms,), dtype=float)
        if "partial_charge" in fr.columns:
            if "atom_num" in fr.columns:
                for _, row in fr.iterrows():
                    aid = int(row["atom_num"])
                    j = atom_to_idx.get(aid)
                    if j is None:
                        continue
                    q[j] = float(row["partial_charge"]) if pd.notna(row["partial_charge"]) else 0.0
            else:
                vals = fr["partial_charge"].to_numpy(dtype=float)
                q[: min(len(vals), n_atoms)] = vals[: min(len(vals), n_atoms)]
        rows.append(q)
        if reporter:
            reporter("load", step_i, n_frames, "Building charge matrix from fort.7")

    charges = np.vstack(rows) if rows else np.empty((0, 0), dtype=float)
    iterations = sim_df["iter"].to_numpy(dtype=int) if "iter" in sim_df.columns else np.arange(len(rows), dtype=int)
    return ChargeData(
        charges=charges,
        total_charge=sim_df["total_charge"].to_numpy(dtype=float) if "total_charge" in sim_df.columns else None,
        simulation=simulation,
        iterations=iterations,
        metadata={"source": "fort7"},
    )


def _electric_field_from_fort78_handler(handler: Fort78Handler) -> ElectricFieldData:
    """Normalize fort.78 data into ``ElectricFieldData``."""
    df = handler.dataframe()
    if df.empty:
        return ElectricFieldData(
            applied_field_values=np.empty((0, 0), dtype=float),
            applied_field_components=(),
            field_energy_values=np.empty((0, 0), dtype=float),
            field_energy_components=(),
            sampled_field_iterations=np.empty((0,), dtype=int),
            metadata={"source": "fort.78", "columns": []},
        )

    components = [c for c in df.columns if c != "iter"]
    if not components:
        return ElectricFieldData(
            applied_field_values=np.empty((len(df), 0), dtype=float),
            applied_field_components=(),
            field_energy_values=np.empty((len(df), 0), dtype=float),
            field_energy_components=(),
            sampled_field_iterations=(
                df["iter"].to_numpy(dtype=int) if "iter" in df.columns else np.arange(len(df), dtype=int)
            ),
            metadata={"source": "fort.78", "columns": list(df.columns)},
        )

    iters = df["iter"].to_numpy(dtype=int) if "iter" in df.columns else np.arange(len(df), dtype=int)
    applied_components = [c for c in components if str(c).startswith("field_")]
    field_energy_components = [c for c in components if str(c) == "E_field" or str(c).startswith("E_field_")]
    applied_vals = (
        df[applied_components].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
        if applied_components
        else np.empty((len(df), 0), dtype=float)
    )


def _simulation_from_summary_handler(handler: SummaryHandler) -> SimulationData:
    """Normalize summary.txt data into SimulationData."""
    df = handler.dataframe()
    n_rows = len(df)
    iterations = df["iter"].to_numpy(dtype=int) if "iter" in df.columns else np.arange(n_rows, dtype=int)
    atom_ids: list[int] = []
    return SimulationData(
        atom_ids=atom_ids,
        iterations=iterations,
        time=(df["time"].to_numpy(dtype=float) if "time" in df.columns else None),
        potential_energy=(df["E_pot"].to_numpy(dtype=float) if "E_pot" in df.columns else None),
        V=(df["V"].to_numpy(dtype=float) if "V" in df.columns else None),
        T=(df["T"].to_numpy(dtype=float) if "T" in df.columns else None),
        P=(df["P"].to_numpy(dtype=float) if "P" in df.columns else None),
        D=(df["D"].to_numpy(dtype=float) if "D" in df.columns else None),
        elap_time=(df["elap_time"].to_numpy(dtype=float) if "elap_time" in df.columns else None),
    )
    energy_vals = (
        df[field_energy_components].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
        if field_energy_components
        else np.empty((len(df), 0), dtype=float)
    )
    return ElectricFieldData(
        applied_field_values=applied_vals,
        applied_field_components=tuple(str(c) for c in applied_components),
        field_energy_values=energy_vals,
        field_energy_components=tuple(str(c) for c in field_energy_components),
        sampled_field_iterations=iters,
        metadata={"source": "fort.78", "columns": list(df.columns)},
    )


def _molecular_analysis_from_molfra_handler(handler: MolFraHandler) -> MolecularAnalysisData:
    """Normalize molfra data into MolecularAnalysisData."""
    species = handler.dataframe().copy()
    totals = handler.totals().copy()
    if not species.empty:
        species["iter"] = pd.to_numeric(species["iter"], errors="coerce").astype(int)
        species["freq"] = pd.to_numeric(species["freq"], errors="coerce").astype(int)
        species["molecular_mass"] = pd.to_numeric(species["molecular_mass"], errors="coerce").astype(float)
    if not totals.empty:
        totals["iter"] = pd.to_numeric(totals["iter"], errors="coerce").astype(int)
        for col in ["total_molecules", "total_atoms"]:
            if col in totals.columns:
                totals[col] = pd.to_numeric(totals[col], errors="coerce").astype(int)
        if "total_molecular_mass" in totals.columns:
            totals["total_molecular_mass"] = pd.to_numeric(totals["total_molecular_mass"], errors="coerce").astype(float)

    if not totals.empty:
        iterations = totals["iter"].to_numpy(dtype=int)
    elif not species.empty:
        iterations = np.asarray(sorted(species["iter"].unique().tolist()), dtype=int)
    else:
        iterations = np.empty((0,), dtype=int)

    return MolecularAnalysisData(
        iterations=iterations,
        totals=totals,
        molecular_species=species,
    )


@register_engine("reaxff")
class ReaxFFAdapter(EngineAdapter):
    """Adapter that loads ReaxFF outputs into domain models."""

    def detect(self, path: str | Path) -> float:
        p = Path(path)
        has_xmol = (p / "xmolout").exists() or p.name == "xmolout"
        return 0.95 if has_xmol else 0.0

    def load_trajectory(self, args: dict, reporter=None) -> TrajectoryData:
        xmol_path = args.get("xmolout") or args.get("input") or "xmolout"
        handler = XmoloutHandler(xmol_path, reporter=reporter)
        trj = _trajectory_from_xmolout_handler(handler)
        trj.simulation = _merge_simulation_data(
            trj.simulation,
            self._load_simulation_from_summary(args, reporter=reporter),
        )
        return trj

    def load_simulation(self, args: dict, reporter=None) -> SimulationData:
        sim = self._load_simulation_from_xmolout(args, reporter=reporter)
        sim = _merge_simulation_data(sim, self._load_simulation_from_summary(args, reporter=reporter))
        if sim is None:
            raise FileNotFoundError("SimulationData for reaxff currently requires xmolout or summary.txt.")
        return sim

    @staticmethod
    def _load_simulation_from_xmolout(args: dict, reporter=None) -> SimulationData | None:
        raw = args.get("xmolout") or args.get("input")
        if not raw:
            return None
        p = Path(raw)
        xmol_path = p / "xmolout" if p.is_dir() else p
        if not xmol_path.exists() or xmol_path.name != "xmolout":
            return None
        handler = XmoloutHandler(xmol_path, reporter=reporter)
        trj = _trajectory_from_xmolout_handler(handler)
        return trj.simulation

    @staticmethod
    def _load_simulation_from_summary(args: dict, reporter=None) -> SimulationData | None:
        candidates = [args.get("summary"), args.get("input"), args.get("xmolout")]
        summary_path = None
        for raw in candidates:
            if not raw:
                continue
            p = Path(raw)
            if p.is_dir():
                candidate = p / "summary.txt"
            elif p.name == "summary.txt":
                candidate = p
            else:
                candidate = p.parent / "summary.txt"
            if candidate.exists() and candidate.name == "summary.txt":
                summary_path = candidate
                break
        if summary_path is None:
            return None
        handler = SummaryHandler(summary_path, reporter=reporter)
        return _simulation_from_summary_handler(handler)

    def load_connectivity(self, args: dict, reporter=None) -> ConnectivityData:
        raw = args.get("fort7") or args.get("connectivity") or args.get("input") or "fort.7"
        p = Path(raw)
        fort7_path = p / "fort.7" if p.is_dir() else p
        handler = Fort7Handler(fort7_path, reporter=reporter)
        conn = _connectivity_from_fort7_handler(handler, reporter=reporter)
        sim = _merge_simulation_data(
            self._load_simulation_from_xmolout(args, reporter=reporter),
            self._load_simulation_from_summary(args, reporter=reporter),
        )
        if sim is not None:
            conn.simulation = _merge_simulation_data(sim, conn.simulation)
            conn.elements = conn.simulation.elements
            conn.atom_ids = conn.simulation.atom_ids
        return conn

    def load_force_field(self, args: dict, reporter=None) -> ForceFieldData:
        raw = args.get("ffield") or args.get("force_field") or args.get("atom_reference") or args.get("input") or "ffield"
        p = Path(raw)
        ffield_path = p / "ffield" if p.is_dir() else p
        handler = FFieldHandler(ffield_path, reporter=reporter)
        out = _force_field_from_ffield_handler(handler)
        return out

    def load_charges(self, args: dict, reporter=None) -> ChargeData:
        raw = args.get("fort7") or args.get("charges") or args.get("input") or "fort.7"
        p = Path(raw)
        fort7_path = p / "fort.7" if p.is_dir() else p
        handler = Fort7Handler(fort7_path, reporter=reporter)
        sim = _merge_simulation_data(
            self._load_simulation_from_xmolout(args, reporter=reporter),
            self._load_simulation_from_summary(args, reporter=reporter),
        )
        return _charges_from_fort7_handler(handler, simulation=sim, reporter=reporter)

    def load_electric_field(self, args: dict, reporter=None) -> ElectricFieldData:
        raw = args.get("fort78") or args.get("electric_field") or args.get("input") or "fort.78"
        p = Path(raw)
        fort78_path = p / "fort.78" if p.is_dir() else p
        # ReaxFF writes fort.78 on the iout1 schedule, which may differ from xmolout/fort.7/summary
        # outputs that are typically written on the iout2 schedule.
        handler = Fort78Handler(fort78_path, reporter=reporter)
        out = _electric_field_from_fort78_handler(handler)
        return out

    def load_molecular_analysis(self, args: dict, reporter=None) -> MolecularAnalysisData:
        raw = args.get("molfra") or args.get("molecular_analysis") or args.get("input") or "molfra.out"
        p = Path(raw)
        if p.is_dir():
            if (p / "molfra.out").exists():
                molfra_path = p / "molfra.out"
            elif (p / "molfra_ig.out").exists():
                molfra_path = p / "molfra_ig.out"
            else:
                molfra_path = p / "molfra.out"
        else:
            molfra_path = p
        handler = MolFraHandler(molfra_path, reporter=reporter)
        return _molecular_analysis_from_molfra_handler(handler)
