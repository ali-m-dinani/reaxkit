"""ReaxFF engine adapter."""

from __future__ import annotations

from pathlib import Path
from time import perf_counter
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from reaxkit.core.engine_registry import register_engine
from reaxkit.domain.data_models import (
    AtomicKinematicsData,
    ChargeData,
    ConnectivityTrajectoryData,
    CoordinationStatusBundleData,
    ConnectivityData,
    ControlParametersData,
    ElectrostaticsData,
    EregimeData,
    ElectricFieldData,
    ForceFieldParametersData,
    ForceFieldOptimizationProgressData,
    ForceFieldOptimizationTrainingSetData,
    ForceFieldOptimizationData,
    ForceFieldOptimizationParameterBundleData,
    ForceFieldOptimizationDiagnosticBundleData,
    ForceFieldOptimizationReportEOSBundleData,
    ForceFieldOptimizationParameterData,
    ForceFieldOptimizationReportData,
    GeometryData,
    GeometryOptimizationProgressData,
    MolecularAnalysisData,
    ForceFieldOptimizationDiagnosticData,
    PartialEnergyData,
    RestraintData,
    GeometrySummaryData,
    SimulationData,
    TrajectoryData,
)
from reaxkit.engine.base import EngineAdapter

if TYPE_CHECKING:
    from reaxkit.engine.reaxff.io.control_handler import ControlHandler
    from reaxkit.engine.reaxff.io.eregime_handler import EregimeHandler
    from reaxkit.engine.reaxff.io.ffield_handler import FFieldHandler
    from reaxkit.engine.reaxff.io.fort7_handler import Fort7Handler
    from reaxkit.engine.reaxff.io.fort13_handler import Fort13Handler
    from reaxkit.engine.reaxff.io.fort57_handler import Fort57Handler
    from reaxkit.engine.reaxff.io.fort73_handler import Fort73Handler
    from reaxkit.engine.reaxff.io.fort74_handler import Fort74Handler
    from reaxkit.engine.reaxff.io.fort76_handler import Fort76Handler
    from reaxkit.engine.reaxff.io.fort78_handler import Fort78Handler
    from reaxkit.engine.reaxff.io.fort79_handler import Fort79Handler
    from reaxkit.engine.reaxff.io.fort99_handler import Fort99Handler
    from reaxkit.engine.reaxff.io.geo_handler import GeoHandler
    from reaxkit.engine.reaxff.io.molfra_handler import MolFraHandler
    from reaxkit.engine.reaxff.io.params_handler import ParamsHandler
    from reaxkit.engine.reaxff.io.summary_handler import SummaryHandler
    from reaxkit.engine.reaxff.io.trainset_handler import TrainsetHandler
    from reaxkit.engine.reaxff.io.vels_handler import VelsHandler
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
    def _reindex_to_iterations(
        values: np.ndarray | None,
        *,
        source_iterations: np.ndarray | None,
        target_iterations: np.ndarray | None,
    ) -> np.ndarray | None:
        if values is None or source_iterations is None or target_iterations is None:
            return values
        src_it = np.asarray(source_iterations, dtype=int)
        tgt_it = np.asarray(target_iterations, dtype=int)
        arr = np.asarray(values)
        if arr.ndim == 0 or arr.shape[0] != src_it.shape[0]:
            return values

        idx_by_iter: dict[int, int] = {}
        for src_i, iter_i in enumerate(src_it):
            idx_by_iter[int(iter_i)] = int(src_i)

        out_shape = (tgt_it.shape[0],) + tuple(arr.shape[1:])
        out_dtype = arr.dtype
        if arr.dtype.kind in "iu":
            out_dtype = float
        out = np.full(out_shape, np.nan, dtype=out_dtype)

        for out_i, iter_i in enumerate(tgt_it):
            src_i = idx_by_iter.get(int(iter_i))
            if src_i is None:
                continue
            out[out_i] = arr[src_i]
        return out

    def _pick(
        base_values: np.ndarray | None,
        extra_values: np.ndarray | None,
        *,
        base_iterations: np.ndarray | None,
        extra_iterations: np.ndarray | None,
    ) -> np.ndarray | None:
        if base_values is not None:
            return base_values
        return _reindex_to_iterations(
            extra_values,
            source_iterations=extra_iterations,
            target_iterations=base_iterations,
        )

    if base is None:
        return extra
    if extra is None:
        return base
    return SimulationData(
        atom_ids=base.atom_ids if base.atom_ids is not None else extra.atom_ids,
        iterations=base.iterations if base.iterations is not None else extra.iterations,
        time=_pick(
            base.time,
            extra.time,
            base_iterations=base.iterations,
            extra_iterations=extra.iterations,
        ),
        elements=base.elements if base.elements is not None else extra.elements,
        num_of_atoms=_pick(
            base.num_of_atoms,
            extra.num_of_atoms,
            base_iterations=base.iterations,
            extra_iterations=extra.iterations,
        ),
        potential_energy=_pick(
            base.potential_energy,
            extra.potential_energy,
            base_iterations=base.iterations,
            extra_iterations=extra.iterations,
        ),
        volume=_pick(
            base.volume,
            extra.volume,
            base_iterations=base.iterations,
            extra_iterations=extra.iterations,
        ),
        temperature=_pick(
            base.temperature,
            extra.temperature,
            base_iterations=base.iterations,
            extra_iterations=extra.iterations,
        ),
        pressure=_pick(
            base.pressure,
            extra.pressure,
            base_iterations=base.iterations,
            extra_iterations=extra.iterations,
        ),
        density=_pick(
            base.density,
            extra.density,
            base_iterations=base.iterations,
            extra_iterations=extra.iterations,
        ),
        elapsed_time=_pick(
            base.elapsed_time,
            extra.elapsed_time,
            base_iterations=base.iterations,
            extra_iterations=extra.iterations,
        ),
        atom_type_nums=_pick(
            base.atom_type_nums,
            extra.atom_type_nums,
            base_iterations=base.iterations,
            extra_iterations=extra.iterations,
        ),
        molecule_nums=_pick(
            base.molecule_nums,
            extra.molecule_nums,
            base_iterations=base.iterations,
            extra_iterations=extra.iterations,
        ),
        cell_lengths=_pick(
            base.cell_lengths,
            extra.cell_lengths,
            base_iterations=base.iterations,
            extra_iterations=extra.iterations,
        ),
        cell_angles=_pick(
            base.cell_angles,
            extra.cell_angles,
            base_iterations=base.iterations,
            extra_iterations=extra.iterations,
        ),
    )


def _union_atom_ids_from_frames(frames_df: list[pd.DataFrame]) -> list[int]:
    """Collect stable atom ids across frames, falling back to 1..max(frame_size)."""
    seen: set[int] = set()
    ordered: list[int] = []
    max_rows = 0
    for fr in frames_df:
        max_rows = max(max_rows, len(fr))
        if "atom_num" in fr.columns:
            vals = pd.to_numeric(fr["atom_num"], errors="coerce").dropna().astype(int).tolist()
            for aid in vals:
                if aid not in seen:
                    seen.add(aid)
                    ordered.append(int(aid))
    if ordered:
        return ordered
    return list(range(1, max_rows + 1))


def _merge_atom_id_lists(primary: list[int], secondary: list[int]) -> list[int]:
    seen: set[int] = set()
    out: list[int] = []
    for aid in primary + secondary:
        a = int(aid)
        if a in seen:
            continue
        seen.add(a)
        out.append(a)
    return out


def _fort7_per_atom_arrays(
    frames_df: list[pd.DataFrame],
    atom_ids: list[int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_frames = len(frames_df)
    n_atoms = len(atom_ids)
    atom_to_idx = {int(a): i for i, a in enumerate(atom_ids)}
    atom_type_nums = np.zeros((n_frames, n_atoms), dtype=int)
    molecule_nums = np.zeros((n_frames, n_atoms), dtype=int)
    num_lone_pairs = np.full((n_frames, n_atoms), np.nan, dtype=float)

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
    if not frames:
        positions = np.empty((0, 0, 3), dtype=float)
        atom_labels = np.empty((0, 0), dtype=object)
        elements: list[str] = []
        atom_ids: list[int] = []
    else:
        n_atoms = max(int(f["coords"].shape[0]) for f in frames)
        atom_ids = list(range(1, n_atoms + 1))
        positions = np.full((n_frames, n_atoms, 3), np.nan, dtype=float)
        atom_labels = np.full((n_frames, n_atoms), "", dtype=object)
        element_map: dict[int, str] = {}

        for fi, frame in enumerate(frames):
            coords = np.asarray(frame.get("coords"), dtype=float)
            if coords.ndim != 2 or coords.shape[1] != 3:
                continue

            raw_types = list(frame.get("atom_types") or [])
            frame_size = min(coords.shape[0], n_atoms)
            positions[fi, :frame_size, :] = coords[:frame_size, :]
            for aj in range(frame_size):
                label = str(raw_types[aj]) if aj < len(raw_types) else ""
                atom_labels[fi, aj] = label
                if label and (aj + 1) not in element_map:
                    element_map[aj + 1] = label

        elements = [element_map.get(aid, "X") for aid in atom_ids]

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
        atom_labels=atom_labels,
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


def _geometry_from_geo_handler(handler: GeoHandler) -> GeometryData:
    """Normalize a ``GeoHandler`` into ``GeometryData``."""
    df = handler.coordinates()
    meta = dict(handler.metadata())
    atom_ids = df["atom_id"].astype(int).tolist() if "atom_id" in df.columns else list(range(1, len(df) + 1))
    elements = df["atom_type"].astype(str).tolist() if "atom_type" in df.columns else []
    cell = handler.cell()
    simulation = SimulationData(
        atom_ids=atom_ids,
        elements=elements,
        num_of_atoms=np.asarray([handler.n_atoms()], dtype=int),
        cell_lengths=np.asarray([[cell["a"], cell["b"], cell["c"]]], dtype=object),
        cell_angles=np.asarray([[cell["alpha"], cell["beta"], cell["gamma"]]], dtype=object),
    )
    return GeometryData(
        coordinates=df,
        atom_ids=atom_ids,
        elements=elements,
        descriptor=str(meta.get("descriptor") or ""),
        remark=str(meta.get("remark") or ""),
        lattice_parameters=cell,
        simulation=simulation,
        metadata=meta,
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

    atom_ids = np.asarray(_union_atom_ids_from_frames(frames_df), dtype=int)
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


def _connectivity_trajectory_from_handlers(
    fort7_handler: Fort7Handler,
    xmolout_handler: XmoloutHandler,
    *,
    summary_simulation: SimulationData | None = None,
    force_field_parameters: ForceFieldParametersData | None = None,
    reporter=None,
) -> ConnectivityTrajectoryData:
    trajectory = _trajectory_from_xmolout_handler(xmolout_handler)
    trajectory.simulation = _merge_simulation_data(trajectory.simulation, summary_simulation)
    connectivity = _connectivity_from_fort7_handler(fort7_handler, reporter=reporter)
    if connectivity.atom_ids is not None and trajectory.atom_ids:
        common_atom_ids = _merge_atom_id_lists(list(trajectory.atom_ids), [int(a) for a in connectivity.atom_ids])
        if common_atom_ids != list(trajectory.atom_ids):
            atom_to_idx = {aid: i for i, aid in enumerate(trajectory.atom_ids)}
            padded_pos = np.full((trajectory.positions.shape[0], len(common_atom_ids), 3), np.nan, dtype=float)
            padded_lbl = np.full((trajectory.positions.shape[0], len(common_atom_ids)), "", dtype=object)
            padded_elements = ["X"] * len(common_atom_ids)
            for new_j, aid in enumerate(common_atom_ids):
                old_j = atom_to_idx.get(aid)
                if old_j is None:
                    continue
                padded_pos[:, new_j, :] = trajectory.positions[:, old_j, :]
                if trajectory.atom_labels is not None and old_j < trajectory.atom_labels.shape[1]:
                    padded_lbl[:, new_j] = trajectory.atom_labels[:, old_j]
                if old_j < len(trajectory.elements):
                    padded_elements[new_j] = trajectory.elements[old_j]
            trajectory.positions = padded_pos
            trajectory.atom_labels = padded_lbl
            trajectory.elements = padded_elements
            trajectory.atom_ids = common_atom_ids
            if trajectory.simulation is not None:
                trajectory.simulation.atom_ids = common_atom_ids
    if trajectory.simulation is not None:
        connectivity.simulation = _merge_simulation_data(trajectory.simulation, connectivity.simulation)
        connectivity.elements = trajectory.elements
        connectivity.atom_ids = trajectory.atom_ids
    if connectivity.iterations is None and trajectory.iterations is not None:
        connectivity.iterations = trajectory.iterations
    return ConnectivityTrajectoryData(
        connectivity=connectivity,
        trajectory=trajectory,
        force_field_parameters=force_field_parameters,
    )


def _force_field_from_ffield_handler(handler: FFieldHandler) -> ForceFieldParametersData:
    """Normalize an ``FFieldHandler`` into ``ForceFieldParametersData``."""
    sections = handler.sections
    meta = handler.metadata()
    section_general = getattr(handler, "SECTION_GENERAL", "general")
    section_atom = getattr(handler, "SECTION_ATOM", "atom")
    section_bond = getattr(handler, "SECTION_BOND", "bond")
    section_off_diagonal = getattr(handler, "SECTION_OFF_DIAGONAL", "off_diagonal")
    section_angle = getattr(handler, "SECTION_ANGLE", "angle")
    section_torsion = getattr(handler, "SECTION_TORSION", "torsion")
    section_hbond = getattr(handler, "SECTION_HBOND", "hbond")

    return ForceFieldParametersData(
        general_parameters=sections.get(section_general, pd.DataFrame()).copy(),
        atom_parameters=sections.get(section_atom, pd.DataFrame()).copy(),
        bond_parameters=sections.get(section_bond, pd.DataFrame()).copy(),
        off_diagonal_parameters=sections.get(section_off_diagonal, pd.DataFrame()).copy(),
        angle_parameters=sections.get(section_angle, pd.DataFrame()).copy(),
        torsion_parameters=sections.get(section_torsion, pd.DataFrame()).copy(),
        hydrogen_bond_parameters=sections.get(section_hbond, pd.DataFrame()).copy(),
        source="reaxff/ffield",
        metadata=dict(meta),
    )


def _force_field_optimization_from_fort13_handler(handler: Fort13Handler) -> ForceFieldOptimizationProgressData:
    """Normalize a ``Fort13Handler`` into ``ForceFieldOptimizationProgressData``."""
    df = handler.dataframe().copy()
    epochs = df["epoch"].to_numpy(dtype=int) if "epoch" in df.columns else np.arange(len(df), dtype=int)
    total_ff_error = (
        df["total_ff_error"].to_numpy(dtype=float)
        if "total_ff_error" in df.columns
        else np.empty((len(df),), dtype=float)
    )
    return ForceFieldOptimizationProgressData(
        epochs=epochs,
        total_ff_error=total_ff_error,
        metadata=dict(handler.metadata()),
    )


def _parameter_optimization_diagnostic_from_fort79_handler(
    handler: Fort79Handler,
) -> ForceFieldOptimizationDiagnosticData:
    """Normalize a ``Fort79Handler`` into ``ForceFieldOptimizationDiagnosticData``."""
    df = handler.dataframe().copy()
    return ForceFieldOptimizationDiagnosticData(
        identifiers=(
            df["identifier"].fillna("").to_numpy(dtype=object)
            if "identifier" in df.columns
            else np.empty((0,), dtype=object)
        ),
        value1=(df["value1"].to_numpy(dtype=float) if "value1" in df.columns else np.empty((len(df),), dtype=float)),
        value2=(df["value2"].to_numpy(dtype=float) if "value2" in df.columns else np.empty((len(df),), dtype=float)),
        value3=(df["value3"].to_numpy(dtype=float) if "value3" in df.columns else np.empty((len(df),), dtype=float)),
        diff1=(df["diff1"].to_numpy(dtype=float) if "diff1" in df.columns else np.empty((len(df),), dtype=float)),
        diff2=(df["diff2"].to_numpy(dtype=float) if "diff2" in df.columns else np.empty((len(df),), dtype=float)),
        diff3=(df["diff3"].to_numpy(dtype=float) if "diff3" in df.columns else np.empty((len(df),), dtype=float)),
        a=(df["a"].to_numpy(dtype=float) if "a" in df.columns else np.empty((len(df),), dtype=float)),
        b=(df["b"].to_numpy(dtype=float) if "b" in df.columns else np.empty((len(df),), dtype=float)),
        c=(df["c"].to_numpy(dtype=float) if "c" in df.columns else np.empty((len(df),), dtype=float)),
        parabol_min=(
            df["parabol_min"].to_numpy(dtype=float)
            if "parabol_min" in df.columns
            else np.empty((len(df),), dtype=float)
        ),
        parabol_min_diff=(
            df["parabol_min_diff"].to_numpy(dtype=float)
            if "parabol_min_diff" in df.columns
            else np.empty((len(df),), dtype=float)
        ),
        value4=(df["value4"].to_numpy(dtype=float) if "value4" in df.columns else np.empty((len(df),), dtype=float)),
        diff4=(df["diff4"].to_numpy(dtype=float) if "diff4" in df.columns else np.empty((len(df),), dtype=float)),
        metadata=dict(handler.metadata()),
    )


def _force_field_optimization_report_from_fort99_handler(
    handler: Fort99Handler,
) -> ForceFieldOptimizationReportData:
    """Normalize a ``Fort99Handler`` into ``ForceFieldOptimizationReportData``."""
    df = handler.dataframe().copy()
    n_rows = len(df)
    return ForceFieldOptimizationReportData(
        linenos=(df["lineno"].to_numpy(dtype=int) if "lineno" in df.columns else np.arange(1, n_rows + 1, dtype=int)),
        sections=(
            df["section"].fillna("").to_numpy(dtype=object)
            if "section" in df.columns
            else np.full((n_rows,), "", dtype=object)
        ),
        titles=(
            df["title"].fillna("").to_numpy(dtype=object)
            if "title" in df.columns
            else np.full((n_rows,), "", dtype=object)
        ),
        ffield_values=(
            df["ffield_value"].to_numpy(dtype=float)
            if "ffield_value" in df.columns
            else np.full((n_rows,), np.nan, dtype=float)
        ),
        qm_values=(
            df["qm_value"].to_numpy(dtype=float)
            if "qm_value" in df.columns
            else np.full((n_rows,), np.nan, dtype=float)
        ),
        weights=(
            df["weight"].to_numpy(dtype=float)
            if "weight" in df.columns
            else np.full((n_rows,), np.nan, dtype=float)
        ),
        errors=(
            df["error"].to_numpy(dtype=float)
            if "error" in df.columns
            else np.full((n_rows,), np.nan, dtype=float)
        ),
        total_ff_error=(
            df["total_ff_error"].to_numpy(dtype=float)
            if "total_ff_error" in df.columns
            else np.full((n_rows,), np.nan, dtype=float)
        ),
        metadata=dict(handler.metadata()),
    )


def _force_field_optimization_training_set_from_trainset_handler(
    handler: TrainsetHandler,
) -> ForceFieldOptimizationTrainingSetData:
    """Normalize a ``TrainsetHandler`` into ``ForceFieldOptimizationTrainingSetData``."""
    meta = dict(handler.metadata())
    tables = dict(meta.get("tables", {}))
    return ForceFieldOptimizationTrainingSetData(
        sections=tuple(str(s) for s in meta.get("sections", [])),
        charge=tables.get("CHARGE", pd.DataFrame()).copy(),
        heatfo=tables.get("HEATFO", pd.DataFrame()).copy(),
        geometry=tables.get("GEOMETRY", pd.DataFrame()).copy(),
        cell_parameters=tables.get("CELL_PARAMETERS", pd.DataFrame()).copy(),
        energy=tables.get("ENERGY", pd.DataFrame()).copy(),
        metadata=meta,
    )


def _force_field_optimization_parameters_from_params_handler(
    handler: ParamsHandler,
) -> ForceFieldOptimizationParameterData:
    """Normalize a ``ParamsHandler`` into ``ForceFieldOptimizationParameterData``."""
    df = handler.dataframe().copy()
    n_rows = len(df)
    return ForceFieldOptimizationParameterData(
        ff_section=(
            df["ff_section"].to_numpy(dtype=int)
            if "ff_section" in df.columns
            else np.empty((n_rows,), dtype=int)
        ),
        ff_section_line=(
            df["ff_section_line"].to_numpy(dtype=int)
            if "ff_section_line" in df.columns
            else np.empty((n_rows,), dtype=int)
        ),
        ff_parameter=(
            df["ff_parameter"].to_numpy(dtype=int)
            if "ff_parameter" in df.columns
            else np.empty((n_rows,), dtype=int)
        ),
        search_interval=(
            df["search_interval"].to_numpy(dtype=float)
            if "search_interval" in df.columns
            else np.full((n_rows,), np.nan, dtype=float)
        ),
        min_value=(
            df["min_value"].to_numpy(dtype=float)
            if "min_value" in df.columns
            else np.full((n_rows,), np.nan, dtype=float)
        ),
        max_value=(
            df["max_value"].to_numpy(dtype=float)
            if "max_value" in df.columns
            else np.full((n_rows,), np.nan, dtype=float)
        ),
        inline_comment=(
            df["inline_comment"].fillna("").to_numpy(dtype=object)
            if "inline_comment" in df.columns
            else np.full((n_rows,), "", dtype=object)
        ),
        metadata=dict(handler.metadata()),
    )


def _partial_energy_from_energy_log_handler(handler: Fort73Handler) -> PartialEnergyData:
    """Normalize a partial-energy log handler into ``PartialEnergyData``."""
    df = handler.dataframe().copy()
    iterations = df["iter"].to_numpy(dtype=int) if "iter" in df.columns else np.arange(len(df), dtype=int)
    components = tuple(str(c) for c in df.columns if str(c) != "iter")
    values = (
        df.loc[:, list(components)].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
        if components
        else np.empty((len(df), 0), dtype=float)
    )
    # The same tabular partial-energy schema may come from fort.73 or energylog for MD,
    # and from fort.58 for MM optimization/minimization outputs.
    return PartialEnergyData(
        iterations=iterations,
        components=components,
        values=values,
        metadata=dict(handler.metadata()),
    )


def _structure_summary_from_fort74_handler(handler: Fort74Handler) -> GeometrySummaryData:
    """Normalize a ``Fort74Handler`` into ``GeometrySummaryData``."""
    df = handler.dataframe().copy()
    return GeometrySummaryData(
        identifiers=df["identifier"].fillna("").to_numpy(dtype=object) if "identifier" in df.columns else np.empty((0,), dtype=object),
        minimum_energy=(df["Emin"].to_numpy(dtype=float) if "Emin" in df.columns else None),
        iterations=(df["iter"].to_numpy(dtype=float) if "iter" in df.columns else None),
        formation_energy=(df["Hf"].to_numpy(dtype=float) if "Hf" in df.columns else None),
        volume=(df["V"].to_numpy(dtype=float) if "V" in df.columns else None),
        density=(df["D"].to_numpy(dtype=float) if "D" in df.columns else None),
        metadata=dict(handler.metadata()),
    )


def _restraint_from_fort76_handler(handler: Fort76Handler) -> RestraintData:
    """Normalize a ``Fort76Handler`` into ``RestraintData``."""
    df = handler.dataframe().copy()
    n_restraints = int(handler.metadata().get("n_restraints", 0))
    target_cols = [f"r{i}_target" for i in range(1, n_restraints + 1) if f"r{i}_target" in df.columns]
    actual_cols = [f"r{i}_actual" for i in range(1, n_restraints + 1) if f"r{i}_actual" in df.columns]
    return RestraintData(
        iterations=(df["iter"].to_numpy(dtype=int) if "iter" in df.columns else np.arange(len(df), dtype=int)),
        restraint_energy=(df["E_res"].to_numpy(dtype=float) if "E_res" in df.columns else None),
        potential_energy=(df["E_pot"].to_numpy(dtype=float) if "E_pot" in df.columns else None),
        target_values=(
            df.loc[:, target_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
            if target_cols
            else np.empty((len(df), 0), dtype=float)
        ),
        actual_values=(
            df.loc[:, actual_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
            if actual_cols
            else np.empty((len(df), 0), dtype=float)
        ),
        metadata=dict(handler.metadata()),
    )


def _geometry_optimization_from_fort57_handler(handler: Fort57Handler) -> GeometryOptimizationProgressData:
    """Normalize a ``Fort57Handler`` into ``GeometryOptimizationProgressData``."""
    df = handler.dataframe().copy()
    optimization_iterations = (
        df["iter"].to_numpy(dtype=int) if "iter" in df.columns else np.arange(len(df), dtype=int)
    )
    return GeometryOptimizationProgressData(
        optimization_iterations=optimization_iterations,
        potential_energy=(df["E_pot"].to_numpy(dtype=float) if "E_pot" in df.columns else None),
        temperature=(df["T"].to_numpy(dtype=float) if "T" in df.columns else None),
        temperature_setpoint=(df["T_set"].to_numpy(dtype=float) if "T_set" in df.columns else None),
        rms_gradient=(df["RMSG"].to_numpy(dtype=float) if "RMSG" in df.columns else None),
        n_force_calls=(df["nfc"].to_numpy(dtype=int) if "nfc" in df.columns else None),
        geo_descriptor=str(handler.geo_descriptor),
        metadata=dict(handler.metadata()),
    )


def _control_parameters_from_control_handler(handler: ControlHandler) -> ControlParametersData:
    """Normalize a ``ControlHandler`` into ``ControlParametersData``."""
    return ControlParametersData(
        general=dict(handler.general_parameters),
        md=dict(handler.md_parameters),
        mm=dict(handler.mm_parameters),
        ff=dict(handler.ff_parameters),
        outdated=dict(handler.outdated_parameters),
        metadata=dict(handler.metadata()),
    )


def _atomic_kinematics_from_vels_handler(handler: VelsHandler) -> AtomicKinematicsData:
    """Normalize a ``VelsHandler`` into ``AtomicKinematicsData``."""
    meta = dict(handler.metadata())
    return AtomicKinematicsData(
        coordinates=handler.section_df(handler.SECTION_COORDS).copy(),
        velocities=handler.section_df(handler.SECTION_VELS).copy(),
        accelerations=handler.section_df(handler.SECTION_ACCELS).copy(),
        previous_accelerations=handler.section_df(handler.SECTION_PREV_ACCELS).copy(),
        lattice_parameters=meta.get("lattice_parameters"),
        md_temperature_K=meta.get("md_temperature_K"),
        metadata=meta,
    )


def _eregime_from_handler(handler: EregimeHandler) -> EregimeData:
    """Normalize an ``EregimeHandler`` into ``EregimeData``."""
    df = handler.dataframe().copy()
    iterations = df["iter"].to_numpy(dtype=int) if "iter" in df.columns else np.arange(len(df), dtype=int)
    field_zones = (
        df["field_zones"].to_numpy(dtype=int)
        if "field_zones" in df.columns
        else np.ones((len(df),), dtype=int)
    )
    field_dir_col = "field_dir" if "field_dir" in df.columns else "field_dir1"
    field_col = "field" if "field" in df.columns else "field1"
    return EregimeData(
        iterations=iterations,
        field_zones=field_zones,
        field_dir=(
            df[field_dir_col].fillna("").to_numpy(dtype=object)
            if field_dir_col in df.columns
            else np.full((len(df),), "", dtype=object)
        ),
        field=(
            pd.to_numeric(df[field_col], errors="coerce").to_numpy(dtype=float)
            if field_col in df.columns
            else np.full((len(df),), np.nan, dtype=float)
        ),
        metadata=dict(handler.metadata()),
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

    discovered_atom_ids = _union_atom_ids_from_frames(frames_df)
    if simulation is not None and simulation.atom_ids:
        atom_ids = _merge_atom_id_lists([int(a) for a in simulation.atom_ids], discovered_atom_ids)
    else:
        atom_ids = discovered_atom_ids
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
        q = np.full((n_atoms,), np.nan, dtype=float)
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
        volume=(df["V"].to_numpy(dtype=float) if "V" in df.columns else None),
        temperature=(df["T"].to_numpy(dtype=float) if "T" in df.columns else None),
        pressure=(df["P"].to_numpy(dtype=float) if "P" in df.columns else None),
        density=(df["D"].to_numpy(dtype=float) if "D" in df.columns else None),
        elapsed_time=(df["elap_time"].to_numpy(dtype=float) if "elap_time" in df.columns else None),
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

    @staticmethod
    def _resolve_reaxff_path(args: dict, *keys: str, default: str) -> Path:
        for key in keys:
            raw = args.get(key)
            if raw:
                path = Path(raw)
                return path / default if path.is_dir() else path

        run_dir = args.get("run_dir")
        if run_dir:
            return Path(run_dir) / default

        input_path = args.get("input")
        if input_path:
            path = Path(input_path)
            return path / default if path.is_dir() else path

        return Path(default)

    @staticmethod
    def _resolve_against_run_dir(args: dict, path: Path) -> Path:
        run_dir = args.get("run_dir")
        if run_dir and not path.is_absolute():
            candidate = Path(run_dir) / path
            if candidate.exists():
                return candidate
        return path

    @staticmethod
    def _quick_n_frames_from_control(control_path: Path) -> int | None:
        if not control_path.exists() or not control_path.is_file():
            return None
        nmdit: int | None = None
        iout2: int | None = None
        try:
            with open(control_path, "r", encoding="utf-8", errors="replace") as fh:
                for raw in fh:
                    line = raw.split("#", 1)[0].strip()
                    if not line:
                        continue
                    parts = line.split()
                    if len(parts) < 2:
                        continue
                    # ReaxFF control commonly uses "<value> <key>" but accept both orders.
                    for key, value in ((parts[1].lower(), parts[0]), (parts[0].lower(), parts[1])):
                        if key not in {"nmdit", "iout2"}:
                            continue
                        try:
                            parsed = int(float(value))
                        except Exception:
                            continue
                        if key == "nmdit":
                            nmdit = parsed
                        else:
                            iout2 = parsed
                    if nmdit is not None and iout2 is not None and iout2 > 0:
                        break
        except Exception:
            return None
        if nmdit is None or iout2 is None or iout2 <= 0:
            return None
        return max(1, int(nmdit / iout2) + 1)

    @staticmethod
    def _quick_n_frames_from_geo_xmol(geo_path: Path, xmol_path: Path) -> int | None:
        if not xmol_path.exists() or not xmol_path.is_file():
            return None
        descriptor = ""
        if geo_path.exists() and geo_path.is_file():
            try:
                from reaxkit.engine.reaxff.io.geo_handler import GeoHandler

                descriptor = str(GeoHandler(geo_path).metadata().get("descriptor") or "").strip()
            except Exception:
                descriptor = ""
        iterations: set[int] = set()
        try:
            with open(xmol_path, "r", encoding="utf-8", errors="replace") as fh:
                for raw in fh:
                    if descriptor and descriptor not in raw:
                        continue
                    vals = raw.strip().split()
                    if len(vals) != 9:
                        continue
                    try:
                        iterations.add(int(float(vals[1])))
                    except Exception:
                        continue
        except Exception:
            return None
        return len(iterations) if iterations else None

    @classmethod
    def quick_n_frames(cls, args: dict) -> int | None:
        """Fast frame-count probe for Web UI metadata updates."""
        control_path = cls._resolve_reaxff_path(args, "control", "control_file", default="control")
        control_path = cls._resolve_against_run_dir(args, control_path)
        n_from_control = cls._quick_n_frames_from_control(control_path)
        if n_from_control is not None:
            return n_from_control

        geo_raw = args.get("geo") or args.get("geometry") or args.get("run_dir") or args.get("input") or "geo"
        geo_path = Path(geo_raw)
        geo_path = geo_path / "geo" if geo_path.is_dir() else geo_path
        geo_path = cls._resolve_against_run_dir(args, geo_path)

        xmol_path = cls._resolve_reaxff_path(args, "xmolout", default="xmolout")
        xmol_path = cls._resolve_against_run_dir(args, xmol_path)
        return cls._quick_n_frames_from_geo_xmol(geo_path, xmol_path)

    def detect(self, path: str | Path) -> float:
        p = Path(path)
        if p.is_dir():
            if (p / "xmolout").exists() or (p / "fort.7").exists():
                return 0.95
            return 0.0
        if p.is_file():
            lower_name = p.name.lower()
            if "xmolout" in lower_name or p.name == "fort.7":
                return 0.95
        return 0.0

    def required_input_files(self, data_type, args: dict) -> tuple[str, ...] | None:
        mapping: dict[object, tuple[str, ...]] = {
            TrajectoryData: ("xmolout", "summary.txt"),
            GeometryData: ("geo",),
            SimulationData: ("xmolout", "summary.txt"),
            ConnectivityData: ("fort.7", "xmolout", "summary.txt"),
            ConnectivityTrajectoryData: ("fort.7", "xmolout", "summary.txt", "ffield"),
            CoordinationStatusBundleData: ("fort.7", "xmolout", "summary.txt", "ffield"),
            ChargeData: ("fort.7", "xmolout", "summary.txt"),
            ElectrostaticsData: ("xmolout", "fort.7", "summary.txt"),
            AtomicKinematicsData: ("vels",),
            ElectricFieldData: ("fort.78",),
            EregimeData: ("eregime.in",),
            ForceFieldParametersData: ("ffield",),
            ForceFieldOptimizationProgressData: ("fort.13",),
            ForceFieldOptimizationTrainingSetData: ("trainset.in",),
            ForceFieldOptimizationData: ("ffield", "params"),
            ForceFieldOptimizationParameterBundleData: ("params", "ffield"),
            ForceFieldOptimizationDiagnosticBundleData: ("fort.79", "ffield"),
            ForceFieldOptimizationReportEOSBundleData: ("fort.99", "fort.74"),
            ForceFieldOptimizationParameterData: ("params",),
            ForceFieldOptimizationReportData: ("fort.99",),
            ForceFieldOptimizationDiagnosticData: ("fort.79",),
            GeometrySummaryData: ("fort.74",),
            PartialEnergyData: ("fort.73", "energylog", "fort.58"),
            RestraintData: ("fort.76",),
            GeometryOptimizationProgressData: ("fort.57",),
            ControlParametersData: ("control",),
            MolecularAnalysisData: ("molfra.out", "molfra_ig.out"),
        }
        if data_type is ElectrostaticsData and str(args.get("command") or "").strip().lower() == "hyst":
            return ("xmolout", "fort.7", "fort.78", "summary.txt")
        return mapping.get(data_type)

    @staticmethod
    def _emit_load_timing(
        args: dict,
        *,
        handler: str,
        source_path: Path | str | None,
        seconds: float,
    ) -> None:
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

    @classmethod
    def _build_handler(
        cls,
        args: dict,
        *,
        handler_name: str,
        source_path: Path | str | None,
        factory,
    ):
        _ = (args, handler_name, source_path)
        return factory()

    @classmethod
    def _time_source(
        cls,
        args: dict,
        *,
        handler_name: str,
        source_path: Path | str | None,
        loader,
    ):
        t0 = perf_counter()
        out = loader()
        cls._emit_load_timing(args, handler=handler_name, source_path=source_path, seconds=perf_counter() - t0)
        return out

    def load_trajectory(self, args: dict, reporter=None) -> TrajectoryData:
        from reaxkit.engine.reaxff.io.xmolout_handler import XmoloutHandler

        xmol_path = self._resolve_reaxff_path(args, "xmolout", default="xmolout")
        handler = self._build_handler(
            args,
            handler_name="XmoloutHandler",
            source_path=xmol_path,
            factory=lambda: XmoloutHandler(xmol_path, reporter=reporter),
        )
        trj = self._time_source(
            args,
            handler_name="XmoloutHandler",
            source_path=xmol_path,
            loader=lambda: _trajectory_from_xmolout_handler(handler),
        )
        trj.simulation = _merge_simulation_data(
            trj.simulation,
            self._load_simulation_from_summary(args, reporter=reporter),
        )
        return trj

    def load_geometry(self, args: dict, reporter=None) -> GeometryData:
        from reaxkit.engine.reaxff.io.geo_handler import GeoHandler

        raw = args.get("geo") or args.get("geometry") or args.get("input") or "geo"
        p = Path(raw)
        geo_path = p / "geo" if p.is_dir() else p
        handler = self._build_handler(
            args,
            handler_name="GeoHandler",
            source_path=geo_path,
            factory=lambda: GeoHandler(geo_path, reporter=reporter),
        )
        return self._time_source(
            args,
            handler_name="GeoHandler",
            source_path=geo_path,
            loader=lambda: _geometry_from_geo_handler(handler),
        )

    def load_simulation(self, args: dict, reporter=None) -> SimulationData:
        sim = self._load_simulation_from_xmolout(args, reporter=reporter)
        sim = _merge_simulation_data(sim, self._load_simulation_from_summary(args, reporter=reporter))
        if sim is None:
            raise FileNotFoundError("SimulationData for reaxff currently requires xmolout or summary.txt.")
        return sim

    @classmethod
    def _load_simulation_from_xmolout(cls, args: dict, reporter=None) -> SimulationData | None:
        from reaxkit.engine.reaxff.io.xmolout_handler import XmoloutHandler

        xmol_path = cls._resolve_reaxff_path(args, "xmolout", default="xmolout")
        if not xmol_path.exists():
            return None
        handler = cls._build_handler(
            args,
            handler_name="XmoloutHandler",
            source_path=xmol_path,
            factory=lambda: XmoloutHandler(xmol_path, reporter=reporter),
        )
        trj = cls._time_source(
            args,
            handler_name="XmoloutHandler",
            source_path=xmol_path,
            loader=lambda: _trajectory_from_xmolout_handler(handler),
        )
        return trj.simulation

    @classmethod
    def _load_simulation_from_summary(cls, args: dict, reporter=None) -> SimulationData | None:
        from reaxkit.engine.reaxff.io.summary_handler import SummaryHandler

        candidates = [args.get("summary"), args.get("xmolout"), args.get("run_dir"), args.get("input")]
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
        handler = cls._build_handler(
            args,
            handler_name="SummaryHandler",
            source_path=summary_path,
            factory=lambda: SummaryHandler(summary_path, reporter=reporter),
        )
        return cls._time_source(
            args,
            handler_name="SummaryHandler",
            source_path=summary_path,
            loader=lambda: _simulation_from_summary_handler(handler),
        )

    def load_connectivity(self, args: dict, reporter=None) -> ConnectivityData:
        from reaxkit.engine.reaxff.io.fort7_handler import Fort7Handler

        raw = args.get("fort7") or args.get("connectivity") or args.get("input") or "fort.7"
        p = Path(raw)
        fort7_path = p / "fort.7" if p.is_dir() else p
        handler = self._build_handler(
            args,
            handler_name="Fort7Handler",
            source_path=fort7_path,
            factory=lambda: Fort7Handler(fort7_path, reporter=reporter),
        )
        conn = self._time_source(
            args,
            handler_name="Fort7Handler",
            source_path=fort7_path,
            loader=lambda: _connectivity_from_fort7_handler(handler, reporter=reporter),
        )
        sim = _merge_simulation_data(
            self._load_simulation_from_xmolout(args, reporter=reporter),
            self._load_simulation_from_summary(args, reporter=reporter),
        )
        if sim is not None:
            conn.simulation = _merge_simulation_data(sim, conn.simulation)
            conn.elements = conn.simulation.elements
            conn.atom_ids = conn.simulation.atom_ids
        return conn

    def load_coordination_status_bundle(self, args: dict, reporter=None) -> CoordinationStatusBundleData:
        return CoordinationStatusBundleData(
            connectivity=self.load_connectivity(args, reporter=reporter),
            force_field_parameters=self.load_force_field(args, reporter=reporter),
        )

    def load_connectivity_trajectory(self, args: dict, reporter=None) -> ConnectivityTrajectoryData:
        from reaxkit.engine.reaxff.io.fort7_handler import Fort7Handler
        from reaxkit.engine.reaxff.io.xmolout_handler import XmoloutHandler

        fort7_raw = args.get("fort7") or args.get("connectivity") or args.get("input") or "fort.7"
        fort7_path = Path(fort7_raw)
        fort7_path = fort7_path / "fort.7" if fort7_path.is_dir() else fort7_path

        xmol_path = self._resolve_reaxff_path(args, "xmolout", default="xmolout")
        fort7_handler = self._build_handler(
            args,
            handler_name="Fort7Handler",
            source_path=fort7_path,
            factory=lambda: Fort7Handler(fort7_path, reporter=reporter),
        )
        xmol_handler = self._build_handler(
            args,
            handler_name="XmoloutHandler",
            source_path=xmol_path,
            factory=lambda: XmoloutHandler(xmol_path, reporter=reporter),
        )
        summary_simulation = self._load_simulation_from_summary(args, reporter=reporter)
        force_field_parameters: ForceFieldParametersData | None = None
        try:
            ff_args = dict(args)
            if not ff_args.get("ffield"):
                ff_args["ffield"] = str(
                    self._resolve_reaxff_path(
                        args,
                        "ffield",
                        "force_field",
                        "atom_reference",
                        default="ffield",
                    )
                )
            force_field_parameters = self.load_force_field(ff_args, reporter=reporter)
        except Exception:
            force_field_parameters = None
        return _connectivity_trajectory_from_handlers(
            fort7_handler,
            xmol_handler,
            summary_simulation=summary_simulation,
            force_field_parameters=force_field_parameters,
            reporter=reporter,
        )

    def load_force_field(self, args: dict, reporter=None) -> ForceFieldParametersData:
        from reaxkit.engine.reaxff.io.ffield_handler import FFieldHandler

        raw = args.get("ffield") or args.get("force_field") or args.get("atom_reference") or args.get("input") or "ffield"
        p = Path(raw)
        ffield_path = p / "ffield" if p.is_dir() else p
        handler = self._build_handler(
            args,
            handler_name="FFieldHandler",
            source_path=ffield_path,
            factory=lambda: FFieldHandler(ffield_path, reporter=reporter),
        )
        return self._time_source(
            args,
            handler_name="FFieldHandler",
            source_path=ffield_path,
            loader=lambda: _force_field_from_ffield_handler(handler),
        )

    def load_force_field_optimization(self, args: dict, reporter=None) -> ForceFieldOptimizationProgressData:
        from reaxkit.engine.reaxff.io.fort13_handler import Fort13Handler

        raw = args.get("fort13") or args.get("force_field_optimization") or args.get("input") or "fort.13"
        p = Path(raw)
        fort13_path = p / "fort.13" if p.is_dir() else p
        handler = self._build_handler(
            args,
            handler_name="Fort13Handler",
            source_path=fort13_path,
            factory=lambda: Fort13Handler(fort13_path, reporter=reporter),
        )
        return self._time_source(
            args,
            handler_name="Fort13Handler",
            source_path=fort13_path,
            loader=lambda: _force_field_optimization_from_fort13_handler(handler),
        )

    def load_force_field_optimization_report(self, args: dict, reporter=None) -> ForceFieldOptimizationReportData:
        from reaxkit.engine.reaxff.io.fort99_handler import Fort99Handler

        raw = args.get("fort99") or args.get("force_field_optimization_report") or args.get("input") or "fort.99"
        p = Path(raw)
        fort99_path = p / "fort.99" if p.is_dir() else p
        handler = self._build_handler(
            args,
            handler_name="Fort99Handler",
            source_path=fort99_path,
            factory=lambda: Fort99Handler(fort99_path, reporter=reporter),
        )
        return self._time_source(
            args,
            handler_name="Fort99Handler",
            source_path=fort99_path,
            loader=lambda: _force_field_optimization_report_from_fort99_handler(handler),
        )

    def load_force_field_optimization_training_set(
        self,
        args: dict,
        reporter=None,
    ) -> ForceFieldOptimizationTrainingSetData:
        from reaxkit.engine.reaxff.io.trainset_handler import TrainsetHandler

        raw = args.get("trainset") or args.get("force_field_optimization_training_set") or args.get("input") or "trainset.in"
        p = Path(raw)
        trainset_path = p / "trainset.in" if p.is_dir() else p
        handler = self._build_handler(
            args,
            handler_name="TrainsetHandler",
            source_path=trainset_path,
            factory=lambda: TrainsetHandler(trainset_path, reporter=reporter),
        )
        return self._time_source(
            args,
            handler_name="TrainsetHandler",
            source_path=trainset_path,
            loader=lambda: _force_field_optimization_training_set_from_trainset_handler(handler),
        )

    def load_force_field_optimization_parameters(
        self,
        args: dict,
        reporter=None,
    ) -> ForceFieldOptimizationParameterData:
        from reaxkit.engine.reaxff.io.params_handler import ParamsHandler

        raw = args.get("params") or args.get("force_field_optimization_parameters") or args.get("input") or "params"
        p = Path(raw)
        params_path = p / "params" if p.is_dir() else p
        handler = self._build_handler(
            args,
            handler_name="ParamsHandler",
            source_path=params_path,
            factory=lambda: ParamsHandler(params_path, reporter=reporter),
        )
        return self._time_source(
            args,
            handler_name="ParamsHandler",
            source_path=params_path,
            loader=lambda: _force_field_optimization_parameters_from_params_handler(handler),
        )

    def load_force_field_optimization_data(
        self,
        args: dict,
        reporter=None,
    ) -> ForceFieldOptimizationData:
        return ForceFieldOptimizationData(
            force_field_parameters=self.load_force_field(args, reporter=reporter),
            optimization_parameters=self.load_force_field_optimization_parameters(args, reporter=reporter),
        )

    def load_force_field_optimization_parameter_bundle(
        self,
        args: dict,
        reporter=None,
    ) -> ForceFieldOptimizationParameterBundleData:
        return ForceFieldOptimizationParameterBundleData(
            optimization_parameters=self.load_force_field_optimization_parameters(args, reporter=reporter),
            force_field_parameters=self.load_force_field(args, reporter=reporter),
        )

    def load_parameter_optimization_diagnostic(
        self,
        args: dict,
        reporter=None,
    ) -> ForceFieldOptimizationDiagnosticData:
        from reaxkit.engine.reaxff.io.fort79_handler import Fort79Handler

        raw = args.get("fort79") or args.get("parameter_optimization_diagnostic") or args.get("input") or "fort.79"
        p = Path(raw)
        fort79_path = p / "fort.79" if p.is_dir() else p
        handler = self._build_handler(
            args,
            handler_name="Fort79Handler",
            source_path=fort79_path,
            factory=lambda: Fort79Handler(fort79_path, reporter=reporter),
        )
        return self._time_source(
            args,
            handler_name="Fort79Handler",
            source_path=fort79_path,
            loader=lambda: _parameter_optimization_diagnostic_from_fort79_handler(handler),
        )

    def load_parameter_optimization_diagnostic_bundle(
        self,
        args: dict,
        reporter=None,
    ) -> ForceFieldOptimizationDiagnosticBundleData:
        return ForceFieldOptimizationDiagnosticBundleData(
            diagnostics=self.load_parameter_optimization_diagnostic(args, reporter=reporter),
            force_field_parameters=self.load_force_field(args, reporter=reporter),
        )

    def load_force_field_optimization_report_eos_bundle(
        self,
        args: dict,
        reporter=None,
    ) -> ForceFieldOptimizationReportEOSBundleData:
        return ForceFieldOptimizationReportEOSBundleData(
            report=self.load_force_field_optimization_report(args, reporter=reporter),
            geometry_summary=self.load_structure_summary(args, reporter=reporter),
        )

    def load_structure_summary(self, args: dict, reporter=None) -> GeometrySummaryData:
        from reaxkit.engine.reaxff.io.fort74_handler import Fort74Handler

        raw = args.get("fort74") or args.get("structure_summary") or args.get("input") or "fort.74"
        p = Path(raw)
        fort74_path = p / "fort.74" if p.is_dir() else p
        handler = self._build_handler(
            args,
            handler_name="Fort74Handler",
            source_path=fort74_path,
            factory=lambda: Fort74Handler(fort74_path, reporter=reporter),
        )
        return self._time_source(
            args,
            handler_name="Fort74Handler",
            source_path=fort74_path,
            loader=lambda: _structure_summary_from_fort74_handler(handler),
        )

    def load_partial_energy(self, args: dict, reporter=None) -> PartialEnergyData:
        from reaxkit.engine.reaxff.io.fort73_handler import Fort73Handler

        raw = args.get("fort73") or args.get("partial_energy") or args.get("input") or "fort.73"
        p = Path(raw)
        if p.is_dir():
            candidates = [p / "fort.73", p / "energylog", p / "fort.58"]
            partial_energy_path = next((candidate for candidate in candidates if candidate.exists()), candidates[0])
        else:
            partial_energy_path = p
        handler = self._build_handler(
            args,
            handler_name="Fort73Handler",
            source_path=partial_energy_path,
            factory=lambda: Fort73Handler(partial_energy_path, reporter=reporter),
        )
        return self._time_source(
            args,
            handler_name="Fort73Handler",
            source_path=partial_energy_path,
            loader=lambda: _partial_energy_from_energy_log_handler(handler),
        )

    def load_restraints(self, args: dict, reporter=None) -> RestraintData:
        from reaxkit.engine.reaxff.io.fort76_handler import Fort76Handler

        raw = args.get("fort76") or args.get("restraints") or args.get("input") or "fort.76"
        p = Path(raw)
        fort76_path = p / "fort.76" if p.is_dir() else p
        handler = self._build_handler(
            args,
            handler_name="Fort76Handler",
            source_path=fort76_path,
            factory=lambda: Fort76Handler(fort76_path, reporter=reporter),
        )
        return self._time_source(
            args,
            handler_name="Fort76Handler",
            source_path=fort76_path,
            loader=lambda: _restraint_from_fort76_handler(handler),
        )

    def load_geometry_optimization(self, args: dict, reporter=None) -> GeometryOptimizationProgressData:
        from reaxkit.engine.reaxff.io.fort57_handler import Fort57Handler

        raw = args.get("fort57") or args.get("geometry_optimization") or args.get("input") or "fort.57"
        p = Path(raw)
        fort57_path = p / "fort.57" if p.is_dir() else p
        handler = self._build_handler(
            args,
            handler_name="Fort57Handler",
            source_path=fort57_path,
            factory=lambda: Fort57Handler(fort57_path, reporter=reporter),
        )
        return self._time_source(
            args,
            handler_name="Fort57Handler",
            source_path=fort57_path,
            loader=lambda: _geometry_optimization_from_fort57_handler(handler),
        )

    def load_control_parameters(self, args: dict, reporter=None) -> ControlParametersData:
        from reaxkit.engine.reaxff.io.control_handler import ControlHandler

        raw = args.get("control") or args.get("control_file") or args.get("input") or "control"
        p = Path(raw)
        control_path = p / "control" if p.is_dir() else p
        handler = self._build_handler(
            args,
            handler_name="ControlHandler",
            source_path=control_path,
            factory=lambda: ControlHandler(control_path, reporter=reporter),
        )
        return self._time_source(
            args,
            handler_name="ControlHandler",
            source_path=control_path,
            loader=lambda: _control_parameters_from_control_handler(handler),
        )

    def load_eregime(self, args: dict, reporter=None) -> EregimeData:
        from reaxkit.engine.reaxff.io.eregime_handler import EregimeHandler

        raw = args.get("eregime") or args.get("eregime_file") or args.get("input") or "eregime.in"
        p = Path(raw)
        eregime_path = p / "eregime.in" if p.is_dir() else p
        handler = self._build_handler(
            args,
            handler_name="EregimeHandler",
            source_path=eregime_path,
            factory=lambda: EregimeHandler(eregime_path, reporter=reporter),
        )
        return self._time_source(
            args,
            handler_name="EregimeHandler",
            source_path=eregime_path,
            loader=lambda: _eregime_from_handler(handler),
        )

    def load_charges(self, args: dict, reporter=None) -> ChargeData:
        from reaxkit.engine.reaxff.io.fort7_handler import Fort7Handler

        raw = args.get("fort7") or args.get("charges") or args.get("input") or "fort.7"
        p = Path(raw)
        fort7_path = p / "fort.7" if p.is_dir() else p
        handler = self._build_handler(
            args,
            handler_name="Fort7Handler",
            source_path=fort7_path,
            factory=lambda: Fort7Handler(fort7_path, reporter=reporter),
        )
        sim = _merge_simulation_data(
            self._load_simulation_from_xmolout(args, reporter=reporter),
            self._load_simulation_from_summary(args, reporter=reporter),
        )
        return self._time_source(
            args,
            handler_name="Fort7Handler",
            source_path=fort7_path,
            loader=lambda: _charges_from_fort7_handler(handler, simulation=sim, reporter=reporter),
        )

    def load_electrostatics(self, args: dict, reporter=None) -> ElectrostaticsData:
        trajectory = self.load_trajectory(args, reporter=reporter)
        charges = self.load_charges(args, reporter=reporter)
        connectivity = self.load_connectivity(args, reporter=reporter)
        electric_field = None
        command = str(args.get("command") or "").strip().lower()
        fort78_path = self._resolve_reaxff_path(args, "fort78", default="fort.78")
        if command == "hyst" or fort78_path.exists():
            try:
                electric_field = self.load_electric_field(args, reporter=reporter)
            except FileNotFoundError:
                if command == "hyst":
                    raise
        return ElectrostaticsData(
            trajectory=trajectory,
            charges=charges,
            connectivity=connectivity,
            electric_field=electric_field,
        )

    def load_atomic_kinematics(self, args: dict, reporter=None) -> AtomicKinematicsData:
        from reaxkit.engine.reaxff.io.vels_handler import VelsHandler

        raw = args.get("vels") or args.get("kinematics") or args.get("input") or "vels"
        p = Path(raw)
        if p.is_dir():
            if (p / "vels").exists():
                vels_path = p / "vels"
            elif (p / "moldyn.vel").exists():
                vels_path = p / "moldyn.vel"
            elif (p / "molsav").exists():
                vels_path = p / "molsav"
            else:
                vels_path = p / "vels"
        else:
            vels_path = p
        handler = self._build_handler(
            args,
            handler_name="VelsHandler",
            source_path=vels_path,
            factory=lambda: VelsHandler(vels_path, reporter=reporter),
        )
        return self._time_source(
            args,
            handler_name="VelsHandler",
            source_path=vels_path,
            loader=lambda: _atomic_kinematics_from_vels_handler(handler),
        )

    def load_electric_field(self, args: dict, reporter=None) -> ElectricFieldData:
        from reaxkit.engine.reaxff.io.fort78_handler import Fort78Handler

        raw = args.get("fort78") or args.get("electric_field") or args.get("input") or "fort.78"
        p = Path(raw)
        fort78_path = p / "fort.78" if p.is_dir() else p
        # ReaxFF writes fort.78 on the iout1 schedule, which may differ from xmolout/fort.7/summary
        # outputs that are typically written on the iout2 schedule.
        handler = self._build_handler(
            args,
            handler_name="Fort78Handler",
            source_path=fort78_path,
            factory=lambda: Fort78Handler(fort78_path, reporter=reporter),
        )
        return self._time_source(
            args,
            handler_name="Fort78Handler",
            source_path=fort78_path,
            loader=lambda: _electric_field_from_fort78_handler(handler),
        )

    def load_molecular_analysis(self, args: dict, reporter=None) -> MolecularAnalysisData:
        from reaxkit.engine.reaxff.io.molfra_handler import MolFraHandler

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
        handler = self._build_handler(
            args,
            handler_name="MolFraHandler",
            source_path=molfra_path,
            factory=lambda: MolFraHandler(molfra_path, reporter=reporter),
        )
        return self._time_source(
            args,
            handler_name="MolFraHandler",
            source_path=molfra_path,
            loader=lambda: _molecular_analysis_from_molfra_handler(handler),
        )

    def write_control(
        self,
        data: ControlParametersData,
        out_path: str | Path,
        args: dict | None = None,
    ):
        from reaxkit.engine.reaxff.generators.control_generator import write_control_from_data

        args = args or {}
        if not isinstance(data, ControlParametersData):
            raise TypeError("write_control expects ControlParametersData.")

        overrides = args.get("overrides") or args.get("control_overrides")
        return write_control_from_data(
            data,
            out_path=out_path,
            overrides=overrides,
        )

    def write_trajectory(self, data: TrajectoryData, out_path: str | Path, args: dict | None = None):
        from reaxkit.engine.reaxff.generators.xmolout_generator import write_xmolout_from_frames

        args = args or {}
        positions = np.asarray(data.positions, dtype=float)
        if positions.ndim != 3:
            raise ValueError("TrajectoryData.positions must have shape (n_frames, n_atoms, 3).")

        n_frames, n_atoms, _ = positions.shape
        if data.atom_labels is not None:
            atom_labels = np.asarray(data.atom_labels, dtype=object)
            if atom_labels.shape != (n_frames, n_atoms):
                raise ValueError("TrajectoryData.atom_labels must have shape (n_frames, n_atoms).")
        else:
            atom_labels = np.tile(np.asarray(data.elements, dtype=object), (n_frames, 1))

        iterations = (
            np.asarray(data.iterations, dtype=int)
            if data.iterations is not None
            else (
                np.asarray(data.simulation.iterations, dtype=int)
                if data.simulation is not None and data.simulation.iterations is not None
                else np.arange(n_frames, dtype=int)
            )
        )
        potential_energy = (
            np.asarray(data.simulation.potential_energy, dtype=float)
            if data.simulation is not None and data.simulation.potential_energy is not None
            else np.zeros((n_frames,), dtype=float)
        )
        cell_lengths = (
            np.asarray(data.simulation.cell_lengths, dtype=float)
            if data.simulation is not None and data.simulation.cell_lengths is not None
            else np.ones((n_frames, 3), dtype=float)
        )
        cell_angles = (
            np.asarray(data.simulation.cell_angles, dtype=float)
            if data.simulation is not None and data.simulation.cell_angles is not None
            else np.full((n_frames, 3), 90.0, dtype=float)
        )

        frames = []
        for fi in range(n_frames):
            frames.append(
                {
                    "iter": int(iterations[fi]) if fi < len(iterations) else int(fi),
                    "coords": positions[fi],
                    "atom_types": [str(label) for label in atom_labels[fi].tolist()],
                    "E_pot": float(potential_energy[fi]) if fi < len(potential_energy) else 0.0,
                    "a": float(cell_lengths[fi, 0]) if fi < len(cell_lengths) else 1.0,
                    "b": float(cell_lengths[fi, 1]) if fi < len(cell_lengths) else 1.0,
                    "c": float(cell_lengths[fi, 2]) if fi < len(cell_lengths) else 1.0,
                    "alpha": float(cell_angles[fi, 0]) if fi < len(cell_angles) else 90.0,
                    "beta": float(cell_angles[fi, 1]) if fi < len(cell_angles) else 90.0,
                    "gamma": float(cell_angles[fi, 2]) if fi < len(cell_angles) else 90.0,
                }
            )

        return write_xmolout_from_frames(
            frames,
            out_path,
            simulation_name=str(args.get("simulation") or "MD"),
            precision=int(args.get("precision", 6)),
        )
