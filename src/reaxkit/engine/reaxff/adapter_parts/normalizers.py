"""Normalization helpers for the ReaxFF engine adapter.

This module contains pure conversion logic that maps ReaxFF handler outputs into
canonical reaxkit domain models. The adapter class imports these helpers to keep
loader methods focused on source selection and orchestration.

**Usage context**

- Data conversion: Transform handler dataframes/frames into typed domain models.
- Connectivity assembly: Build sparse connectivity and charge/electrostatics views.
- Adapter internals: Consumed by `ReaxFFAdapter`; not a public API module.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from reaxkit.domain.data_models import (
    AtomicKinematicsData,
    ChargeData,
    ConnectivityTrajectoryData,
    ConnectivityData,
    ControlParametersData,
    EregimeData,
    ElectricFieldData,
    ForceFieldParametersData,
    ForceFieldOptimizationProgressData,
    ForceFieldOptimizationTrainingSetData,
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

if TYPE_CHECKING:
    from reaxkit.engine.reaxff.io.control_handler import ControlHandler
    from reaxkit.engine.reaxff.io.eregime_handler import EregimeHandler
    from reaxkit.engine.common.io.ffield_handler import FFieldHandler
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
        """Init."""
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
        """Shape.

        Parameters
        ----------
        None

        Returns
        -------
        tuple[int, int]
            Return value.

        Examples
        --------
        ```python
        # Example
        shape(...)
        ```
        """
        return (self.n_atoms, self.n_atoms)

    def toarray(self) -> np.ndarray:
        """Toarray.

        Parameters
        ----------
        None

        Returns
        -------
        np.ndarray
            Return value.

        Examples
        --------
        ```python
        # Example
        toarray(...)
        ```
        """
        mat = np.zeros((self.n_atoms, self.n_atoms), dtype=float)
        if self._vals.size:
            mat[self._rows, self._cols] = self._vals
        return mat

    def todense(self) -> np.ndarray:
        """Todense.

        Parameters
        ----------
        None

        Returns
        -------
        np.ndarray
            Return value.

        Examples
        --------
        ```python
        # Example
        todense(...)
        ```
        """
        return self.toarray()

    def sum(self, axis=None):
        """Sum.

        Parameters
        ----------
        axis : Any, optional
            Input parameter.

        Returns
        -------
        Any
            Return value.

        Examples
        --------
        ```python
        # Example
        sum(...)
        ```
        """
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
    """Merge simulation data."""
    def _reindex_to_iterations(
        values: np.ndarray | None,
        *,
        source_iterations: np.ndarray | None,
        target_iterations: np.ndarray | None,
    ) -> np.ndarray | None:
        """Reindex to iterations."""
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
        """Pick."""
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
    """Merge atom id lists."""
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
    """Fort7 per atom arrays."""
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


def _geometry_from_geo_handler(
    handler: GeoHandler,
    *,
    source_file: str,
    geometry_role: str,
) -> GeometryData:
    """Normalize a ``GeoHandler`` into ``GeometryData``."""
    df = handler.coordinates()
    connectivity_df = handler.connectivity()
    meta = dict(handler.metadata())
    meta["source_file"] = str(source_file)
    meta["geometry_role"] = str(geometry_role)
    meta.setdefault("source", "geo_handler")
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
        connectivity=connectivity_df,
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
    """Connectivity trajectory from handlers."""
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


