"""Constant-memory frame streaming for ReaxFF trajectory analyses."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from reaxkit.domain.data_models import (
    ChargeData,
    ConnectivityData,
    ConnectivityTrajectoryData,
    CoordinationStatusBundleData,
    ElectrostaticsData,
    SimulationData,
    TrajectoryData,
)
from reaxkit.engine.reaxff.adapter_parts.normalizers import _SparseFrame
from reaxkit.engine.reaxff.io.fort7_handler import Fort7Handler
from reaxkit.engine.reaxff.io.xmolout_handler import XmoloutHandler


STREAMABLE_REAXFF_TYPES = {
    TrajectoryData,
    ChargeData,
    ConnectivityData,
    ConnectivityTrajectoryData,
    CoordinationStatusBundleData,
    ElectrostaticsData,
}


def _trajectory_frame(record: dict[str, Any]) -> TrajectoryData:
    table = record["frame"]
    coords = table[["x", "y", "z"]].to_numpy(dtype=float)
    elements = table["atom_type"].astype(str).tolist()
    atom_ids = list(range(1, len(elements) + 1))
    iteration = int(record["iter"])
    source_index = int(record["source_index"])
    cell_lengths = np.asarray([record["cell_lengths"]], dtype=float)
    cell_angles = np.asarray([record["cell_angles"]], dtype=float)
    simulation = SimulationData(
        atom_ids=atom_ids,
        iterations=np.asarray([iteration], dtype=int),
        elements=elements,
        num_of_atoms=np.asarray([len(atom_ids)], dtype=int),
        potential_energy=np.asarray([record["potential_energy"]], dtype=float),
        cell_lengths=cell_lengths,
        cell_angles=cell_angles,
    )
    return TrajectoryData(
        positions=coords[np.newaxis, :, :],
        elements=elements,
        atom_ids=atom_ids,
        simulation=simulation,
        iterations=np.asarray([iteration], dtype=int),
        atom_labels=np.asarray([elements], dtype=object),
        source_frame_indices=np.asarray([source_index], dtype=int),
    )


def _fort7_frame(
    record: dict[str, Any],
    *,
    atom_ids: list[int] | None = None,
    elements: list[str] | None = None,
) -> tuple[ConnectivityData, ChargeData]:
    table: pd.DataFrame = record["frame"]
    discovered = (
        pd.to_numeric(table["atom_num"], errors="coerce").dropna().astype(int).tolist()
        if "atom_num" in table.columns
        else list(range(1, len(table) + 1))
    )
    ids = list(atom_ids or [])
    for atom_id in discovered:
        if atom_id not in ids:
            ids.append(int(atom_id))
    if not ids:
        ids = list(range(1, int(record["num_of_atoms"]) + 1))
    n_atoms = len(ids)
    atom_to_index = {int(atom_id): index for index, atom_id in enumerate(ids)}
    labels = list(elements or [])
    labels.extend(["X"] * (n_atoms - len(labels)))

    atom_type_nums = np.zeros((1, n_atoms), dtype=int)
    molecule_nums = np.zeros((1, n_atoms), dtype=int)
    lone_pairs = np.full((1, n_atoms), np.nan, dtype=float)
    charges = np.full((1, n_atoms), np.nan, dtype=float)
    bo_pairs: dict[tuple[int, int], float] = {}
    conn_pairs: dict[tuple[int, int], float] = {}
    bo_columns = [column for column in table.columns if str(column).startswith("BO")]

    for row_number, row in table.iterrows():
        atom_id = int(row["atom_num"]) if "atom_num" in table.columns else int(row_number) + 1
        source = atom_to_index.get(atom_id)
        if source is None:
            continue
        if "atom_type_num" in table.columns and pd.notna(row["atom_type_num"]):
            atom_type_nums[0, source] = int(row["atom_type_num"])
        if "molecule_num" in table.columns and pd.notna(row["molecule_num"]):
            molecule_nums[0, source] = int(row["molecule_num"])
        if "num_LPs" in table.columns and pd.notna(row["num_LPs"]):
            lone_pairs[0, source] = float(row["num_LPs"])
        if "partial_charge" in table.columns and pd.notna(row["partial_charge"]):
            charges[0, source] = float(row["partial_charge"])
        for bo_column in bo_columns:
            slot = str(bo_column)[2:]
            neighbor_column = f"atom_cnn{slot}"
            if neighbor_column not in table.columns or pd.isna(row[neighbor_column]):
                continue
            neighbor_id = int(row[neighbor_column])
            target = atom_to_index.get(neighbor_id)
            if target is None or neighbor_id <= 0:
                continue
            conn_pairs[(source, target)] = 1.0
            bond_order = float(row[bo_column]) if pd.notna(row[bo_column]) else 0.0
            if bond_order > 0.0:
                bo_pairs[(source, target)] = max(bo_pairs.get((source, target), 0.0), bond_order)

    bond_orders = _SparseFrame(n_atoms, bo_pairs)
    connectivity = _SparseFrame(n_atoms, conn_pairs)
    iteration = int(record["iter"])
    source_index = int(record["source_index"])
    simulation = SimulationData(
        atom_ids=ids,
        iterations=np.asarray([iteration], dtype=int),
        elements=labels,
        num_of_atoms=np.asarray([n_atoms], dtype=int),
        atom_type_nums=atom_type_nums,
        molecule_nums=molecule_nums,
    )
    totals = list(record.get("totals") or [])
    connectivity_data = ConnectivityData(
        connectivity=[connectivity],
        bond_orders=[bond_orders],
        sum_bond_orders=np.asarray(bond_orders.sum(axis=1), dtype=float)[np.newaxis, :],
        num_lone_pairs=lone_pairs,
        num_of_bonds=np.asarray([record["num_of_bonds"]], dtype=int),
        total_bond_order=np.asarray([totals[0]], dtype=float) if len(totals) > 0 else None,
        total_lone_pairs=np.asarray([totals[1]], dtype=float) if len(totals) > 1 else None,
        total_bond_order_uncorrected=np.asarray([totals[2]], dtype=float) if len(totals) > 2 else None,
        atom_ids=ids,
        elements=labels,
        simulation=simulation,
        iterations=np.asarray([iteration], dtype=int),
        metadata={
            "source": "fort7",
            "streaming": True,
            "bond_orders_format": "sparse_frame_list",
            "connectivity_incomplete": bool(record.get("connectivity_incomplete", False)),
        },
        source_frame_indices=np.asarray([source_index], dtype=int),
    )
    charge_data = ChargeData(
        charges=charges,
        total_charge=np.asarray([totals[3]], dtype=float) if len(totals) > 3 else None,
        simulation=simulation,
        iterations=np.asarray([iteration], dtype=int),
        metadata={
            "source": "fort7",
            "streaming": True,
            "source_frame_indices": [source_index],
        },
    )
    return connectivity_data, charge_data


def _aligned_records(
    coordinate_records: Iterator[dict[str, Any]],
    connectivity_records: Iterator[dict[str, Any]],
) -> Iterator[tuple[dict[str, Any], dict[str, Any]]]:
    """Merge two source-index ordered iterators while retaining two frames."""
    coordinates = next(coordinate_records, None)
    connectivity = next(connectivity_records, None)
    while coordinates is not None and connectivity is not None:
        coord_index = int(coordinates["source_index"])
        conn_index = int(connectivity["source_index"])
        if coord_index == conn_index:
            yield coordinates, connectivity
            coordinates = next(coordinate_records, None)
            connectivity = next(connectivity_records, None)
        elif coord_index < conn_index:
            coordinates = next(coordinate_records, None)
        else:
            connectivity = next(connectivity_records, None)


def iter_reaxff_data(adapter, data_type, args: dict, reporter=None) -> Iterator[Any]:
    """Yield one canonical ReaxFF frame bundle at a time."""
    selected = args.get("_frame_indices")
    xmol_path = adapter._resolve_reaxff_path(args, "xmolout", default="xmolout")
    coordinate_records = XmoloutHandler(
        xmol_path,
        frame_indices=selected,
        reporter=reporter,
    ).stream_file_frames()

    if data_type is TrajectoryData:
        for coordinate_record in coordinate_records:
            yield _trajectory_frame(coordinate_record)
        return

    fort7_path = adapter._resolve_reaxff_path(
        args,
        "fort7",
        "connectivity",
        "charges",
        default="fort.7",
    )
    if not Path(fort7_path).is_file():
        raise FileNotFoundError(f"ReaxFF streaming requires fort.7: {fort7_path}")
    connectivity_records = Fort7Handler(
        fort7_path,
        frame_indices=selected,
        reporter=None,
    ).stream_file_frames(
        charges_only=(
            data_type is ChargeData
            or (
                data_type is ElectrostaticsData
                and str(args.get("scope") or "total").strip().lower() == "total"
            )
        )
    )

    electric_field = None
    if data_type is ElectrostaticsData:
        command = str(args.get("command") or "").strip().lower()
        fort78_path = adapter._resolve_reaxff_path(args, "fort78", default="fort.78")
        if command == "hyst" or fort78_path.exists():
            try:
                electric_field = adapter.load_electric_field(
                    {**args, "fort78": str(fort78_path)}, reporter=None
                )
            except FileNotFoundError:
                if command == "hyst":
                    raise

    if data_type in {ConnectivityData, CoordinationStatusBundleData}:
        force_field = (
            adapter.load_force_field(args, reporter=None)
            if data_type is CoordinationStatusBundleData
            else None
        )
        for connectivity_record in connectivity_records:
            connectivity, charges = _fort7_frame(connectivity_record)
            if data_type is ConnectivityData:
                yield connectivity
            else:
                yield CoordinationStatusBundleData(
                    connectivity=connectivity,
                    force_field_parameters=force_field,
                    metadata={"streaming": True},
                )
        return

    for coordinate_record, connectivity_record in _aligned_records(
        iter(coordinate_records),
        iter(connectivity_records),
    ):
        trajectory = _trajectory_frame(coordinate_record)
        connectivity, charges = _fort7_frame(
            connectivity_record,
            atom_ids=list(trajectory.atom_ids),
            elements=list(trajectory.elements),
        )
        # The coordinate file is the public iteration axis used by existing
        # materialized electrostatics data.
        charges.iterations = trajectory.iterations
        charges.simulation = trajectory.simulation
        connectivity.elements = trajectory.elements
        connectivity.atom_ids = trajectory.atom_ids
        if connectivity.simulation is not None:
            connectivity.simulation.elements = trajectory.elements

        if data_type is ChargeData:
            yield charges
        elif data_type is ConnectivityTrajectoryData:
            yield ConnectivityTrajectoryData(
                connectivity=connectivity,
                trajectory=trajectory,
            )
        elif data_type is ElectrostaticsData:
            yield ElectrostaticsData(
                trajectory=trajectory,
                charges=charges,
                connectivity=connectivity,
                electric_field=electric_field,
            )
        else:  # pragma: no cover - guarded by EngineAdapter
            raise ValueError(f"Unsupported ReaxFF streaming type: {data_type}")
