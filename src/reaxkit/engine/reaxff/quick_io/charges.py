"""Charge-only fort.7 readers that skip connectivity and pandas tables."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from reaxkit.domain.data_models import ChargeData, SimulationData
from reaxkit.engine.reaxff.io.fort7_handler import Fort7Handler
from reaxkit.engine.reaxff.quick_io.xmolout_identity import iter_xmolout_atom_identities


def iter_fort7_charge_frames(
        path: str | Path,
        *,
        frame_indices: Sequence[int] | None = None,
        reporter=None,
        include_atom_types: bool = True,
        input_cache: bool = True,
) -> Iterator[dict[str, Any]]:
    """Yield compact records containing only ids, types, charges, and frame metadata."""

    yield from Fort7Handler(
        path,
        frame_indices=frame_indices,
        reporter=reporter,
        input_cache=input_cache,
    ).stream_file_frames(
        charge_arrays_only=True,
        include_atom_types=include_atom_types,
    )


def charge_data_from_record(
        record: dict[str, Any],
        *,
        elements: Sequence[str] | None = None,
) -> ChargeData:
    """Convert one compact fort.7 charge record into canonical ``ChargeData``."""

    atom_ids = np.asarray(record["charge_atom_ids"], dtype=int)
    values = np.asarray(record["charges"], dtype=float)
    atom_types = np.asarray(record.get("charge_atom_type_nums", []), dtype=int)
    if atom_ids.shape != values.shape:
        raise ValueError("fort.7 charge atom ids and values must have matching shapes.")
    n_atoms = int(record["num_of_atoms"])
    canonical_ids = list(range(1, n_atoms + 1))
    charges = np.full(n_atoms, np.nan, dtype=float)
    type_numbers = np.zeros(n_atoms, dtype=int)
    valid = (atom_ids >= 1) & (atom_ids <= n_atoms)
    charges[atom_ids[valid] - 1] = values[valid]
    if atom_types.shape == atom_ids.shape:
        type_numbers[atom_ids[valid] - 1] = atom_types[valid]

    labels = [str(value) for value in elements] if elements is not None else None
    if labels is not None and len(labels) != n_atoms:
        labels = None
    iteration = int(record["iter"])
    source_index = int(record["source_index"])
    simulation = SimulationData(
        atom_ids=canonical_ids,
        iterations=np.asarray([iteration], dtype=int),
        elements=labels,
        num_of_atoms=np.asarray([n_atoms], dtype=int),
        atom_type_nums=type_numbers[np.newaxis, :],
    )
    totals = list(record.get("totals") or [])
    return ChargeData(
        charges=charges[np.newaxis, :],
        total_charge=np.asarray([totals[3]], dtype=float) if len(totals) > 3 else None,
        simulation=simulation,
        iterations=np.asarray([iteration], dtype=int),
        metadata={
            "source": "fort7",
            "streaming": True,
            "source_frame_indices": [source_index],
            "charges_only": True,
        },
    )


def iter_charge_data_quick(
        fort7_path: str | Path,
        *,
        xmolout_path: str | Path | None = None,
        frame_indices: Sequence[int] | None = None,
        reporter=None,
        input_cache: bool = True,
) -> Iterator[ChargeData]:
    """Yield charge frames aligned with each frame's lightweight identities."""

    identity_records = None
    identity_record = None
    if xmolout_path is not None and Path(xmolout_path).is_file():
        identity_records = iter_xmolout_atom_identities(
            xmolout_path,
            frame_indices=frame_indices,
            reporter=None,
        )
        identity_record = next(identity_records, None)

    for charge_record in iter_fort7_charge_frames(
            fort7_path,
            frame_indices=frame_indices,
            reporter=reporter,
            input_cache=input_cache,
    ):
        charge_index = int(charge_record["source_index"])
        while identity_record is not None and int(identity_record["source_index"]) < charge_index:
            identity_record = next(identity_records, None)
        elements = None
        if identity_record is not None and int(identity_record["source_index"]) == charge_index:
            elements = identity_record["elements"]
            identity_record = next(identity_records, None)
        yield charge_data_from_record(charge_record, elements=elements)


def load_charge_data_quick(
        fort7_path: str | Path,
        *,
        xmolout_path: str | Path | None = None,
        frame_indices: Sequence[int] | None = None,
        reporter=None,
        input_cache: bool = True,
) -> ChargeData:
    """Materialize canonical charges while parsing no connectivity or coordinates."""

    frames = list(
        iter_charge_data_quick(
            fort7_path,
            xmolout_path=xmolout_path,
            frame_indices=frame_indices,
            reporter=reporter,
            input_cache=input_cache,
        )
    )
    if not frames:
        return ChargeData(charges=np.empty((0, 0), dtype=float), iterations=np.empty(0, dtype=int))

    max_atoms = max(frame.charges.shape[1] for frame in frames)
    padded_charges = np.full((len(frames), max_atoms), np.nan, dtype=float)
    padded_types = np.zeros((len(frames), max_atoms), dtype=int)
    merged_elements = [""] * max_atoms
    for frame_index, frame in enumerate(frames):
        atom_count = frame.charges.shape[1]
        padded_charges[frame_index, :atom_count] = frame.charges[0]
        padded_types[frame_index, :atom_count] = frame.simulation.atom_type_nums[0]
        for atom_index, element in enumerate(frame.simulation.elements or []):
            if element:
                merged_elements[atom_index] = str(element)
    totals = None
    if all(frame.total_charge is not None for frame in frames):
        totals = np.concatenate([np.asarray(frame.total_charge, dtype=float) for frame in frames])
    return ChargeData(
        charges=padded_charges,
        total_charge=totals,
        simulation=SimulationData(
            atom_ids=list(range(1, max_atoms + 1)),
            elements=merged_elements if any(merged_elements) else None,
            atom_type_nums=padded_types,
        ),
        iterations=np.concatenate([frame.iterations for frame in frames]),
        metadata={
            "source": "fort7",
            "source_frame_indices": [
                int(frame.metadata["source_frame_indices"][0]) for frame in frames
            ],
            "charges_only": True,
        },
    )


__all__ = [
    "charge_data_from_record",
    "iter_charge_data_quick",
    "iter_fort7_charge_frames",
    "load_charge_data_quick",
]
