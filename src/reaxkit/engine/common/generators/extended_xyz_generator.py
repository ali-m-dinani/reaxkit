"""Streaming Extended XYZ generation for atom-resolved trajectory properties."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TextIO

import numpy as np


@dataclass
class ExtendedXYZFrame:
    """One Extended XYZ frame with arbitrary scalar or vector atom properties."""

    species: Sequence[str]
    positions: np.ndarray
    properties: Mapping[str, Any] = field(default_factory=dict)
    frame: int | None = None
    iteration: int | None = None
    lattice: np.ndarray | None = None
    pbc: tuple[bool, bool, bool] | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


def lattice_from_lengths_angles(
        lengths: Sequence[float],
        angles_degrees: Sequence[float],
) -> np.ndarray:
    """Convert crystallographic lengths and angles to a row-vector lattice."""

    a, b, c = np.asarray(lengths, dtype=float)
    alpha, beta, gamma = np.deg2rad(np.asarray(angles_degrees, dtype=float))
    sin_gamma = np.sin(gamma)
    if not np.isfinite([a, b, c, alpha, beta, gamma]).all() or min(a, b, c) <= 0:
        raise ValueError("Cell lengths and angles must be finite with positive lengths.")
    if abs(sin_gamma) < 1e-12:
        raise ValueError("Cell gamma angle produces a singular lattice.")
    ax = np.asarray([a, 0.0, 0.0])
    bx = np.asarray([b * np.cos(gamma), b * sin_gamma, 0.0])
    cx_x = c * np.cos(beta)
    cx_y = c * (np.cos(alpha) - np.cos(beta) * np.cos(gamma)) / sin_gamma
    cx_z_sq = c * c - cx_x * cx_x - cx_y * cx_y
    cx = np.asarray([cx_x, cx_y, np.sqrt(max(0.0, cx_z_sq))])
    return np.vstack([ax, bx, cx])


def _property_array(name: str, values: Any, n_atoms: int) -> np.ndarray:
    if not name or any(character.isspace() or character == ":" for character in name):
        raise ValueError(f"Invalid Extended XYZ property name: {name!r}.")
    array = np.asarray(values)
    if array.ndim == 1:
        array = array[:, np.newaxis]
    if array.ndim != 2 or array.shape[0] != n_atoms:
        raise ValueError(
            f"Extended XYZ property '{name}' must have shape (n_atoms,) or "
            "(n_atoms, n_components)."
        )
    return array


def _property_kind(array: np.ndarray) -> str:
    if np.issubdtype(array.dtype, np.bool_) or np.issubdtype(array.dtype, np.integer):
        return "I"
    if np.issubdtype(array.dtype, np.number):
        return "R"
    return "S"


def _quoted(value: Any) -> str:
    text = str(value).replace('"', '\\"')
    return f'"{text}"' if any(character.isspace() for character in text) else text


class ExtendedXYZWriter:
    """Incrementally write Extended XYZ frames without materializing a trajectory."""

    def __init__(self, out_path: str | Path, *, precision: int = 8):
        if int(precision) < 1:
            raise ValueError("precision must be at least 1.")
        self.path = Path(out_path)
        self.precision = int(precision)
        self._handle: TextIO | None = None

    def __enter__(self) -> "ExtendedXYZWriter":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self.path.open("w", encoding="utf-8", newline="\n")
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None

    def write_frame(self, frame: ExtendedXYZFrame) -> None:
        if self._handle is None:
            raise RuntimeError("ExtendedXYZWriter must be used as a context manager.")
        positions = np.asarray(frame.positions, dtype=float)
        if positions.ndim != 2 or positions.shape[1] != 3:
            raise ValueError("ExtendedXYZFrame.positions must have shape (n_atoms, 3).")
        n_atoms = positions.shape[0]
        species = [str(value) for value in frame.species]
        if len(species) != n_atoms:
            raise ValueError("ExtendedXYZFrame.species length must match its positions.")

        properties: list[tuple[str, np.ndarray, str]] = []
        descriptor = ["species:S:1", "pos:R:3"]
        for name, raw_values in frame.properties.items():
            if name in {"species", "pos"}:
                raise ValueError(f"'{name}' is a reserved Extended XYZ property name.")
            array = _property_array(str(name), raw_values, n_atoms)
            kind = _property_kind(array)
            descriptor.append(f"{name}:{kind}:{array.shape[1]}")
            properties.append((str(name), array, kind))

        comment = [f"Properties={':'.join(descriptor)}"]
        if frame.lattice is not None:
            lattice = np.asarray(frame.lattice, dtype=float)
            if lattice.shape != (3, 3):
                raise ValueError("ExtendedXYZFrame.lattice must have shape (3, 3).")
            # Extended XYZ stores the three lattice vectors column-major.
            lattice_text = " ".join(
                self._format_real(value) for value in lattice.reshape(-1, order="F")
            )
            comment.append(f'Lattice="{lattice_text}"')
        if frame.pbc is not None:
            pbc_text = " ".join("T" if value else "F" for value in frame.pbc)
            comment.append(f'pbc="{pbc_text}"')
        if frame.frame is not None:
            comment.append(f"frame={int(frame.frame)}")
        if frame.iteration is not None:
            comment.append(f"iter={int(frame.iteration)}")
        for name, value in frame.metadata.items():
            if name not in {"Properties", "Lattice", "pbc", "frame", "iter"}:
                comment.append(f"{name}={_quoted(value)}")

        self._handle.write(f"{n_atoms}\n")
        self._handle.write(" ".join(comment) + "\n")
        for atom_index in range(n_atoms):
            values = [
                species[atom_index],
                *(self._format_real(value) for value in positions[atom_index]),
            ]
            for _, array, kind in properties:
                values.extend(self._format_value(value, kind) for value in array[atom_index])
            self._handle.write(" ".join(values) + "\n")

    def _format_real(self, value: Any) -> str:
        return f"{float(value):.{self.precision}g}"

    def _format_value(self, value: Any, kind: str) -> str:
        if kind == "R":
            return self._format_real(value)
        if kind == "I":
            return str(int(value))
        return _quoted(value)


def write_extended_xyz_trajectory(
        frames: Iterable[ExtendedXYZFrame],
        out_path: str | Path,
        *,
        precision: int = 8,
) -> Path:
    """Stream an iterable of frames to an Extended XYZ trajectory."""

    with ExtendedXYZWriter(out_path, precision=precision) as writer:
        for frame in frames:
            writer.write_frame(frame)
    return Path(out_path)


__all__ = [
    "ExtendedXYZFrame",
    "ExtendedXYZWriter",
    "lattice_from_lengths_angles",
    "write_extended_xyz_trajectory",
]
