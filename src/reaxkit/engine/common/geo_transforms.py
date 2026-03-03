"""
Geometry transformation utilities.

This module contains ASE-based helpers for building and transforming structures
before they are written to disk or converted into GEO/XTLGRF format.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Literal, Optional, Tuple

import numpy as np
from ase import Atoms
from ase.build import make_supercell as ase_make_supercell, surface as ase_surface
from ase.geometry import cellpar_to_cell

from reaxkit.engine.common.geo_io import read_structure


TerminationSide = Literal["top", "bottom"]
Axis = Literal[0, 1, 2]

__all__ = [
    "Axis",
    "TerminationSide",
    "build_surface",
    "make_supercell",
    "orthogonalize_hexagonal_cell",
    "place2",
]


def build_surface(
    bulk: Atoms,
    miller: Tuple[int, int, int],
    layers: int,
    vacuum: float = 15.0,
    center: bool = True,
) -> Atoms:
    """
    Build a surface slab from a bulk structure using Miller indices.
    """
    slab = ase_surface(bulk, miller, layers=layers, vacuum=vacuum)
    if center:
        slab.center(axis=2)
    return slab


def make_supercell(
    atoms: Atoms,
    transform: Iterable[Iterable[int]] | Tuple[int, int, int],
) -> Atoms:
    """
    Generate a supercell or apply a lattice transformation.
    """
    if isinstance(transform, tuple) and len(transform) == 3:
        return atoms * transform

    matrix = np.array(transform, dtype=int)
    if matrix.shape != (3, 3):
        raise ValueError("transform must be either (nx, ny, nz) or a 3x3 integer matrix.")
    return ase_make_supercell(atoms, matrix)


def orthogonalize_hexagonal_cell(atoms: Atoms) -> Atoms:
    """
    Convert a hexagonal unit cell into an orthorhombic cell.
    """
    transform = np.array([
        [1, -1, 0],
        [1, 1, 0],
        [0, 0, 1],
    ])
    return ase_make_supercell(atoms, transform)


def place2(
    insert_molecule: str | Path,
    base_structure: Optional[str | Path] = None,
    *,
    n_copies: int,
    box_length_x: float,
    box_length_y: float,
    box_length_z: float,
    alpha: float,
    beta: float,
    gamma: float,
    min_interatomic_distance: float,
    base_structure_placement_mode: str = "as-is",
    max_placement_attempts_per_copy: int = 50000,
    random_seed: int | None = None,
) -> Atoms:
    """
    Randomly place multiple copies of a molecule into a simulation cell.
    """
    rng = np.random.default_rng(random_seed)
    cell = cellpar_to_cell([box_length_x, box_length_y, box_length_z, alpha, beta, gamma])
    inv_cell = np.linalg.inv(cell)

    insert_atoms = read_structure(insert_molecule)
    insert_symbols = insert_atoms.get_chemical_symbols()
    insert_coords = insert_atoms.get_positions()
    insert_center = insert_coords.mean(axis=0)
    insert_coords_centered = insert_coords - insert_center

    system_symbols: List[str] = []
    system_coords = np.empty((0, 3), dtype=float)

    if base_structure is not None:
        base_atoms = read_structure(base_structure)
        base_coords = base_atoms.get_positions()
        base_symbols = base_atoms.get_chemical_symbols()

        if base_structure_placement_mode not in {"as-is", "center", "origin"}:
            raise ValueError('base_structure_placement_mode must be one of {"as-is", "center", "origin"}')

        if base_structure_placement_mode in {"center", "origin"}:
            base_center = base_coords.mean(axis=0)
            if base_structure_placement_mode == "center":
                box_center = np.array([box_length_x / 2, box_length_y / 2, box_length_z / 2])
                shift = box_center - base_center
            else:
                shift = -base_center
            base_coords = base_coords + shift

        system_symbols.extend(base_symbols)
        system_coords = np.vstack([system_coords, base_coords])

    def random_rotation_matrix() -> np.ndarray:
        phi = 2 * np.pi * rng.random()
        theta = 2 * np.pi * rng.random()
        psi = 2 * np.pi * rng.random()

        c1, s1 = np.cos(phi), np.sin(phi)
        c2, s2 = np.cos(theta), np.sin(theta)
        c3, s3 = np.cos(psi), np.sin(psi)

        rz1 = np.array([[c1, -s1, 0], [s1, c1, 0], [0, 0, 1]])
        ry = np.array([[c2, 0, s2], [0, 1, 0], [-s2, 0, c2]])
        rz2 = np.array([[c3, -s3, 0], [s3, c3, 0], [0, 0, 1]])
        return rz2 @ ry @ rz1

    def min_distance(new: np.ndarray, existing: np.ndarray) -> float:
        if existing.size == 0:
            return float("inf")

        diff = new[:, None, :] - existing[None, :, :]
        frac = diff @ inv_cell
        frac -= np.round(frac)
        cart = frac @ cell
        d2 = np.sum(cart ** 2, axis=-1)
        return float(np.sqrt(d2.min()))

    for n in range(n_copies):
        attempts = 0
        while True:
            attempts += 1
            if attempts > max_placement_attempts_per_copy:
                raise RuntimeError(
                    f"Could not place copy {n + 1} after {max_placement_attempts_per_copy} attempts."
                )

            rotation = random_rotation_matrix()
            rotated = insert_coords_centered @ rotation.T
            translate = np.array([
                box_length_x * rng.random(),
                box_length_y * rng.random(),
                box_length_z * rng.random(),
            ])
            new_coords = rotated + translate

            if min_distance(new_coords, system_coords) > min_interatomic_distance:
                system_coords = np.vstack([system_coords, new_coords])
                system_symbols.extend(insert_symbols)
                break

    return Atoms(symbols=system_symbols, positions=system_coords, cell=cell, pbc=True)
