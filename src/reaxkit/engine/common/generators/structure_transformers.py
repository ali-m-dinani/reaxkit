"""Build and transform structures with ASE-based geometry utilities.

This module provides reusable structure-construction and transformation helpers
for slab generation, supercell/lattice transforms, and random molecular
placement inside periodic cells. It is focused on geometry preparation and does
not perform engine-specific simulation logic.

**Usage context**

- Pre-processing: Build slabs/supercells before simulation input generation.
- Cell transformations: Re-map lattices into alternate simulation-friendly forms.
- Packing workflows: Place molecules stochastically under distance constraints.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Literal, Optional, Tuple

import numpy as np
from ase import Atoms
from ase.build import make_supercell as ase_make_supercell, surface as ase_surface
from ase.geometry import cellpar_to_cell

from reaxkit.engine.common.io.geo_io import read_structure


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
    """Build a surface slab from a bulk structure using Miller indices.

    Parameters
    ----------
    bulk : Atoms
        Input bulk structure.
    miller : Tuple[int, int, int]
        Miller index triple defining the surface orientation.
    layers : int
        Number of atomic layers to include in the slab.
    vacuum : float, optional
        Vacuum padding along the surface normal.
    center : bool, optional
        Whether to center slab atoms along ``z`` after slab construction.

    Returns
    -------
    Atoms
        Generated slab structure.

    Examples
    --------
    ```python
    slab = build_surface(bulk, (1, 1, 1), layers=6, vacuum=15.0)
    ```
    """
    slab = ase_surface(bulk, miller, layers=layers, vacuum=vacuum)
    if center:
        slab.center(axis=2)
    return slab


def make_supercell(
    atoms: Atoms,
    transform: Iterable[Iterable[int]] | Tuple[int, int, int],
) -> Atoms:
    """Generate a supercell or apply a lattice transformation matrix.

    Parameters
    ----------
    atoms : Atoms
        Input structure to transform.
    transform : Iterable[Iterable[int]] | Tuple[int, int, int]
        Either repetition counts ``(nx, ny, nz)`` or a ``3x3`` integer matrix.

    Returns
    -------
    Atoms
        Transformed supercell structure.

    Examples
    --------
    ```python
    sc = make_supercell(atoms, (2, 2, 1))
    ```
    """
    if isinstance(transform, tuple) and len(transform) == 3:
        return atoms * transform

    matrix = np.array(transform, dtype=int)
    if matrix.shape != (3, 3):
        raise ValueError("transform must be either (nx, ny, nz) or a 3x3 integer matrix.")
    return ase_make_supercell(atoms, matrix)


def orthogonalize_hexagonal_cell(atoms: Atoms) -> Atoms:
    """Convert a hexagonal unit cell into an orthorhombic supercell mapping.

    Parameters
    ----------
    atoms : Atoms
        Input structure with a hexagonal-like cell representation.

    Returns
    -------
    Atoms
        Structure transformed with a fixed integer supercell matrix.

    Examples
    --------
    ```python
    ortho = orthogonalize_hexagonal_cell(hex_atoms)
    ```
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
    """Randomly place molecule copies into a periodic simulation cell.

    Generates random rigid-body rotations/translations for ``n_copies`` of an
    inserted molecule and accepts placements that satisfy a minimum
    interatomic-distance constraint under periodic boundary conditions.

    Parameters
    ----------
    insert_molecule : str | Path
        Structure file for the molecule to insert repeatedly.
    base_structure : Optional[str | Path], optional
        Optional initial structure preloaded into the target cell.
    n_copies : int
        Number of inserted molecule copies to place.
    box_length_x : float
        Cell length along ``x``.
    box_length_y : float
        Cell length along ``y``.
    box_length_z : float
        Cell length along ``z``.
    alpha : float
        Cell angle ``alpha`` in degrees.
    beta : float
        Cell angle ``beta`` in degrees.
    gamma : float
        Cell angle ``gamma`` in degrees.
    min_interatomic_distance : float
        Minimum allowed distance between inserted and existing atoms.
    base_structure_placement_mode : str, optional
        Placement mode for base structure: ``"as-is"``, ``"center"``, or
        ``"origin"``.
    max_placement_attempts_per_copy : int, optional
        Maximum random attempts before failing one copy placement.
    random_seed : int | None, optional
        Seed for deterministic random placement.

    Returns
    -------
    Atoms
        Combined packed structure with periodic boundary conditions enabled.

    Examples
    --------
    ```python
    packed = place2(
        "h2o.xyz",
        n_copies=50,
        box_length_x=30.0,
        box_length_y=30.0,
        box_length_z=30.0,
        alpha=90.0,
        beta=90.0,
        gamma=90.0,
        min_interatomic_distance=1.4,
    )
    ```
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
        """Generate a random 3D rotation matrix from Euler-angle sampling."""
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
        """Compute minimum-image nearest distance between new and existing sets."""
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
