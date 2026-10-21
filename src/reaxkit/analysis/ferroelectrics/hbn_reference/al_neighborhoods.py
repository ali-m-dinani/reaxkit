"""Fixed Al-centered, coplanar nitrogen neighborhoods in the hBN reference."""

from dataclasses import dataclass
import itertools

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree


@dataclass(frozen=True)
class AlNeighborhoods:
    al_indices: np.ndarray
    nitrogen_indices: np.ndarray
    nitrogen_weights: np.ndarray
    reference_centers: np.ndarray
    reference_vectors: np.ndarray
    layer_ids: np.ndarray
    cell_ids: np.ndarray

    def neighbor_displacements(self, displacement, cell, periodic):
        """Keep relative displacements on the image branch nearest the reference."""
        differences = displacement[self.nitrogen_indices] - displacement[self.al_indices, None, :]
        fractional = differences @ np.linalg.inv(cell)
        for axis, enabled in enumerate(periodic):
            if enabled:
                fractional[:, :, axis] -= np.round(fractional[:, :, axis])
        return fractional @ cell

    def relative_displacement(self, displacement, cell, periodic):
        return -self.neighbor_displacements(displacement, cell, periodic).mean(axis=1)

    def current_vectors(self, displacement, reference_cell, current_cell, periodic):
        reference_vectors = self.reference_vectors @ np.linalg.inv(reference_cell) @ current_cell
        return reference_vectors + self.neighbor_displacements(displacement, current_cell, periodic)


def neighborhood_hull_volumes(vectors):
    """Exact four-point convex hull volumes, with coplanar neighborhoods zero."""
    volumes = np.abs(np.linalg.det(vectors)) / 6.0
    scale = np.max(np.linalg.norm(vectors, axis=2), axis=1)
    degenerate = volumes <= 64.0 * np.finfo(float).eps * scale ** 3
    return np.where(degenerate, 0.0, volumes)


def build_al_neighborhoods(prepared) -> AlNeighborhoods:
    """Select three nearest coplanar N images, retaining periodic multiplicity."""
    cached = getattr(prepared, "_al_neighborhoods", None)
    if cached is not None:
        return cached
    atoms = prepared.atoms
    symbols = np.asarray(atoms.get_chemical_symbols())
    if not np.isin(symbols, ["Al", "N"]).all():
        raise ValueError("Al-centered hBN neighborhoods require Al/N reference sites.")
    al_reference = np.flatnonzero(symbols == "Al")
    nitrogen_reference = np.flatnonzero(symbols == "N")
    if not len(al_reference) or not len(nitrogen_reference):
        raise ValueError("Al-centered hBN neighborhoods require both Al and N sites.")
    fractional = atoms.get_scaled_positions(wrap=False)
    for axis, periodic in enumerate(atoms.pbc):
        if periodic:
            fractional[:, axis] %= 1.0
    planes = np.round(fractional[:, 2], 7)
    if atoms.pbc[2]:
        planes %= 1.0
    coordinates = fractional @ atoms.cell.array
    shifts = np.asarray(list(itertools.product(
        (-1, 0, 1) if atoms.pbc[0] else (0,),
        (-1, 0, 1) if atoms.pbc[1] else (0,),
        (0,),
    ))) @ atoms.cell.array
    neighbors = np.empty((len(al_reference), 3), dtype=int)
    vectors = np.empty((len(al_reference), 3, 3), dtype=float)
    for plane in np.unique(planes[al_reference]):
        center_rows = np.flatnonzero(planes[al_reference] == plane)
        candidates = nitrogen_reference[planes[nitrogen_reference] == plane]
        if not len(candidates):
            raise ValueError("Reference Al plane contains no nitrogen sites.")
        image_positions = (coordinates[candidates][None, :, :] + shifts[:, None, :]).reshape(-1, 3)
        image_indices = np.tile(candidates, len(shifts))
        distances, indices = cKDTree(image_positions).query(coordinates[al_reference[center_rows]], k=4)
        valid = (np.isfinite(distances[:, :3]).all(axis=1)
                 & (distances[:, 0] > 1.0e-8)
                 & (distances[:, 2] < 1.5 * distances[:, 0])
                 & (distances[:, 3] > 1.05 * distances[:, 2]))
        if not valid.all():
            raise ValueError("Reference must have three unambiguous nearest coplanar N images per Al; "
                             "check in-plane periodicity, reference geometry, and missing neighbors.")
        neighbors[center_rows] = image_indices[indices[:, :3]]
        vectors[center_rows] = image_positions[indices[:, :3]] - coordinates[al_reference[center_rows], None, :]
    sharing = np.bincount(neighbors.ravel(), minlength=len(atoms))
    if (sharing[nitrogen_reference] == 0).any():
        raise ValueError("Some reference N atoms have no Al neighborhood; partition would lose contributions.")
    result = AlNeighborhoods(
        al_indices=prepared.reference_to_simulation[al_reference],
        nitrogen_indices=prepared.reference_to_simulation[neighbors],
        nitrogen_weights=1.0 / sharing[neighbors],
        reference_centers=prepared.aligned_positions[al_reference],
        reference_vectors=vectors,
        layer_ids=prepared.local_layer_ids[al_reference],
        cell_ids=prepared.local_cell_ids[al_reference],
    )
    prepared._al_neighborhoods = result
    return result


def neighborhood_mapping(reference_result) -> pd.DataFrame:
    """Export each Al-N image edge and its partition weight for inspection."""
    neighborhoods = build_al_neighborhoods(reference_result.reference)
    atom_ids = reference_result.mapping.sort_values("atom_index")["atom_id"].to_numpy(int)
    table = pd.DataFrame({
        "local_neighborhood_id": np.repeat(neighborhoods.layer_ids, 3),
        "al_atom_id": np.repeat(atom_ids[neighborhoods.al_indices], 3),
        "n_atom_id": atom_ids[neighborhoods.nitrogen_indices].ravel(),
        "neighbor_slot": np.tile(np.arange(3), len(neighborhoods.al_indices)),
        "partition_weight": neighborhoods.nitrogen_weights.ravel(),
        "neutral_center_weight": -1.0 / 3.0,
    })
    for axis, name in enumerate("xyz"):
        table[f"reference_delta_{name} (angstrom)"] = neighborhoods.reference_vectors[:, :, axis].ravel()
    return table
