"""
Trainset strained-geometry generation utilities.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from ase import Atoms
from ase.geometry import cellpar_to_cell, cell_to_cellpar

from reaxkit.engine.reaxff.generators.geo_generator import xtob
from reaxkit.engine.common.geo_io import read_structure, write_structure
from reaxkit.engine.reaxff.generators.trainset_elastic_energy import CellSpec


GEOMETRY_MODE_ORDER = ["bulk", "c11", "c22", "c33", "c12", "c13", "c23", "c44", "c55", "c66"]


@dataclass(frozen=True)
class StrainedGeometrySpec:
    elastic_xyz: str | Path
    bulk_xyz: Optional[str | Path]
    elastic_cell: CellSpec
    bulk_cell: CellSpec
    max_strain_elastic: float
    dstrain_elastic: float
    max_strain_bulk_linear: float
    dstrain_bulk_linear: float
    sort_by: Optional[str] = None


@dataclass(frozen=True)
class StrainedGeometryRecord:
    mode: str
    title: str
    atoms: Atoms
    box_lengths: tuple[float, float, float]
    box_angles: tuple[float, float, float]
    xyz_filename: str
    geo_filename: str


@dataclass(frozen=True)
class StrainedGeometryResult:
    records_by_mode: Dict[str, List[StrainedGeometryRecord]]


def _deformation_matrix(mode: str, eps: float) -> np.ndarray:
    identity = np.eye(3, dtype=float)
    if mode == "bulk":
        return np.diag([1.0 + eps, 1.0 + eps, 1.0 + eps])
    if mode == "c11":
        d = identity.copy()
        d[0, 0] = 1.0 + eps
        return d
    if mode == "c22":
        d = identity.copy()
        d[1, 1] = 1.0 + eps
        return d
    if mode == "c33":
        d = identity.copy()
        d[2, 2] = 1.0 + eps
        return d
    if mode in {"c12", "c13", "c23"}:
        u = 1.0 / np.sqrt(max(1e-30, 1.0 - eps * eps))
        d = identity.copy()
        if mode == "c12":
            d[0, 0] = u * (1.0 + eps)
            d[1, 1] = u * (1.0 - eps)
        elif mode == "c13":
            d[0, 0] = u * (1.0 + eps)
            d[2, 2] = u * (1.0 - eps)
        else:
            d[1, 1] = u * (1.0 + eps)
            d[2, 2] = u * (1.0 - eps)
        return d
    if mode in {"c44", "c55", "c66"}:
        u = 1.0 / (max(1e-30, 1.0 - eps * eps) ** (1.0 / 3.0))
        d = identity.copy()
        if mode == "c44":
            d[1, 2] = eps
            d[2, 1] = eps
        elif mode == "c55":
            d[0, 2] = eps
            d[2, 0] = eps
        else:
            d[0, 1] = eps
            d[1, 0] = eps
        return u * d
    raise ValueError(f"Unknown mode: {mode!r}")


def _symmetric_strain_grid(max_abs: float, step: float) -> List[float]:
    n = int(np.ceil(max_abs / step))
    grid = [k * step for k in range(-n, n + 1)]
    grid = [x for x in grid if abs(x) <= max_abs + 1e-12]
    if 0.0 not in grid:
        grid.append(0.0)
        grid.sort()
    return grid


def _strain_title(prefix: str, eps: float, idx_abs: int) -> str:
    if abs(eps) < 1e-15:
        return f"{prefix}_0"
    return f"{prefix}_{'c' if eps < 0 else 'e'}{idx_abs:04d}"


def _make_base_atoms_from_xyz_and_cell(xyz_path: str | Path, cell: np.ndarray) -> Atoms:
    atoms = read_structure(xyz_path, format="xyz")
    atoms.set_cell(cell, scale_atoms=False)
    atoms.set_pbc(True)
    return atoms


def generate_strained_geometries(spec: StrainedGeometrySpec) -> StrainedGeometryResult:
    def idx_abs_from_eps(eps: float, step: float) -> int:
        if abs(eps) < 1e-15:
            return 0
        return abs(int(round(eps / step)))

    cell_e = cellpar_to_cell([
        spec.elastic_cell.a,
        spec.elastic_cell.b,
        spec.elastic_cell.c,
        spec.elastic_cell.alpha,
        spec.elastic_cell.beta,
        spec.elastic_cell.gamma,
    ])
    cell_b = cellpar_to_cell([
        spec.bulk_cell.a,
        spec.bulk_cell.b,
        spec.bulk_cell.c,
        spec.bulk_cell.alpha,
        spec.bulk_cell.beta,
        spec.bulk_cell.gamma,
    ])

    base_e = _make_base_atoms_from_xyz_and_cell(spec.elastic_xyz, cell_e)
    frac_e = base_e.get_scaled_positions(wrap=False)

    if spec.bulk_xyz is None:
        base_b = base_e.copy()
        base_b.set_cell(cell_b, scale_atoms=False)
        base_b.set_pbc(True)
    else:
        base_b = _make_base_atoms_from_xyz_and_cell(spec.bulk_xyz, cell_b)
    frac_b = base_b.get_scaled_positions(wrap=False)

    out: Dict[str, List[StrainedGeometryRecord]] = {mode: [] for mode in GEOMETRY_MODE_ORDER}

    for eps in _symmetric_strain_grid(spec.max_strain_bulk_linear, spec.dstrain_bulk_linear):
        new_cell = _deformation_matrix("bulk", eps) @ cell_b
        a, b, c, alpha, beta, gamma = cell_to_cellpar(new_cell)
        title = _strain_title("bulk", eps, idx_abs_from_eps(eps, spec.dstrain_bulk_linear))
        atoms = base_b.copy()
        atoms.set_cell(new_cell, scale_atoms=False)
        atoms.set_scaled_positions(frac_b)
        out["bulk"].append(
            StrainedGeometryRecord(
                mode="bulk",
                title=title,
                atoms=atoms,
                box_lengths=(float(a), float(b), float(c)),
                box_angles=(float(alpha), float(beta), float(gamma)),
                xyz_filename=f"{title}.xyz",
                geo_filename=f"{title}.bgf",
            )
        )

    for mode in GEOMETRY_MODE_ORDER[1:]:
        for eps in _symmetric_strain_grid(spec.max_strain_elastic, spec.dstrain_elastic):
            new_cell = _deformation_matrix(mode, eps) @ cell_e
            a, b, c, alpha, beta, gamma = cell_to_cellpar(new_cell)
            title = _strain_title(mode, eps, idx_abs_from_eps(eps, spec.dstrain_elastic))
            atoms = base_e.copy()
            atoms.set_cell(new_cell, scale_atoms=False)
            atoms.set_scaled_positions(frac_e)
            out[mode].append(
                StrainedGeometryRecord(
                    mode=mode,
                    title=title,
                    atoms=atoms,
                    box_lengths=(float(a), float(b), float(c)),
                    box_angles=(float(alpha), float(beta), float(gamma)),
                    xyz_filename=f"{title}.xyz",
                    geo_filename=f"{title}.geo",
                )
            )

    return StrainedGeometryResult(records_by_mode=out)


def write_strained_geometries(
    result: StrainedGeometryResult,
    *,
    out_dir: str | Path,
    sort_by: Optional[str] = None,
) -> Dict[str, List[Path]]:
    out_dir = Path(out_dir)
    xyz_dir = out_dir / "xyz_strained"
    geo_dir = out_dir / "geo_strained"
    xyz_dir.mkdir(parents=True, exist_ok=True)
    geo_dir.mkdir(parents=True, exist_ok=True)

    written: Dict[str, List[Path]] = {mode: [] for mode in result.records_by_mode}
    for mode, records in result.records_by_mode.items():
        for record in records:
            xyz_path = xyz_dir / record.xyz_filename
            geo_path = geo_dir / record.geo_filename
            write_structure(record.atoms, xyz_path, format="xyz", comment=record.title)
            xtob(
                xyz_file=xyz_path,
                geo_file=geo_path,
                box_lengths=record.box_lengths,
                box_angles=record.box_angles,
                sort_by=sort_by,
                ascending=True,
            )
            written[mode].append(geo_path)
    return written


def generate_strained_geometries_with_xtob(
    *,
    elastic_xyz: str | Path,
    bulk_xyz: Optional[str | Path],
    elastic_cell: Dict[str, float],
    bulk_cell: Dict[str, float],
    max_strain_elastic: float,
    dstrain_elastic: float,
    max_strain_bulk_linear: float,
    dstrain_bulk_linear: float,
    out_dir: str | Path,
    sort_by: Optional[str] = None,
) -> Dict[str, List[Path]]:
    spec = StrainedGeometrySpec(
        elastic_xyz=elastic_xyz,
        bulk_xyz=bulk_xyz,
        elastic_cell=CellSpec(**elastic_cell),
        bulk_cell=CellSpec(**bulk_cell),
        max_strain_elastic=max_strain_elastic,
        dstrain_elastic=dstrain_elastic,
        max_strain_bulk_linear=max_strain_bulk_linear,
        dstrain_bulk_linear=dstrain_bulk_linear,
        sort_by=sort_by,
    )
    return write_strained_geometries(generate_strained_geometries(spec), out_dir=out_dir, sort_by=sort_by)
