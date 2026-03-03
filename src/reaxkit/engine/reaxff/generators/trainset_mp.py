"""
Materials Project helpers for trainset settings generation.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Literal, Optional
import os

from mp_api.client import MPRester

from reaxkit.engine.common.geo_io import read_structure, write_structure
from reaxkit.engine.reaxff.generators.trainset_energy import CellSpec
from reaxkit.engine.reaxff.generators.trainset_yaml import write_trainset_settings_yaml


BulkModulusMode = Literal["voigt", "reuss", "vrh"]


@dataclass(frozen=True)
class MaterialsProjectTrainsetSpec:
    mp_id: str
    out_yaml: str | Path
    structure_dir: Optional[str | Path] = None
    bulk_mode: BulkModulusMode = "vrh"
    api_key: Optional[str] = None
    verbose: bool = True


def _tensor6x6_to_cij_dict(t6: list[list[float]]) -> Dict[str, float]:
    if t6 is None or len(t6) != 6 or any(len(row) != 6 for row in t6):
        raise ValueError("Elastic tensor must be a 6x6 matrix.")
    f = lambda i, j: float(t6[i][j])
    return {
        "c11": f(0, 0), "c22": f(1, 1), "c33": f(2, 2),
        "c12": f(0, 1), "c13": f(0, 2), "c23": f(1, 2),
        "c44": f(3, 3), "c55": f(4, 4), "c66": f(5, 5),
    }


def _extract_tensor6(elastic_tensor_obj: Any):
    if elastic_tensor_obj is None:
        return None
    if hasattr(elastic_tensor_obj, "ieee_format") and elastic_tensor_obj.ieee_format is not None:
        return elastic_tensor_obj.ieee_format
    if hasattr(elastic_tensor_obj, "raw") and elastic_tensor_obj.raw is not None:
        return elastic_tensor_obj.raw
    if isinstance(elastic_tensor_obj, (list, tuple)):
        return list(elastic_tensor_obj)
    return None


def _pick_bulk_modulus(bm: Any, mode: BulkModulusMode) -> Optional[float]:
    if bm is None:
        return None
    value = getattr(bm, mode, None)
    return None if value is None else float(value)


def write_trainset_settings_from_mp(spec: MaterialsProjectTrainsetSpec) -> Dict[str, str]:
    api_key = spec.api_key or os.getenv("MP_API_KEY")
    if not api_key:
        raise RuntimeError("Set MP_API_KEY env var (or pass api_key=...).")

    out_yaml = Path(spec.out_yaml)
    out_yaml.parent.mkdir(parents=True, exist_ok=True)
    structure_dir = Path(spec.structure_dir) if spec.structure_dir is not None else out_yaml.parent
    structure_dir.mkdir(parents=True, exist_ok=True)

    base = spec.mp_id.replace(":", "_")
    cif_path = structure_dir / f"{base}.cif"
    xyz_path = structure_dir / f"{base}.xyz"

    with MPRester(api_key) as mpr:
        sdoc = mpr.materials.summary.search(
            material_ids=[spec.mp_id],
            fields=["material_id", "formula_pretty", "structure"],
        )[0]
        structure = sdoc.structure
        lat = structure.lattice
        name = getattr(sdoc, "formula_pretty", None) or spec.mp_id
        cell = CellSpec(
            a=float(lat.a),
            b=float(lat.b),
            c=float(lat.c),
            alpha=float(lat.alpha),
            beta=float(lat.beta),
            gamma=float(lat.gamma),
        )

        edocs = mpr.materials.elasticity.search(
            material_ids=[spec.mp_id],
            fields=["material_id", "elastic_tensor", "bulk_modulus"],
        )
        if not edocs:
            raise ValueError(f"No elasticity data for {spec.mp_id} (cannot populate elastic/bulk).")
        edoc = edocs[0]
        tensor6 = _extract_tensor6(getattr(edoc, "elastic_tensor", None))
        if tensor6 is None:
            raise ValueError(f"{spec.mp_id}: elastic_tensor missing/unreadable.")
        cij = _tensor6x6_to_cij_dict(tensor6)
        bulk_modulus = _pick_bulk_modulus(getattr(edoc, "bulk_modulus", None), spec.bulk_mode)
        if bulk_modulus is None:
            raise ValueError(f"{spec.mp_id}: bulk_modulus.{spec.bulk_mode} missing/unreadable.")

    structure.to(filename=str(cif_path), fmt="cif")
    atoms = read_structure(cif_path, format="cif")
    write_structure(atoms, xyz_path, format="xyz", comment=spec.mp_id)

    relative_xyz = xyz_path.resolve().relative_to(out_yaml.resolve().parent).as_posix()
    write_trainset_settings_yaml(
        out_path=str(out_yaml),
        name=f"{name} ({spec.mp_id})",
        source="materials_project",
        mp_id=spec.mp_id,
        cij_gpa=cij,
        B0_gpa=bulk_modulus,
        elastic_cell=cell.as_dict(),
        bulk_cell=cell.as_dict(),
        elastic_xyz=str(relative_xyz),
        bulk_xyz=str(relative_xyz),
        geo_enable=True,
    )

    if spec.verbose:
        print(f"[MP] CIF:  {cif_path}")
        print(f"[MP] XYZ:  {xyz_path}")
        print(f"[MP] YAML: {out_yaml}")

    return {"cif": str(cif_path), "xyz": str(xyz_path), "yaml": str(out_yaml)}


def generate_trainset_settings_yaml_from_mp_simple(
    *,
    mp_id: str,
    out_yaml: str | Path,
    structure_dir: Optional[str | Path] = None,
    bulk_mode: BulkModulusMode = "vrh",
    api_key: Optional[str] = None,
    verbose: bool = True,
) -> Dict[str, str]:
    return write_trainset_settings_from_mp(
        MaterialsProjectTrainsetSpec(
            mp_id=mp_id,
            out_yaml=out_yaml,
            structure_dir=structure_dir,
            bulk_mode=bulk_mode,
            api_key=api_key,
            verbose=verbose,
        )
    )
