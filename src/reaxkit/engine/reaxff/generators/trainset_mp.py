"""
Materials Project helpers for trainset generation.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, Literal, Optional
import os

from mp_api.client import MPRester

from reaxkit.engine.common.geo_io import read_structure, write_structure
from reaxkit.engine.reaxff.generators.trainset_elastic_energy import CellSpec
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


@dataclass(frozen=True)
class MaterialsProjectHeatFoDocs:
    docs: list[object]
    elements: list[str]


def mp_to_plain_dict(doc_obj) -> dict:
    if hasattr(doc_obj, "model_dump"):
        return dict(doc_obj.model_dump())
    if hasattr(doc_obj, "dict"):
        return dict(doc_obj.dict())
    if isinstance(doc_obj, dict):
        return doc_obj
    return {}


def mp_doc_field(doc_obj, key: str, default=None):
    if hasattr(doc_obj, key):
        value = getattr(doc_obj, key)
        if value is not None:
            return value
    doc = mp_to_plain_dict(doc_obj)
    return doc.get(key, default)


def mp_search_summary_docs_by_elements(
    *,
    mpr: MPRester,
    elements: list[str],
    exact_element_count: bool,
    max_materials: Optional[int] = None,
    fields: Optional[list[str]] = None,
) -> list[object]:
    fields = fields or [
        "material_id",
        "formula_pretty",
        "symmetry",
        "structure",
        "formation_energy_per_atom",
        "elements",
        "nelements",
    ]
    allowed = set(elements)
    subset_sizes = [len(elements)] if exact_element_count else list(range(1, len(elements) + 1))
    subsets = [list(combo) for size in subset_sizes for combo in combinations(elements, size)]

    picked: dict[str, object] = {}
    for subset in subsets:
        docs = mpr.materials.summary.search(
            elements=subset,
            num_elements=len(subset),
            fields=fields,
        )
        for doc in docs:
            material_id = str(mp_doc_field(doc, "material_id", "")).strip()
            if not material_id:
                continue

            doc_elements = mp_doc_field(doc, "elements", None)
            if doc_elements:
                doc_element_set = {str(e) for e in doc_elements}
            else:
                structure = mp_doc_field(doc, "structure", None)
                if structure is None:
                    continue
                doc_element_set = {str(e) for e in structure.composition.get_el_amt_dict().keys()}

            if not doc_element_set.issubset(allowed):
                continue
            if exact_element_count and doc_element_set != allowed:
                continue

            picked[material_id] = doc
            if max_materials is not None and len(picked) >= max_materials:
                return list(picked.values())
    return list(picked.values())


def mp_fetch_summary_docs_by_material_ids(
    *,
    mpr: MPRester,
    material_ids: list[str],
    fields: Optional[list[str]] = None,
) -> list[object]:
    fields = fields or [
        "material_id",
        "formula_pretty",
        "symmetry",
        "structure",
        "formation_energy_per_atom",
        "elements",
        "nelements",
    ]
    docs = mpr.materials.summary.search(material_ids=material_ids, fields=fields)
    found_map = {str(mp_doc_field(doc, "material_id", "")).strip(): doc for doc in docs}
    missing = [mid for mid in material_ids if mid not in found_map]
    if missing:
        raise ValueError(f"Materials Project ids not found or inaccessible: {missing}")
    return [found_map[mid] for mid in material_ids]


def mp_derive_elements_from_summary_docs(docs: list[object]) -> list[str]:
    ordered: list[str] = []
    seen = set()
    for doc in docs:
        structure = mp_doc_field(doc, "structure", None)
        if structure is None:
            continue
        for element in structure.composition.get_el_amt_dict().keys():
            e = str(element)
            if e not in seen:
                ordered.append(e)
                seen.add(e)
    return ordered


def mp_pick_unary_reference_doc(*, mpr: MPRester, element: str) -> object:
    fields = [
        "material_id",
        "formula_pretty",
        "symmetry",
        "structure",
        "formation_energy_per_atom",
        "energy_above_hull",
        "is_stable",
        "elements",
        "nelements",
    ]
    docs = mpr.materials.summary.search(
        elements=[element],
        num_elements=1,
        fields=fields,
    )
    if not docs:
        raise ValueError(f"No unary systems found for element {element}.")

    def _energy_above_hull(doc_obj) -> float:
        eah = mp_doc_field(doc_obj, "energy_above_hull", None)
        if eah is not None:
            return float(eah)
        if bool(mp_doc_field(doc_obj, "is_stable", None)):
            return 0.0
        return float("inf")

    candidates: list[tuple[float, int, object]] = []
    for idx, doc in enumerate(docs):
        fep = mp_doc_field(doc, "formation_energy_per_atom", None)
        structure = mp_doc_field(doc, "structure", None)
        if fep is None or structure is None:
            continue
        if abs(float(fep)) > 1e-8:
            continue
        candidates.append((_energy_above_hull(doc), idx, doc))

    if not candidates:
        raise ValueError(
            f"No unary reference for {element} satisfies formation_energy_per_atom = 0."
        )

    candidates.sort(key=lambda item: (item[0], item[1]))
    return candidates[0][2]


def mp_search_material_ids_by_elements(
    *,
    api_key: str,
    elements: list[str],
    exact_element_count: bool,
    max_materials: Optional[int] = None,
) -> list[str]:
    with MPRester(api_key) as mpr:
        docs = mp_search_summary_docs_by_elements(
            mpr=mpr,
            elements=elements,
            exact_element_count=exact_element_count,
            max_materials=max_materials,
            fields=["material_id", "elements", "structure", "nelements"],
        )
    ids: list[str] = []
    seen = set()
    for doc in docs:
        mid = str(mp_doc_field(doc, "material_id", "")).strip()
        if not mid or mid in seen:
            continue
        ids.append(mid)
        seen.add(mid)
    return ids


def mp_collect_heatfo_docs(
    *,
    api_key: str,
    elements: list[str],
    material_ids: Optional[list[str]] = None,
    exact_element_count: bool = True,
    max_materials: Optional[int] = None,
) -> MaterialsProjectHeatFoDocs:
    with MPRester(api_key) as mpr:
        if material_ids:
            docs = mp_fetch_summary_docs_by_material_ids(mpr=mpr, material_ids=material_ids)
            resolved_elements = list(elements) if elements else mp_derive_elements_from_summary_docs(docs)
        else:
            if not elements:
                raise ValueError("Heatfo batch mode requires elements.")
            docs = mp_search_summary_docs_by_elements(
                mpr=mpr,
                elements=elements,
                exact_element_count=exact_element_count,
                max_materials=max_materials,
            )
            resolved_elements = list(elements)
    return MaterialsProjectHeatFoDocs(docs=docs, elements=resolved_elements)


def mp_pick_unary_reference_docs(
    *,
    api_key: str,
    elements: list[str],
) -> Dict[str, object]:
    picked: Dict[str, object] = {}
    with MPRester(api_key) as mpr:
        for element in elements:
            picked[element] = mp_pick_unary_reference_doc(mpr=mpr, element=element)
    return picked


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
        sdoc = mp_fetch_summary_docs_by_material_ids(
            mpr=mpr,
            material_ids=[spec.mp_id],
            fields=["material_id", "formula_pretty", "structure"],
        )[0]
        structure = mp_doc_field(sdoc, "structure")
        if structure is None:
            raise ValueError(f"{spec.mp_id}: structure missing/unreadable.")
        lat = structure.lattice
        name = mp_doc_field(sdoc, "formula_pretty", None) or spec.mp_id
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
