"""
Materials Project helpers for trainset generation.

**Usage context**

- Template generation: Produce canonical text payloads for ReaxFF artifacts.
- File writing: Persist generated outputs to disk with stable formatting.
- Workflow integration: Support higher-level ReaxKit workflow commands.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, Literal, Optional
import os

from mp_api.client import MPRester

from reaxkit.engine.common.io.geo_io import read_structure, write_structure
from reaxkit.engine.reaxff.generators.trainset_elastic_energy import CellSpec
from reaxkit.engine.reaxff.generators.trainset_yaml import _write_trainset_settings_yaml


BulkModulusMode = Literal["voigt", "reuss", "vrh"]
CrystallographicSettingConversion = Literal["to-conventional", "to-primitive"]


@dataclass(frozen=True)
class MaterialsProjectTrainsetSpec:
    """Represent MaterialsProjectTrainsetSpec.

    Public class used by ReaxFF generator components.

    Fields
    ------
    mp_id : str
        Dataclass field.
    out_yaml : str | Path
        Dataclass field.
    structure_dir : Optional[str | Path]
        Dataclass field.
    bulk_mode : BulkModulusMode
        Dataclass field.
    crystallographic_setting_conversion : CrystallographicSettingConversion
        Dataclass field.
    api_key : Optional[str]
        Dataclass field.
    verbose : bool
        Dataclass field.
    """
    mp_id: str
    out_yaml: str | Path
    structure_dir: Optional[str | Path] = None
    bulk_mode: BulkModulusMode = "vrh"
    crystallographic_setting_conversion: CrystallographicSettingConversion = "to-primitive"
    api_key: Optional[str] = None
    verbose: bool = True


@dataclass(frozen=True)
class MaterialsProjectHeatFoDocs:
    """Represent MaterialsProjectHeatFoDocs.

    Public class used by ReaxFF generator components.

    Fields
    ------
    docs : list[object]
        Dataclass field.
    elements : list[str]
        Dataclass field.
    """
    docs: list[object]
    elements: list[str]


def _mp_to_plain_dict(doc_obj) -> dict:
    """Mp to plain dict."""
    if hasattr(doc_obj, "model_dump"):
        return dict(doc_obj.model_dump())
    if hasattr(doc_obj, "dict"):
        return dict(doc_obj.dict())
    if isinstance(doc_obj, dict):
        return doc_obj
    return {}


def _mp_doc_field(doc_obj, key: str, default=None):
    """Mp doc field."""
    if hasattr(doc_obj, key):
        value = getattr(doc_obj, key)
        if value is not None:
            return value
    doc = _mp_to_plain_dict(doc_obj)
    return doc.get(key, default)


def _mp_search_summary_docs_by_elements(
    *,
    mpr: MPRester,
    elements: list[str],
    exact_element_count: bool,
    max_materials: Optional[int] = None,
    fields: Optional[list[str]] = None,
) -> list[object]:
    """Mp search summary docs by elements."""
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
            material_id = str(_mp_doc_field(doc, "material_id", "")).strip()
            if not material_id:
                continue

            doc_elements = _mp_doc_field(doc, "elements", None)
            if doc_elements:
                doc_element_set = {str(e) for e in doc_elements}
            else:
                structure = _mp_doc_field(doc, "structure", None)
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


def _mp_fetch_summary_docs_by_material_ids(
    *,
    mpr: MPRester,
    material_ids: list[str],
    fields: Optional[list[str]] = None,
    log_retrieval: bool = True,
) -> list[object]:
    """Mp fetch summary docs by material ids."""
    fields = fields or [
        "material_id",
        "formula_pretty",
        "symmetry",
        "structure",
        "formation_energy_per_atom",
        "elements",
        "nelements",
    ]
    if log_retrieval:
        for idx, mid in enumerate(material_ids):
            if idx > 0:
                print("")
            print(f"[RetrievalStart] Retrieving data for material ID [{mid}]")
            print("Retrieving SummaryDoc documents")
    docs = mpr.materials.summary.search(material_ids=material_ids, fields=fields)
    found_map = {str(_mp_doc_field(doc, "material_id", "")).strip(): doc for doc in docs}
    missing = [mid for mid in material_ids if mid not in found_map]
    if missing:
        raise ValueError(f"Materials Project ids not found or inaccessible: {missing}")
    return [found_map[mid] for mid in material_ids]


def _mp_derive_elements_from_summary_docs(docs: list[object]) -> list[str]:
    """Mp derive elements from summary docs."""
    ordered: list[str] = []
    seen = set()
    for doc in docs:
        structure = _mp_doc_field(doc, "structure", None)
        if structure is None:
            continue
        for element in structure.composition.get_el_amt_dict().keys():
            e = str(element)
            if e not in seen:
                ordered.append(e)
                seen.add(e)
    return ordered


def _mp_pick_unary_reference_doc(*, mpr: MPRester, element: str) -> object:
    """Mp pick unary reference doc."""
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
        """Energy above hull."""
        eah = _mp_doc_field(doc_obj, "energy_above_hull", None)
        if eah is not None:
            return float(eah)
        if bool(_mp_doc_field(doc_obj, "is_stable", None)):
            return 0.0
        return float("inf")

    candidates: list[tuple[float, int, object]] = []
    for idx, doc in enumerate(docs):
        fep = _mp_doc_field(doc, "formation_energy_per_atom", None)
        structure = _mp_doc_field(doc, "structure", None)
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


def _mp_search_material_ids_by_elements(
    *,
    api_key: str,
    elements: list[str],
    exact_element_count: bool,
    max_materials: Optional[int] = None,
) -> list[str]:
    """Mp search material ids by elements."""
    with MPRester(api_key, mute_progress_bars=True) as mpr:
        docs = _mp_search_summary_docs_by_elements(
            mpr=mpr,
            elements=elements,
            exact_element_count=exact_element_count,
            max_materials=max_materials,
            fields=["material_id", "elements", "structure", "nelements"],
        )
    ids: list[str] = []
    seen = set()
    for doc in docs:
        mid = str(_mp_doc_field(doc, "material_id", "")).strip()
        if not mid or mid in seen:
            continue
        ids.append(mid)
        seen.add(mid)
    return ids


def _mp_collect_heatfo_docs(
    *,
    api_key: str,
    elements: list[str],
    material_ids: Optional[list[str]] = None,
    exact_element_count: bool = True,
    max_materials: Optional[int] = None,
) -> MaterialsProjectHeatFoDocs:
    """Mp collect heatfo docs."""
    with MPRester(api_key, mute_progress_bars=True) as mpr:
        if material_ids:
            docs = _mp_fetch_summary_docs_by_material_ids(mpr=mpr, material_ids=material_ids)
            resolved_elements = list(elements) if elements else _mp_derive_elements_from_summary_docs(docs)
        else:
            if not elements:
                raise ValueError("Heatfo batch mode requires elements.")
            docs = _mp_search_summary_docs_by_elements(
                mpr=mpr,
                elements=elements,
                exact_element_count=exact_element_count,
                max_materials=max_materials,
            )
            resolved_elements = list(elements)
    return MaterialsProjectHeatFoDocs(docs=docs, elements=resolved_elements)


def _mp_pick_unary_reference_docs(
    *,
    api_key: str,
    elements: list[str],
) -> Dict[str, object]:
    """Mp pick unary reference docs."""
    picked: Dict[str, object] = {}
    with MPRester(api_key, mute_progress_bars=True) as mpr:
        for element in elements:
            picked[element] = _mp_pick_unary_reference_doc(mpr=mpr, element=element)
    return picked


def _tensor6x6_to_cij_dict(t6: list[list[float]]) -> Dict[str, float]:
    """Tensor6x6 to cij dict."""
    if t6 is None or len(t6) != 6 or any(len(row) != 6 for row in t6):
        raise ValueError("Elastic tensor must be a 6x6 matrix.")
    f = lambda i, j: float(t6[i][j])
    return {
        "c11": f(0, 0), "c22": f(1, 1), "c33": f(2, 2),
        "c12": f(0, 1), "c13": f(0, 2), "c23": f(1, 2),
        "c44": f(3, 3), "c55": f(4, 4), "c66": f(5, 5),
    }


def _extract_tensor6(elastic_tensor_obj: Any):
    """Extract tensor6."""
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
    """Pick bulk modulus."""
    if bm is None:
        return None
    value = getattr(bm, mode, None)
    return None if value is None else float(value)


def _extract_crystal_system(symmetry_obj: Any) -> Optional[str]:
    """Extract crystal system."""
    if symmetry_obj is None:
        return None
    crystal = getattr(symmetry_obj, "crystal_system", None)
    if crystal is None and isinstance(symmetry_obj, dict):
        crystal = symmetry_obj.get("crystal_system")
    if crystal is None:
        return None
    if hasattr(crystal, "value"):
        crystal = crystal.value
    text = str(crystal).strip()
    return text or None


def _mp_fetch_material_summary_metadata(
    *,
    api_key: str,
    material_id: str,
) -> Dict[str, str]:
    """Mp fetch material summary metadata."""
    with MPRester(api_key, mute_progress_bars=True) as mpr:
        doc = _mp_fetch_summary_docs_by_material_ids(
            mpr=mpr,
            material_ids=[material_id],
            fields=["material_id", "formula_pretty", "symmetry"],
            log_retrieval=False,
        )[0]
    return {
        "material_id": str(_mp_doc_field(doc, "material_id", material_id) or material_id),
        "formula_pretty": str(_mp_doc_field(doc, "formula_pretty", "") or ""),
        "crystal_system": str(_extract_crystal_system(_mp_doc_field(doc, "symmetry", None)) or ""),
    }


def _convert_structure_setting(structure: Any, conversion: str):
    """Convert structure setting."""
    conversion_mode = str(conversion).strip().lower()
    if conversion_mode not in {"to-conventional", "to-primitive"}:
        raise ValueError(
            "crystallographic_setting_conversion must be one of: "
            "'to-conventional', 'to-primitive'."
        )
    if conversion_mode == "to-conventional":
        converted = structure.to_conventional()
    else:
        conventional = structure.to_conventional()
        converted = conventional.get_primitive_structure()
    return _normalize_non_orthogonal_angle_to_gamma(converted)


def _is_close_90(value: float, tol: float = 1e-6) -> bool:
    """Is close 90."""
    return abs(float(value) - 90.0) <= tol


def _normalize_non_orthogonal_angle_to_gamma(structure: Any):
    """Normalize non orthogonal angle to gamma."""
    lat = structure.lattice
    alpha = float(lat.alpha)
    beta = float(lat.beta)
    gamma = float(lat.gamma)

    # If exactly one lattice angle is non-orthogonal, canonicalize axis order so that
    # the non-90 angle appears as gamma (a,b), matching expected ReaxFF-style ordering.
    non_orth = [not _is_close_90(alpha), not _is_close_90(beta), not _is_close_90(gamma)]
    if sum(non_orth) != 1:
        return structure
    if non_orth[2]:
        return structure

    if non_orth[0]:
        # alpha = angle(b,c) -> make it new gamma by (a', b', c') = (b, c, a)
        perm = (1, 2, 0)
    else:
        # beta = angle(a,c) -> make it new gamma by (a', b', c') = (a, c, b)
        perm = (0, 2, 1)

    m = lat.matrix
    new_matrix = [m[perm[0]], m[perm[1]], m[perm[2]]]
    frac = structure.frac_coords
    new_frac = [[f[perm[0]], f[perm[1]], f[perm[2]]] for f in frac]
    return structure.__class__(
        lattice=new_matrix,
        species=structure.species,
        coords=new_frac,
        coords_are_cartesian=False,
        site_properties=structure.site_properties,
    )


def _write_trainset_settings_from_mp(spec: MaterialsProjectTrainsetSpec) -> Dict[str, str]:
    """Write trainset settings from mp."""
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

    with MPRester(api_key, mute_progress_bars=True) as mpr:
        sdoc = _mp_fetch_summary_docs_by_material_ids(
            mpr=mpr,
            material_ids=[spec.mp_id],
            fields=["material_id", "formula_pretty", "structure", "symmetry"],
        )[0]
        structure = _mp_doc_field(sdoc, "structure")
        if structure is None:
            raise ValueError(f"{spec.mp_id}: structure missing/unreadable.")
        structure = _convert_structure_setting(structure, spec.crystallographic_setting_conversion)
        lat = structure.lattice
        formula_pretty = _mp_doc_field(sdoc, "formula_pretty", None)
        name = formula_pretty or spec.mp_id
        crystal_system = _extract_crystal_system(_mp_doc_field(sdoc, "symmetry", None))
        cell = CellSpec(
            a=float(lat.a),
            b=float(lat.b),
            c=float(lat.c),
            alpha=float(lat.alpha),
            beta=float(lat.beta),
            gamma=float(lat.gamma),
        )

        print("Retrieving ElasticityDoc documents")
        edocs = mpr.materials.elasticity.search(
            material_ids=[spec.mp_id],
            fields=["material_id", "elastic_tensor", "bulk_modulus"],
        )
        if not edocs:
            print("[RetrievalResult] No elastic data found!")
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
    _write_trainset_settings_yaml(
        out_path=str(out_yaml),
        name=f"{name} ({spec.mp_id})",
        source="materials_project",
        mp_id=spec.mp_id,
        formula_pretty=(str(formula_pretty) if formula_pretty is not None else None),
        crystal_system=crystal_system,
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


def _generate_trainset_settings_yaml_from_mp_simple(
    *,
    mp_id: str,
    out_yaml: str | Path,
    structure_dir: Optional[str | Path] = None,
    bulk_mode: BulkModulusMode = "vrh",
    crystallographic_setting_conversion: CrystallographicSettingConversion = "to-primitive",
    api_key: Optional[str] = None,
    verbose: bool = True,
) -> Dict[str, str]:
    """Generate trainset settings yaml from mp simple."""
    return _write_trainset_settings_from_mp(
        MaterialsProjectTrainsetSpec(
            mp_id=mp_id,
            out_yaml=out_yaml,
            structure_dir=structure_dir,
            bulk_mode=bulk_mode,
            crystallographic_setting_conversion=crystallographic_setting_conversion,
            api_key=api_key,
            verbose=verbose,
        )
    )
