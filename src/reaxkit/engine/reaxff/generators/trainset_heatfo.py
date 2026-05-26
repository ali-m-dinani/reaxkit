"""
Materials Project heat-of-formation trainset generation utilities.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional
import os
import re

from reaxkit.core.constants import const
from reaxkit.engine.common.io.geo_io import read_structure, write_structure
from reaxkit.engine.reaxff.generators.geo_generator import xtob
from reaxkit.engine.reaxff.generators.trainset_mp import (
    _convert_structure_setting,
    _mp_collect_heatfo_docs,
    _mp_doc_field,
    _mp_pick_unary_reference_docs,
)


_SUBSCRIPT_TRANSLATION = str.maketrans(
    "\u2080\u2081\u2082\u2083\u2084\u2085\u2086\u2087\u2088\u2089",
    "0123456789",
)


@dataclass(frozen=True)
class HeatFoReferenceSpec:
    iden: str
    atoms_per_structure: int


@dataclass(frozen=True)
class MaterialsProjectHeatFoSpec:
    out_dir: str | Path
    elements: List[str] = field(default_factory=list)
    material_ids: Optional[List[str]] = None
    references_by_element: Dict[str, HeatFoReferenceSpec] = field(default_factory=dict)
    exact_element_count: bool = True
    api_key: Optional[str] = None
    max_materials: Optional[int] = None
    crystallographic_setting_conversion: str = "to-primitive"
    weight: float = 1.0
    trainset_filename: str = "trainset_heatfo.in"
    concatenated_geo_filename: str = "geo"
    verbose: bool = True


@dataclass(frozen=True)
class MaterialsProjectHeatFoResult:
    out_dir: Path
    trainset_path: Path
    concatenated_geo_path: Path
    sources_dir: Path
    generated_count: int
    identifiers: List[str]


@dataclass(frozen=True)
class HeatFoTrainsetRunResult:
    result: MaterialsProjectHeatFoResult
    metadata: Dict[str, Any]


@dataclass(frozen=True)
class _WrittenStructure:
    identifier: str
    geo_path: Path
    composition_counts: Dict[str, int]
    total_atoms: int


def _parse_elements_csv(value: str) -> List[str]:
    elements = [token.strip() for token in re.split(r"[,\s]+", str(value)) if token.strip()]
    if not elements:
        raise ValueError("No elements were provided.")
    normalized: List[str] = []
    seen = set()
    for element in elements:
        e = element[:1].upper() + element[1:].lower()
        if e not in seen:
            normalized.append(e)
            seen.add(e)
    return normalized


def _parse_mp_ids_csv(value: str) -> List[str]:
    ids = [token.strip() for token in re.split(r"[,\s]+", str(value)) if token.strip()]
    out: List[str] = []
    seen = set()
    for material_id in ids:
        mid = str(material_id).strip()
        if not mid:
            continue
        if mid not in seen:
            out.append(mid)
            seen.add(mid)
    if not out:
        raise ValueError("No Materials Project ids were provided.")
    return out


def _parse_heatfo_references(value: str) -> Dict[str, HeatFoReferenceSpec]:
    refs: Dict[str, HeatFoReferenceSpec] = {}
    chunks = [token.strip() for token in re.split(r"[;,]", str(value)) if token.strip()]
    if not chunks:
        raise ValueError("No references were provided.")
    for chunk in chunks:
        if "=" not in chunk or ":" not in chunk:
            raise ValueError(
                f"Invalid reference entry {chunk!r}. "
                "Expected format element=identifier:atoms_per_structure."
            )
        element_part, right = chunk.split("=", 1)
        iden_part, atoms_part = right.rsplit(":", 1)
        element = element_part.strip()[:1].upper() + element_part.strip()[1:].lower()
        iden = iden_part.strip()
        if not iden:
            raise ValueError(f"Missing identifier in reference entry {chunk!r}.")
        try:
            atoms_per_structure = int(atoms_part.strip())
        except ValueError as exc:
            raise ValueError(f"Invalid atom count in reference entry {chunk!r}.") from exc
        if atoms_per_structure <= 0:
            raise ValueError(f"atoms_per_structure must be > 0 in {chunk!r}.")
        refs[element] = HeatFoReferenceSpec(iden=iden, atoms_per_structure=atoms_per_structure)
    return refs


def _parse_heatfo_reference_map(ref_obj) -> Dict[str, HeatFoReferenceSpec]:
    if ref_obj is None:
        return {}
    if isinstance(ref_obj, str):
        return _parse_heatfo_references(ref_obj)
    if not isinstance(ref_obj, dict):
        raise ValueError("heatfo YAML references must be a string or mapping.")
    out: Dict[str, HeatFoReferenceSpec] = {}
    for element_raw, value in ref_obj.items():
        element = str(element_raw).strip()
        if not element:
            continue
        e_norm = element[:1].upper() + element[1:].lower()
        if isinstance(value, str):
            text = value.strip()
            if ":" not in text:
                raise ValueError(
                    f"Invalid reference for element {element!r}. "
                    "Expected 'identifier:atoms_per_structure'."
                )
            iden, atoms = text.rsplit(":", 1)
            out[e_norm] = HeatFoReferenceSpec(iden=iden.strip(), atoms_per_structure=int(atoms.strip()))
        elif isinstance(value, dict):
            iden = str(value.get("iden", "")).strip()
            atoms_per_structure = int(value.get("atoms_per_structure"))
            if not iden:
                raise ValueError(f"Missing iden in reference for element {element!r}.")
            out[e_norm] = HeatFoReferenceSpec(iden=iden, atoms_per_structure=atoms_per_structure)
        else:
            raise ValueError(f"Unsupported reference value for element {element!r}: {value!r}")
    return out


def _load_heatfo_yaml(path: str | Path) -> dict:
    try:
        import yaml
    except ImportError as exc:
        raise ImportError("PyYAML is required for heatfo YAML mode. Install with: pip install pyyaml") from exc
    yaml_path = Path(path)
    if not yaml_path.exists():
        raise FileNotFoundError(f"YAML file does not exist: {yaml_path}")
    data = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Heatfo YAML root must be a mapping.")
    return data


def _clean_formula(formula_pretty: str) -> str:
    formula_ascii = str(formula_pretty).translate(_SUBSCRIPT_TRANSLATION)
    formula_ascii = formula_ascii.replace(" ", "")
    formula_ascii = formula_ascii.replace("(", "_").replace(")", "_")
    formula_ascii = re.sub(r"[^A-Za-z0-9_]+", "", formula_ascii)
    formula_ascii = re.sub(r"_+", "_", formula_ascii).strip("_")
    return formula_ascii or "unknown_formula"


def _extract_crystal_system(doc_obj) -> str:
    symmetry = _mp_doc_field(doc_obj, "symmetry")
    if symmetry is None:
        return "unknown"
    crystal = None
    if hasattr(symmetry, "crystal_system"):
        crystal = getattr(symmetry, "crystal_system")
    elif isinstance(symmetry, dict):
        crystal = symmetry.get("crystal_system")
    if crystal is None:
        return "unknown"
    if hasattr(crystal, "value"):
        crystal = crystal.value
    return re.sub(r"[^a-z0-9]+", "", str(crystal).lower()) or "unknown"


def _material_id_numeric_part(material_id: str) -> str:
    text = str(material_id).strip()
    if text.startswith("mp-"):
        text = text[3:]
    text = re.sub(r"[^0-9A-Za-z]+", "", text)
    return text or "unknown_id"


def _build_identifier(*, formula_pretty: str, crystal_system: str, material_id: str) -> str:
    return f"{_clean_formula(formula_pretty)}_{crystal_system}_{_material_id_numeric_part(material_id)}"


def _format_denominator(value: float) -> str:
    rounded = round(float(value))
    if abs(float(value) - rounded) < 1e-10:
        return str(int(rounded))
    return f"{float(value):.4f}".rstrip("0").rstrip(".")


def _extract_integer_composition_counts(structure_obj) -> Dict[str, int]:
    comp = structure_obj.composition.get_el_amt_dict()
    counts: Dict[str, int] = {}
    for element, amount in comp.items():
        value = float(amount)
        rounded = int(round(value))
        if abs(value - rounded) > 1e-6:
            raise ValueError(
                f"Non-integer composition detected for element {element}: {value}. "
                "Cannot build heatfo balancing coefficients safely."
            )
        if rounded > 0:
            counts[str(element)] = rounded
    return counts


def _write_structure_triplet(
    *,
    doc_obj,
    cif_dir: Path,
    xyz_dir: Path,
    geo_dir: Path,
    crystallographic_setting_conversion: str = "to-primitive",
) -> _WrittenStructure:
    material_id = str(_mp_doc_field(doc_obj, "material_id"))
    formula_pretty = str(_mp_doc_field(doc_obj, "formula_pretty", material_id))
    crystal_system = _extract_crystal_system(doc_obj)
    structure = _mp_doc_field(doc_obj, "structure")
    if structure is None:
        raise ValueError(f"{material_id}: missing structure.")
    structure = _convert_structure_setting(structure, crystallographic_setting_conversion)

    identifier = _build_identifier(
        formula_pretty=formula_pretty,
        crystal_system=crystal_system,
        material_id=material_id,
    )
    cif_path = cif_dir / f"{identifier}.cif"
    xyz_path = xyz_dir / f"{identifier}.xyz"
    geo_path = geo_dir / f"{identifier}.geo"

    structure.to(filename=str(cif_path), fmt="cif")
    atoms = read_structure(cif_path, format="cif")
    write_structure(atoms, xyz_path, format="xyz", comment=identifier)

    lattice = structure.lattice
    xtob(
        xyz_file=xyz_path,
        geo_file=geo_path,
        box_lengths=(float(lattice.a), float(lattice.b), float(lattice.c)),
        box_angles=(float(lattice.alpha), float(lattice.beta), float(lattice.gamma)),
    )

    composition_counts = _extract_integer_composition_counts(structure)
    total_atoms = int(sum(composition_counts.values()))
    if total_atoms <= 0:
        raise ValueError(f"{material_id}: invalid atom count ({total_atoms}).")

    return _WrittenStructure(
        identifier=identifier,
        geo_path=geo_path,
        composition_counts=composition_counts,
        total_atoms=total_atoms,
    )


def _build_energy_line(
    *,
    identifier: str,
    composition_counts: Dict[str, int],
    element_order: List[str],
    total_atoms: int,
    per_atom_heat_kcal_mol: float,
    weight: float,
    references_by_element: Dict[str, HeatFoReferenceSpec],
) -> str:
    terms = [f"+   {identifier}/{_format_denominator(total_atoms)}"]
    for element in element_order:
        if element not in composition_counts:
            continue
        if element not in references_by_element:
            raise ValueError(
                f"Missing reference for element {element!r}. "
                "Provide references for all elements appearing in selected systems."
            )
        ref = references_by_element[element]
        numerator = float(composition_counts[element])
        denominator = (float(total_atoms) * float(ref.atoms_per_structure)) / numerator
        terms.append(f"- {ref.iden}/{_format_denominator(denominator)}")
    terms_joined = " ".join(terms)
    return f"{weight:7.3f}  {terms_joined}   {per_atom_heat_kcal_mol:11.4f}"


def _generate_heatfo_trainset_from_mp(spec: MaterialsProjectHeatFoSpec) -> MaterialsProjectHeatFoResult:
    material_ids = _parse_mp_ids_csv(",".join(spec.material_ids)) if spec.material_ids else None
    elements = _parse_elements_csv(",".join(spec.elements)) if spec.elements else []
    references_by_element = {
        str(k)[:1].upper() + str(k)[1:].lower(): v for k, v in spec.references_by_element.items()
    }
    if spec.max_materials is not None and spec.max_materials <= 0:
        raise ValueError("max_materials must be > 0 when provided.")

    api_key = spec.api_key or os.getenv("MP_API_KEY")
    if not api_key:
        raise RuntimeError("Set MP_API_KEY env var (or pass api_key=...).")

    out_dir = Path(spec.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    sources_dir = out_dir / "sources"
    cif_dir = sources_dir / "cif"
    xyz_dir = sources_dir / "xyz"
    geo_dir = sources_dir / "geo"
    for d in (sources_dir, cif_dir, xyz_dir, geo_dir):
        d.mkdir(parents=True, exist_ok=True)

    kcalmol_to_ev = const("energy_kcalmol_to_eV")
    if kcalmol_to_ev is None or kcalmol_to_ev <= 0:
        raise RuntimeError("Missing/invalid constant: energy_kcalmol_to_eV.")
    ev_to_kcal_mol = 1.0 / float(kcalmol_to_ev)

    generated_identifiers: List[str] = []
    energy_lines: List[str] = []
    generated_geo_paths: List[Path] = []
    written_geo_seen: set[Path] = set()

    collected = _mp_collect_heatfo_docs(
        api_key=api_key,
        elements=elements,
        material_ids=material_ids,
        exact_element_count=bool(spec.exact_element_count),
        max_materials=spec.max_materials,
    )
    docs = collected.docs
    elements = collected.elements
    if not elements:
        raise ValueError("Could not determine element set for selected systems.")

    missing_refs = [e for e in elements if e not in references_by_element]
    if missing_refs:
        if spec.verbose:
            print(f"[MP] Auto-discovering unary references for: {missing_refs}")
        reference_docs = _mp_pick_unary_reference_docs(api_key=api_key, elements=missing_refs)
        for element in missing_refs:
            ref_written = _write_structure_triplet(
                doc_obj=reference_docs[element],
                cif_dir=cif_dir,
                xyz_dir=xyz_dir,
                geo_dir=geo_dir,
                crystallographic_setting_conversion=spec.crystallographic_setting_conversion,
            )
            ref_count = ref_written.composition_counts.get(element, 0)
            if ref_count <= 0:
                raise ValueError(
                    f"Auto reference for {element} has invalid composition: {ref_written.composition_counts}"
                )
            references_by_element[element] = HeatFoReferenceSpec(
                iden=ref_written.identifier,
                atoms_per_structure=ref_count,
            )
            if ref_written.geo_path not in written_geo_seen:
                generated_geo_paths.append(ref_written.geo_path)
                written_geo_seen.add(ref_written.geo_path)
            if spec.verbose:
                print(
                    f"[MP][ref] {element}: {ref_written.identifier} "
                    f"(atoms_per_structure={ref_count})"
                )

    if spec.verbose:
        if material_ids:
            print(f"[MP] Selected {len(docs)} systems for material_ids={material_ids}")
        else:
            print(f"[MP] Selected {len(docs)} systems for elements={elements}")

    for doc in docs:
        try:
            written = _write_structure_triplet(
                doc_obj=doc,
                cif_dir=cif_dir,
                xyz_dir=xyz_dir,
                geo_dir=geo_dir,
                crystallographic_setting_conversion=spec.crystallographic_setting_conversion,
            )

            formation_e_pa = float(_mp_doc_field(doc, "formation_energy_per_atom"))
            formation_e_total_kcal_mol = formation_e_pa * float(written.total_atoms) * ev_to_kcal_mol
            per_atom_heat_kcal_mol = formation_e_total_kcal_mol / float(written.total_atoms)

            energy_lines.append(
                _build_energy_line(
                    identifier=written.identifier,
                    composition_counts=written.composition_counts,
                    element_order=elements,
                    total_atoms=written.total_atoms,
                    per_atom_heat_kcal_mol=per_atom_heat_kcal_mol,
                    weight=float(spec.weight),
                    references_by_element=references_by_element,
                )
            )
            generated_identifiers.append(written.identifier)
            if written.geo_path not in written_geo_seen:
                generated_geo_paths.append(written.geo_path)
                written_geo_seen.add(written.geo_path)
        except Exception as exc:
            if spec.verbose:
                mid = str(_mp_doc_field(doc, "material_id", "unknown"))
                print(f"[MP][skip] {mid}: {exc}")

    trainset_path = out_dir / spec.trainset_filename
    element_label = ",".join(elements)
    trainset_content = "\n".join(
        [
            "ENERGY",
            f"# Heat of formation targets from Materials Project for systems containing {element_label}",
            *energy_lines,
            "ENDENERGY",
            "",
        ]
    )
    trainset_path.write_text(trainset_content, encoding="utf-8")

    concatenated_geo_path = out_dir / spec.concatenated_geo_filename
    with concatenated_geo_path.open("w", encoding="utf-8") as fout:
        for geo_path in sorted(generated_geo_paths):
            text = geo_path.read_text(encoding="utf-8").rstrip()
            if text:
                fout.write(text + "\n\n")

    return MaterialsProjectHeatFoResult(
        out_dir=out_dir,
        trainset_path=trainset_path,
        concatenated_geo_path=concatenated_geo_path,
        sources_dir=sources_dir,
        generated_count=len(generated_identifiers),
        identifiers=generated_identifiers,
    )


def gen_heatfo_trainset(
    *,
    out_dir: str | Path,
    source: str = "mp",
    input_mode: str = "batch",
    yaml_path: str | Path | None = None,
    mat_id: str | None = None,
    elements: str | None = None,
    references: str | None = None,
    element_count_scope: str = "exact",
    max_materials: int | None = None,
    crystallographic_setting_conversion: str = "to-primitive",
    weight: float = 1.0,
    trainset_file: str = "trainset_heatfo.in",
    geo_file: str = "geo",
    api_key: str | None = None,
    verbose: bool = False,
) -> HeatFoTrainsetRunResult:
    """
    Public entrypoint for heat-of-formation trainset generation.

    Supports:
    - yaml mode
    - material-id mode
    - batch mode
    """
    from reaxkit.engine.reaxff.generators.trainset_source_adapter import (
        HeatFoTrainsetRequest,
        _get_trainset_source_adapter,
    )

    out_dir_path = Path(out_dir)
    out_dir_path.mkdir(parents=True, exist_ok=True)
    mode = str(input_mode).strip().lower()

    if mode == "yaml":
        if not yaml_path:
            raise ValueError("yaml mode requires yaml_path.")
        data = _load_heatfo_yaml(yaml_path)
        yaml_source = str(data.get("source", source))
        source_adapter = _get_trainset_source_adapter(yaml_source)
        yaml_mode = str(data.get("input_mode", "batch")).strip().lower()

        yaml_elements = data.get("elements", [])
        if isinstance(yaml_elements, str):
            yaml_elements = _parse_elements_csv(yaml_elements)
        elif isinstance(yaml_elements, list):
            yaml_elements = _parse_elements_csv(",".join(str(x) for x in yaml_elements))
        else:
            yaml_elements = []

        material_ids = None
        if yaml_mode == "material-id":
            if isinstance(data.get("material_ids"), list) and data.get("material_ids"):
                material_ids = [str(x).strip() for x in data.get("material_ids") if str(x).strip()]
            elif data.get("mat_id"):
                material_ids = [str(data.get("mat_id")).strip()]
            elif data.get("mp_id"):
                material_ids = [str(data.get("mp_id")).strip()]
            if not material_ids:
                raise ValueError("heatfo YAML material-id mode requires mat_id (or legacy mp_id) or material_ids.")
        elif yaml_mode != "batch":
            raise ValueError("heatfo YAML input_mode must be 'batch' or 'material-id'.")

        refs = _parse_heatfo_reference_map(data.get("references"))
        result = source_adapter.generate_heatfo_trainset(
            HeatFoTrainsetRequest(
                out_dir=str(out_dir_path),
                elements=yaml_elements,
                material_ids=material_ids,
                references_by_element=refs,
                exact_element_count=str(data.get("element_count_scope", "exact")).strip().lower() == "exact",
                api_key=str(data.get("api_key")) if data.get("api_key") else (api_key or os.getenv("MP_API_KEY")),
                max_materials=(int(data["max_materials"]) if data.get("max_materials") is not None else None),
                crystallographic_setting_conversion=str(
                    data.get("crystallographic_setting_conversion", crystallographic_setting_conversion)
                ),
                weight=float(data.get("weight", 1.0)),
                trainset_filename=str(data.get("trainset_file", "trainset_heatfo.in")),
                concatenated_geo_filename=str(data.get("geo_file", "geo")),
                verbose=bool(data.get("verbose", verbose)),
            )
        )
        return HeatFoTrainsetRunResult(
            result=result,
            metadata={"mode": "yaml", "yaml_path": str(yaml_path), "generated_count": int(result.generated_count)},
        )

    source_adapter = _get_trainset_source_adapter(str(source))
    resolved_api_key = api_key or os.getenv("MP_API_KEY")

    if mode == "material-id":
        if not mat_id:
            raise ValueError("material-id mode requires mat_id.")
        refs = _parse_heatfo_references(str(references)) if references else {}
        parsed_elements = _parse_elements_csv(str(elements)) if elements else []
        result = source_adapter.generate_heatfo_trainset(
            HeatFoTrainsetRequest(
                out_dir=str(out_dir_path),
                elements=parsed_elements,
                material_ids=[str(mat_id)],
                references_by_element=refs,
                exact_element_count=True,
                api_key=resolved_api_key,
                max_materials=None,
                crystallographic_setting_conversion=str(crystallographic_setting_conversion),
                weight=float(weight),
                trainset_filename=str(trainset_file),
                concatenated_geo_filename=str(geo_file),
                verbose=bool(verbose),
            )
        )
        return HeatFoTrainsetRunResult(
            result=result,
            metadata={"mode": "material-id", "mat_id": str(mat_id), "generated_count": int(result.generated_count)},
        )

    if mode == "batch":
        if not elements:
            raise ValueError("batch mode requires elements.")
        refs = _parse_heatfo_references(str(references)) if references else {}
        parsed_elements = _parse_elements_csv(str(elements))
        result = source_adapter.generate_heatfo_trainset(
            HeatFoTrainsetRequest(
                out_dir=str(out_dir_path),
                elements=parsed_elements,
                material_ids=None,
                references_by_element=refs,
                exact_element_count=str(element_count_scope).strip().lower() == "exact",
                api_key=resolved_api_key,
                max_materials=max_materials,
                crystallographic_setting_conversion=str(crystallographic_setting_conversion),
                weight=float(weight),
                trainset_filename=str(trainset_file),
                concatenated_geo_filename=str(geo_file),
                verbose=bool(verbose),
            )
        )
        return HeatFoTrainsetRunResult(
            result=result,
            metadata={"mode": "batch", "elements": parsed_elements, "generated_count": int(result.generated_count)},
        )

    raise ValueError(f"Unsupported input mode: {mode!r}")
