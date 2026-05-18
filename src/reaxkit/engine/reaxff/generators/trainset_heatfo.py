"""
Materials Project heat-of-formation trainset generation utilities.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Optional
import os
import re

from mp_api.client import MPRester

from reaxkit.core.constants import const
from reaxkit.engine.common.geo_io import read_structure, write_structure
from reaxkit.engine.reaxff.generators.geo_generator import xtob


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
    elements: List[str]
    references_by_element: Dict[str, HeatFoReferenceSpec]
    out_dir: str | Path
    exact_element_count: bool = True
    api_key: Optional[str] = None
    max_materials: Optional[int] = None
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
class _WrittenStructure:
    identifier: str
    geo_path: Path
    composition_counts: Dict[str, int]
    total_atoms: int


def parse_elements_csv(value: str) -> List[str]:
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


def parse_heatfo_references(value: str) -> Dict[str, HeatFoReferenceSpec]:
    """
    Parse references in the form:
        "Ba=Babcc_opt:2,B=B_alp:12,O=O2:2"
    """
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


def _to_plain_dict(doc_obj) -> dict:
    if hasattr(doc_obj, "model_dump"):
        return dict(doc_obj.model_dump())
    if hasattr(doc_obj, "dict"):
        return dict(doc_obj.dict())
    if isinstance(doc_obj, dict):
        return doc_obj
    return {}


def _doc_field(doc_obj, key: str, default=None):
    if hasattr(doc_obj, key):
        value = getattr(doc_obj, key)
        if value is not None:
            return value
    doc = _to_plain_dict(doc_obj)
    return doc.get(key, default)


def _clean_formula(formula_pretty: str) -> str:
    formula_ascii = str(formula_pretty).translate(_SUBSCRIPT_TRANSLATION)
    formula_ascii = formula_ascii.replace(" ", "")
    formula_ascii = formula_ascii.replace("(", "_").replace(")", "_")
    formula_ascii = re.sub(r"[^A-Za-z0-9_]+", "", formula_ascii)
    formula_ascii = re.sub(r"_+", "_", formula_ascii).strip("_")
    return formula_ascii or "unknown_formula"


def _extract_crystal_system(doc_obj) -> str:
    symmetry = _doc_field(doc_obj, "symmetry")
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


def _search_summary_docs(
    mpr: MPRester,
    *,
    elements: List[str],
    exact_element_count: bool,
    max_materials: Optional[int],
) -> List[object]:
    fields = [
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

    picked: Dict[str, object] = {}
    for subset in subsets:
        docs = mpr.materials.summary.search(
            elements=subset,
            num_elements=len(subset),
            fields=fields,
        )
        for doc in docs:
            material_id = str(_doc_field(doc, "material_id", "")).strip()
            if not material_id:
                continue

            doc_elements = _doc_field(doc, "elements", None)
            if doc_elements:
                doc_element_set = {str(e) for e in doc_elements}
            else:
                structure = _doc_field(doc, "structure", None)
                if structure is None:
                    continue
                doc_element_set = set(_extract_integer_composition_counts(structure).keys())

            if not doc_element_set.issubset(allowed):
                continue
            if exact_element_count and doc_element_set != allowed:
                continue

            formation_e = _doc_field(doc, "formation_energy_per_atom", None)
            if formation_e is None:
                continue
            if _doc_field(doc, "structure", None) is None:
                continue

            picked[material_id] = doc
            if max_materials is not None and len(picked) >= max_materials:
                return list(picked.values())
    return list(picked.values())


def _extract_energy_above_hull(doc_obj) -> float:
    eah = _doc_field(doc_obj, "energy_above_hull", None)
    if eah is not None:
        return float(eah)
    # Fallback for stable entries when explicit E_hull is missing.
    is_stable = _doc_field(doc_obj, "is_stable", None)
    if bool(is_stable):
        return 0.0
    return float("inf")


def _pick_unary_reference_doc(mpr: MPRester, *, element: str) -> object:
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

    candidates = []
    for idx, doc in enumerate(docs):
        fep = _doc_field(doc, "formation_energy_per_atom", None)
        structure = _doc_field(doc, "structure", None)
        if fep is None or structure is None:
            continue
        if abs(float(fep)) > 1e-8:
            continue
        eah = _extract_energy_above_hull(doc)
        candidates.append((eah, idx, doc))

    if not candidates:
        raise ValueError(
            f"No unary reference for {element} satisfies formation_energy_per_atom = 0."
        )

    candidates.sort(key=lambda item: (item[0], item[1]))
    return candidates[0][2]


def _write_structure_triplet(
    *,
    doc_obj,
    cif_dir: Path,
    xyz_dir: Path,
    geo_dir: Path,
) -> _WrittenStructure:
    material_id = str(_doc_field(doc_obj, "material_id"))
    formula_pretty = str(_doc_field(doc_obj, "formula_pretty", material_id))
    crystal_system = _extract_crystal_system(doc_obj)
    structure = _doc_field(doc_obj, "structure")
    if structure is None:
        raise ValueError(f"{material_id}: missing structure.")

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


def generate_heatfo_trainset_from_mp(spec: MaterialsProjectHeatFoSpec) -> MaterialsProjectHeatFoResult:
    elements = parse_elements_csv(",".join(spec.elements))
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

    with MPRester(api_key) as mpr:
        missing_refs = [e for e in elements if e not in references_by_element]
        if missing_refs:
            if spec.verbose:
                print(f"[MP] Auto-discovering unary references for: {missing_refs}")
            for element in missing_refs:
                ref_doc = _pick_unary_reference_doc(mpr, element=element)
                ref_written = _write_structure_triplet(
                    doc_obj=ref_doc,
                    cif_dir=cif_dir,
                    xyz_dir=xyz_dir,
                    geo_dir=geo_dir,
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

        docs = _search_summary_docs(
            mpr,
            elements=elements,
            exact_element_count=bool(spec.exact_element_count),
            max_materials=spec.max_materials,
        )

        if spec.verbose:
            print(f"[MP] Selected {len(docs)} systems for elements={elements}")

        for doc in docs:
            try:
                written = _write_structure_triplet(
                    doc_obj=doc,
                    cif_dir=cif_dir,
                    xyz_dir=xyz_dir,
                    geo_dir=geo_dir,
                )

                formation_e_pa = float(_doc_field(doc, "formation_energy_per_atom"))
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
                    mid = str(_doc_field(doc, "material_id", "unknown"))
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
                fout.write(text + "\n")

    return MaterialsProjectHeatFoResult(
        out_dir=out_dir,
        trainset_path=trainset_path,
        concatenated_geo_path=concatenated_geo_path,
        sources_dir=sources_dir,
        generated_count=len(generated_identifiers),
        identifiers=generated_identifiers,
    )
