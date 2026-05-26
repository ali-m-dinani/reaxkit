"""
Trainset YAML settings generation and orchestration utilities.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional
import csv
import os
import shutil

from reaxkit.engine.reaxff.generators.trainset_elastic_energy import (
    BulkEnergySpec,
    CellSpec,
    ElasticEnergySpec,
    _generate_trainset_energy_with_source_note,
    _write_trainset_energy,
)
from reaxkit.engine.reaxff.generators.trainset_elastic_geometry import (
    StrainedGeometrySpec,
    _generate_strained_geometries,
    _write_strained_geometries,
)


DEFAULT_CIJ_GPA = {
    "c11": 287,
    "c22": 287,
    "c33": 219,
    "c12": 100,
    "c13": 144,
    "c23": 144,
    "c44": 76,
    "c55": 76,
    "c66": 50,
}

DEFAULT_CELL = CellSpec(a=2.85086, b=2.85086, c=3.49456, alpha=90.0, beta=90.0, gamma=90.0)
DEFAULT_TABLES = {
    "bulk": "EvsStrain_bulk.dat",
    "c11": "EvsStrain_c11.dat",
    "c22": "EvsStrain_c22.dat",
    "c33": "EvsStrain_c33.dat",
    "c12": "EvsStrain_c12.dat",
    "c13": "EvsStrain_c13.dat",
    "c23": "EvsStrain_c23.dat",
    "c44": "EvsStrain_c44.dat",
    "c55": "EvsStrain_c55.dat",
    "c66": "EvsStrain_c66.dat",
}


def _concat_geo_strained(out_dir: Path) -> Path | None:
    geo_dir = out_dir / "structures" / "geo_strained"
    all_geo_file = out_dir / "geo"
    if not geo_dir.exists():
        return None
    geo_files = sorted([*geo_dir.glob("*.bgf"), *geo_dir.glob("*.geo")])
    if not geo_files:
        return None
    with all_geo_file.open("w", encoding="utf-8") as fout:
        for geo_file in geo_files:
            text = geo_file.read_text(encoding="utf-8").rstrip()
            if text:
                fout.write(text + "\n\n")
    return all_geo_file


def _normalize_formula_for_eos_label(formula: str) -> str:
    text = str(formula or "").strip()
    if not text:
        return "UnknownFormula"
    text = text.replace(" ", "")
    if "(" not in text and ")" not in text:
        return text
    text = text.replace("(", "_").replace(")", "_")
    out: list[str] = []
    for i, ch in enumerate(text):
        if i > 0:
            prev = text[i - 1]
            if (prev.isalpha() and ch.isdigit()) or (prev.isdigit() and ch.isalpha()):
                if out and out[-1] != "_":
                    out.append("_")
        out.append(ch)
    normalized = "".join(out)
    while "__" in normalized:
        normalized = normalized.replace("__", "_")
    return normalized.strip("_") or "UnknownFormula"


def _build_eos_source_note(cfg: dict) -> str | None:
    metadata = cfg.get("metadata", {}) or {}
    mp_id = str(metadata.get("mp_id", "") or "").strip()
    formula = str(metadata.get("formula_pretty", "") or "").strip()
    crystal = str(metadata.get("crystal_system", "") or "").strip()
    if not (mp_id and formula and crystal):
        return None
    crystal_label = crystal[:1].upper() + crystal[1:].lower()
    formula_label = _normalize_formula_for_eos_label(formula)
    return f"# EOS data based on {formula_label} {crystal_label} {mp_id}"


def _extract_material_metadata_from_yaml(yaml_path: str | Path) -> Dict[str, str]:
    try:
        import yaml
    except ImportError:
        return {"formula_pretty": "", "crystal_system": "", "mp_id": ""}
    path = Path(yaml_path)
    if not path.exists():
        return {"formula_pretty": "", "crystal_system": "", "mp_id": ""}
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        return {"formula_pretty": "", "crystal_system": "", "mp_id": ""}
    meta = data.get("metadata", {}) or {}
    return {
        "formula_pretty": str(meta.get("formula_pretty", "") or "").strip(),
        "crystal_system": str(meta.get("crystal_system", "") or "").strip(),
        "mp_id": str(meta.get("mp_id", "") or "").strip(),
    }


def _collect_cell_warnings_from_yaml(yaml_path: str | Path) -> list[str]:
    try:
        import yaml
    except ImportError:
        return []
    path = Path(yaml_path)
    if not path.exists():
        return []
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        return []
    bulk_cfg = data.get("bulk", {}) or {}
    elastic_cfg = data.get("elastic", {}) or {}
    bulk_cell_cfg = bulk_cfg.get("cell")
    elastic_cell_cfg = elastic_cfg.get("cell", bulk_cell_cfg)
    if not isinstance(bulk_cell_cfg, dict) or not isinstance(elastic_cell_cfg, dict):
        return []

    warnings_list: list[str] = []
    try:
        bulk_cell = CellSpec(**bulk_cell_cfg)
        if not _is_orthogonal_cell(bulk_cell):
            warnings_list.append(
                f"Bulk cell is non-orthogonal (angles = [{bulk_cell.alpha}, {bulk_cell.beta}, {bulk_cell.gamma}]). "
                "Elastic energy targets assume an orthogonal lattice."
            )
    except Exception:
        pass
    try:
        elastic_cell = CellSpec(**elastic_cell_cfg)
        if not _is_orthogonal_cell(elastic_cell):
            warnings_list.append(
                f"Elastic cell is non-orthogonal (angles = [{elastic_cell.alpha}, {elastic_cell.beta}, {elastic_cell.gamma}]). "
                "Elastic energy targets assume an orthogonal lattice."
            )
    except Exception:
        pass
    return warnings_list


def _is_orthogonal_cell(cell: CellSpec, *, tol: float = 1e-6) -> bool:
    return (
        abs(float(cell.alpha) - 90.0) <= tol
        and abs(float(cell.beta) - 90.0) <= tol
        and abs(float(cell.gamma) - 90.0) <= tol
    )


@dataclass(frozen=True)
class TrainsetSettingsSpec:
    out_path: str
    name: str = "AlN example"
    source: str = "manual"
    mp_id: Optional[str] = None
    formula_pretty: Optional[str] = None
    crystal_system: Optional[str] = None
    elastic_max_strain_percent: float = 3.0
    elastic_dstrain: float = 0.005
    cij_gpa: Dict[str, float] = field(default_factory=lambda: dict(DEFAULT_CIJ_GPA))
    elastic_cell: CellSpec = DEFAULT_CELL
    B0_gpa: float = 174.0
    B0_prime: float = 1.5
    bulk_max_volumetric_strain_percent: float = 6.0
    bulk_dstrain_linear: float = 0.004
    bulk_cell: CellSpec = DEFAULT_CELL
    trainset_file: str = "trainset_elastic.in"
    tables: Dict[str, str] = field(default_factory=lambda: dict(DEFAULT_TABLES))
    elastic_xyz: Optional[str | Path] = "ground_elastic.xyz"
    bulk_xyz: Optional[str | Path] = "null"
    geo_enable: bool = True
    geo_sort_by: Optional[str] = None


def _generate_trainset_settings_yaml_text(spec: TrainsetSettingsSpec) -> str:
    required_cij = ("c11", "c22", "c33", "c12", "c13", "c23", "c44", "c55", "c66")
    missing = [key for key in required_cij if key not in spec.cij_gpa]
    if missing:
        raise ValueError(f"cij_gpa is missing required keys: {missing}")

    def _q(value: str) -> str:
        escaped = value.replace("\\", "\\\\").replace('"', '\\"')
        return f'"{escaped}"'

    out_path = Path(spec.out_path)
    mp_id_yaml = "null" if spec.mp_id is None else _q(spec.mp_id)
    formula_pretty_yaml = "null" if spec.formula_pretty is None else _q(spec.formula_pretty)
    crystal_system_yaml = "null" if spec.crystal_system is None else _q(spec.crystal_system)
    geo_sort = "null" if spec.geo_sort_by is None else _q(str(spec.geo_sort_by))
    elastic_xyz = spec.elastic_xyz if spec.elastic_xyz is not None else "null"
    bulk_xyz = spec.bulk_xyz if spec.bulk_xyz is not None else "null"

    lines = [
        f"# {out_path.name}",
        "",
        "# This is the settings file used by ReaxKit's trainset generator.",
        "# Edit the values below (especially strains/moduli/cell) to match your material/system.",
        "",
        "metadata:",
        f"  name: {_q(spec.name)}",
        f"  source: {_q(spec.source)}  # 'manual' or 'materials_project'",
        f"  mp_id: {mp_id_yaml}  # Optional: e.g. \"mp-661\"",
        f"  formula_pretty: {formula_pretty_yaml}",
        f"  crystal_system: {crystal_system_yaml}",
        "",
        "units:",
        '  elastic_constants: "GPa"',
        '  bulk_modulus: "GPa"',
        '  angles: "deg"',
        '  lengths: "angstrom"',
        '  strain: "percent"',
        "",
        "# Elastic section: generates energy-vs-strain targets for c11..c66 (small linear strains).",
        "# Use this for harmonic elastic response around the reference cell.",
        "elastic:",
        f"  max_strain_percent: {spec.elastic_max_strain_percent}",
        f"  dstrain: {spec.elastic_dstrain}  # Strain step size (unitless). Default = 0.5% = 0.005",
        "  cij_gpa:",
    ]
    for key in required_cij:
        lines.append(f"    {key}: {spec.cij_gpa[key]}")
    lines.extend(
        [
            "",
            "  cell:",
            f"    a: {spec.elastic_cell.a}",
            f"    b: {spec.elastic_cell.b}",
            f"    c: {spec.elastic_cell.c}",
            f"    alpha: {spec.elastic_cell.alpha}",
            f"    beta: {spec.elastic_cell.beta}",
            f"    gamma: {spec.elastic_cell.gamma}",
            "",
            "# Input structures (XYZ). Used when geo.enable=true.",
            "structure 1:",
            f"  elastic_xyz: {elastic_xyz}  # required if geo.enable=true",
            "",
            "# Bulk section: generates energy-vs-volume targets using an EOS (Vinet) over a wider strain range.",
            "# Use this to constrain compressibility (B0, B0') around the reference volume.",
            "bulk:",
            f"  B0_gpa: {spec.B0_gpa}",
            f"  B0_prime: {spec.B0_prime}",
            f"  max_volumetric_strain_percent: {spec.bulk_max_volumetric_strain_percent}",
            (
                f"  dstrain_linear: {spec.bulk_dstrain_linear}  # Linear isotropic strain step "
                f"(unitless). Volume uses V=V0*(1+e)^3."
            ),
            "",
            "  cell:",
            f"    a: {spec.bulk_cell.a}",
            f"    b: {spec.bulk_cell.b}",
            f"    c: {spec.bulk_cell.c}",
            f"    alpha: {spec.bulk_cell.alpha}",
            f"    beta: {spec.bulk_cell.beta}",
            f"    gamma: {spec.bulk_cell.gamma}",
            "",
            "structure 2:",
            f"  bulk_xyz: {bulk_xyz}  # optional; if null, reuse elastic_xyz",
            "",
            "# Geometry generation options.",
            "geo:",
            f"  enable: {spec.geo_enable}",
            f"  sort_by: {geo_sort}",
            "",
            "# Output section: you can usually keep this as-is unless you want different filenames.",
            "output:",
            f"  trainset_file: {_q(spec.trainset_file)}",
            f"  xyz_strained_dir: {_q('structures/xyz_strained')}",
            f"  geo_strained_dir: {_q('structures/geo_strained')}",
            "  tables:",
        ]
    )
    for key in ("bulk", "c11", "c22", "c33", "c12", "c13", "c23", "c44", "c55", "c66"):
        lines.append(f"    {key}: {_q(spec.tables[key])}")
    lines.append("")
    return "\n".join(lines)


def gen_template_yaml_for_elastic_settings(
    spec: TrainsetSettingsSpec | None = None,
    *,
    out_path: str | Path | None = None,
) -> str | Path:
    """
    Public entrypoint for elastic settings template generation.

    - If `spec` is provided and `out_path` is omitted: returns YAML text.
    - If `out_path` is provided: writes YAML file and returns the output path.
    """
    if spec is not None and out_path is None:
        return _generate_trainset_settings_yaml_text(spec)

    if out_path is None:
        raise ValueError("Either provide spec (for text generation) or out_path (for file generation).")

    if spec is None:
        _write_trainset_settings_yaml(out_path=str(out_path))
    else:
        _write_trainset_settings_yaml(
            out_path=str(out_path),
            name=spec.name,
            source=spec.source,
            mp_id=spec.mp_id,
            elastic_max_strain_percent=spec.elastic_max_strain_percent,
            elastic_dstrain=spec.elastic_dstrain,
            cij_gpa=spec.cij_gpa,
            elastic_cell=spec.elastic_cell.as_dict(),
            B0_gpa=spec.B0_gpa,
            B0_prime=spec.B0_prime,
            bulk_max_volumetric_strain_percent=spec.bulk_max_volumetric_strain_percent,
            bulk_dstrain_linear=spec.bulk_dstrain_linear,
            bulk_cell=spec.bulk_cell.as_dict(),
            trainset_file=spec.trainset_file,
            tables=spec.tables,
            elastic_xyz=spec.elastic_xyz,
            bulk_xyz=spec.bulk_xyz,
            geo_enable=spec.geo_enable,
            geo_sort_by=spec.geo_sort_by,
        )
    return Path(out_path)


def _write_trainset_settings_yaml(
    *,
    out_path: str,
    name: str = "AlN example",
    source: str = "manual",
    mp_id: Optional[str] = None,
    formula_pretty: Optional[str] = None,
    crystal_system: Optional[str] = None,
    elastic_max_strain_percent: float = 3.0,
    elastic_dstrain: float = 0.005,
    cij_gpa: Optional[Dict[str, float]] = None,
    elastic_cell: Optional[Dict[str, float]] = None,
    B0_gpa: float = 174.0,
    B0_prime: float = 1.5,
    bulk_max_volumetric_strain_percent: float = 6.0,
    bulk_dstrain_linear: float = 0.004,
    bulk_cell: Optional[Dict[str, float]] = None,
    trainset_file: str = "trainset_elastic.in",
    tables: Optional[Dict[str, str]] = None,
    elastic_xyz: Optional[str | Path] = "ground_elastic.xyz",
    bulk_xyz: Optional[str | Path] = "null",
    geo_enable: bool = True,
    geo_sort_by: Optional[str] = None,
) -> None:
    spec = TrainsetSettingsSpec(
        out_path=out_path,
        name=name,
        source=source,
        mp_id=mp_id,
        formula_pretty=formula_pretty,
        crystal_system=crystal_system,
        elastic_max_strain_percent=elastic_max_strain_percent,
        elastic_dstrain=elastic_dstrain,
        cij_gpa=dict(DEFAULT_CIJ_GPA if cij_gpa is None else cij_gpa),
        elastic_cell=DEFAULT_CELL if elastic_cell is None else CellSpec(**elastic_cell),
        B0_gpa=B0_gpa,
        B0_prime=B0_prime,
        bulk_max_volumetric_strain_percent=bulk_max_volumetric_strain_percent,
        bulk_dstrain_linear=bulk_dstrain_linear,
        bulk_cell=DEFAULT_CELL if bulk_cell is None else CellSpec(**bulk_cell),
        trainset_file=trainset_file,
        tables=dict(DEFAULT_TABLES if tables is None else tables),
        elastic_xyz=elastic_xyz,
        bulk_xyz=bulk_xyz,
        geo_enable=geo_enable,
        geo_sort_by=geo_sort_by,
    )
    out_path_obj = Path(out_path)
    out_path_obj.parent.mkdir(parents=True, exist_ok=True)
    out_path_obj.write_text(_generate_trainset_settings_yaml_text(spec), encoding="utf-8")


def _generate_heatfo_settings_yaml_text() -> str:
    lines = [
        "# Heatfo trainset settings for `make-trainset-heatfo --input-mode yaml`",
        "# Edit values to match your system.",
        "",
        "source: mp  # Data source: mp (jarvis can be added later)",
        "input_mode: batch  # batch or material-id",
        "",
        "# Used in batch mode",
        "elements: [Ba, B, O]  # Allowed element pool",
        "element_count_scope: exact  # exact or up-to",
        "max_materials: null  # Optional integer cap",
        "crystallographic_setting_conversion: to-primitive  # to-conventional or to-primitive",
        "",
        "# Used in material-id mode",
        "mat_id: null  # Single material id for selected source, e.g. mp-661",
        "material_ids: null  # Optional list, e.g. [mp-661, mp-1234]",
        "",
        "# Optional explicit references; if null, unary references are auto-discovered from MP",
        "references: null  # or {Ba: \"Babcc_opt:2\", B: \"B_alp:12\", O: \"O2:2\"}",
        "",
        "weight: 1.0  # Trainset weight",
        "trainset_file: trainset_heatfo.in  # Output trainset filename",
        "geo_file: geo  # Output concatenated geo filename",
        "",
        "api_key: null  # Optional MP API key; if null, use MP_API_KEY env var",
        "verbose: false",
        "",
    ]
    return "\n".join(lines)


def _write_heatfo_settings_yaml(*, out_path: str | Path) -> Path:
    out_path_obj = Path(out_path)
    out_path_obj.parent.mkdir(parents=True, exist_ok=True)
    out_path_obj.write_text(_generate_heatfo_settings_yaml_text(), encoding="utf-8")
    return out_path_obj


def gen_template_yaml_for_heatfo_settings(*, out_path: str | Path) -> Path:
    """
    Public entrypoint for generating heatfo settings YAML template files.
    """
    return _write_heatfo_settings_yaml(out_path=out_path)


def _read_trainset_settings_yaml(yaml_path: str) -> dict:
    try:
        import yaml
    except ImportError as exc:
        raise ImportError("PyYAML is required to read trainset YAML files. Install with: pip install pyyaml") from exc

    yaml_path_obj = Path(yaml_path)
    if not yaml_path_obj.exists():
        raise FileNotFoundError(f"YAML file does not exist: {yaml_path_obj}")

    data = yaml.safe_load(yaml_path_obj.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("YAML root must be a mapping/dictionary.")

    missing = [key for key in ("elastic", "bulk", "output") if key not in data]
    if missing:
        raise ValueError(f"YAML is missing required sections: {missing}")

    geo_cfg = data.get("geo", {}) or {}
    if bool(geo_cfg.get("enable", False)):
        s1 = data.get("structure 1")
        if not isinstance(s1, dict):
            raise ValueError("Missing required section: 'structure 1' (required when geo.enable=true)")
        elastic_xyz = s1.get("elastic_xyz")
        if not elastic_xyz:
            raise ValueError("Missing required key: structure 1.elastic_xyz (required when geo.enable=true)")
        elastic_xyz_path = Path(elastic_xyz)
        if not elastic_xyz_path.is_absolute():
            elastic_xyz_path = (yaml_path_obj.parent / elastic_xyz_path).resolve()
        if not elastic_xyz_path.exists():
            raise FileNotFoundError(f"structure 1.elastic_xyz does not exist: {elastic_xyz_path}")
        data["structure 1"]["elastic_xyz"] = str(elastic_xyz_path)

        s2 = data.get("structure 2", {}) or {}
        if not isinstance(s2, dict):
            raise ValueError("'structure 2' must be a mapping if provided")
        bulk_xyz = s2.get("bulk_xyz")
        if bulk_xyz:
            bulk_xyz_path = Path(bulk_xyz)
            if not bulk_xyz_path.is_absolute():
                bulk_xyz_path = (yaml_path_obj.parent / bulk_xyz_path).resolve()
            if not bulk_xyz_path.exists():
                raise FileNotFoundError(f"structure 2.bulk_xyz does not exist: {bulk_xyz_path}")
            data.setdefault("structure 2", {})["bulk_xyz"] = str(bulk_xyz_path)
    return data


def _generate_trainset_from_yaml(
    yaml_path: str,
    out_dir: str,
    *,
    place_all_outputs_in_out_dir: bool = True,
    copy_input_xyz_into_out_dir: bool = True,
    skip_no_orthogonal: bool = False,
):
    cfg = _read_trainset_settings_yaml(yaml_path)
    yaml_path_p = Path(yaml_path).resolve()
    out_dir_p = Path(out_dir).resolve()
    out_dir_p.mkdir(parents=True, exist_ok=True)

    bulk_cfg = cfg["bulk"]
    elastic_cfg = cfg["elastic"]
    metadata = cfg.get("metadata", {}) or {}
    material_id = str(metadata.get("mp_id", "") or "").strip()
    material_prefix = f"for material ID [{material_id}] " if material_id else ""
    bulk_cell = CellSpec(**bulk_cfg["cell"])
    elastic_cell = CellSpec(**elastic_cfg.get("cell", bulk_cfg["cell"]))

    if skip_no_orthogonal:
        is_elastic_ortho = _is_orthogonal_cell(elastic_cell)
        is_bulk_ortho = _is_orthogonal_cell(bulk_cell)
        if not is_elastic_ortho:
            print(
                f"[Warning] {material_prefix}Elastic cell is non-orthogonal "
                f"(angles = [{elastic_cell.alpha}, {elastic_cell.beta}, {elastic_cell.gamma}]). "
                "Elastic energy targets assume an orthogonal lattice."
            )
        if not is_bulk_ortho:
            print(
                f"[Warning] {material_prefix}Bulk cell is non-orthogonal "
                f"(angles = [{bulk_cell.alpha}, {bulk_cell.beta}, {bulk_cell.gamma}]). "
                "Elastic energy targets assume an orthogonal lattice."
            )
        if (not is_elastic_ortho) or (not is_bulk_ortho):
            print("[Skip] Lattice is non-orthogonal; skipping this lattice.\n")
            return False

    source_note = _build_eos_source_note(cfg)
    energy_result = _generate_trainset_energy_with_source_note(
        BulkEnergySpec(
            bulk_modulus_gpa=bulk_cfg["B0_gpa"],
            bulk_modulus_pressure_derivative=bulk_cfg["B0_prime"],
            max_volumetric_strain_percent=bulk_cfg["max_volumetric_strain_percent"],
            cell=bulk_cell,
            linear_strain_step=bulk_cfg.get("dstrain_linear", 0.004),
        ),
        ElasticEnergySpec(
            elastic_constants_gpa=dict(elastic_cfg["cij_gpa"]),
            max_strain_percent=elastic_cfg["max_strain_percent"],
            volume_reference_cell=elastic_cell,
            strain_step=elastic_cfg.get("dstrain", 0.005),
        ),
        source_note=source_note,
    )
    for message in energy_result.warnings:
        print(f"[Warning] {material_prefix}{message}")

    written_energy_paths = _write_trainset_energy(
        energy_result,
        out_dir=out_dir_p,
        trainset_filename=cfg.get("output", {}).get("trainset_file", "trainset_elastic.in"),
    )
    dat_dir = out_dir_p / "volume energy data"
    dat_dir.mkdir(parents=True, exist_ok=True)
    for key, path in written_energy_paths.items():
        if key == "trainset":
            continue
        target_path = dat_dir / path.name
        if path.resolve() != target_path.resolve():
            shutil.move(str(path), str(target_path))

    geo_cfg = cfg.get("geo", {}) or {}
    if not bool(geo_cfg.get("enable", False)):
        return True

    elastic_xyz = Path(cfg["structure 1"]["elastic_xyz"])
    bulk_xyz_val = (cfg.get("structure 2", {}) or {}).get("bulk_xyz")
    bulk_xyz = Path(bulk_xyz_val) if bulk_xyz_val and str(bulk_xyz_val).lower() != "null" else None
    geo_out_dir = out_dir_p if place_all_outputs_in_out_dir else yaml_path_p.parent
    structures_dir = geo_out_dir / "structures"
    downloaded_structures_dir = structures_dir / "downloaded_structures"
    structures_dir.mkdir(parents=True, exist_ok=True)
    downloaded_structures_dir.mkdir(parents=True, exist_ok=True)

    if copy_input_xyz_into_out_dir:
        elastic_dst = downloaded_structures_dir / elastic_xyz.name
        if elastic_xyz.resolve() != elastic_dst.resolve():
            shutil.copy2(elastic_xyz, elastic_dst)
        elastic_xyz = elastic_dst
        if bulk_xyz is not None:
            bulk_dst = downloaded_structures_dir / bulk_xyz.name
            if bulk_xyz.resolve() != bulk_dst.resolve():
                shutil.copy2(bulk_xyz, bulk_dst)
            bulk_xyz = bulk_dst

    max_vol = bulk_cfg["max_volumetric_strain_percent"] / 100.0
    geometry_result = _generate_strained_geometries(
        StrainedGeometrySpec(
            elastic_xyz=str(elastic_xyz),
            bulk_xyz=None if bulk_xyz is None else str(bulk_xyz),
            elastic_cell=elastic_cell,
            bulk_cell=bulk_cell,
            max_strain_elastic=elastic_cfg["max_strain_percent"] / 100.0,
            dstrain_elastic=elastic_cfg.get("dstrain", 0.005),
            max_strain_bulk_linear=(1.0 + max_vol) ** (1.0 / 3.0) - 1.0,
            dstrain_bulk_linear=bulk_cfg.get("dstrain_linear", 0.004),
            sort_by=geo_cfg.get("sort_by"),
        )
    )
    _write_strained_geometries(geometry_result, out_dir=structures_dir, sort_by=geo_cfg.get("sort_by"))
    return True


def _gen_elastic_trainset_from_yaml_mode(
    *,
    yaml_path: str,
    out_dir: Path,
    skip_no_orthogonal: bool = False,
) -> dict[str, Any]:
    generated = _generate_trainset_from_yaml(
        yaml_path=yaml_path,
        out_dir=str(out_dir),
        skip_no_orthogonal=skip_no_orthogonal,
    )
    if not generated:
        print(f"[Done] Skipped non-orthogonal lattice for YAML: {yaml_path}")
        return {"mode": "yaml", "yaml_path": str(yaml_path), "skipped_non_orthogonal": True}
    geo_path = _concat_geo_strained(out_dir)
    if geo_path is not None:
        print(f"[Done] Concatenated strained geometries to: {geo_path}")
    print(f"[Done] Elastic trainset written to: {out_dir}")
    return {"mode": "yaml", "yaml_path": str(yaml_path)}


def _run_single_material_id_elastic_trainset(
    *,
    source_adapter,
    mat_id: str,
    out_dir: Path,
    out_yaml: str,
    structure_dir: str | Path | None,
    bulk_mode: str,
    crystallographic_setting_conversion: str,
    api_key: str,
    skip_no_orthogonal: bool,
    verbose: bool,
) -> tuple[str, bool]:
    out_yaml_path = out_dir / Path(str(out_yaml)).name
    structure_dir_path = Path(structure_dir) if structure_dir else (out_dir / "structures" / "downloaded_structures")
    structure_dir_path.mkdir(parents=True, exist_ok=True)
    result = source_adapter.generate_elastic_settings_yaml_from_material_id(
        mat_id=mat_id,
        out_yaml=str(out_yaml_path),
        structure_dir=str(structure_dir_path),
        bulk_mode=bulk_mode,
        crystallographic_setting_conversion=crystallographic_setting_conversion,
        api_key=api_key,
        verbose=verbose,
    )
    generated = _generate_trainset_from_yaml(
        yaml_path=result["yaml"],
        out_dir=str(out_dir),
        skip_no_orthogonal=skip_no_orthogonal,
    )
    if not generated:
        return result["yaml"], False
    _concat_geo_strained(out_dir)
    return result["yaml"], True


def _gen_elastic_trainset_from_material_id_mode(
    *,
    source_adapter,
    out_dir: Path,
    mat_id: str,
    out_yaml: str,
    structure_dir: str | Path | None,
    bulk_mode: str,
    crystallographic_setting_conversion: str,
    api_key: str,
    skip_no_orthogonal: bool,
    verbose: bool,
) -> dict[str, Any]:
    yaml_path, generated = _run_single_material_id_elastic_trainset(
        source_adapter=source_adapter,
        mat_id=mat_id,
        out_dir=out_dir,
        out_yaml=out_yaml,
        structure_dir=structure_dir,
        bulk_mode=bulk_mode,
        crystallographic_setting_conversion=crystallographic_setting_conversion,
        api_key=api_key,
        skip_no_orthogonal=skip_no_orthogonal,
        verbose=verbose,
    )
    if not generated:
        print(f"[Done] Generated settings from source '{source_adapter.source_name}': {yaml_path}")
        print(f"[Done] Skipped non-orthogonal lattice: {mat_id}")
        return {
            "mode": "material-id",
            "yaml_path": str(yaml_path),
            "mat_id": str(mat_id),
            "skipped_non_orthogonal": True,
        }
    print(f"[Done] Generated settings from source '{source_adapter.source_name}': {yaml_path}")
    print(f"[Done] Elastic trainset written to: {out_dir}")
    return {"mode": "material-id", "yaml_path": str(yaml_path), "mat_id": str(mat_id)}


def _gen_elastic_trainset_batch_mode(
    *,
    source_adapter,
    out_dir: Path,
    elements_csv: str,
    element_count_scope: str,
    max_materials: int | None,
    out_yaml: str,
    structure_dir: str | Path | None,
    bulk_mode: str,
    crystallographic_setting_conversion: str,
    api_key: str,
    skip_no_orthogonal: bool,
    verbose: bool,
) -> dict[str, Any]:
    from reaxkit.engine.reaxff.generators.trainset_heatfo import _parse_elements_csv
    from reaxkit.engine.reaxff.generators.trainset_mp import _mp_fetch_material_summary_metadata

    elements = _parse_elements_csv(str(elements_csv))
    mat_ids = source_adapter.search_material_ids_by_elements(
        api_key=api_key,
        elements=elements,
        exact_element_count=str(element_count_scope).strip().lower() == "exact",
        max_materials=max_materials,
    )
    if not mat_ids:
        raise ValueError(f"No source systems found for batch query in source '{source_adapter.source_name}'.")

    successful_root = out_dir / "successful"
    skipped_root = out_dir / "skipped"
    successful_root.mkdir(parents=True, exist_ok=True)
    skipped_root.mkdir(parents=True, exist_ok=True)

    ok = 0
    skipped = 0
    skipped_non_orthogonal = 0
    status_rows: list[dict[str, str]] = []
    for idx, mat_id in enumerate(mat_ids):
        if idx > 0:
            print("")
        target_dir = successful_root / mat_id
        try:
            yaml_path, generated = _run_single_material_id_elastic_trainset(
                source_adapter=source_adapter,
                mat_id=mat_id,
                out_dir=target_dir,
                out_yaml=out_yaml,
                structure_dir=structure_dir,
                bulk_mode=bulk_mode,
                crystallographic_setting_conversion=crystallographic_setting_conversion,
                api_key=api_key,
                skip_no_orthogonal=skip_no_orthogonal,
                verbose=verbose,
            )
            meta = _extract_material_metadata_from_yaml(yaml_path)
            warnings_list = _collect_cell_warnings_from_yaml(yaml_path)
            warning_text = " | ".join(warnings_list)
            if generated:
                ok += 1
                status_rows.append(
                    {
                        "chemical_formula": meta.get("formula_pretty", ""),
                        "crystal_system": meta.get("crystal_system", ""),
                        "material_id": meta.get("mp_id", mat_id) or mat_id,
                        "status": "success",
                        "warning": warning_text,
                    }
                )
            else:
                skipped_non_orthogonal += 1
                skipped_target = skipped_root / mat_id
                if skipped_target.exists():
                    shutil.rmtree(skipped_target)
                if target_dir.exists():
                    shutil.move(str(target_dir), str(skipped_target))
                status_rows.append(
                    {
                        "chemical_formula": meta.get("formula_pretty", ""),
                        "crystal_system": meta.get("crystal_system", ""),
                        "material_id": meta.get("mp_id", mat_id) or mat_id,
                        "status": "skip",
                        "warning": warning_text,
                    }
                )
        except Exception as exc:
            skipped += 1
            skipped_target = skipped_root / mat_id
            if skipped_target.exists():
                shutil.rmtree(skipped_target)
            if target_dir.exists():
                shutil.move(str(target_dir), str(skipped_target))
            else:
                skipped_target.mkdir(parents=True, exist_ok=True)

            formula = ""
            crystal = ""
            material_id_for_row = mat_id
            if str(getattr(source_adapter, "source_name", "")).strip().lower() == "mp":
                try:
                    meta = _mp_fetch_material_summary_metadata(api_key=api_key, material_id=mat_id)
                    formula = meta.get("formula_pretty", "")
                    crystal = meta.get("crystal_system", "")
                    material_id_for_row = meta.get("material_id", mat_id) or mat_id
                except Exception:
                    pass
            status_rows.append(
                {
                    "chemical_formula": formula,
                    "crystal_system": crystal,
                    "material_id": material_id_for_row,
                    "status": "skip",
                    "warning": "",
                }
            )
            if verbose:
                print(f"[{source_adapter.source_name.upper()}][skip] {mat_id}: {exc}")

    csv_path = out_dir / "materials_status.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=["chemical_formula", "crystal_system", "material_id", "status", "warning"],
        )
        writer.writeheader()
        writer.writerows(status_rows)

    print(
        f"[Done] Elastic batch completed ({source_adapter.source_name}): "
        f"success={ok}, skipped={skipped}, skipped_non_orthogonal={skipped_non_orthogonal}, total={len(mat_ids)}"
    )
    print(f"[Done] Saved material status CSV: {csv_path}")
    return {
        "mode": "batch",
        "elements": elements,
        "mat_ids_total": len(mat_ids),
        "mat_ids_success": ok,
        "mat_ids_skipped": skipped,
        "mat_ids_skipped_non_orthogonal": skipped_non_orthogonal,
        "status_csv": str(csv_path),
    }


def gen_elastic_trainset(
    *,
    out_dir: str | Path,
    source: str = "mp",
    input_mode: str = "yaml",
    yaml_path: str | None = None,
    mat_id: str | None = None,
    elements: str | None = None,
    element_count_scope: str = "exact",
    max_materials: int | None = None,
    api_key: str | None = None,
    bulk_mode: str = "voigt",
    crystallographic_setting_conversion: str = "to-primitive",
    out_yaml: str = "trainset_settings_source.yaml",
    structure_dir: str | Path | None = None,
    skip_no_orthogonal: bool = False,
    verbose: bool = False,
) -> dict[str, Any]:
    """
    Public entrypoint for elastic trainset generation.

    Supports:
    - yaml mode
    - material-id mode
    - batch mode
    """
    from reaxkit.engine.reaxff.generators.trainset_source_adapter import _get_trainset_source_adapter

    source_adapter = _get_trainset_source_adapter(str(source))
    out_dir_path = Path(out_dir)
    out_dir_path.mkdir(parents=True, exist_ok=True)
    mode = str(input_mode).strip().lower()

    if mode == "yaml":
        if not yaml_path:
            raise ValueError("yaml mode requires yaml_path.")
        return _gen_elastic_trainset_from_yaml_mode(
            yaml_path=str(yaml_path),
            out_dir=out_dir_path,
            skip_no_orthogonal=bool(skip_no_orthogonal),
        )

    resolved_api_key = api_key or os.getenv("MP_API_KEY")
    if not resolved_api_key:
        raise ValueError(
            f"Missing API key for source '{source_adapter.source_name}'. "
            "For MP, provide --api-key or set MP_API_KEY."
        )

    if mode == "material-id":
        if not mat_id:
            raise ValueError("material-id mode requires mat_id.")
        return _gen_elastic_trainset_from_material_id_mode(
            source_adapter=source_adapter,
            out_dir=out_dir_path,
            mat_id=str(mat_id),
            out_yaml=out_yaml,
            structure_dir=structure_dir,
            bulk_mode=bulk_mode,
            crystallographic_setting_conversion=crystallographic_setting_conversion,
            api_key=resolved_api_key,
            skip_no_orthogonal=bool(skip_no_orthogonal),
            verbose=bool(verbose),
        )

    if mode == "batch":
        if not elements:
            raise ValueError("batch mode requires elements.")
        return _gen_elastic_trainset_batch_mode(
            source_adapter=source_adapter,
            out_dir=out_dir_path,
            elements_csv=str(elements),
            element_count_scope=element_count_scope,
            max_materials=max_materials,
            out_yaml=out_yaml,
            structure_dir=structure_dir,
            bulk_mode=bulk_mode,
            crystallographic_setting_conversion=crystallographic_setting_conversion,
            api_key=resolved_api_key,
            skip_no_orthogonal=bool(skip_no_orthogonal),
            verbose=bool(verbose),
        )

    raise ValueError(f"Unsupported input mode: {mode!r}")


def _gen_elastic_trainset_from_yaml(
    yaml_path: str,
    out_dir: str,
    *,
    place_all_outputs_in_out_dir: bool = True,
    copy_input_xyz_into_out_dir: bool = True,
):
    """
    Internal helper for elastic trainset generation from YAML.
    """
    return _generate_trainset_from_yaml(
        yaml_path=yaml_path,
        out_dir=out_dir,
        place_all_outputs_in_out_dir=place_all_outputs_in_out_dir,
        copy_input_xyz_into_out_dir=copy_input_xyz_into_out_dir,
    )
