"""
Trainset YAML settings generation and orchestration utilities.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional
import shutil

from reaxkit.engine.reaxff.generators.trainset_elastic_energy import (
    BulkEnergySpec,
    CellSpec,
    ElasticEnergySpec,
    generate_trainset_energy,
    write_trainset_energy,
)
from reaxkit.engine.reaxff.generators.trainset_elastic_geometry import (
    StrainedGeometrySpec,
    generate_strained_geometries,
    write_strained_geometries,
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


@dataclass(frozen=True)
class TrainsetSettingsSpec:
    out_path: str
    name: str = "AlN example"
    source: str = "manual"
    mp_id: Optional[str] = None
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


def generate_trainset_settings_yaml(spec: TrainsetSettingsSpec) -> str:
    required_cij = ("c11", "c22", "c33", "c12", "c13", "c23", "c44", "c55", "c66")
    missing = [key for key in required_cij if key not in spec.cij_gpa]
    if missing:
        raise ValueError(f"cij_gpa is missing required keys: {missing}")

    def _q(value: str) -> str:
        escaped = value.replace("\\", "\\\\").replace('"', '\\"')
        return f'"{escaped}"'

    out_path = Path(spec.out_path)
    mp_id_yaml = "null" if spec.mp_id is None else _q(spec.mp_id)
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
            f"  xyz_strained_dir: {_q('xyz_strained')}",
            f"  geo_strained_dir: {_q('geo_strained')}",
            "  tables:",
        ]
    )
    for key in ("bulk", "c11", "c22", "c33", "c12", "c13", "c23", "c44", "c55", "c66"):
        lines.append(f"    {key}: {_q(spec.tables[key])}")
    lines.append("")
    return "\n".join(lines)


def write_trainset_settings_yaml(
    *,
    out_path: str,
    name: str = "AlN example",
    source: str = "manual",
    mp_id: Optional[str] = None,
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
    out_path_obj.write_text(generate_trainset_settings_yaml(spec), encoding="utf-8")


def generate_heatfo_settings_yaml_text() -> str:
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


def write_heatfo_settings_yaml(*, out_path: str | Path) -> Path:
    out_path_obj = Path(out_path)
    out_path_obj.parent.mkdir(parents=True, exist_ok=True)
    out_path_obj.write_text(generate_heatfo_settings_yaml_text(), encoding="utf-8")
    return out_path_obj


def read_trainset_settings_yaml(yaml_path: str) -> dict:
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


def generate_trainset_from_yaml(
    yaml_path: str,
    out_dir: str,
    *,
    place_all_outputs_in_out_dir: bool = True,
    copy_input_xyz_into_out_dir: bool = True,
):
    cfg = read_trainset_settings_yaml(yaml_path)
    yaml_path_p = Path(yaml_path).resolve()
    out_dir_p = Path(out_dir).resolve()
    out_dir_p.mkdir(parents=True, exist_ok=True)

    bulk_cfg = cfg["bulk"]
    elastic_cfg = cfg["elastic"]
    bulk_cell = CellSpec(**bulk_cfg["cell"])
    elastic_cell = CellSpec(**elastic_cfg.get("cell", bulk_cfg["cell"]))

    energy_result = generate_trainset_energy(
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
    )
    write_trainset_energy(
        energy_result,
        out_dir=out_dir_p,
        trainset_filename=cfg.get("output", {}).get("trainset_file", "trainset_elastic.in"),
    )

    geo_cfg = cfg.get("geo", {}) or {}
    if not bool(geo_cfg.get("enable", False)):
        return

    elastic_xyz = Path(cfg["structure 1"]["elastic_xyz"])
    bulk_xyz_val = (cfg.get("structure 2", {}) or {}).get("bulk_xyz")
    bulk_xyz = Path(bulk_xyz_val) if bulk_xyz_val and str(bulk_xyz_val).lower() != "null" else None
    geo_out_dir = out_dir_p if place_all_outputs_in_out_dir else yaml_path_p.parent

    if copy_input_xyz_into_out_dir:
        elastic_dst = geo_out_dir / elastic_xyz.name
        if elastic_xyz.resolve() != elastic_dst.resolve():
            shutil.copy2(elastic_xyz, elastic_dst)
        elastic_xyz = elastic_dst
        if bulk_xyz is not None:
            bulk_dst = geo_out_dir / bulk_xyz.name
            if bulk_xyz.resolve() != bulk_dst.resolve():
                shutil.copy2(bulk_xyz, bulk_dst)
            bulk_xyz = bulk_dst

    max_vol = bulk_cfg["max_volumetric_strain_percent"] / 100.0
    geometry_result = generate_strained_geometries(
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
    write_strained_geometries(geometry_result, out_dir=geo_out_dir, sort_by=geo_cfg.get("sort_by"))
