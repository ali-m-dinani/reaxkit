"""Direct command workflows for trainset export and generation utilities."""

from __future__ import annotations

import argparse
import os
from itertools import combinations
from pathlib import Path
from typing import Dict, List

from mp_api.client import MPRester

from reaxkit.core.generator_runtime import (
    maybe_copy_output_to_dot,
    persist_generator_metadata,
    prepare_generator_output,
    print_saved_dirs,
)
from reaxkit.core.storage_layout import add_storage_cli_arguments
from reaxkit.domain.data_models import ForceFieldOptimizationTrainingSetData
from reaxkit.engine.reaxff.adapter import ReaxFFAdapter
from reaxkit.engine.reaxff.generators.trainset_heatfo import (
    HeatFoReferenceSpec,
    MaterialsProjectHeatFoSpec,
    generate_heatfo_trainset_from_mp,
    parse_elements_csv,
    parse_heatfo_references,
)
from reaxkit.engine.reaxff.generators.trainset_mp import generate_trainset_settings_yaml_from_mp_simple
from reaxkit.engine.reaxff.generators.trainset_yaml import generate_trainset_from_yaml, write_trainset_settings_yaml

TRAINSET_FILE_COMMANDS = (
    "export-trainset",
    "make-trainset-settings",
    "make-trainset-elastic",
    "make-trainset-heatfo",
)


def _load_trainset_tables(path: str) -> dict[str, object]:
    data = ReaxFFAdapter().load(
        ForceFieldOptimizationTrainingSetData,
        {"trainset": path, "input": path},
    )
    return {
        "CHARGE": data.charge,
        "HEATFO": data.heatfo,
        "GEOMETRY": data.geometry,
        "CELL_PARAMETERS": data.cell_parameters,
        "ENERGY": data.energy,
    }


def _ensure_source_mp(source: str) -> None:
    if str(source).strip().lower() != "mp":
        raise NotImplementedError(
            f"Unsupported source={source!r}. Supported today: 'mp'. "
            "Other sources (for example JARVIS) can be added later."
        )


def _concat_geo_strained(out_dir: Path) -> Path | None:
    geo_dir = out_dir / "geo_strained"
    all_geo_file = geo_dir / "all_trainset_geo.bgf"
    if not geo_dir.exists():
        return None
    bgf_files = sorted(geo_dir.glob("*.bgf"))
    if not bgf_files:
        return None
    with all_geo_file.open("w", encoding="utf-8") as fout:
        for bgf in bgf_files:
            fout.write(f"# ===== BEGIN {bgf.name} =====\n")
            with bgf.open("r", encoding="utf-8") as fin:
                fout.write(fin.read())
            fout.write(f"\n# ===== END {bgf.name} =====\n\n")
    return all_geo_file


def _search_mp_material_ids_by_elements(
    *,
    api_key: str,
    elements: List[str],
    element_count_scope: str,
    max_materials: int | None,
) -> List[str]:
    allowed = set(elements)
    exact = str(element_count_scope).strip().lower() == "exact"
    subset_sizes = [len(elements)] if exact else list(range(1, len(elements) + 1))
    subsets = [list(combo) for size in subset_sizes for combo in combinations(elements, size)]
    picked: Dict[str, str] = {}
    with MPRester(api_key) as mpr:
        for subset in subsets:
            docs = mpr.materials.summary.search(
                elements=subset,
                num_elements=len(subset),
                fields=["material_id", "elements", "structure"],
            )
            for doc in docs:
                material_id = str(getattr(doc, "material_id", "")).strip()
                if not material_id:
                    continue
                doc_elements = getattr(doc, "elements", None)
                if doc_elements is not None:
                    e_set = {str(e) for e in doc_elements}
                else:
                    structure = getattr(doc, "structure", None)
                    if structure is None:
                        continue
                    e_set = {str(e) for e in structure.composition.get_el_amt_dict().keys()}
                if not e_set.issubset(allowed):
                    continue
                if exact and e_set != allowed:
                    continue
                picked[material_id] = material_id
                if max_materials is not None and len(picked) >= max_materials:
                    return list(picked.keys())
    return list(picked.keys())


def _parse_heatfo_reference_map(ref_obj) -> Dict[str, HeatFoReferenceSpec]:
    if ref_obj is None:
        return {}
    if isinstance(ref_obj, str):
        return parse_heatfo_references(ref_obj)
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


def build_parser(parser: argparse.ArgumentParser, *, command: str) -> argparse.ArgumentParser:
    parser.formatter_class = argparse.RawTextHelpFormatter

    if command == "export-trainset":
        parser.description = (
            "Export trainset sections to CSV files.\n\n"
            "Examples:\n"
            "  reaxkit export-trainset --section all --output trainset_export\n"
            "  reaxkit export-trainset --section energy --trainset trainset.in --output energy_export"
        )
        parser.add_argument("--trainset", "--file", dest="trainset", default="trainset.in", help="Path to trainset.in")
        parser.add_argument("--section", default="all", help="Section: all, charge, heatfo, geometry, cell_parameters, energy")
        parser.add_argument("--output", default="trainset_export", help="Output directory for exported CSV files")
        parser.add_argument("--copy-to-dot", action="store_true", help="Also copy generated output to current directory")
    elif command == "make-trainset-settings":
        parser.description = (
            "Write a sample trainset settings YAML.\n\n"
            "Examples:\n"
            "  reaxkit make-trainset-settings\n"
            "  reaxkit make-trainset-settings --output trainset_settings.yaml"
        )
        parser.add_argument("--output", default="trainset_settings.yaml", help="Output YAML path")
        parser.add_argument("--copy-to-dot", action="store_true", help="Also copy generated output to current directory")
    elif command == "make-trainset-elastic":
        parser.description = (
            "Generate elastic trainsets.\n\n"
            "input-mode options:\n"
            "  yaml: use existing trainset YAML\n"
            "  material-id: fetch one MP id\n"
            "  batch: fetch many MP systems by elements"
        )
        parser.add_argument("--source", choices=["mp", "jarvis"], default="mp", help="Data source.")
        parser.add_argument("--input-mode", choices=["yaml", "material-id", "batch"], default="yaml")
        parser.add_argument("--yaml", default=None, help="Existing trainset_settings.yaml file (yaml mode).")
        parser.add_argument("--mp-id", default=None, help="Materials Project material id (material-id mode).")
        parser.add_argument("--elements", default=None, help="Comma-separated elements for batch mode, for example Ba,B,O")
        parser.add_argument("--element-count-scope", choices=["exact", "up-to"], default="exact")
        parser.add_argument("--max-materials", type=int, default=None, help="Optional cap for batch mode.")
        parser.add_argument("--api-key", default=None, help="Materials Project API key or set MP_API_KEY.")
        parser.add_argument("--bulk-mode", default="voigt", choices=["voigt", "reuss", "vrh"], help="MP bulk modulus source")
        parser.add_argument("--out-yaml", default="trainset_settings_mp.yaml", help="Generated YAML filename in MP modes.")
        parser.add_argument("--structure-dir", default=None, help="Directory for MP-downloaded structures.")
        parser.add_argument("--verbose", action="store_true", help="Verbose MP fetching/logging")
        parser.add_argument("--output", default="trainset_elastic_generated", help="Directory for outputs.")
        parser.add_argument("--copy-to-dot", action="store_true", help="Also copy generated output to current directory")
    elif command == "make-trainset-heatfo":
        parser.description = (
            "Generate heat-of-formation trainsets.\n\n"
            "input-mode options:\n"
            "  yaml: use heatfo YAML config\n"
            "  material-id: use one MP id\n"
            "  batch: fetch many MP systems by elements"
        )
        parser.add_argument("--source", choices=["mp", "jarvis"], default="mp", help="Data source.")
        parser.add_argument("--input-mode", choices=["yaml", "material-id", "batch"], default="batch")
        parser.add_argument("--yaml", default=None, help="Heatfo YAML settings file (yaml mode).")
        parser.add_argument("--mp-id", default=None, help="Materials Project material id (material-id mode).")
        parser.add_argument("--elements", default=None, help="Comma-separated elements for batch mode, for example Ba,B,O")
        parser.add_argument(
            "--references",
            default=None,
            help=(
                "Optional reference map: element=identifier:atoms,... "
                '(example: "Ba=Babcc_opt:2,B=B_alp:12,O=O2:2"). '
                "If omitted, unary references are auto-selected from MP."
            ),
        )
        parser.add_argument("--element-count-scope", choices=["exact", "up-to"], default="exact")
        parser.add_argument("--max-materials", type=int, default=None, help="Optional cap for batch mode.")
        parser.add_argument("--weight", type=float, default=1.0, help="Weight used for heatfo ENERGY lines.")
        parser.add_argument("--trainset-file", default="trainset_heatfo.in", help="Output trainset filename.")
        parser.add_argument("--geo-file", default="geo", help="Output concatenated geo filename.")
        parser.add_argument("--api-key", default=None, help="Materials Project API key or set MP_API_KEY.")
        parser.add_argument("--verbose", action="store_true", help="Verbose MP fetching/logging")
        parser.add_argument("--output", default="trainset_heatfo_generated", help="Directory for outputs.")
        parser.add_argument("--copy-to-dot", action="store_true", help="Also copy generated output to current directory")
    else:
        raise KeyError(f"Unsupported trainset file-tool command {command!r}.")

    add_storage_cli_arguments(parser)
    return parser


def _run_export_trainset(args: argparse.Namespace) -> int:
    outdir, layout = prepare_generator_output(args, command="export-trainset", output_value=str(args.output))
    tables = _load_trainset_tables(args.trainset)
    section = str(args.section).strip().lower()
    outdir.mkdir(parents=True, exist_ok=True)

    if section == "all":
        items = [(name, df) for name, df in tables.items() if df is not None and not df.empty]
    else:
        section_key = section.upper()
        if section_key in {"CELL", "CELL PARAMETERS"}:
            section_key = "CELL_PARAMETERS"
        df = tables.get(section_key)
        if df is None:
            raise ValueError(f"Unknown trainset section {args.section!r}.")
        items = [(section_key, df)]

    stem = Path(args.trainset).stem
    for sec_name, df in items:
        outpath = outdir / f"{stem}_{sec_name.lower()}.csv"
        df.to_csv(outpath, index=False)
        print(f"[Done] Exported section {sec_name} to {outpath}")
    persist_generator_metadata(
        args,
        command="export-trainset",
        output_path=outdir,
        layout=layout,
        copy_to_dot=bool(getattr(args, "copy_to_dot", False)),
    )
    copied = maybe_copy_output_to_dot(outdir, enabled=bool(getattr(args, "copy_to_dot", False)))
    dirs = [outdir]
    if copied is not None:
        dirs.append(copied.parent)
    print_saved_dirs(dirs)
    return 0


def _run_make_trainset_settings(args: argparse.Namespace) -> int:
    out, layout = prepare_generator_output(args, command="make-trainset-settings", output_value=str(args.output))
    write_trainset_settings_yaml(out_path=str(out))
    persist_generator_metadata(
        args,
        command="make-trainset-settings",
        output_path=out,
        layout=layout,
        copy_to_dot=bool(getattr(args, "copy_to_dot", False)),
    )
    copied = maybe_copy_output_to_dot(out, enabled=bool(getattr(args, "copy_to_dot", False)))
    dirs = [out.parent]
    if copied is not None:
        dirs.append(copied.parent)
    print_saved_dirs(dirs)
    return 0


def _run_make_trainset_elastic(args: argparse.Namespace) -> int:
    _ensure_source_mp(args.source)
    out_dir, layout = prepare_generator_output(args, command="make-trainset-elastic", output_value=str(args.output))
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    mode = str(args.input_mode).strip().lower()

    def run_single_mp_id(mp_id: str, target_dir: Path) -> str:
        api_key = args.api_key or os.getenv("MP_API_KEY")
        if not api_key:
            raise ValueError("Missing Materials Project API key. Provide --api-key or set MP_API_KEY.")
        out_yaml = target_dir / Path(str(args.out_yaml)).name
        structure_dir = Path(args.structure_dir) if args.structure_dir else (target_dir / "downloaded_structures")
        structure_dir.mkdir(parents=True, exist_ok=True)
        res = generate_trainset_settings_yaml_from_mp_simple(
            mp_id=mp_id,
            out_yaml=str(out_yaml),
            structure_dir=str(structure_dir),
            bulk_mode=args.bulk_mode,
            api_key=api_key,
            verbose=bool(args.verbose),
        )
        generate_trainset_from_yaml(yaml_path=res["yaml"], out_dir=str(target_dir))
        _concat_geo_strained(target_dir)
        return res["yaml"]

    extra = {"mode": mode}
    if mode == "yaml":
        if not args.yaml:
            raise ValueError("yaml mode requires --yaml.")
        generate_trainset_from_yaml(yaml_path=args.yaml, out_dir=str(out_dir))
        geo_path = _concat_geo_strained(out_dir)
        if geo_path is not None:
            print(f"[Done] Concatenated strained geometries to: {geo_path}")
        print(f"[Done] Elastic trainset written to: {out_dir}")
        extra["yaml_path"] = str(args.yaml)
    elif mode == "material-id":
        if not args.mp_id:
            raise ValueError("material-id mode requires --mp-id.")
        yaml_path = run_single_mp_id(str(args.mp_id), out_dir)
        print(f"[Done] Generated settings from Materials Project: {yaml_path}")
        print(f"[Done] Elastic trainset written to: {out_dir}")
        extra["yaml_path"] = str(yaml_path)
        extra["mp_id"] = str(args.mp_id)
    elif mode == "batch":
        if not args.elements:
            raise ValueError("batch mode requires --elements.")
        api_key = args.api_key or os.getenv("MP_API_KEY")
        if not api_key:
            raise ValueError("Missing Materials Project API key. Provide --api-key or set MP_API_KEY.")
        elements = parse_elements_csv(str(args.elements))
        mp_ids = _search_mp_material_ids_by_elements(
            api_key=api_key,
            elements=elements,
            element_count_scope=str(args.element_count_scope),
            max_materials=args.max_materials,
        )
        if not mp_ids:
            raise ValueError("No Materials Project systems found for the requested batch query.")
        ok = 0
        skipped = 0
        for mp_id in mp_ids:
            target_dir = out_dir / mp_id
            try:
                run_single_mp_id(mp_id, target_dir)
                ok += 1
            except Exception as exc:
                skipped += 1
                if args.verbose:
                    print(f"[MP][skip] {mp_id}: {exc}")
        print(f"[Done] Elastic batch completed: success={ok}, skipped={skipped}, total={len(mp_ids)}")
        extra.update(
            {
                "elements": elements,
                "mp_ids_total": len(mp_ids),
                "mp_ids_success": ok,
                "mp_ids_skipped": skipped,
            }
        )
    else:
        raise ValueError(f"Unsupported input mode: {mode!r}")

    persist_generator_metadata(
        args,
        command="make-trainset-elastic",
        output_path=out_dir,
        layout=layout,
        copy_to_dot=bool(getattr(args, "copy_to_dot", False)),
        extra=extra,
    )
    copied = maybe_copy_output_to_dot(out_dir, enabled=bool(getattr(args, "copy_to_dot", False)))
    dirs = [out_dir]
    if copied is not None:
        dirs.append(copied.parent)
    print_saved_dirs(dirs)
    return 0


def _run_make_trainset_heatfo(args: argparse.Namespace) -> int:
    _ensure_source_mp(args.source)
    out_dir, layout = prepare_generator_output(args, command="make-trainset-heatfo", output_value=str(args.output))
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    mode = str(args.input_mode).strip().lower()

    if mode == "yaml":
        if not args.yaml:
            raise ValueError("yaml mode requires --yaml.")
        data = _load_heatfo_yaml(args.yaml)
        yaml_source = str(data.get("source", args.source))
        _ensure_source_mp(yaml_source)
        yaml_mode = str(data.get("input_mode", "batch")).strip().lower()
        yaml_elements = data.get("elements", [])
        if isinstance(yaml_elements, str):
            yaml_elements = parse_elements_csv(yaml_elements)
        elif isinstance(yaml_elements, list):
            yaml_elements = parse_elements_csv(",".join(str(x) for x in yaml_elements))
        else:
            yaml_elements = []
        material_ids = None
        if yaml_mode == "material-id":
            if isinstance(data.get("material_ids"), list) and data.get("material_ids"):
                material_ids = [str(x).strip() for x in data.get("material_ids") if str(x).strip()]
            elif data.get("mp_id"):
                material_ids = [str(data.get("mp_id")).strip()]
            if not material_ids:
                raise ValueError("heatfo YAML material-id mode requires mp_id or material_ids.")
        elif yaml_mode != "batch":
            raise ValueError("heatfo YAML input_mode must be 'batch' or 'material-id'.")
        refs = _parse_heatfo_reference_map(data.get("references"))
        spec = MaterialsProjectHeatFoSpec(
            out_dir=str(out_dir),
            elements=yaml_elements,
            material_ids=material_ids,
            references_by_element=refs,
            exact_element_count=str(data.get("element_count_scope", "exact")).strip().lower() == "exact",
            api_key=str(data.get("api_key")) if data.get("api_key") else (args.api_key or os.getenv("MP_API_KEY")),
            max_materials=(int(data["max_materials"]) if data.get("max_materials") is not None else None),
            weight=float(data.get("weight", 1.0)),
            trainset_filename=str(data.get("trainset_file", "trainset_heatfo.in")),
            concatenated_geo_filename=str(data.get("geo_file", "geo")),
            verbose=bool(data.get("verbose", args.verbose)),
        )
        result = generate_heatfo_trainset_from_mp(spec)
        extra = {"mode": "yaml", "yaml_path": str(args.yaml), "generated_count": int(result.generated_count)}
    elif mode == "material-id":
        if not args.mp_id:
            raise ValueError("material-id mode requires --mp-id.")
        refs = parse_heatfo_references(str(args.references)) if args.references else {}
        elements = parse_elements_csv(str(args.elements)) if args.elements else []
        result = generate_heatfo_trainset_from_mp(
            MaterialsProjectHeatFoSpec(
                out_dir=str(out_dir),
                elements=elements,
                material_ids=[str(args.mp_id)],
                references_by_element=refs,
                exact_element_count=True,
                api_key=args.api_key or os.getenv("MP_API_KEY"),
                max_materials=None,
                weight=float(args.weight),
                trainset_filename=str(args.trainset_file),
                concatenated_geo_filename=str(args.geo_file),
                verbose=bool(args.verbose),
            )
        )
        extra = {"mode": "material-id", "mp_id": str(args.mp_id), "generated_count": int(result.generated_count)}
    elif mode == "batch":
        if not args.elements:
            raise ValueError("batch mode requires --elements.")
        refs = parse_heatfo_references(str(args.references)) if args.references else {}
        elements = parse_elements_csv(str(args.elements))
        result = generate_heatfo_trainset_from_mp(
            MaterialsProjectHeatFoSpec(
                out_dir=str(out_dir),
                elements=elements,
                material_ids=None,
                references_by_element=refs,
                exact_element_count=str(args.element_count_scope).strip().lower() == "exact",
                api_key=args.api_key or os.getenv("MP_API_KEY"),
                max_materials=args.max_materials,
                weight=float(args.weight),
                trainset_filename=str(args.trainset_file),
                concatenated_geo_filename=str(args.geo_file),
                verbose=bool(args.verbose),
            )
        )
        extra = {"mode": "batch", "elements": elements, "generated_count": int(result.generated_count)}
    else:
        raise ValueError(f"Unsupported input mode: {mode!r}")

    print(f"[Done] Heatfo trainset written to: {result.trainset_path}")
    print(f"[Done] Concatenated geo written to: {result.concatenated_geo_path}")
    print(f"[Done] Individual source files written under: {result.sources_dir}")
    print(f"[Done] Systems included: {result.generated_count}")

    persist_generator_metadata(
        args,
        command="make-trainset-heatfo",
        output_path=out_dir,
        layout=layout,
        copy_to_dot=bool(getattr(args, "copy_to_dot", False)),
        extra=extra,
    )
    copied = maybe_copy_output_to_dot(out_dir, enabled=bool(getattr(args, "copy_to_dot", False)))
    dirs = [out_dir]
    if copied is not None:
        dirs.append(copied.parent)
    print_saved_dirs(dirs)
    return 0


def run_main(command: str, args: argparse.Namespace) -> int:
    if command == "export-trainset":
        return _run_export_trainset(args)
    if command == "make-trainset-settings":
        return _run_make_trainset_settings(args)
    if command == "make-trainset-elastic":
        return _run_make_trainset_elastic(args)
    if command == "make-trainset-heatfo":
        return _run_make_trainset_heatfo(args)
    raise KeyError(f"Unsupported trainset file-tool command {command!r}.")
