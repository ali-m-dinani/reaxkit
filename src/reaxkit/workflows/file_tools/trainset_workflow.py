"""Direct command workflows for trainset export and generation utilities."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

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
    MaterialsProjectHeatFoSpec,
    generate_heatfo_trainset_from_mp,
    parse_elements_csv,
    parse_heatfo_references,
)
from reaxkit.engine.reaxff.generators.trainset_mp import generate_trainset_settings_yaml_from_mp_simple
from reaxkit.engine.reaxff.generators.trainset_yaml import generate_trainset_from_yaml, write_trainset_settings_yaml

TRAINSET_FILE_COMMANDS = ("export-trainset", "make-trainset-settings", "make-trainset")


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
    elif command == "make-trainset":
        parser.description = (
            "Generate an elastic-energy trainset from YAML or Materials Project metadata.\n\n"
            "Examples:\n"
            "  reaxkit make-trainset --yaml trainset_settings.yaml\n"
            "  reaxkit make-trainset --mp-id mp-661 --api-key YOUR_KEY\n"
            "  reaxkit make-trainset --yaml trainset_settings.yaml --output trainset_generated\n"
            "  reaxkit make-trainset --heatfo-elements Ba,B,O --heatfo-references "
            "Ba=Babcc_opt:2,B=B_alp:12,O=O2:2 --api-key YOUR_KEY --output heatfo_trainset"
        )
        parser.add_argument("--yaml", default=None, help="Existing trainset_settings.yaml file")
        parser.add_argument("--mp-id", default=None, help="Materials Project material id, for example mp-661")
        parser.add_argument("--api-key", default=None, help="Materials Project API key or set MP_API_KEY")
        parser.add_argument("--bulk-mode", default="voigt", choices=["voigt", "reuss", "vrh"], help="MP bulk modulus source")
        parser.add_argument("--out-yaml", default="trainset_settings_mp.yaml", help="Generated YAML path in MP mode")
        parser.add_argument("--structure-dir", default=None, help="Directory for MP-downloaded structures")
        parser.add_argument("--verbose", action="store_true", help="Verbose MP fetching/logging")
        parser.add_argument("--output", default="trainset_generated", help="Directory for trainset outputs")
        parser.add_argument(
            "--heatfo-elements",
            default=None,
            help="Comma-separated elements for heat-of-formation trainset mode, for example Ba,B,O",
        )
        parser.add_argument(
            "--heatfo-references",
            default=None,
            help=(
                "Optional reference map in the form element=identifier:atoms, ... "
                '(example: "Ba=Babcc_opt:2,B=B_alp:12,O=O2:2"). '
                "If omitted, unary references are auto-selected from Materials Project."
            ),
        )
        parser.add_argument(
            "--heatfo-element-count-scope",
            choices=["exact", "up-to"],
            default="exact",
            help=(
                "Element-count filter mode for Materials Project search: "
                "'exact' means exactly N elements (N=len(heatfo-elements)); "
                "'up-to' includes subset systems with up to N elements."
            ),
        )
        parser.add_argument(
            "--heatfo-max-materials",
            type=int,
            default=None,
            help="Optional cap on selected Materials Project systems in heatfo mode.",
        )
        parser.add_argument("--heatfo-weight", type=float, default=1.0, help="Weight used for heatfo ENERGY lines.")
        parser.add_argument(
            "--heatfo-trainset-file",
            default="trainset_heatfo.in",
            help="Output trainset filename in heatfo mode.",
        )
        parser.add_argument(
            "--heatfo-geo-file",
            default="geo",
            help="Output filename for concatenated geo data in heatfo mode.",
        )
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


def _run_make_trainset(args: argparse.Namespace) -> int:
    out_dir, layout = prepare_generator_output(args, command="make-trainset", output_value=str(args.output))

    if args.heatfo_elements:
        elements = parse_elements_csv(str(args.heatfo_elements))
        references = parse_heatfo_references(str(args.heatfo_references)) if args.heatfo_references else {}
        result = generate_heatfo_trainset_from_mp(
            MaterialsProjectHeatFoSpec(
                elements=elements,
                references_by_element=references,
                out_dir=str(out_dir),
                exact_element_count=(str(args.heatfo_element_count_scope).strip().lower() == "exact"),
                api_key=args.api_key or os.getenv("MP_API_KEY"),
                max_materials=args.heatfo_max_materials,
                weight=float(args.heatfo_weight),
                trainset_filename=str(args.heatfo_trainset_file),
                concatenated_geo_filename=str(args.heatfo_geo_file),
                verbose=bool(args.verbose),
            )
        )
        print(f"[Done] Heatfo trainset written to: {result.trainset_path}")
        print(f"[Done] Concatenated geo written to: {result.concatenated_geo_path}")
        print(f"[Done] Individual source files written under: {result.sources_dir}")
        print(f"[Done] Systems included: {result.generated_count}")
        persist_generator_metadata(
            args,
            command="make-trainset",
            output_path=out_dir,
            layout=layout,
            copy_to_dot=bool(getattr(args, "copy_to_dot", False)),
            extra={
                "mode": "heatfo",
                "generated_count": int(result.generated_count),
                "trainset_path": str(result.trainset_path),
                "concatenated_geo_path": str(result.concatenated_geo_path),
            },
        )
        copied = maybe_copy_output_to_dot(out_dir, enabled=bool(getattr(args, "copy_to_dot", False)))
        dirs = [out_dir]
        if copied is not None:
            dirs.append(copied.parent)
        print_saved_dirs(dirs)
        return 0

    yaml_path = args.yaml
    if not yaml_path:
        if not args.mp_id:
            raise ValueError("Provide either --yaml or --mp-id.")
        api_key = args.api_key or os.getenv("MP_API_KEY")
        if not api_key:
            raise ValueError("Missing Materials Project API key. Provide --api-key or set MP_API_KEY.")

        out_yaml = out_dir.parent / Path(str(args.out_yaml)).name
        structure_dir = args.structure_dir or str(out_dir.parent / "downloaded_structures")
        Path(structure_dir).mkdir(parents=True, exist_ok=True)
        res = generate_trainset_settings_yaml_from_mp_simple(
            mp_id=args.mp_id,
            out_yaml=str(out_yaml),
            structure_dir=structure_dir,
            bulk_mode=args.bulk_mode,
            api_key=api_key,
            verbose=bool(args.verbose),
        )
        yaml_path = res["yaml"]
        print(f"[Done] Generated settings from Materials Project: {yaml_path}")

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    generate_trainset_from_yaml(yaml_path=yaml_path, out_dir=str(out_dir))
    print(f"[Done] Elastic-energy trainset + tables written to: {out_dir}")

    geo_dir = Path(out_dir) / "geo_strained"
    all_geo_file = geo_dir / "all_trainset_geo.bgf"
    if geo_dir.exists():
        bgf_files = sorted(geo_dir.glob("*.bgf"))
        if bgf_files:
            with open(all_geo_file, "w", encoding="utf-8") as fout:
                for bgf in bgf_files:
                    fout.write(f"# ===== BEGIN {bgf.name} =====\n")
                    with open(bgf, "r", encoding="utf-8") as fin:
                        fout.write(fin.read())
                    fout.write(f"\n# ===== END {bgf.name} =====\n\n")
            print(f"[Done] Concatenated strained geometries to: {all_geo_file}")
    persist_generator_metadata(
        args,
        command="make-trainset",
        output_path=out_dir,
        layout=layout,
        copy_to_dot=bool(getattr(args, "copy_to_dot", False)),
        extra={"yaml_path": str(yaml_path)},
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
    if command == "make-trainset":
        return _run_make_trainset(args)
    raise KeyError(f"Unsupported trainset file-tool command {command!r}.")
