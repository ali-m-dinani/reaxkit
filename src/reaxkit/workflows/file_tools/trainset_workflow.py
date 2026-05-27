"""Direct command workflows for trainset export and generation utilities."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

from reaxkit.analysis import force_field as _force_field_tasks  # noqa: F401
from reaxkit.analysis.force_field.trainset import GetTrainsetDataRequest, TrainsetGroupCommentsRequest
from reaxkit.core.analysis_executor import AnalysisExecutor
from reaxkit.core.analysis_task_registry import TASK_REGISTRY
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
    gen_heatfo_trainset,
)
from reaxkit.engine.reaxff.generators.trainset_yaml import (
    gen_elastic_trainset,
    gen_template_yaml_for_elastic_settings,
    gen_template_yaml_for_heatfo_settings,
)
from reaxkit.presentation.dispatcher import present_result

TRAINSET_FILE_COMMANDS = (
    "export-trainset",
    "gen_template_yaml_for_elastic_settings",
    "gen_template_yaml_for_heatfo_settings",
    "gen_elastic_trainset",
    "gen_heatfo_trainset",
    "make-trainset-settings",
    "make-trainset-settings-heatfo",
    "make-trainset-elastic",
    "make-trainset-heatfo",
)

TRAINSET_ANALYSIS_COMMANDS = (
    "get_trainset_data",
    "get_trainset_group_comments",
)

ALL_TRAINSET_COMMANDS = TRAINSET_FILE_COMMANDS + TRAINSET_ANALYSIS_COMMANDS

WORKFLOW_TASK_NAME_MAP = {
    "get_trainset_data": "trainset_data",
    "get_trainset_group_comments": "trainset_group_comments",
}


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

    if command == "get_trainset_data":
        parser.description = (
            "Read trainset entries from one section or all sections and return them as a table.\n\n"
            "Examples:\n"
            "  reaxkit get_trainset_data --section all --export trainset_data.csv\n"
            "  reaxkit get_trainset_data --section geometry --export geometry_trainset_data.csv"
        )
        parser.add_argument("--run-dir", "--dir", dest="run_dir", default=".", help="Run directory fallback for engine detection")
        parser.add_argument("--trainset", default="trainset.in", help="Path to trainset file")
        parser.add_argument("--log", choices=["verbose", "quiet"], default=None, help="Logging level")
        parser.add_argument("--plot", choices=["single", "subplot"], default=None, help="Render a plot")
        parser.add_argument("--show", action="store_true", help="Show the generated plot window")
        parser.add_argument("--save", default=None, help="Save the generated plot to a file path")
        parser.add_argument("--export", default=None, help="Write the result table to CSV")
        parser.add_argument("--grid", default=None, help="Subplot grid like 2x2 or 2*2")
        parser.add_argument("--xaxis", default=None, help="Optional x-axis column override")
        parser.add_argument("--section", default="all", help="Section to keep: all, charge, heatfo, geometry, cell_parameters, energy.")
        add_storage_cli_arguments(parser)
        return parser

    if command == "get_trainset_group_comments":
        parser.description = (
            "Read grouped/comment metadata from trainset sections.\n\n"
            "Examples:\n"
            "  reaxkit get_trainset_group_comments --section all --export trainset_group_comments.csv\n"
            "  reaxkit get_trainset_group_comments --section geometry --export geometry_group_comments.csv"
        )
        parser.add_argument("--run-dir", "--dir", dest="run_dir", default=".", help="Run directory fallback for engine detection")
        parser.add_argument("--trainset", default="trainset.in", help="Path to trainset file")
        parser.add_argument("--log", choices=["verbose", "quiet"], default=None, help="Logging level")
        parser.add_argument("--plot", choices=["single", "subplot"], default=None, help="Render a plot")
        parser.add_argument("--show", action="store_true", help="Show the generated plot window")
        parser.add_argument("--save", default=None, help="Save the generated plot to a file path")
        parser.add_argument("--export", default=None, help="Write the result table to CSV")
        parser.add_argument("--grid", default=None, help="Subplot grid like 2x2 or 2*2")
        parser.add_argument("--xaxis", default=None, help="Optional x-axis column override")
        parser.add_argument("--section", default="all", help="Section to keep: all, charge, heatfo, geometry, cell_parameters, energy.")
        add_storage_cli_arguments(parser)
        return parser

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
    elif command in {"gen_template_yaml_for_elastic_settings", "make-trainset-settings"}:
        parser.description = (
            "Write a sample trainset settings YAML.\n\n"
            "Examples:\n"
            "  reaxkit gen_template_yaml_for_elastic_settings\n"
            "  reaxkit gen_template_yaml_for_elastic_settings --output trainset_settings.yaml"
        )
        parser.add_argument("--output", default="trainset_settings.yaml", help="Output YAML path")
        parser.add_argument("--copy-to-dot", action="store_true", help="Also copy generated output to current directory")
    elif command in {"gen_template_yaml_for_heatfo_settings", "make-trainset-settings-heatfo"}:
        parser.description = (
            "Write a sample heatfo trainset settings YAML.\n\n"
            "Example:\n"
            "  reaxkit gen_template_yaml_for_heatfo_settings --output trainset_heatfo_settings.yaml"
        )
        parser.add_argument("--output", default="trainset_heatfo_settings.yaml", help="Output YAML path")
        parser.add_argument("--copy-to-dot", action="store_true", help="Also copy generated output to current directory")
    elif command in {"gen_elastic_trainset", "make-trainset-elastic"}:
        parser.description = (
            "Generate elastic trainsets.\n\n"
            "input-mode options:\n"
            "  yaml: use existing trainset YAML\n"
            "  material-id: fetch one source material id\n"
            "  batch: fetch many source systems by elements"
        )
        parser.add_argument("--source", choices=["mp", "jarvis"], default="mp", help="Data source.")
        parser.add_argument("--input-mode", choices=["yaml", "material-id", "batch"], default="yaml")
        parser.add_argument("--yaml", default=None, help="Existing trainset_settings.yaml file (yaml mode).")
        parser.add_argument("--mat-id", "--mp-id", dest="mat_id", default=None, help="Material id (material-id mode).")
        parser.add_argument("--elements", default=None, help="Comma-separated elements for batch mode, for example Ba,B,O")
        parser.add_argument("--element-count-scope", choices=["exact", "up-to"], default="exact")
        parser.add_argument("--max-materials", type=int, default=None, help="Optional cap for batch mode.")
        parser.add_argument("--api-key", default=None, help="Source API key (MP uses --api-key or MP_API_KEY).")
        parser.add_argument("--bulk-mode", default="voigt", choices=["voigt", "reuss", "vrh"], help="Bulk modulus mode for supported sources.")
        parser.add_argument(
            "--crystallographic-setting-conversion",
            choices=["to-conventional", "to-primitive"],
            default="to-primitive",
            help="Convert fetched crystal structure setting before generating files. For more information, "
                 "see the article on material's project website "
                 "https://docs.materialsproject.org/methodology/materials-methodology/understanding-structures-and-properties-in-the-materials-project",
        )
        parser.add_argument("--out-yaml", default="trainset_settings_source.yaml", help="Generated YAML filename in source-backed modes.")
        parser.add_argument("--structure-dir", default=None, help="Directory for downloaded source structures.")
        parser.add_argument(
            "--skip-not-orthogonal",
            action="store_true",
            help="Skip lattices with non-orthogonal cell angles (alpha/beta/gamma not all 90).",
        )
        parser.add_argument("--verbose", action="store_true", help="Verbose source fetching/logging")
        parser.add_argument("--output", default="trainset_elastic_generated", help="Directory for outputs.")
        parser.add_argument("--copy-to-dot", action="store_true", help="Also copy generated output to current directory")
    elif command in {"gen_heatfo_trainset", "make-trainset-heatfo"}:
        parser.description = (
            "Generate heat-of-formation trainsets.\n\n"
            "input-mode options:\n"
            "  yaml: use heatfo YAML config\n"
            "  material-id: use one source material id\n"
            "  batch: fetch many source systems by elements"
        )
        parser.add_argument("--source", choices=["mp", "jarvis"], default="mp", help="Data source.")
        parser.add_argument("--input-mode", choices=["yaml", "material-id", "batch"], default="batch")
        parser.add_argument("--yaml", default=None, help="Heatfo YAML settings file (yaml mode).")
        parser.add_argument("--mat-id", "--mp-id", dest="mat_id", default=None, help="Material id (material-id mode).")
        parser.add_argument("--elements", default=None, help="Comma-separated elements for batch mode, for example Ba,B,O")
        parser.add_argument(
            "--references",
            default=None,
            help=(
                "Optional reference map: element=identifier:atoms,... "
                '(example: "Ba=Babcc_opt:2,B=B_alp:12,O=O2:2"). '
                "If omitted, unary references are auto-selected from the source."
            ),
        )
        parser.add_argument("--element-count-scope", choices=["exact", "up-to"], default="exact")
        parser.add_argument("--max-materials", type=int, default=None, help="Optional cap for batch mode.")
        parser.add_argument(
            "--crystallographic-setting-conversion",
            choices=["to-conventional", "to-primitive"],
            default="to-primitive",
            help="Convert fetched crystal structure setting before generating files. For more information, "
                 "see the article on material's project website "
                 "https://docs.materialsproject.org/methodology/materials-methodology/understanding-structures-and-properties-in-the-materials-project",
        )
        parser.add_argument("--weight", type=float, default=1.0, help="Weight used for heatfo ENERGY lines.")
        parser.add_argument("--trainset-file", default="trainset_heatfo.in", help="Output trainset filename.")
        parser.add_argument("--geo-file", default="geo", help="Output concatenated geo filename.")
        parser.add_argument("--api-key", default=None, help="Source API key (MP uses --api-key or MP_API_KEY).")
        parser.add_argument("--verbose", action="store_true", help="Verbose source fetching/logging")
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


def _run_make_trainset_settings(args: argparse.Namespace, *, command_name: str) -> int:
    out, layout = prepare_generator_output(args, command=command_name, output_value=str(args.output))
    gen_template_yaml_for_elastic_settings(out_path=str(out))
    persist_generator_metadata(
        args,
        command=command_name,
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


def _run_make_trainset_settings_heatfo(args: argparse.Namespace, *, command_name: str) -> int:
    out, layout = prepare_generator_output(args, command=command_name, output_value=str(args.output))
    gen_template_yaml_for_heatfo_settings(out_path=str(out))
    persist_generator_metadata(
        args,
        command=command_name,
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


def _run_make_trainset_elastic(args: argparse.Namespace, *, command_name: str) -> int:
    out_dir, layout = prepare_generator_output(args, command=command_name, output_value=str(args.output))
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    extra = gen_elastic_trainset(
        out_dir=str(out_dir),
        source=str(args.source),
        input_mode=str(args.input_mode),
        yaml_path=args.yaml,
        mat_id=args.mat_id,
        elements=args.elements,
        element_count_scope=str(args.element_count_scope),
        max_materials=args.max_materials,
        api_key=args.api_key,
        bulk_mode=str(args.bulk_mode),
        crystallographic_setting_conversion=str(args.crystallographic_setting_conversion),
        out_yaml=str(args.out_yaml),
        structure_dir=args.structure_dir,
        skip_no_orthogonal=bool(getattr(args, "skip_not_orthogonal", False)),
        verbose=bool(args.verbose),
    )

    persist_generator_metadata(
        args,
        command=command_name,
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


def _run_make_trainset_heatfo(args: argparse.Namespace, *, command_name: str) -> int:
    out_dir, layout = prepare_generator_output(args, command=command_name, output_value=str(args.output))
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    run_result = gen_heatfo_trainset(
        out_dir=str(out_dir),
        source=str(args.source),
        input_mode=str(args.input_mode),
        yaml_path=args.yaml,
        mat_id=args.mat_id,
        elements=args.elements,
        references=args.references,
        element_count_scope=str(args.element_count_scope),
        max_materials=args.max_materials,
        crystallographic_setting_conversion=str(args.crystallographic_setting_conversion),
        weight=float(args.weight),
        trainset_file=str(args.trainset_file),
        geo_file=str(args.geo_file),
        api_key=args.api_key,
        verbose=bool(args.verbose),
    )
    result = run_result.result
    extra = run_result.metadata

    print(f"[Done] Heatfo trainset written to: {result.trainset_path}")
    print(f"[Done] Concatenated geo written to: {result.concatenated_geo_path}")
    print(f"[Done] Individual source files written under: {result.sources_dir}")
    print(f"[Done] Systems included: {result.generated_count}")

    persist_generator_metadata(
        args,
        command=command_name,
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


def _task_name_for_command(command: str) -> str:
    return WORKFLOW_TASK_NAME_MAP.get(command, command)


def _build_get_trainset_data_request(args: argparse.Namespace) -> GetTrainsetDataRequest:
    return GetTrainsetDataRequest(section=str(getattr(args, "section", "all")))


def _build_trainset_group_comments_request(args: argparse.Namespace) -> TrainsetGroupCommentsRequest:
    return TrainsetGroupCommentsRequest(section=str(getattr(args, "section", "all")))


REQUEST_BUILDERS = {
    "get_trainset_data": _build_get_trainset_data_request,
    "get_trainset_group_comments": _build_trainset_group_comments_request,
}


def _run_trainset_analysis_main(command: str, args: argparse.Namespace) -> int:
    task_cls = TASK_REGISTRY[_task_name_for_command(command)]
    request = REQUEST_BUILDERS[command](args)
    executor = AnalysisExecutor()
    result = executor.run(task_cls(), request, vars(args))
    present_result(command, result, args)
    return 0


def run_main(command: str, args: argparse.Namespace) -> int:
    if command in TRAINSET_ANALYSIS_COMMANDS:
        return _run_trainset_analysis_main(command, args)
    if command == "export-trainset":
        return _run_export_trainset(args)
    if command in {"gen_template_yaml_for_elastic_settings", "make-trainset-settings"}:
        return _run_make_trainset_settings(args, command_name=command)
    if command in {"gen_template_yaml_for_heatfo_settings", "make-trainset-settings-heatfo"}:
        return _run_make_trainset_settings_heatfo(args, command_name=command)
    if command in {"gen_elastic_trainset", "make-trainset-elastic"}:
        return _run_make_trainset_elastic(args, command_name=command)
    if command in {"gen_heatfo_trainset", "make-trainset-heatfo"}:
        return _run_make_trainset_heatfo(args, command_name=command)
    raise KeyError(f"Unsupported trainset file-tool command {command!r}.")
