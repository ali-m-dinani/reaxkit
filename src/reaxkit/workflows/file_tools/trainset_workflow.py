"""Direct command workflows for trainset export and generation utilities."""

from __future__ import annotations

import argparse
from pathlib import Path

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

WORKFLOW_TASK_NAME_MAP: dict[str, str] = {
    "get_trainset_data": "trainset_data",
    "get_trainset_group_comments": "trainset_group_comments",
}


def build_parser(parser: argparse.ArgumentParser, *, command: str) -> argparse.ArgumentParser:
    parser.formatter_class = argparse.RawTextHelpFormatter

    if command == "get_trainset_data":
        parser.description = (
            "Read trainset entries from one section or all sections and return them as a table.\n"
            "There are multiple sections in a training set file such as ENRGY, CHARGET, etc., and they are "
            "separated by lines starting with keyword END. \n"
            "Examples:\n"
            " 1. Getting all training sets in all sections:\n"
            "   reaxkit get_trainset_data --section all --export trainset_data.csv\n\n"
            " 2. Getting training sets in a specific section, for example geometry:\n"
            "  reaxkit get_trainset_data --section geometry --export geometry_trainset_data.csv\n\n"
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
            "Read grouped/comment metadata from trainset sections.\n"
            "In each section of training set files, different data are separated by line comments above them which shows "
            "what those data are exactly (for example separating the EOS data for a material from the reaction barriers in "
            "the ENERGY seciton.\n"
            "Getting these group comments helps user get a summary of training set and understand what the ffield was trained against.\n\n"
            "Examples:\n"
            " 1. Getting all group comments in all sections:\n"
            "   reaxkit get_trainset_group_comments --section all --export trainset_group_comments.csv\n\n"
            " 2. Getting group comments in a specific section, for example geometry:\n"
            "   reaxkit get_trainset_group_comments --section geometry --export geometry_group_comments.csv"
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

    if command in {"gen_template_yaml_for_elastic_settings", "make-trainset-settings"}:
        parser.description = (
            "Write a sample trainset settings YAML for generating elastic-based training set (i.e., EOS).\n"
            "This command only writes the template YAML file but does not generate any trainset data. Once you have the YAML file, "
            "you can edit it to specify what materials/systems you want to generate elastic training data for and then run 'gen_elastic_trainset' "
            "with '--input-mode yaml' to generate the trainset based on the YAML config.\n\n"
            "Examples:\n"
            "  1. Generate a template YAML with default name 'trainset_settings.yaml':\n"
            "  reaxkit gen_template_yaml_for_elastic_settings\n\n"
            "  2. Generate a template YAML with a custom name:\n"
            "  reaxkit gen_template_yaml_for_elastic_settings --output trainset_settings.yaml\n\n"
        )
        parser.add_argument("--output", default="trainset_settings.yaml", help="Output YAML path")
        parser.add_argument("--copy-to-dot", action="store_true", help="Also copy generated output to current directory")

    elif command in {"gen_template_yaml_for_heatfo_settings", "make-trainset-settings-heatfo"}:
        parser.description = (
            "Write a sample trainset settings YAML for generating heat-of-formation-based-training data.\n"
            "This command only writes the template YAML file but does not generate any trainset data. Once you have the YAML file, "
            "you can edit it to specify what materials/systems you want to generate heatfo (i.e., heat of formation) training "
            "data for and then run 'gen_heatfo_trainset' "
            "with '--input-mode yaml' to generate the trainset based on the YAML config.\n\n"
            "Example:\n"
            "  1. Generate a template YAML with default name 'trainset_heatfo_settings.yaml':\n"
            "  reaxkit gen_template_yaml_for_heatfo_settings --output trainset_heatfo_settings.yaml\n\n"
        )
        parser.add_argument("--output", default="trainset_heatfo_settings.yaml", help="Output YAML path")
        parser.add_argument("--copy-to-dot", action="store_true", help="Also copy generated output to current directory")

    elif command in {"gen_elastic_trainset", "make-trainset-elastic"}:
        parser.description = (
            "Generate elastic trainsets (i.e., EOS data).\n"
            "This comamnd supports 3 input-mode options:\n"
            "  1. yaml: which needs an existing trainset YAML. \n"
            "           This trainset YAML can be generated using command 'gen_template_yaml_for_elastic_settings' \n"
            "  2. material-id: fetch one source material id (i.e., material ID [mp-1234] from material's project website)\n"
            "  3. batch: fetch many source systems by elements (i.e., all materials with Ba, B, and O)\n\n"
            
            "[NOTE] To use the source-backed modes (material-id or batch), you need to provide the your API key "
            "and specify the source (MP (material's project) or Jarvis, where default is MP). You API-key can be obtained from the source website. For "
            "example, for MP, you can: \n"
            " 1. login to your account on MP website\n"
            " 2. on the top right of the page, near your account logo, click on the API access page link,\n"
            " 3. this brings you to https://next-gen.materialsproject.org/api"
            " 4. you can now copy your personal API key, which you will provide it to this command using --api-key flag "
            "or set it as an environment variable MP_API_KEY. \n\n"
            
            "Examples:\n"
            "  1. YAML mode:\n"
            "    reaxkit gen_elastic_trainset --input-mode yaml --yaml trainset_settings.yaml --output trainset_elastic_generated\n"
            "  2. Material-id mode:\n"
            "    reaxkit gen_elastic_trainset --input-mode material-id --mat-id mp-1234 --output trainset_elastic_mp-1234 --api-key YOUR_KEY\n"
            "  3. Batch mode:\n"
            "    - for materials containing only and exactly Ba, B, O elements as in Ba2B2O5:\n"
            "       reaxkit gen_elastic_trainset --input-mode batch --elements Ba,B,O --api-key YOUR_KEY\n"
            "    - for materials any or all of Ba, B, O elements (now, BaO10 is also acceptable):\n"
            "       reaxkit gen_elastic_trainset --input-mode batch --elements Ba,B,O --api-key YOUR_KEY --element-count-scope up-to\n "
            "    - for materials containing any or all of Ba, B, O elements but with a cap of 100 materials to prevent large training set genration:\n"
            "       reaxkit gen_elastic_trainset --input-mode batch --elements Ba,B,O --api-key YOUR_KEY --element-count-scope up-to --max-materials 100\n\n"
            
            "[NOTE] As the documentation on https://docs.materialsproject.org/methodology/materials-methodology/understanding-structures-and-properties-in-the-materials-project shows,  "
            "retrieved structures from the new Materials Project (MP) API may have different lattice parameters and angles than"
            "that of conventional or primitive unit cells you might expect from textbooks or the legacy MP database (i.e., seen on the website). "
            "For this purpose, we have a flag --crystallographic-setting-conversion which can convert the fetched crystal structure "
            "setting before generating files. By default, it is set to 'to-primitive' to convert the fetched structure to its primitive setting, "
            "but you can also set it to 'to-conventional' to convert the fetched structure to its conventional setting.\n\n"

            "[Note] As you may know, the trainset generator for elastic data is developed only for orthogonal systems (i.e., with alpha=beta=gamma=90). "
            "If you use the source-backed modes to fetch structures from sources like MP, you may encounter some non-orthogonal structures. "
            "If you want to skip those non-orthogonal structures, you can use the flag --skip-not-orthogonal to automatically skip them and only generate training data for orthogonal structures. \n\n"
            
            ""
        )
        parser.add_argument("--input-mode", choices=["yaml", "material-id", "batch"], default="yaml")
        parser.add_argument("--source", choices=["mp", "jarvis"], default="mp", help="Data source.")
        parser.add_argument("--yaml", default=None, help="Existing trainset_settings.yaml file (yaml mode).")
        parser.add_argument("--mat-id", "--mp-id", dest="mat_id", default=None, help="Material id (material-id mode).")
        parser.add_argument("--elements", default=None, help="Comma-separated elements for batch mode, for example Ba,B,O")
        parser.add_argument("--element-count-scope", choices=["exact", "up-to"], default="exact")
        parser.add_argument("--max-materials", type=int, default=None, help="Optional cap for batch mode.")
        parser.add_argument("--api-key", required=True, help="Source API key (MP uses --api-key or MP_API_KEY).")
        parser.add_argument("--bulk-mode", default="voigt", choices=["voigt", "reuss", "vrh"], help="Bulk modulus mode for supported sources.")
        parser.add_argument(
            "--crystallographic-setting-conversion",
            choices=["to-conventional", "to-primitive"],
            default="to-primitive",
            help="Convert fetched crystal structure setting before generating files",
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
            "Generate heat-of-formation training sets.\n"
            "This command gets the heat of formation data and balances the equation element-wise.\n\n"
            
            "This comamnd supports 3 input-mode options:\n"
            "  1. yaml: which needs an existing heatfo trainset YAML. \n"
            "           This trainset YAML can be generated using command 'gen_template_yaml_for_heatfo_settings' \n"
            "  2. material-id: fetch one source material id (i.e., material ID [mp-1234] from material's project website)\n"
            "  3. batch: fetch many source systems by elements (i.e., all materials with Ba, B, and O)\n\n"
            
            "[NOTE] To use the source-backed modes (material-id or batch), you need to provide the your API key "
            "and specify the source (MP (material's project) or Jarvis, where default is MP). You API-key can be obtained from the source website. For "
            "example, for MP, you can: \n"
            " 1. login to your account on MP website\n"
            " 2. on the top right of the page, near your account logo, click on the API access page link,\n"
            " 3. this brings you to https://next-gen.materialsproject.org/api"
            " 4. you can now copy your personal API key, which you will provide it to this command using --api-key flag "
            "or set it as an environment variable MP_API_KEY. \n\n"
            
            "Examples:\n"
            "  1. YAML mode:\n"
            "    reaxkit gen_heatfo_trainset --input-mode yaml --yaml trainset_heatfo_settings.yaml --output trainset_heatfo_generated\n"
            "  2.   Material-id mode:\n"
            "    reaxkit gen_heatfo_trainset --input-mode material-id --mat-id mp-1234 --output trainset_heatfo_mp-1234 --api-key YOUR_KEY\n"
            "  3. Batch mode:\n"
            "    - for materials containing only and exactly Ba, B, O elements as in Ba2B2O5:\n"
            "       reaxkit gen_heatfo_trainset --input-mode batch --elements Ba,B,O --api-key YOUR_KEY\n"
            "    - for materials any or all of Ba, B, O elements (now, BaO10 is also acceptable):\n"
            "       reaxkit gen_heatfo_trainset --input-mode batch --elements Ba,B,O --api-key YOUR_KEY --element-count-scope up-to\n"
            "     - same as the first one but this time passing reference list:\n"
            "       This means that for balancing the heat of formation equation, the reference geo files will be geo file Babcc_opt with 2 atoms for Ba, "
            "       geo file B_alp with 12 atoms for B, and the geo file O2 with 2 atoms for O. "
            "       If you don't provide the reference list, the command will automatically find the most stable structure of elemnts from the source"
            "       website and uses them for balancing purposes:\n"
            "       reaxkit gen_heatfo_trainset --input-mode batch --elements Ba,B,O --api-key YOUR_KEY --references Ba=Babcc_opt:2,B=B_alp:12,O=O2:2\n"
            "    - for materials containing any or all of Ba, B, O elements but with a cap of 100 materials to prevent large training set genration:\n"
            "       reaxkit gen_heatfo_trainset --input-mode batch --elements Ba,B,O --api-key YOUR_KEY --element-count-scope up-to --max-materials 100 \n\n"

            "[NOTE] As the documentation on https://docs.materialsproject.org/methodology/materials-methodology/understanding-structures-and-properties-in-the-materials-project shows,  "
            "retrieved structures from the new Materials Project (MP) API may have different lattice parameters and angles than"
            "that of conventional or primitive unit cells you might expect from textbooks or the legacy MP database (i.e., seen on the website). "
            "For this purpose, we have a flag --crystallographic-setting-conversion which can convert the fetched crystal structure "
            "setting before generating files. By default, it is set to 'to-primitive' to convert the fetched structure to its primitive setting, "
            "but you can also set it to 'to-conventional' to convert the fetched structure to its conventional setting.\n\n"

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
        parser.add_argument("--weight", type=float, default=1.0, help="Weight used for heatfo ENERGY lines in the training set.")
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
    if command in {"gen_template_yaml_for_elastic_settings", "make-trainset-settings"}:
        return _run_make_trainset_settings(args, command_name=command)
    if command in {"gen_template_yaml_for_heatfo_settings", "make-trainset-settings-heatfo"}:
        return _run_make_trainset_settings_heatfo(args, command_name=command)
    if command in {"gen_elastic_trainset", "make-trainset-elastic"}:
        return _run_make_trainset_elastic(args, command_name=command)
    if command in {"gen_heatfo_trainset", "make-trainset-heatfo"}:
        return _run_make_trainset_heatfo(args, command_name=command)
    raise KeyError(f"Unsupported trainset file-tool command {command!r}.")
