"""Direct command workflows for ReaxFF ``ffield`` tools and force-field analyses."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Callable

import pandas as pd

from reaxkit.analysis import force_field as _force_field_tasks  # noqa: F401
from reaxkit.analysis.force_field.diagnostics import ParameterOptimizationDiagnosticRequest
from reaxkit.analysis.force_field.force_field import ForceFieldDataRequest, ForceFieldDataTask
from reaxkit.analysis.force_field.optimization import ForceFieldOptimizationRequest
from reaxkit.analysis.force_field.report import (
    ForceFieldOptimizationReportBulkModulusRequest,
    ForceFieldOptimizationReportEOSRequest,
    ForceFieldOptimizationReportRequest,
)
from reaxkit.analysis.force_field.structure_summary import StructureSummaryRequest
from reaxkit.analysis.force_field.trainset import GetTrainsetDataRequest, TrainsetGroupCommentsRequest
from reaxkit.cli.path import resolve_output_path
from reaxkit.core.alias import normalize_choice, resolve_alias_from_columns
from reaxkit.core.analysis_executor import AnalysisExecutor
from reaxkit.core.analysis_task_registry import TASK_REGISTRY
from reaxkit.core.command_alias_resolver import resolve_command_name
from reaxkit.core.engine_registry import resolve_engine
from reaxkit.core.generator_runtime import (
    maybe_copy_output_to_dot,
    persist_generator_metadata,
    prepare_generator_output,
    print_saved_dirs,
)
from reaxkit.core.storage_layout import add_storage_cli_arguments, normalize_storage_args
from reaxkit.domain.data_models import (
    ForceFieldOptimizationReportData as ForceFieldOptimizationResultsData,
    ForceFieldParametersData,
    GeometrySummaryData as EnergyMinimizationSummaryData,
)
from reaxkit.engine.reaxff.generators.ffield_generator import (
    add_element_to_ffield,
    add_term_to_ffield,
    merge_ffields,
)
from reaxkit.presentation.dispatcher import export_result_csv, present_result


def _parse_csv_items(value: str) -> list[str]:
    return [token.strip() for token in str(value).replace(";", ",").split(",") if token.strip()]


def _write_snapshot_text(path: Path, title: str, labels_by_field: dict[str, list[str]], blocks_by_field: dict[str, list[str]]) -> Path:
    names = {
        "atom": "Atom",
        "bond": "Bond",
        "off_diagonal": "Off-diagonal",
        "angle": "Angle",
        "torsion": "Torsion",
        "hbond": "H-bond",
    }
    lines: list[str] = [title, ""]
    for field in ("atom", "bond", "off_diagonal", "angle", "torsion", "hbond"):
        lines.append(f"{names[field]}:")
        lines.append("|")
        labels = labels_by_field.get(field, [])
        lines.append((" | ".join(labels) + " |") if labels else "Not Added")
        lines.append("")
    lines.append("Formatted ffield blocks:")
    lines.append("")
    for field in ("atom", "bond", "off_diagonal", "angle", "torsion", "hbond"):
        lines.append(f"{names[field]}:")
        lines.append("|")
        blocks = blocks_by_field.get(field, [])
        if not blocks:
            lines.append("Not Added")
        else:
            for block in blocks:
                lines.append(block)
                lines.append("")
        lines.append("")
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return path


def _write_snapshot_field_csvs(root: Path, labels_by_field: dict[str, list[str]], blocks_by_field: dict[str, list[str]]) -> list[Path]:
    files: list[Path] = []
    for field in ("atom", "bond", "off_diagonal", "angle", "torsion", "hbond"):
        out = root / f"{field}_entries.csv"
        with out.open("w", encoding="utf-8", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(["entry_index", "label", "formatted_block"])
            labels = labels_by_field.get(field, [])
            blocks = blocks_by_field.get(field, [])
            n = max(len(labels), len(blocks))
            for i in range(n):
                label = labels[i] if i < len(labels) else ""
                block = blocks[i] if i < len(blocks) else ""
                writer.writerow([i + 1, label, block])
        files.append(out)
    return files


def _write_similarity_summary(path: Path, details: dict[str, object]) -> Path:
    lines: list[str] = ["Similarity selection", ""]
    lines.append(f"Mode: {details.get('mode', '')}")
    target = details.get("target", {})
    chosen = details.get("chosen", {})
    if isinstance(target, dict):
        lines.append(f"Target: {target.get('symbol', '')}")
        lines.append(f"  group: {target.get('group', '')}")
        lines.append(f"  family: {target.get('family', '')}")
    if isinstance(chosen, dict):
        lines.append(f"Chosen template: {chosen.get('symbol', '')}")
        lines.append(f"  group: {chosen.get('group', '')}")
        lines.append(f"  family: {chosen.get('family', '')}")
    lines.append("")
    lines.append("Candidates:")
    candidates = details.get("candidates", [])
    if isinstance(candidates, list) and candidates:
        for candidate in candidates:
            if not isinstance(candidate, dict):
                continue
            selected = "*" if bool(candidate.get("selected")) else " "
            lines.append(
                f"{selected} {candidate.get('symbol', '')}: "
                f"group={candidate.get('group', '')}, "
                f"family={candidate.get('family', '')}, "
                f"radius_distance={candidate.get('radius_distance', '')}"
            )
        lines.append("")
        lines.append(
            "radius_distance = mean absolute difference across the selected radius metrics "
            "(metrics with missing values are ignored; if all are missing, distance is infinite)."
        )
    else:
        lines.append("No candidate diagnostics available.")
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return path


def _write_similarity_summaries(path: Path, details_by_atom: dict[str, dict[str, object]]) -> Path:
    lines: list[str] = ["Similarity selection (merge template fill)", ""]
    if not details_by_atom:
        lines.append("No similarity details available.")
        path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
        return path
    for atom in sorted(details_by_atom.keys()):
        details = details_by_atom.get(atom, {})
        lines.append(f"Target atom: {atom}")
        lines.append(f"Mode: {details.get('mode', '')}")
        target = details.get("target", {})
        chosen = details.get("chosen", {})
        if isinstance(target, dict):
            lines.append(f"  target symbol: {target.get('symbol', '')}")
            lines.append(f"  target group: {target.get('group', '')}")
            lines.append(f"  target family: {target.get('family', '')}")
        if isinstance(chosen, dict):
            lines.append(f"  chosen template: {chosen.get('symbol', '')}")
            lines.append(f"  chosen group: {chosen.get('group', '')}")
            lines.append(f"  chosen family: {chosen.get('family', '')}")
        lines.append("  candidates:")
        candidates = details.get("candidates", [])
        if isinstance(candidates, list) and candidates:
            for candidate in candidates:
                if not isinstance(candidate, dict):
                    continue
                selected = "*" if bool(candidate.get("selected")) else " "
                lines.append(
                    f"  {selected} {candidate.get('symbol', '')}: "
                    f"group={candidate.get('group', '')}, "
                    f"family={candidate.get('family', '')}, "
                    f"radius_distance={candidate.get('radius_distance', '')}"
                )
        else:
            lines.append("  No candidate diagnostics available.")
        lines.append("")
    lines.append(
        "radius_distance = mean absolute difference across the selected radius metrics "
        "(metrics with missing values are ignored; if all are missing, distance is infinite)."
    )
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return path


FFIELD_TOOL_COMMANDS = (
    "merge-ffield",
    "add-element-to-ffield",
    "add_element_to_ffield",
    "add-term-to-ffield",
    "add_term_to_ffield",
)

FFIELD_ANALYSIS_COMMANDS = (
    "get_ffield_data",
    "get_ffield_opt_progress_data",
    "get_energy_min_summary_data",
    "get_ffield_diagnostic_data",
    "get_ffield_opt_results",
    "get_ffield_opt_eos",
    "ffield_opt_bulk_modulus",
    "get_trainset_data",
    "get_trainset_group_comments",
)

LEGACY_FORCE_FIELD_ALIASES = {
    "force_field_data": "get_ffield_data",
    "force_field_optimization": "get_ffield_opt_progress_data",
    "force_field_optimization_report": "get_ffield_opt_results",
    "force_field_optimization_report_eos": "get_ffield_opt_eos",
    "force_field_optimization_report_bulk_modulus": "ffield_opt_bulk_modulus",
    "ffield_data": "get_ffield_data",
    "ffield_optimization": "get_ffield_opt_progress_data",
    "structure_summary_data": "get_energy_min_summary_data",
    "parameter_optimization_diagnostic": "get_ffield_diagnostic_data",
    "ffield_optimization_report": "get_ffield_opt_results",
    "ffield_optimization_report_eos": "get_ffield_opt_eos",
    "ffield_optimization_report_bulk_modulus": "ffield_opt_bulk_modulus",
    "trainset_data": "get_trainset_data",
    "trainset_group_comments": "get_trainset_group_comments",
    "parameter_optimization_most_sensitive": "get_ffield_diagnostic_data",
    "parameter_optimization_tornado": "get_ffield_diagnostic_data",
}

WORKFLOW_TASK_NAME_MAP = {
    "get_ffield_data": "force_field_data",
    "get_ffield_opt_progress_data": "force_field_optimization",
    "get_energy_min_summary_data": "structure_summary_data",
    "get_ffield_diagnostic_data": "parameter_optimization_diagnostic",
    "get_ffield_opt_results": "force_field_optimization_report",
    "get_ffield_opt_eos": "force_field_optimization_report_eos",
    "ffield_opt_bulk_modulus": "force_field_optimization_report_bulk_modulus",
    "get_trainset_data": "trainset_data",
    "get_trainset_group_comments": "trainset_group_comments",
}

ALL_FFIELD_COMMANDS = FFIELD_TOOL_COMMANDS + FFIELD_ANALYSIS_COMMANDS


def _resolve_workflow_command(command: str) -> str:
    canonical = resolve_command_name(
        command,
        task_names=ALL_FFIELD_COMMANDS + tuple(LEGACY_FORCE_FIELD_ALIASES.keys()),
    )
    return LEGACY_FORCE_FIELD_ALIASES.get(canonical, canonical)


def _task_name_for_command(command: str) -> str:
    return WORKFLOW_TASK_NAME_MAP.get(command, command)


def _build_parser(parser: argparse.ArgumentParser, *, command: str) -> argparse.ArgumentParser:
    command = _resolve_workflow_command(command)
    parser.formatter_class = argparse.RawTextHelpFormatter
    parser.set_defaults(command=command)

    if command in FFIELD_ANALYSIS_COMMANDS:
        parser.set_defaults(progress=True)
        _add_runtime_arguments(parser)
        _add_presentation_arguments(parser)

        if command == "get_ffield_data":
            parser.description = (
                "Load, filter, and export raw or interpreted ffield sections.\n"
                "Interpreted means converting numeric values of atom types into Atom types. For example, 1 will be "
                "interepreted as C if carbon is the first element in the ffield. \n\n"
                "Examples:\n"
                " 1. Reading bond section of the ffield, obtaining the C-H bond parameters in interpreted format, and exporting to CSV:\n"
                "    reaxkit get_ffield_data --field bond --term C-H --format interpreted --export CH_bond.csv\n\n"
                " 2. Reading angle section of the ffield, obtaining the C-C-H angle parameters in interpreted format, and exporting to CSV:\n"
                "    reaxkit get_ffield_data --field angle --term CCH --any-order --format interpreted --export CCH_angles.csv\n\n"
                " 3. Reading all sections of the ffield, obtaining all parameters in interpreted format, and exporting to them as "
                "individual CSV files (one per section):\n"
                "    reaxkit get_ffield_data --format interpreted --outdir ffield_export\n\n"
            )
            parser.add_argument(
                "--field",
                default=None,
                help="Single section to query: general, atom, bond, off_diagonal, angle, torsion, hbond.",
            )
            parser.add_argument(
                "--format",
                choices=["raw", "indices", "interpreted"],
                default="interpreted",
                help="Output format: raw/indices atom ids or interpreted atom symbols.",
            )
            parser.add_argument(
                "--term",
                default=None,
                help="Optional term filter, for example C-H, CCH, C-C-H, or 1-2.",
            )
            parser.add_argument(
                "--ordered-2body",
                action="store_true",
                help="For bond/off_diagonal terms, treat i-j and j-i as distinct.",
            )
            parser.add_argument(
                "--any-order",
                action="store_true",
                help="For angle/torsion/hbond terms, match any atom-order permutation.",
            )
            parser.add_argument(
                "--outdir",
                default=None,
                help="Write per-section CSV exports into this output directory.",
            )
        elif command == "get_ffield_opt_progress_data":
            parser.description = (
                "Return total force-field optimization error versus epoch.\n"
                "If you do the ffield optimization using ReaxFF's Successive One-Parameter Parabolic Interpolation (SOPPI) method, "
                "this will simply be the total force field error vs epoch.\n"
                "Examples:\n"
                " 1. Reading optimization output from ReaxFF fort.13 file, plotting error vs epoch, and saving the plot:\n"
                "   reaxkit get_ffield_opt_progress_data --fort13 fort.13 --plot single --save ffield_opt.png\n\n"
                " 2. Reading optimization output, exporting error vs epoch data to CSV for epochs 1 5 10 as CSV:\n"
                "  reaxkit get_ffield_opt_progress_data --epochs 1 5 10 --export ffield_opt.csv\n"
            )
            parser.add_argument(
                "--epochs",
                type=int,
                nargs="*",
                default=None,
                help="Optional epoch numbers to keep; default uses all available epochs.",
            )
        elif command == "get_energy_min_summary_data":
            parser.description = (
                "Get energy minimization summary data (energy, heat of formation, volume, density, etc.) per geo file, "
                "which can be obtained from fort.74 file if using ReaxFF standalone code for ffield optimization.\n\n"
                "Examples:\n"
                " 1. Getting the data and exporting all columns to CSV:\n"
                "   reaxkit get_energy_min_summary_data --export fort74.csv\n\n"
                " 2. Getting only density data, and exporting to CSV:\n"
                "  reaxkit get_energy_min_summary_data --col density --export fort74_density.csv\n\n"
            )
            parser.add_argument(
                "--col",
                default="all",
                help="Single column to keep (identifier is retained when present), or 'all'.",
            )
        elif command == "get_ffield_diagnostic_data":
            parser.description = (
                "Get per-parameter optimization diagnostics from force-field optimization diagnostic output. "
                "Diagnostics data is simply the data in fort.79 which shows how the optimizer (i.e., Successive One-Parameter Parabolic Interpolation (SOPPI) method) "
                "has gone through the parameter space during optimization, and how the error has changed when each parameter was perturbed. \n\n"
                "Examples:\n"
                "  1. Getting all diagnostic data and exporting to CSV:\n"
                "   reaxkit get_ffield_diagnostic_data --export fort79_diag.csv\n\n"
                
                "  2. Getting the diagnostics data along with the data related to the most sensitive parameter during ffield optimization:\n"
                "   reaxkit get_ffield_diagnostic_data --report-most-sensitive --export most_sensitive.csv --export-all fort79_all.csv\n\n"
                
                " 3. Getting the diagnostics data, plotting a tornado plot for the top 10 most sensitive parameters, and adding a vertical guide line at x=1.0:\n"
                "  This plot shows the relative sensitivity of the parameters in a tornado format, where the bars represent the span between the error at the current parameter value and the error at the perturbed parameter value. "
                "  This is helpful for understanding the marginal effect of each parameter on the total error, and for identifying which parameters are the most sensitive ones during optimization. \n"
                "   reaxkit get_ffield_diagnostic_data --plot tornado --top 10 --vline 1.0 --export tornado.csv\n"
                
                "  4. Getting the diagnostics data, plotting a beeswarm plot for all parameters, and saving the plot:\n"
                "   This plot is very similar to the tornado plot, but instead of showing the span between the current and perturbed error as a bar, "
                "it shows the actual distribution of the error values at the current and perturbed parameter values as swarm points. \n"
                "   A really good example of this plot can be found at: "
                "reaxkit get_ffield_diagnostic_data --plot beeswarm --save diagnostic_beeswarm.png\n\n"
            )
            parser.add_argument(
                "--interpret",
                action="store_true",
                help="Interpret identifier triplets with ffield symbol mapping when possible.",
            )
            parser.add_argument(
                "--report-most-sensitive",
                action="store_true",
                help="Return only the minimum-sensitivity parameter view.",
            )
            parser.add_argument(
                "--export-all",
                default=None,
                help="Optional CSV path to export the full diagnostic table (useful with --report-most-sensitive).",
            )
            parser.add_argument(
                "--top",
                type=int,
                default=0,
                help="For tornado view, keep top-N widest spans; 0 keeps all.",
            )
            parser.add_argument(
                "--vline",
                type=float,
                default=1.0,
                help="For tornado view, reference x-value for the guide line.",
            )
        elif command == "get_ffield_opt_results":
            parser.description = (
                "Get force-field optimization report data, which can be obtained from fort.99 file if using ReaxFF standalone code for ffield optimization. \n"
                "This includes the values in training set and the values generated by the optimized ffield. \n\n"
                "Examples:\n"
                "  1. Getting all optimization report data and exporting to CSV:\n"
                "    reaxkit get_ffield_opt_results --export fort99.csv\n\n"
            )
        elif command == "get_ffield_opt_eos":
            parser.description = (
                "Build energy-vs-volume EOS data from optimization outputs.\n\n"
                "Examples:\n"
                "  1. Getting the EOS data for a specific identifier (for example, MgO) and plotting energy vs volume curve:\n"
                "    reaxkit get_ffield_opt_eos --iden MgO --plot single\n\n"
                "  2. Getting the EOS data for all available identifiers, exporting to CSV, and plotting all EOS curves in subplots:\n"
                "    reaxkit get_ffield_opt_eos --iden all --export eos.csv\n\n"
                "  3. Getting the EOS data for all available identifiers, plotting all EOS curves in subplots, and saving the plot:\n"
                "    reaxkit get_ffield_opt_eos --iden all --plot subplot --save eos.png"
            )
            parser.add_argument(
                "--iden",
                default=None,
                help="Identifier (or base identifier) to keep; use 'all' for all rows.",
            )
            parser.add_argument(
                "--flip-sign",
                action="store_true",
                help="Flip sign of energy values before plotting/export.",
            )
        elif command == "ffield_opt_bulk_modulus":
            parser.description = (
                "Fit a Vinet bulk modulus from optimization report energy-volume data.\n\n"
                "Examples:\n"
                "  1. Fitting bulk modulus for a specific identifier (for example, MgO) and plotting the fitted curve:\n"
                "  reaxkit ffield_opt_bulk_modulus --iden bulk_0\n\n"
                "  2. Fitting bulk modulus for all available identifiers, exporting the fitted parameters to CSV:\n"
                "  reaxkit ffield_opt_bulk_modulus --iden all --export bulk_modulus.csv\n\n"
                "  3. Fitting bulk modulus for all available identifiers, plotting the fitted curves, and saving the plot.\n"
                "Here we have used --flip-sign flag since some values where negative (sign convention) and we couldn't "
                "get the bulk modulus from them:\n"
                "  reaxkit ffield_opt_bulk_modulus --flip-sign --iden all --plot subplot --save bulk_modulus.png\n\n"
            )
            parser.add_argument(
                "--iden",
                default=None,
                help="Optional base identifier to fit; use 'all' for all eligible bases.",
            )
            parser.add_argument(
                "--no-shift-min-to-zero",
                action="store_true",
                help="Do not shift minimum energy to zero before fitting.",
            )
            parser.add_argument(
                "--flip-sign",
                action="store_true",
                help="Flip sign of energy values before fitting.",
            )
            parser.add_argument(
                "--min-points",
                type=int,
                default=6,
                help="Minimum number of finite points required per base identifier.",
            )
        elif command == "get_trainset_data":
            parser.description = (
                "Return trainset rows for one section or all sections.\n\n"
                "Examples:\n"
                "  1. reaxkit get_trainset_data --section all --export trainset_all.csv\n"
                "  2. reaxkit get_trainset_data --section energy --export trainset_energy.csv"
            )
            parser.add_argument(
                "--section",
                default="all",
                help="Trainset section: all, charge, heatfo, geometry, cell_parameters, or energy.",
            )
        elif command == "get_trainset_group_comments":
            parser.description = (
                "Return unique trainset group comments by section.\n\n"
                "Examples:\n"
                "  1. reaxkit get_trainset_group_comments --section all --export group_comments.csv\n"
                "  2. reaxkit get_trainset_group_comments --section geometry --export geometry_group_comments.csv"
            )
            parser.add_argument(
                "--section",
                default="all",
                help="Trainset section: all, charge, heatfo, geometry, cell_parameters, or energy.",
            )
        else:
            raise KeyError(f"Unsupported ffield command '{command}'.")
        return parser

    if command in {"add-element-to-ffield", "add_element_to_ffield"}:
        parser.description = (
            "Add one atom type to an existing ffield and assign parameters of a very similar atom in the "
            "ffield as template parameters. This is intended as a quick way to expand coverage of an existing ffield "
            "by cloning the most similar existing atom terms.\n\n"
            "Examples:\n"
            " 1. Adding element 'Al' to a ffield by copying parameters of the most similar existing atom, automatically selected by group/family/radius similarity:\n"
            "   reaxkit add-element-to-ffield --dest ffield --element Al --output ffield_al\n\n"
            
            " 2. Same as above, but this time the similarity measure is explicitly passed using --similarity flag. "
            "Now, the most similar atom is the one with the same periodic-table group (column):\n"
            "   reaxkit add-element-to-ffield --dest ffield --element Al --similarity group --fields atom,bond,angle\n\n"
            
            " 3. This time, not all fields are selected to get copied for the new element:\n"
            "   reaxkit add-element-to-ffield --dest ffield --element Al --fields atom,bond,angle\n\n"
            
            " 4. the most similar atom is explicitly selected using --closest-atom flag\n"
            "   reaxkit add-element-to-ffield --dest ffield --element Al --closest-atom B\n\n"
        )
        parser.add_argument("--destination", "--dest", required=True, dest="destination", help="Destination ffield path")
        parser.add_argument("--output", default="ffield_with_element", help="Output expanded ffield path")
        parser.add_argument("--element", required=True, help="Element symbol to add, for example: Al")
        parser.add_argument(
            "--similarity",
            default="group",
            choices=["group", "family", "radius"],
            help=(
                "Similarity rule for template-atom selection: \n"
                " 1. 'family' = chemical family match "
                "(transition_metal, lanthanoid, actinoid, alkali_metal, alkaline_earth_metal, "
                "halogen, noble_gas, metalloid, post_transition_metal, other),\n"
                " 2. 'group' = same periodic-table group number (column),\n"
                " 3. 'radius' = closest by atomic/covalent-proxy/van-der-Waals radii distance.\n\n"
                "Priority order for selecting the single template atom is:\n"
                " 1. manual override by --closest-atom,\n"
                " 2. similarity by --similarity mode, where priority is family > group > radius, meaning for example "
                "that if --similarity group is selected, the most similar atom will be the one with the same group number, "
                "and if multiple candidates have the same group number, then similarity by radius will be used to break ties, and so on. "
                " 3. if multiple candidates are tied by similarity, the one with the smallest radius distance (if radius metrics are available) is chosen"
            ),
        )
        parser.add_argument(
            "--radius-metrics",
            default="atomic_radius,covalent_radius,van_der_waals_radius",
            help=(
                "Comma-separated radius metrics used when --similarity radius is selected. "
                "Use 'all' to include every supported metric. \n"
                "Options: \n"
                " 1. atomic_radius (empirical neutral-atom radius), \n"
                " 2. covalent_radius (mapped to pymatgen atomic_radius_calculated proxy), \n"
                " 3. van_der_waals_radius (non-bonded contact radius), \n"
                " 4. atomic_radius_calculated (theoretical neutral-atom radius), \n"
                " 5. average_ionic_radius (mean ionic radius over known oxidation states), \n"
                " 6. average_cationic_radius (mean radius over positive oxidation states), \n"
                " 7. average_anionic_radius (mean radius over negative oxidation states).\n"
            ),
        )
        parser.add_argument(
            "--closest-atom",
            default=None,
            help="Override automatic selection and force template atom symbol, for example: B",
        )
        parser.add_argument(
            "--replace-existing",
            action="store_true",
            help="Replace destination rows when the same atom tuple already exists.",
        )
    elif command in {"add-term-to-ffield", "add_term_to_ffield"}:
        parser.description = (
            "Add one specific missing term (bond/off_diagonal/angle/torsion/hbond) to an existing ffield by "
            "copying parameters from a similar existing template term.\n\n"
            "Examples:\n"
            " 1. Adding angle term 'Al-N-Al' to a ffield by copying parameters of the most similar existing angle, automatically selected by similarity:\n"
            "   reaxkit add-term-to-ffield --dest ffield --field angle --term Al-N-Al --output ffield_with_term\n\n"
            
            " 2. Same as above, but this time manual mapping for Al atom is done to B:\n"
            "   reaxkit add-term-to-ffield --dest ffield --field angle --term Al-N-Al --template-map Al:B\n\n"
            
            " 3. This time, the most similar template term is explicitly selected using --closest-term flag\n"
            "   reaxkit add-term-to-ffield --dest ffield --field angle --term Al-N-Al --closest-term B-N-B\n\n"
            
            " 4. Restrict candidate template terms to those with the same general order pattern (X-Y-X vs X-Y-Z), "
            "which can be important for angle and torsion terms. "
            "For example, if --same-general-order is selected, then the template term for Al-N-Al will be restricted to "
            "angle terms of the form X-Y-X, and angle terms of the form X-Y-Z will not be considered as templates even "
            "if they are similar by other criteria.\n"
            "   reaxkit add-term-to-ffield --dest ffield --field angle --term Al-N-Al --same-general-order --output ffield_with_term\n\n"
        )
        parser.add_argument("--destination", "--dest", required=True, dest="destination", help="Destination ffield path")
        parser.add_argument("--output", default="ffield_with_term", help="Output expanded ffield path")
        parser.add_argument(
            "--field",
            required=True,
            choices=["bond", "off_diagonal", "angle", "torsion", "hbond"],
            help="Target section for the term.",
        )
        parser.add_argument(
            "--term",
            required=True,
            help="Hyphen-separated atom symbols, for example: Al-N-Al",
        )
        parser.add_argument(
            "--closest-term",
            "--closest_term",
            dest="closest_term",
            default=None,
            help="Explicit template term override, for example: B-N-B",
        )
        parser.add_argument(
            "--template-map",
            default="",
            help="Optional per-atom manual template mapping CSV, for example: Al:B,N:N",
        )
        parser.add_argument(
            "--similarity",
            default="group",
            choices=["family", "group", "radius"],
            help=(
                "Similarity rule for template-atom selection: \n"
                " 1. 'family' = chemical family match "
                "(transition_metal, lanthanoid, actinoid, alkali_metal, alkaline_earth_metal, "
                "halogen, noble_gas, metalloid, post_transition_metal, other),\n"
                " 2. 'group' = same periodic-table group number (column),\n"
                " 3. 'radius' = closest by atomic/covalent-proxy/van-der-Waals radii distance.\n\n"
                "Priority order for selecting the single template atom is:\n"
                " 1. manual override by --closest-atom,\n"
                " 2. similarity by --similarity mode, where priority is family > group > radius, meaning for example "
                "that if --similarity group is selected, the most similar atom will be the one with the same group number, "
                "and if multiple candidates have the same group number, then similarity by radius will be used to break ties, and so on. "
                " 3. if multiple candidates are tied by similarity, the one with the smallest radius distance (if radius metrics are available) is chosen"
            ),
        )
        parser.add_argument(
            "--radius-metrics",
            default="atomic_radius,covalent_radius,van_der_waals_radius",
            help=(
                "Comma-separated radius metrics used when --similarity radius is selected. "
                "Use 'all' to include every supported metric."
            ),
        )
        parser.add_argument(
            "--replace-existing",
            action="store_true",
            help="Replace destination row when the same atom tuple already exists.",
        )
        parser.add_argument(
            "--same-general-order",
            action="store_true",
            help=(
                "Restrict candidate template terms to those with the same equality/order pattern as --term "
                "(example for angle: X-Y-X vs X-Y-Y)."
            ),
        )
    else:
        parser.description = (
            "Merge selected atom-type parameter blocks from one ffield into another.\n"
            "For eaxmple, if ffield 1 contains 'C,H,O,N,S' elements and ffield 2 contains 'C,H,O,N,Al,He' elements, "
            "then merging Al from ffield 2 into ffield 1 leads to adding all C-Al, H-Al, O-Al, N-Al, and Al-Al bond terms "
            "(and same for other fields like angle, etc.) but not S-Al by default.\n\n"
            
            "Examples:\n"
            " 1. Merging all blocks for atom type W from ffield_src into ffield_dst:\n"
            "   reaxkit merge-ffield --src ffield_src --dest ffield_dst --output ffield_merged --atom-types W\n\n"
            " 2. Merging all blocks for atom types W and Mo from ffield_src into ffield_dst, but only for selected fields:\n"
            "  reaxkit merge-ffield --source f_src --destination f_dst --output merged --atom-types W,Mo --fields atom,bond,angle,torsion\n\n"
            " 3. Same as above, but with automatic filling of missing terms for the merged atom types by templating from "
            "the most similar atom in destination, where similarity is defined by belonging to the same chemical family "
            "(for example, transition metal, halogen, noble gas, etc. See --template-similarity options for more details):\n"
            "  reaxkit merge-ffield --source f_src --destination f_dst --output ffield_merged --atom-types W,Mo --fields atom,bond,angle,torsion --fill-missing-with-template --template-similarity family\n\n"
        )
        parser.add_argument("--source", "--src", required=True, dest="source", help="Source ffield path")
        parser.add_argument("--destination", "--dest", required=True, dest="destination", help="Destination ffield path")
        parser.add_argument("--output", default="ffield_merged", help="Output merged ffield path")
        parser.add_argument(
            "--atom-types",
            required=True,
            help="Comma-separated source atom symbols to merge, for example: W or W,Mo",
        )
        parser.add_argument(
            "--replace-existing",
            action="store_true",
            help="Replace destination rows when the same atom tuple already exists.",
        )
        parser.add_argument(
            "--fill-missing-with-template",
            action="store_true",
            help="After direct merge, fill missing terms for merged atom-types by templating from the most similar atom in destination.",
        )
        parser.add_argument(
            "--template-similarity",
            default="group",
            choices=["family", "group", "radius"],
            help=(
                "Similarity rule for template-atom selection: \n"
                " 1. 'family' = chemical family match "
                "(transition_metal, lanthanoid, actinoid, alkali_metal, alkaline_earth_metal, "
                "halogen, noble_gas, metalloid, post_transition_metal, other),\n"
                " 2. 'group' = same periodic-table group number (column),\n"
                " 3. 'radius' = closest by atomic/covalent-proxy/van-der-Waals radii distance.\n\n"
                "Priority order for selecting the single template atom is:\n"
                " 1. manual override by --closest-atom,\n"
                " 2. similarity by --similarity mode, where priority is family > group > radius, meaning for example "
                "that if --similarity group is selected, the most similar atom will be the one with the same group number, "
                "and if multiple candidates have the same group number, then similarity by radius will be used to break ties, and so on. "
                " 3. if multiple candidates are tied by similarity, the one with the smallest radius distance (if radius metrics are available) is chosen"
            ),
        )
        parser.add_argument(
            "--template-closest-atom",
            default=None,
            help="Manual destination template atom override for --fill-missing-with-template, for example: B",
        )
        parser.add_argument(
            "--template-radius-metrics",
            default="atomic_radius,covalent_radius,van_der_waals_radius",
            help=(
                "Radius metrics for template selection when --template-similarity radius. "
                "Use 'all' or a CSV subset of: atomic_radius,covalent_radius,van_der_waals_radius,"
                "atomic_radius_calculated,average_ionic_radius,average_cationic_radius,average_anionic_radius."
            ),
        )
    parser.add_argument(
        "--fields",
        default="atom,bond,off_diagonal,angle,torsion,hbond",
        help="Comma-separated fields to process: atom,bond,off_diagonal,angle,torsion,hbond",
    )
    parser.add_argument(
        "--disallow-torsion-wildcard",
        action="store_true",
        help="Reject torsion rows containing atom index 0 wildcard.",
    )
    parser.add_argument(
        "--report-format",
        choices=["none", "txt", "csv", "both"],
        default="both",
        help="Write merge-detail report files next to output ffield.",
    )
    parser.add_argument("--copy-to-dot", action="store_true", help="Also copy generated output to current directory")
    add_storage_cli_arguments(parser)
    return parser


def _run_ffield_main(command: str, args: argparse.Namespace) -> int:
    out_path, layout = prepare_generator_output(args, command=command, output_value=str(args.output))
    fields = _parse_csv_items(args.fields)

    report_paths: list[Path] = []
    if command in {"add-element-to-ffield", "add_element_to_ffield"}:
        summary = add_element_to_ffield(
            destination=args.destination,
            output=out_path,
            element=args.element,
            fields=fields,
            similarity_mode=str(args.similarity),
            closest_atom=args.closest_atom,
            radius_metrics=_parse_csv_items(args.radius_metrics),
            replace_existing=bool(args.replace_existing),
            allow_torsion_wildcard=not bool(args.disallow_torsion_wildcard),
        )

        print(f"[Done] Expanded ffield written to {summary.output_path}")
        print(f"  element: {summary.element}")
        print(f"  template atom: {summary.template_atom}")
        print(f"  similarity: {summary.similarity_mode}")
        print(f"  fields: {', '.join(summary.fields)}")
        print(f"  appended: {summary.appended}")
        if any(summary.updated.values()):
            print(f"  updated: {summary.updated}")
        if any(summary.skipped_existing.values()):
            print(f"  skipped_existing: {summary.skipped_existing}")
        if any(summary.skipped_incompatible.values()):
            print(f"  skipped_incompatible: {summary.skipped_incompatible}")

        base = Path(summary.output_path)
        report_root = base.parent / f"{base.stem}_add_element_audit"
        template_root = report_root / "template_snapshot"
        destination_root = report_root / "destination_projection"
        template_root.mkdir(parents=True, exist_ok=True)
        destination_root.mkdir(parents=True, exist_ok=True)

        if args.report_format in {"txt", "both"}:
            report_paths.append(_write_similarity_summary(report_root / "similarity_summary.txt", summary.similarity_details))
            report_paths.append(
                _write_snapshot_text(
                    template_root / "template_snapshot_summary.txt",
                    f"Template snapshot of entries projected from {summary.template_atom}",
                    summary.template_labels,
                    summary.template_blocks,
                )
            )
            report_paths.append(
                _write_snapshot_text(
                    destination_root / "destination_projection_summary.txt",
                    f"Destination projection of entries added for {summary.element}",
                    summary.destination_labels,
                    summary.destination_blocks,
                )
            )
        if args.report_format in {"csv", "both"}:
            report_paths.extend(_write_snapshot_field_csvs(template_root, summary.template_labels, summary.template_blocks))
            report_paths.extend(_write_snapshot_field_csvs(destination_root, summary.destination_labels, summary.destination_blocks))
        for report_path in report_paths:
            print(f"[Done] Add-element report written to {report_path}")

        persist_generator_metadata(
            args,
            command=command,
            output_path=summary.output_path,
            layout=layout,
            extra={
                "destination": str(args.destination),
                "element": summary.element,
                "template_atom": summary.template_atom,
                "similarity_mode": summary.similarity_mode,
                "similarity_details": summary.similarity_details,
                "fields": list(summary.fields),
                "report_format": str(args.report_format),
                "report_paths": [str(p) for p in report_paths],
                "appended": summary.appended,
                "updated": summary.updated,
                "skipped_existing": summary.skipped_existing,
                "skipped_incompatible": summary.skipped_incompatible,
            },
            copy_to_dot=bool(getattr(args, "copy_to_dot", False)),
        )
    elif command in {"add-term-to-ffield", "add_term_to_ffield"}:
        template_map: dict[str, str] = {}
        for item in _parse_csv_items(getattr(args, "template_map", "")):
            if ":" not in item:
                continue
            key, value = item.split(":", 1)
            k = key.strip()
            v = value.strip()
            if k and v:
                template_map[k] = v

        summary = add_term_to_ffield(
            destination=args.destination,
            output=out_path,
            field=str(args.field),
            term=str(args.term),
            closest_term=getattr(args, "closest_term", None),
            template_atom_map=template_map if template_map else None,
            similarity_mode=str(args.similarity),
            radius_metrics=_parse_csv_items(args.radius_metrics),
            same_general_order=bool(getattr(args, "same_general_order", False)),
            replace_existing=bool(args.replace_existing),
        )
        print(f"[Done] Updated ffield written to {summary.output_path}")
        print(f"  field: {summary.field}")
        print(f"  term: {summary.term}")
        print(f"  template term: {summary.template_term}")
        print(f"  similarity: {summary.similarity_mode}")
        print(f"  template atoms: {summary.template_atoms}")
        print(f"  appended: {summary.appended}")
        if summary.updated:
            print(f"  updated: {summary.updated}")
        if summary.skipped_existing:
            print(f"  skipped_existing: {summary.skipped_existing}")

        persist_generator_metadata(
            args,
            command=command,
            output_path=summary.output_path,
            layout=layout,
            extra={
                "destination": str(args.destination),
                "field": summary.field,
                "term": summary.term,
                "closest_term": str(getattr(args, "closest_term", "") or ""),
                "template_term": summary.template_term,
                "template_atoms": summary.template_atoms,
                "similarity_mode": summary.similarity_mode,
                "similarity_details": summary.similarity_details,
                "same_general_order": bool(getattr(args, "same_general_order", False)),
                "appended": summary.appended,
                "updated": summary.updated,
                "skipped_existing": summary.skipped_existing,
            },
            copy_to_dot=bool(getattr(args, "copy_to_dot", False)),
        )
    else:
        atom_types = _parse_csv_items(args.atom_types)
        summary = merge_ffields(
            source=args.source,
            destination=args.destination,
            output=out_path,
            atom_types=atom_types,
            fields=fields,
            replace_existing=bool(args.replace_existing),
            allow_torsion_wildcard=not bool(args.disallow_torsion_wildcard),
            fill_missing_with_template=bool(getattr(args, "fill_missing_with_template", False)),
            template_similarity_mode=str(getattr(args, "template_similarity", "group")),
            template_closest_atom=getattr(args, "template_closest_atom", None),
            template_radius_metrics=_parse_csv_items(getattr(args, "template_radius_metrics", "")),
        )

        print(f"[Done] Merged ffield written to {summary.output_path}")
        print(f"  atom types: {', '.join(summary.atom_types_merged)}")
        print(f"  fields: {', '.join(summary.fields)}")
        print(f"  appended: {summary.appended}")
        if any(summary.updated.values()):
            print(f"  updated: {summary.updated}")
        if any(summary.skipped_existing.values()):
            print(f"  skipped_existing: {summary.skipped_existing}")
        if any(summary.skipped_incompatible.values()):
            print(f"  skipped_incompatible: {summary.skipped_incompatible}")
        if any(summary.template_generated.values()):
            print(f"  template_generated: {summary.template_generated}")
        if summary.template_choices:
            print(f"  template_choices: {summary.template_choices}")

        base = Path(summary.output_path)
        report_root = base.parent / f"{base.stem}_merge_audit"
        source_root = report_root / "source_snapshot"
        template_root = report_root / "template_snapshot"
        skipped_root = report_root / "skipped_existing"
        destination_root = report_root / "destination_projection"
        source_root.mkdir(parents=True, exist_ok=True)
        template_root.mkdir(parents=True, exist_ok=True)
        skipped_root.mkdir(parents=True, exist_ok=True)
        destination_root.mkdir(parents=True, exist_ok=True)

        if args.report_format in {"txt", "both"}:
            if bool(getattr(args, "fill_missing_with_template", False)):
                report_paths.append(
                    _write_similarity_summaries(
                        report_root / "similarity_summary.txt",
                        summary.template_similarity_details,
                    )
                )
            report_paths.append(
                _write_snapshot_text(
                    source_root / "source_snapshot_summary.txt",
                    "Source snapshot of transferred entries",
                    summary.source_labels,
                    summary.source_blocks,
                )
            )
            if bool(getattr(args, "fill_missing_with_template", False)):
                report_paths.append(
                    _write_snapshot_text(
                        template_root / "template_snapshot_summary.txt",
                        "Template snapshot of destination entries used for template generation",
                        summary.template_labels,
                        summary.template_blocks,
                    )
                )
            if any(summary.skipped_existing.values()):
                report_paths.append(
                    _write_snapshot_text(
                        skipped_root / "skipped_existing_summary.txt",
                        "Entries skipped because matching rows already existed in destination",
                        summary.skipped_existing_labels,
                        summary.skipped_existing_blocks,
                    )
                )
            report_paths.append(
                _write_snapshot_text(
                    destination_root / "destination_projection_summary.txt",
                    "Destination projection of transferred entries",
                    summary.destination_labels,
                    summary.destination_blocks,
                )
            )
        if args.report_format in {"csv", "both"}:
            report_paths.extend(_write_snapshot_field_csvs(source_root, summary.source_labels, summary.source_blocks))
            if bool(getattr(args, "fill_missing_with_template", False)):
                report_paths.extend(_write_snapshot_field_csvs(template_root, summary.template_labels, summary.template_blocks))
            if any(summary.skipped_existing.values()):
                report_paths.extend(_write_snapshot_field_csvs(skipped_root, summary.skipped_existing_labels, summary.skipped_existing_blocks))
            report_paths.extend(_write_snapshot_field_csvs(destination_root, summary.destination_labels, summary.destination_blocks))
        for report_path in report_paths:
            print(f"[Done] Merge report written to {report_path}")

        persist_generator_metadata(
            args,
            command=command,
            output_path=summary.output_path,
            layout=layout,
            extra={
                "source": str(args.source),
                "destination": str(args.destination),
                "atom_types": list(summary.atom_types_merged),
                "fields": list(summary.fields),
                "report_format": str(args.report_format),
                "report_paths": [str(p) for p in report_paths],
                "appended": summary.appended,
                "updated": summary.updated,
                "skipped_existing": summary.skipped_existing,
                "skipped_incompatible": summary.skipped_incompatible,
                "template_generated": summary.template_generated,
                "template_choices": summary.template_choices,
                "template_similarity_details": summary.template_similarity_details,
            },
            copy_to_dot=bool(getattr(args, "copy_to_dot", False)),
        )

    copied = maybe_copy_output_to_dot(summary.output_path, enabled=bool(getattr(args, "copy_to_dot", False)))
    dirs = [summary.output_path.parent]
    if copied is not None:
        dirs.append(copied.parent)
    if bool(getattr(args, "copy_to_dot", False)):
        for report_path in report_paths:
            copied_report = maybe_copy_output_to_dot(report_path, enabled=True)
            if copied_report is not None:
                dirs.append(copied_report.parent)
    print_saved_dirs(dirs)
    return 0


_FORCE_FIELD_TERM_SECTIONS = {
    "bond": ("i", "j"),
    "off_diagonal": ("i", "j"),
    "angle": ("i", "j", "k"),
    "torsion": ("i", "j", "k", "l"),
    "hbond": ("i", "j", "k"),
}


def _normalize_force_field_format(value: str) -> str:
    fmt = str(value).strip().lower()
    if fmt == "indices":
        return "raw"
    if fmt not in {"raw", "interpreted"}:
        raise ValueError("Force-field format must be 'raw', 'indices', or 'interpreted'.")
    return fmt


def _split_term_string(term: str) -> list[str]:
    text = str(term).strip()
    if not text:
        return []
    for sep in ("-", ",", " ", "_"):
        if sep in text:
            return [token for token in text.replace(",", " ").replace("-", " ").replace("_", " ").split() if token]
    if text.isdigit():
        return [text]

    tokens: list[str] = []
    i = 0
    while i < len(text):
        ch = text[i]
        if ch.isupper():
            if i + 1 < len(text) and text[i + 1].islower():
                tokens.append(text[i : i + 2])
                i += 2
            else:
                tokens.append(ch)
                i += 1
        else:
            tokens.append(ch)
            i += 1
    return tokens


def _atom_maps_from_data(data: ForceFieldParametersData) -> tuple[dict[int, str], dict[str, int]]:
    atom_df = data.atom_parameters
    if atom_df.empty or "symbol" not in atom_df.columns:
        raise KeyError("ForceFieldParametersData.atom_parameters must include a non-empty 'symbol' column.")

    idx_to_sym: dict[int, str] = {}
    sym_to_idx: dict[str, int] = {}
    for idx, row in atom_df.iterrows():
        atom_idx = int(idx)
        symbol = str(row["symbol"]).strip()
        idx_to_sym[atom_idx] = symbol
        sym_to_idx[symbol] = atom_idx
    return idx_to_sym, sym_to_idx


def _make_term_series_indices(df: pd.DataFrame, cols: tuple[str, ...], *, unordered_2body: bool) -> pd.Series:
    if len(cols) == 2 and unordered_2body:
        a = df[cols[0]].astype("Int64")
        b = df[cols[1]].astype("Int64")
        lo = a.where(a <= b, b)
        hi = b.where(a <= b, a)
        return lo.astype(str) + "-" + hi.astype(str)

    parts = [df[col].astype("Int64").astype(str) for col in cols]
    out = parts[0]
    for part in parts[1:]:
        out = out + "-" + part
    return out


def _filter_force_field_table_by_term(
    data: ForceFieldParametersData,
    section: str,
    df: pd.DataFrame,
    *,
    term: str,
    unordered_2body: bool = True,
    any_order: bool = False,
) -> pd.DataFrame:
    if section not in _FORCE_FIELD_TERM_SECTIONS:
        raise ValueError(f"--term is only valid for sections: {', '.join(sorted(_FORCE_FIELD_TERM_SECTIONS))}.")

    cols = _FORCE_FIELD_TERM_SECTIONS[section]
    tokens = _split_term_string(term)
    if not tokens:
        return df

    if all(token.isdigit() for token in tokens):
        idxs = [int(token) for token in tokens]
    else:
        _, sym_to_idx = _atom_maps_from_data(data)
        idxs = []
        for token in tokens:
            if token not in sym_to_idx:
                raise KeyError(f"Unknown atom symbol {token!r}. Available: {sorted(sym_to_idx)}")
            idxs.append(int(sym_to_idx[token]))

    if len(idxs) != len(cols):
        raise ValueError(f"Term {term!r} implies {len(idxs)} atoms, but section '{section}' requires {len(cols)}.")

    is_2body = len(cols) == 2
    if is_2body and (unordered_2body or any_order):
        a, b = sorted(idxs)
        wanted = f"{a}-{b}"
        key = _make_term_series_indices(df, cols, unordered_2body=True)
        return df.loc[key == wanted].copy()

    if any_order and not is_2body:
        wanted_sorted = sorted(idxs)
        sub = df.loc[:, list(cols)].astype("Int64")

        def _row_matches(row: pd.Series) -> bool:
            values = [int(value) for value in row.tolist()]
            values.sort()
            return values == wanted_sorted

        mask = sub.apply(_row_matches, axis=1)
        return df.loc[mask].copy()

    wanted = "-".join(str(idx) for idx in idxs)
    key = _make_term_series_indices(df, cols, unordered_2body=False)
    return df.loc[key == wanted].copy()


def _resolve_engine_from_args(normalized: dict, args: argparse.Namespace):
    detection_path = (
        normalized.get("_snapshot_source_dir")
        or normalized.get("input")
        or normalized.get("run_dir")
        or "."
    )
    return resolve_engine(
        str(detection_path),
        engine=getattr(args, "engine", None),
    )


def _load_force_field_data(args: argparse.Namespace) -> ForceFieldParametersData:
    normalized = normalize_storage_args(vars(args))
    adapter = _resolve_engine_from_args(normalized, args)
    return adapter.load(ForceFieldParametersData, normalized)


def _export_force_field_tables(tables: dict[str, pd.DataFrame], outdir: str | Path, *, fmt: str) -> None:
    out_path = Path(outdir)
    out_path.mkdir(parents=True, exist_ok=True)
    suffix = "interpreted" if fmt == "interpreted" else "indices"
    for section, table in tables.items():
        table.to_csv(out_path / f"{section}_{suffix}.csv", index=True)


def _add_runtime_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--engine", choices=["reaxff", "ams", "lammps"], default=None)
    parser.add_argument("--input", default=".", help="Input file or directory for engine resolution")
    parser.add_argument("--run-dir", "--dir", dest="run_dir", default=".", help="Run directory fallback for engine detection")
    parser.add_argument("--ffield", default="ffield", help="Path to ffield")
    parser.add_argument("--fort13", default="fort.13", help="Path to fort.13")
    parser.add_argument("--fort79", default="fort.79", help="Path to fort.79")
    parser.add_argument("--fort99", default="fort.99", help="Path to fort.99")
    parser.add_argument("--fort74", default="fort.74", help="Path to fort.74")
    parser.add_argument("--trainset", default="trainset.in", help="Path to trainset file")
    parser.add_argument("--log", choices=["verbose", "quiet"], default=None, help="Logging level")
    add_storage_cli_arguments(parser)


def _add_presentation_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--plot", choices=["single", "subplot", "tornado", "beeswarm"], default=None, help="Render a plot")
    parser.add_argument("--show", action="store_true", help="Show the generated plot window")
    parser.add_argument("--save", default=None, help="Save the generated plot to a file path")
    parser.add_argument("--export", default=None, help="Write the result table to CSV")
    parser.add_argument("--grid", default=None, help="Subplot grid like 2x2 or 2*2")
    parser.add_argument("--xaxis", default=None, help="Optional x-axis column override")


def _build_force_field_data_request(args: argparse.Namespace) -> ForceFieldDataRequest:
    section = args.field if args.field else None
    fmt = _normalize_force_field_format(args.format)
    return ForceFieldDataRequest(
        section=section,
        interpret=(fmt == "interpreted"),
    )


def _build_force_field_optimization_request(args: argparse.Namespace) -> ForceFieldOptimizationRequest:
    return ForceFieldOptimizationRequest(epochs=args.epochs)


def _build_structure_summary_request(args: argparse.Namespace) -> StructureSummaryRequest:
    return StructureSummaryRequest()


def _build_parameter_optimization_diagnostic_request(args: argparse.Namespace) -> ParameterOptimizationDiagnosticRequest:
    return ParameterOptimizationDiagnosticRequest(
        interpret=bool(getattr(args, "interpret", False)),
    )


def _build_force_field_optimization_report_request(args: argparse.Namespace) -> ForceFieldOptimizationReportRequest:
    return ForceFieldOptimizationReportRequest()


def _build_force_field_optimization_report_eos_request(args: argparse.Namespace) -> ForceFieldOptimizationReportEOSRequest:
    return ForceFieldOptimizationReportEOSRequest(
        iden=args.iden,
    )


def _build_force_field_optimization_report_bulk_modulus_request(
    args: argparse.Namespace,
) -> ForceFieldOptimizationReportBulkModulusRequest:
    return ForceFieldOptimizationReportBulkModulusRequest(
        iden=args.iden,
        shift_min_to_zero=not args.no_shift_min_to_zero,
        flip_sign=args.flip_sign,
        min_points=int(args.min_points),
    )


def _build_trainset_group_comments_request(args: argparse.Namespace) -> TrainsetGroupCommentsRequest:
    return TrainsetGroupCommentsRequest(section=str(getattr(args, "section", "all")))


def _build_trainset_data_request(args: argparse.Namespace) -> GetTrainsetDataRequest:
    return GetTrainsetDataRequest(section=str(getattr(args, "section", "all")))


REQUEST_BUILDERS: dict[str, Callable[[argparse.Namespace], object]] = {
    "get_ffield_data": _build_force_field_data_request,
    "get_ffield_opt_progress_data": _build_force_field_optimization_request,
    "get_energy_min_summary_data": _build_structure_summary_request,
    "get_ffield_diagnostic_data": _build_parameter_optimization_diagnostic_request,
    "get_ffield_opt_results": _build_force_field_optimization_report_request,
    "get_ffield_opt_eos": _build_force_field_optimization_report_eos_request,
    "ffield_opt_bulk_modulus": _build_force_field_optimization_report_bulk_modulus_request,
    "get_trainset_data": _build_trainset_data_request,
    "get_trainset_group_comments": _build_trainset_group_comments_request,
}

def _prepare_result(command: str, result) -> object:
    if command == "get_ffield_data" and getattr(result, "table", None) is None and getattr(result, "tables", None):
        frames = []
        for section, table in result.tables.items():
            df = table.copy()
            df.insert(0, "section", section)
            frames.append(df)
        result.table = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return result


def _filter_structure_summary_columns(result, requested_col: str) -> None:
    table = getattr(result, "table", None)
    if not isinstance(table, pd.DataFrame) or table.empty:
        return
    raw = str(requested_col).strip()
    if not raw or raw.lower() == "all":
        return
    canonical = normalize_choice(raw)
    resolved = resolve_alias_from_columns(table.columns, canonical)
    if resolved is None:
        raise ValueError(f"Column {raw!r} not found. Available columns: {', '.join(str(col) for col in table.columns)}")
    cols = []
    if "identifier" in table.columns:
        cols.append("identifier")
    if resolved != "identifier":
        cols.append(resolved)
    result.table = table.loc[:, cols].copy()
    if resolved != raw and resolved in result.table.columns:
        result.table = result.table.rename(columns={resolved: raw})


def _build_most_sensitive_result(base_result):
    table = getattr(base_result, "table", None)
    if not isinstance(table, pd.DataFrame) or table.empty:
        return argparse.Namespace(table=pd.DataFrame(), values={}, metadata={})
    df = table.copy()
    idx_min = df["min_sensitivity"].idxmin()
    min_identifier = df.loc[idx_min, "identifier"]
    min_value = float(df.loc[idx_min, "min_sensitivity"])
    print(f"[Done] Minimum sensitivity value: {min_value:.6f}")
    print(f"Corresponding identifier: {min_identifier}")
    df["epoch_set"] = df.index + 1
    ratio_cols = [col for col in ("sensitivity1/3", "sensitivity2/3", "sensitivity4/3") if col in df.columns]
    subset = df.loc[df["identifier"] == min_identifier].copy()
    if ratio_cols:
        subset_long = (
            subset[["epoch_set", "identifier", "min_sensitivity", "max_sensitivity"] + ratio_cols]
            .melt(
                id_vars=["epoch_set", "identifier", "min_sensitivity", "max_sensitivity"],
                value_vars=ratio_cols,
                var_name="sensitivity_name",
                value_name="sensitivity",
            )
            .dropna(subset=["sensitivity"])
            .reset_index(drop=True)
        )
    else:
        subset_long = subset.loc[:, ["epoch_set", "identifier", "min_sensitivity", "max_sensitivity"]].copy()
    return argparse.Namespace(
        table=subset_long,
        values={"identifier": min_identifier, "min_sensitivity": min_value},
        metadata={"full_table": df},
    )


def _build_tornado_result(base_result, top: int):
    table = getattr(base_result, "table", None)
    if not isinstance(table, pd.DataFrame) or table.empty:
        return argparse.Namespace(table=pd.DataFrame(), values={}, metadata={})
    sensitivities = table.copy()
    grouped = (
        sensitivities.groupby("identifier", dropna=True)
        .agg(min_eff=("min_sensitivity", "min"), max_eff=("max_sensitivity", "max"))
        .reset_index()
    )
    eff_union = (
        sensitivities.melt(
            id_vars=["identifier"],
            value_vars=["min_sensitivity", "max_sensitivity"],
            var_name="kind",
            value_name="eff",
        )
        .dropna(subset=["eff"])
    )
    grouped["median_eff"] = grouped["identifier"].map(eff_union.groupby("identifier", dropna=True)["eff"].median())
    grouped["span"] = grouped["max_eff"] - grouped["min_eff"]
    grouped = grouped.sort_values("span", ascending=False).reset_index(drop=True)
    if top and top > 0:
        grouped = grouped.head(int(top)).copy()
    return argparse.Namespace(table=grouped, values={}, metadata={})


def _prepare_eos_table(result, *, flip_sign: bool) -> None:
    table = getattr(result, "table", None)
    if not isinstance(table, pd.DataFrame) or table.empty or not flip_sign:
        return
    for col in ("E_other_iden", "ffield_value", "qm_value"):
        if col in table.columns:
            result.table[col] = -pd.to_numeric(table[col], errors="coerce")


def _plot_payload(command: str, result, args: argparse.Namespace) -> dict[str, object] | None:
    table = getattr(result, "table", None)
    if not isinstance(table, pd.DataFrame) or table.empty:
        return None

    if command == "get_ffield_opt_progress_data":
        return {
            "plot_type": "single_plot",
            "x": table["epoch"].tolist(),
            "y": pd.to_numeric(table["total_ff_error"], errors="coerce").tolist(),
            "xlabel": "Epoch",
            "ylabel": "Total FF Error",
            "title": "Force-Field Optimization",
        }

    if command == "get_ffield_diagnostic_data":
        if getattr(args, "plot", None) == "beeswarm":
            if "identifier" not in table.columns:
                return None
            candidate_cols = [col for col in table.columns if "sensitivity" in str(col).lower()]
            if not candidate_cols:
                return None
            long = (
                table[["identifier"] + candidate_cols]
                .melt(id_vars=["identifier"], value_vars=candidate_cols, var_name="metric", value_name="value")
                .dropna(subset=["value"])
            )
            if long.empty:
                return None
            long["value"] = pd.to_numeric(long["value"], errors="coerce")
            long = long.dropna(subset=["value"])
            if long.empty:
                return None
            top = int(getattr(args, "top", 0) or 0)
            if top > 0:
                spans = (
                    long.groupby("identifier", dropna=False)["value"]
                    .agg(lambda s: float(s.max() - s.min()))
                    .sort_values(ascending=False)
                )
                keep = set(spans.head(top).index.tolist())
                long = long.loc[long["identifier"].isin(keep)].copy()
                if long.empty:
                    return None
            return {
                "plot_type": "beeswarm_plot",
                "x": long["value"].tolist(),
                "y": long["identifier"].astype(str).tolist(),
                "hue": long["value"].tolist(),
                "palette": "coolwarm",
                "size": 5,
                "legend": False,
                "xlabel": "Sensitivity",
                "ylabel": "",
                "title": "Parameter Sensitivity Beeswarm",
            }
        if getattr(args, "plot", None) == "tornado":
            if not {"min_eff", "max_eff", "median_eff", "identifier"}.issubset(set(table.columns)):
                return None
            return {
                "plot_type": "tornado_plot",
                "labels": table["identifier"].tolist(),
                "min_vals": pd.to_numeric(table["min_eff"], errors="coerce").tolist(),
                "max_vals": pd.to_numeric(table["max_eff"], errors="coerce").tolist(),
                "median_vals": pd.to_numeric(table["median_eff"], errors="coerce").tolist(),
                "xlabel": "Sensitivity (Error Response)",
                "ylabel": "Parameter",
                "title": "Parameter Sensitivity Tornado Plot",
                "vline": getattr(args, "vline", None),
            }
        if getattr(args, "report_most_sensitive", False):
            if "sensitivity" not in table.columns:
                return None
            series = []
            if "sensitivity_name" in table.columns:
                for name, group in table.groupby("sensitivity_name", dropna=False):
                    series.append(
                        {
                            "x": pd.to_numeric(group["epoch_set"], errors="coerce").tolist(),
                            "y": pd.to_numeric(group["sensitivity"], errors="coerce").tolist(),
                            "label": str(name),
                            "marker": "o",
                            "linewidth": 0,
                            "markersize": 20,
                            "alpha": 1,
                        }
                    )
            else:
                series.append(
                    {
                        "x": pd.to_numeric(table["epoch_set"], errors="coerce").tolist(),
                        "y": pd.to_numeric(table["sensitivity"], errors="coerce").tolist(),
                        "label": str(table["identifier"].iloc[0]) if "identifier" in table.columns and not table.empty else "most_sensitive",
                        "marker": "o",
                        "linewidth": 0,
                        "markersize": 20,
                        "alpha": 1,
                    }
                )
            return {
                "plot_type": "single_plot",
                "series": series,
                "xlabel": "Epoch Set",
                "ylabel": "Sensitivity (Error Response)",
                "title": "Most Sensitive Parameter Ratios",
                "legend": True,
            }
        return {
            "plot_type": "single_plot",
            "series": [
                {"x": table["identifier"].tolist(), "y": pd.to_numeric(table["min_sensitivity"], errors="coerce").tolist(), "label": "min"},
                {"x": table["identifier"].tolist(), "y": pd.to_numeric(table["max_sensitivity"], errors="coerce").tolist(), "label": "max"},
            ],
            "xlabel": "Identifier",
            "ylabel": "Sensitivity",
            "title": "Parameter Optimization Diagnostic",
            "legend": True,
        }

    if command == "get_ffield_opt_results":
        x_col = getattr(args, "xaxis", None) or "lineno"
        if x_col not in table.columns:
            x_col = "lineno"
        y_col = "qm_ff_difference" if "qm_ff_difference" in table.columns else "error"
        return {
            "plot_type": "single_plot",
            "x": table[x_col].tolist(),
            "y": pd.to_numeric(table[y_col], errors="coerce").tolist(),
            "xlabel": x_col,
            "ylabel": y_col,
            "title": "Force-Field Optimization Report",
        }

    if command == "get_ffield_opt_eos":
        work = table.copy()
        for col in ("V_other_iden", "E_other_iden"):
            if col in work.columns:
                work[col] = pd.to_numeric(work[col], errors="coerce")
        work = work.sort_values(["base_iden", "V_other_iden"]).reset_index(drop=True)
        groups = [(iden, grp.copy()) for iden, grp in work.groupby("base_iden", dropna=False)]
        if getattr(args, "plot", None) == "subplot" and len(groups) > 1:
            return {
                "plot_type": "multi_subplots",
                "subplots": [
                    [
                        {"x": grp["V_other_iden"].tolist(), "y": grp["E_other_iden"].tolist(), "label": f"{iden} energy"},
                    ]
                    for iden, grp in groups
                ],
                "xlabel": "Volume",
                "ylabel": "Energy",
                "title": "EOS Energy vs Volume",
                "legend": True,
                "grid": getattr(args, "grid", None),
            }
        series = []
        for iden, grp in groups:
            series.append({"x": grp["V_other_iden"].tolist(), "y": grp["E_other_iden"].tolist(), "label": f"{iden} energy"})
        return {
            "plot_type": "single_plot",
            "series": series,
            "xlabel": "Volume",
            "ylabel": "Energy",
            "title": "EOS Energy vs Volume",
            "legend": True,
        }

    if command == "get_energy_min_summary_data":
        x_col = getattr(args, "xaxis", None) or "identifier"
        if x_col not in table.columns:
            x_col = "identifier"
        numeric_cols = [col for col in table.columns if col != x_col and pd.api.types.is_numeric_dtype(table[col])]
        if not numeric_cols:
            return None
        if getattr(args, "plot", None) == "subplot":
            subplots = []
            for col in numeric_cols:
                subplots.append([{"x": table[x_col].tolist(), "y": pd.to_numeric(table[col], errors="coerce").tolist(), "label": col}])
            return {
                "plot_type": "multi_subplots",
                "subplots": subplots,
                "xlabel": x_col,
                "ylabel": "Value",
                "title": "Structure Summary",
                "legend": False,
                "grid": getattr(args, "grid", None),
            }
        series = [{"x": table[x_col].tolist(), "y": pd.to_numeric(table[col], errors="coerce").tolist(), "label": col} for col in numeric_cols]
        return {
            "plot_type": "single_plot",
            "series": series,
            "xlabel": x_col,
            "ylabel": "Value",
            "title": "Structure Summary",
            "legend": True,
        }

    if command == "get_ffield_data":
        x_col = getattr(args, "xaxis", None)
        if x_col is None or x_col not in table.columns:
            x_col = table.columns[0]
        numeric_cols = [col for col in table.columns if col != x_col and pd.api.types.is_numeric_dtype(table[col])]
        if not numeric_cols:
            return None
        first = numeric_cols[0]
        return {
            "plot_type": "single_plot",
            "x": table[x_col].tolist(),
            "y": pd.to_numeric(table[first], errors="coerce").tolist(),
            "xlabel": x_col,
            "ylabel": first,
            "title": "Force-Field Data",
        }

    return None


def _run_ffield_data(args: argparse.Namespace) -> int:
    data = _load_force_field_data(args)
    request = REQUEST_BUILDERS["get_ffield_data"](args)
    task = ForceFieldDataTask()
    if args.term:
        if request.section is None:
            raise ValueError("--term requires exactly one selected section via --field.")
        selected_section = request.section
        raw_result = task.run(data, ForceFieldDataRequest(section=selected_section, interpret=False))
        raw_table = raw_result.table
        filtered_raw = _filter_force_field_table_by_term(
            data,
            selected_section,
            raw_table,
            term=args.term,
            unordered_2body=not args.ordered_2body,
            any_order=args.any_order,
        )
        if request.interpret:
            interpreted_result = task.run(data, ForceFieldDataRequest(section=selected_section, interpret=True))
            table = interpreted_result.table.loc[filtered_raw.index].copy()
        else:
            table = filtered_raw
        result = task.run(data, ForceFieldDataRequest(section=selected_section, interpret=request.interpret))
        result.table = table
    else:
        result = task.run(data, request)
    result = _prepare_result("get_ffield_data", result)
    if args.outdir:
        export_tables: dict[str, pd.DataFrame] = {}
        if isinstance(getattr(result, "table", None), pd.DataFrame) and not result.table.empty:
            if "section" in result.table.columns:
                for section_name, group in result.table.groupby("section", dropna=False):
                    section_key = str(section_name)
                    export_tables[section_key] = group.drop(columns=["section"]).copy()
            else:
                selected = request.section or "ffield"
                export_tables[str(selected)] = result.table.copy()
        export_fmt = "interpreted" if request.interpret else "raw"
        _export_force_field_tables(export_tables, args.outdir, fmt=export_fmt)
        print(f"[Done] Exported section tables to {args.outdir}")
    present_result("get_ffield_data", result, args, plot_payload_builder=_plot_payload)
    return 0


def _run_ffield_analysis_main(command: str, args: argparse.Namespace) -> int:
    canonical = _resolve_workflow_command(command)
    if canonical == "get_ffield_data":
        return _run_ffield_data(args)

    if canonical == "get_ffield_diagnostic_data":
        executor = AnalysisExecutor()
        base_result = executor.run(
            TASK_REGISTRY["parameter_optimization_diagnostic"](),
            REQUEST_BUILDERS["get_ffield_diagnostic_data"](args),
            vars(args),
        )
        if bool(getattr(args, "report_most_sensitive", False)):
            result = _build_most_sensitive_result(base_result)
            if getattr(args, "export_all", None):
                out_all = resolve_output_path(
                    args.export_all,
                    canonical,
                    run_id=getattr(args, "run_id", None),
                    project_root=getattr(args, "project_root", "."),
                    analysis_id=getattr(args, "analysis_id", None),
                )
                export_result_csv(argparse.Namespace(table=result.metadata["full_table"]), str(out_all))
                print(f"[Done] Exported full diagnostic table to {out_all}")
        elif getattr(args, "plot", None) == "tornado":
            result = _build_tornado_result(base_result, top=int(args.top))
        else:
            result = base_result
        present_result(canonical, result, args, plot_payload_builder=_plot_payload)
        return 0

    task_cls = TASK_REGISTRY[_task_name_for_command(canonical)]
    request = REQUEST_BUILDERS[canonical](args)
    executor = AnalysisExecutor()
    result = executor.run(task_cls(), request, vars(args))
    result = _prepare_result(canonical, result)
    if canonical == "get_energy_min_summary_data":
        _filter_structure_summary_columns(result, getattr(args, "col", "all"))
    elif canonical == "get_ffield_opt_eos":
        _prepare_eos_table(result, flip_sign=bool(getattr(args, "flip_sign", False)))
    present_result(canonical, result, args, plot_payload_builder=_plot_payload)
    return 0


def build_parser(parser: argparse.ArgumentParser, *, command: str) -> argparse.ArgumentParser:
    return _build_parser(parser, command=command)


def run_main(command: str, args: argparse.Namespace) -> int:
    canonical = _resolve_workflow_command(command)
    if canonical in FFIELD_ANALYSIS_COMMANDS:
        return _run_ffield_analysis_main(canonical, args)
    return _run_ffield_main(canonical, args)
