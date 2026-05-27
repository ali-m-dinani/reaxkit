"""Direct command workflow for molecular analysis tasks."""

from __future__ import annotations

import argparse
from types import SimpleNamespace
from typing import Callable

import pandas as pd

from reaxkit.analysis import molecular_analysis as _molecular_tasks  # noqa: F401
from reaxkit.analysis.molecular_analysis.molecular_analysis import (
    DominantSpeciesRequest,
    LargestMoleculeByMassRequest,
    LargestMoleculeCompositionRequest,
    MoleculeLifetimeRequest,
)
from reaxkit.core.analysis_executor import AnalysisExecutor
from reaxkit.core.analysis_task_registry import TASK_REGISTRY
from reaxkit.core.command_alias_resolver import resolve_command_name
from reaxkit.core.frame_utils import parse_frame_indices
from reaxkit.core.storage_layout import add_storage_cli_arguments
from reaxkit.presentation.dispatcher import present_result

ALL_COMMANDS = (
    "get_dominant_species",
    "get_largest_molecule_by_mass",
    "get_largest_molecule_composition",
    "get_molecule_lifetime",
)
ALL_LEGACY_COMMANDS = (
    "dominant_species",
    "largest_molecule_by_mass",
    "largest_molecule_composition",
    "molecule_lifetime",
    "get-dominant-species",
    "get-largest-molecule-by-mass",
    "get-largest-molecule-composition",
    "get-molecule-lifetime",
)


def _add_runtime_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--engine", choices=["reaxff", "ams", "lammps"], default=None, help="Engine override. Example: --engine reaxff, which forces ReaxFF parser/loader behavior.")
    parser.add_argument("--input", default=".", help="Input file or directory for engine resolution. Example: --input runs/job1, which sets data-loading context.")
    parser.add_argument("--run-dir", "--dir", dest="run_dir", default=".", help="Run directory fallback for engine detection. Example: --run-dir runs/job1, which acts as backup lookup path.")
    parser.add_argument("--molfra", "--file", dest="molfra", default="molfra.out", help="Molecular analysis file path. Example: --molfra molfra.out, which reads species-frequency data from that file.")
    parser.add_argument("--log", choices=["verbose", "quiet"], default=None, help="Logging level. Example: --log verbose, which prints more runtime details.")
    add_storage_cli_arguments(parser)


def _add_presentation_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--plot", choices=["single", "subplot"], default=None, help="Render a plot. Example: --plot single, which draws one combined chart.")
    parser.add_argument("--show", action="store_true", help="Show the generated plot window. Example: --show, which opens the plot interactively.")
    parser.add_argument("--save", default=None, help="Save the generated plot to a file path. Example: --save dominant_species.png, which writes the figure image.")
    parser.add_argument("--export", default=None, help="Write the result table to CSV. Example: --export dominant_species.csv, which saves tabular output.")
    parser.add_argument("--grid", default=None, help="Subplot grid like 2x2 or 2*2. Example: --grid 2x2, which arranges subplots in a 2-by-2 layout.")
    parser.add_argument("--xaxis", choices=["frame", "iter"], default="frame", help="Quantity on x-axis. Example: --xaxis iter, which uses iteration values on horizontal axis.")


def _add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--frames",
        nargs="*",
        default=None,
        help='Frame selector syntax. Example: --frames 0:20:2, which selects frames 0,2,4,...,20.',
    )
    parser.add_argument("--every", type=int, default=1, help="Frame stride. Example: --every 5, which keeps every fifth selected frame.")


def _build_dominant_species_request(args: argparse.Namespace) -> DominantSpeciesRequest:
    return DominantSpeciesRequest(
        frames=parse_frame_indices(args.frames),
        every=args.every,
        top_n=args.top_n,
        min_freq=args.min_freq,
    )


def _build_largest_molecule_by_mass_request(args: argparse.Namespace) -> LargestMoleculeByMassRequest:
    return LargestMoleculeByMassRequest(
        frames=parse_frame_indices(args.frames),
        every=args.every,
    )


def _build_largest_molecule_composition_request(args: argparse.Namespace) -> LargestMoleculeCompositionRequest:
    return LargestMoleculeCompositionRequest(
        frames=parse_frame_indices(args.frames),
        every=args.every,
    )


def _build_molecule_lifetime_request(args: argparse.Namespace) -> MoleculeLifetimeRequest:
    return MoleculeLifetimeRequest(
        molecules=args.molecules,
        frames=parse_frame_indices(args.frames),
        every=args.every,
        min_freq=args.min_freq,
    )


REQUEST_BUILDERS: dict[str, Callable[[argparse.Namespace], object]] = {
    "get_dominant_species": _build_dominant_species_request,
    "get_largest_molecule_by_mass": _build_largest_molecule_by_mass_request,
    "get_largest_molecule_composition": _build_largest_molecule_composition_request,
    "get_molecule_lifetime": _build_molecule_lifetime_request,
}


def build_parser(parser: argparse.ArgumentParser, *, command: str) -> argparse.ArgumentParser:
    canonical = resolve_command_name(command, task_names=ALL_COMMANDS)
    parser.set_defaults(command=canonical)
    parser.set_defaults(progress=True)
    parser.formatter_class = argparse.RawTextHelpFormatter

    _add_runtime_arguments(parser)
    _add_presentation_arguments(parser)
    _add_common_arguments(parser)

    if canonical == "get_dominant_species":
        parser.description = (
            "Return dominant molecular species for selected frames.\n"
            "This command ranks species by frequency per frame and can return multiple top ranks,\n"
            "with optional frequency threshold filtering.\n\n"
            "Examples:\n"
            "  1. Export top 3 species per frame:\n"
            "   reaxkit get_dominant_species --top-n 3 --export dominant_species.csv\n\n"
            "  2. Analyze selected frames with minimum frequency and plot:\n"
            "   reaxkit get_dominant_species --frames 0 10 20 --min-freq 2 --plot single\n\n"
            "  3. Use frame stride and iteration axis in saved figure:\n"
            "   reaxkit get_dominant_species --every 5 --xaxis iter --save dominant_species.png"
        )
        parser.add_argument("--top-n", type=int, default=1, help="Number of ranked species per frame. Example: --top-n 3, which returns first/second/third dominant species.")
        parser.add_argument("--min-freq", type=float, default=0.0, help="Minimum species frequency to include. Example: --min-freq 2, which filters out low-frequency species.")
    elif canonical == "get_largest_molecule_by_mass":
        parser.description = (
            "Return the heaviest molecular species for selected frames.\n"
            "Use this command to track how the maximum molecular mass evolves over trajectory frames.\n\n"
            "Examples:\n"
            "  1. Export largest-mass species table:\n"
            "   reaxkit get_largest_molecule_by_mass --export largest_mass.csv\n\n"
            "  2. Plot largest-mass trend on selected frames:\n"
            "   reaxkit get_largest_molecule_by_mass --frames 0 20 40 --plot single\n\n"
            "  3. Subsample frames and save iteration-axis plot:\n"
            "   reaxkit get_largest_molecule_by_mass --every 10 --xaxis iter --save largest_mass.png"
        )
    elif canonical == "get_largest_molecule_composition":
        parser.description = (
            "Return elemental composition of the heaviest molecular species per frame.\n"
            "This command reports element counts for the dominant-by-mass molecule in each frame,\n"
            "which helps track composition shifts over time.\n\n"
            "Examples:\n"
            "  1. Export composition table:\n"
            "   reaxkit get_largest_molecule_composition --export composition.csv\n\n"
            "  2. Plot selected frames with subplot layout:\n"
            "   reaxkit get_largest_molecule_composition --frames 0 10 20 --plot subplot\n\n"
            "  3. Subsample frames and save iteration-axis plot:\n"
            "   reaxkit get_largest_molecule_composition --every 5 --xaxis iter --save composition.png"
        )
    elif canonical == "get_molecule_lifetime":
        parser.description = (
            "Compute lifetimes of molecular species across selected frames.\n"
            "You can restrict to target formulas and filter by minimum activity frequency before\n"
            "lifetime statistics are reported.\n\n"
            "Examples:\n"
            "  1. Compute lifetimes for selected molecules and export:\n"
            "   reaxkit get_molecule_lifetime --molecules H2O OH --export lifetimes.csv\n\n"
            "  2. Compute and plot lifetimes for all detected molecules:\n"
            "   reaxkit get_molecule_lifetime --plot single\n\n"
            "  3. Apply frequency threshold on selected frames and save plot:\n"
            "   reaxkit get_molecule_lifetime --min-freq 2 --frames 0 50 100 --save molecule_lifetimes.png"
        )
        parser.add_argument("--molecules", nargs="*", default=None, help="Restrict to selected molecular formulae. Example: --molecules H2O OH, which limits analysis to water and hydroxyl.")
        parser.add_argument("--min-freq", type=float, default=1.0, help="Minimum frequency for an active molecule. Example: --min-freq 2, which treats only sufficiently frequent molecules as active.")
    else:
        raise KeyError(f"Unsupported molecular analysis command '{canonical}'.")

    return parser


def _selected_table(command: str, result, args: argparse.Namespace) -> pd.DataFrame:
    return result.table


def _plot_payload(command: str, result, args: argparse.Namespace) -> dict[str, object] | None:
    table = _selected_table(command, result, args)
    if not isinstance(table, pd.DataFrame) or table.empty:
        return None

    x_col = "iter" if getattr(args, "xaxis", "frame") == "iter" and "iter" in table.columns else "frame_index"
    xlabel = "Iteration" if x_col == "iter" else "Frame Index"

    if command == "get_dominant_species":
        series = []
        subplots = []
        for label, group in table.groupby("molecular_formula", sort=True):
            group = group.sort_values(["frame_index", "rank"])
            payload = {
                "x": group[x_col].tolist(),
                "y": pd.to_numeric(group["freq"], errors="coerce").tolist(),
                "label": str(label),
            }
            series.append(payload)
            subplots.append([payload])
        if getattr(args, "plot", None) == "subplot":
            return {
                "plot_type": "multi_subplots",
                "subplots": subplots,
                "xlabel": xlabel,
                "ylabel": "Frequency",
                "title": "Dominant Species",
                "legend": False,
                "grid": getattr(args, "grid", None),
            }
        return {
            "plot_type": "single_plot",
            "series": series,
            "xlabel": xlabel,
            "ylabel": "Frequency",
            "title": "Dominant Species",
            "legend": True,
        }

    if command == "get_largest_molecule_by_mass":
        return {
            "plot_type": "single_plot",
            "x": table[x_col].tolist(),
            "y": pd.to_numeric(table["molecular_mass"], errors="coerce").tolist(),
            "xlabel": xlabel,
            "ylabel": "Molecular Mass",
            "title": "Largest Molecule By Mass",
        }

    if command == "get_largest_molecule_composition":
        series = []
        subplots = []
        for element, group in table.groupby("element", sort=True):
            group = group.sort_values(["frame_index", "element"])
            payload = {
                "x": group[x_col].tolist(),
                "y": pd.to_numeric(group["count"], errors="coerce").tolist(),
                "label": str(element),
            }
            series.append(payload)
            subplots.append([payload])
        if getattr(args, "plot", None) == "subplot":
            return {
                "plot_type": "multi_subplots",
                "subplots": subplots,
                "xlabel": xlabel,
                "ylabel": "Element Count",
                "title": "Largest Molecule Composition",
                "legend": False,
                "grid": getattr(args, "grid", None),
            }
        return {
            "plot_type": "single_plot",
            "series": series,
            "xlabel": xlabel,
            "ylabel": "Element Count",
            "title": "Largest Molecule Composition",
            "legend": True,
        }

    if command == "get_molecule_lifetime":
        return {
            "plot_type": "single_plot",
            "x": table[x_col.replace("frame_index", "start_frame_index").replace("iter", "start_iter")].tolist(),
            "y": pd.to_numeric(table["sampled_step_count"], errors="coerce").tolist(),
            "xlabel": "Start Iteration" if x_col == "iter" else "Start Frame Index",
            "ylabel": "Samples In Lifetime",
            "title": "Molecule Lifetimes",
        }

    return None


def run_main(command: str, args: argparse.Namespace) -> int:
    canonical = resolve_command_name(command, task_names=ALL_COMMANDS)
    task_cls = TASK_REGISTRY[canonical]
    request = REQUEST_BUILDERS[canonical](args)

    executor = AnalysisExecutor()
    result = executor.run(task_cls(), request, vars(args))
    table = _selected_table(canonical, result, args)
    result_for_presentation = SimpleNamespace(table=table)
    present_result(canonical, result_for_presentation, args, plot_payload_builder=lambda cmd, _res, a: _plot_payload(cmd, result, a))
    return 0
