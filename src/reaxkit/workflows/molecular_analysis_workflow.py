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
from reaxkit.presentation.dispatcher import present_result

MOLECULAR_ANALYSIS_COMMANDS = (
    "dominant_species",
    "largest_molecule_by_mass",
    "largest_molecule_composition",
    "molecule_lifetime",
)


def _add_runtime_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--engine", choices=["reaxff", "ams", "lammps"], default=None)
    parser.add_argument("--input", default=".", help="Input file or directory for engine resolution")
    parser.add_argument("--run-dir", "--dir", dest="run_dir", default=".", help="Run directory fallback for engine detection")
    parser.add_argument("--molfra", "--file", dest="molfra", default="molfra.out", help="Molecular analysis file path")
    parser.add_argument("--log", choices=["verbose", "quiet"], default=None, help="Logging level")


def _add_presentation_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--plot", choices=["single", "subplot"], default=None, help="Render a plot")
    parser.add_argument("--show", action="store_true", help="Show the generated plot window")
    parser.add_argument("--save", default=None, help="Save the generated plot to a file path")
    parser.add_argument("--export", default=None, help="Write the result table to CSV")
    parser.add_argument("--grid", default=None, help="Subplot grid like 2x2 or 2*2")
    parser.add_argument("--xaxis", choices=["frame", "iter"], default="frame", help="Quantity on x-axis")


def _add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--frames", type=int, nargs="*", default=None, help="Selected frame indices")
    parser.add_argument("--every", type=int, default=1, help="Frame stride")


def _build_dominant_species_request(args: argparse.Namespace) -> DominantSpeciesRequest:
    return DominantSpeciesRequest(
        frames=args.frames,
        every=args.every,
        top_n=args.top_n,
        min_freq=args.min_freq,
    )


def _build_largest_molecule_by_mass_request(args: argparse.Namespace) -> LargestMoleculeByMassRequest:
    return LargestMoleculeByMassRequest(
        frames=args.frames,
        every=args.every,
    )


def _build_largest_molecule_composition_request(args: argparse.Namespace) -> LargestMoleculeCompositionRequest:
    return LargestMoleculeCompositionRequest(
        frames=args.frames,
        every=args.every,
    )


def _build_molecule_lifetime_request(args: argparse.Namespace) -> MoleculeLifetimeRequest:
    return MoleculeLifetimeRequest(
        molecules=args.molecules,
        frames=args.frames,
        every=args.every,
        min_freq=args.min_freq,
    )


REQUEST_BUILDERS: dict[str, Callable[[argparse.Namespace], object]] = {
    "dominant_species": _build_dominant_species_request,
    "largest_molecule_by_mass": _build_largest_molecule_by_mass_request,
    "largest_molecule_composition": _build_largest_molecule_composition_request,
    "molecule_lifetime": _build_molecule_lifetime_request,
}


def build_parser(parser: argparse.ArgumentParser, *, command: str) -> argparse.ArgumentParser:
    canonical = resolve_command_name(command, task_names=MOLECULAR_ANALYSIS_COMMANDS)
    parser.set_defaults(command=canonical)
    parser.formatter_class = argparse.RawTextHelpFormatter

    _add_runtime_arguments(parser)
    _add_presentation_arguments(parser)
    _add_common_arguments(parser)

    if canonical == "dominant_species":
        parser.description = (
            "Return the dominant molecular species for selected frames.\n\n"
            "Examples:\n"
            "  reaxkit dominant_species --top-n 3 --export dominant_species.csv\n"
            "  reaxkit dominant_species --frames 0 10 20 --min-freq 2 --plot single\n"
            "  reaxkit dominant_species --every 5 --xaxis iter --save dominant_species.png"
        )
        parser.add_argument("--top-n", type=int, default=1, help="Number of ranked species per frame")
        parser.add_argument("--min-freq", type=float, default=0.0, help="Minimum species frequency to include")
    elif canonical == "largest_molecule_by_mass":
        parser.description = (
            "Return the heaviest molecular species for selected frames.\n\n"
            "Examples:\n"
            "  reaxkit largest_molecule_by_mass --export largest_mass.csv\n"
            "  reaxkit largest_molecule_by_mass --frames 0 20 40 --plot single\n"
            "  reaxkit largest_molecule_by_mass --every 10 --xaxis iter --save largest_mass.png"
        )
    elif canonical == "largest_molecule_composition":
        parser.description = (
            "Return element counts for the heaviest molecular species per frame.\n\n"
            "Examples:\n"
            "  reaxkit largest_molecule_composition --export composition.csv\n"
            "  reaxkit largest_molecule_composition --frames 0 10 20 --plot subplot\n"
            "  reaxkit largest_molecule_composition --every 5 --xaxis iter --save composition.png"
        )
    elif canonical == "molecule_lifetime":
        parser.description = (
            "Compute molecule lifetimes and birth/death events.\n\n"
            "Examples:\n"
            "  reaxkit molecule_lifetime --molecules H2O OH --export lifetimes.csv\n"
            "  reaxkit molecule_lifetime --table events --plot single\n"
            "  reaxkit molecule_lifetime --min-freq 2 --frames 0 50 100 --save molecule_events.png"
        )
        parser.add_argument("--molecules", nargs="*", default=None, help="Restrict to selected molecular formulae")
        parser.add_argument("--min-freq", type=float, default=1.0, help="Minimum frequency for an active molecule")
        parser.add_argument("--table", choices=["lifetimes", "events"], default="lifetimes", help="Which result table to present")
    else:
        raise KeyError(f"Unsupported molecular analysis command '{canonical}'.")

    return parser


def _selected_table(command: str, result, args: argparse.Namespace) -> pd.DataFrame:
    if command != "molecule_lifetime":
        return result.table
    table_name = getattr(args, "table", "lifetimes")
    return result.events if table_name == "events" else result.lifetimes


def _plot_payload(command: str, result, args: argparse.Namespace) -> dict[str, object] | None:
    table = _selected_table(command, result, args)
    if not isinstance(table, pd.DataFrame) or table.empty:
        return None

    x_col = "iter" if getattr(args, "xaxis", "frame") == "iter" and "iter" in table.columns else "frame_index"
    xlabel = "Iteration" if x_col == "iter" else "Frame Index"

    if command == "dominant_species":
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

    if command == "largest_molecule_by_mass":
        return {
            "plot_type": "single_plot",
            "x": table[x_col].tolist(),
            "y": pd.to_numeric(table["molecular_mass"], errors="coerce").tolist(),
            "xlabel": xlabel,
            "ylabel": "Molecular Mass",
            "title": "Largest Molecule By Mass",
        }

    if command == "largest_molecule_composition":
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

    if command == "molecule_lifetime":
        if getattr(args, "table", "lifetimes") == "events":
            series = []
            for event, group in table.groupby("event", sort=True):
                series.append(
                    {
                        "x": group[x_col].tolist(),
                        "y": pd.to_numeric(group["freq"], errors="coerce").tolist(),
                        "label": str(event),
                    }
                )
            return {
                "plot_type": "single_plot",
                "series": series,
                "xlabel": xlabel,
                "ylabel": "Frequency",
                "title": "Molecule Lifetime Events",
                "legend": True,
                "plot_type_style": "scatter",
            }
        return {
            "plot_type": "single_plot",
            "x": table[x_col.replace("frame_index", "start_frame_index").replace("iter", "start_iter")].tolist(),
            "y": pd.to_numeric(table["n_samples"], errors="coerce").tolist(),
            "xlabel": "Start Iteration" if x_col == "iter" else "Start Frame Index",
            "ylabel": "Samples In Lifetime",
            "title": "Molecule Lifetimes",
        }

    return None


def run_main(command: str, args: argparse.Namespace) -> int:
    canonical = resolve_command_name(command, task_names=MOLECULAR_ANALYSIS_COMMANDS)
    task_cls = TASK_REGISTRY[canonical]
    request = REQUEST_BUILDERS[canonical](args)

    executor = AnalysisExecutor()
    result = executor.run(task_cls(), request, vars(args))
    table = _selected_table(canonical, result, args)
    result_for_presentation = SimpleNamespace(table=table)
    present_result(canonical, result_for_presentation, args, plot_payload_builder=lambda cmd, _res, a: _plot_payload(cmd, result, a))
    return 0
