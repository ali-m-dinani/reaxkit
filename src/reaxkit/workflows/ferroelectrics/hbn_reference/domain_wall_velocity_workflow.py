"""CSV-only workflows for fraction growth, crossing times, and local wall tracks."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

from reaxkit.analysis.ferroelectrics.hbn_reference.domain_wall_velocity import (
    VelocityRequest, calculate_domain_wall_velocity,
)
from reaxkit.core.storage.storage_layout import default_project_root, generate_run_id
from reaxkit.presentation.workflow_artifacts import write_workflow_csv


METHODS = {
    "get-hbn-domain-wall-fraction-velocity": "fraction-growth",
    "get-hbn-domain-wall-crossing-velocity": "crossing",
    "get-hbn-domain-wall-tracking-velocity": "tracking",
}
ALL_COMMANDS = tuple(METHODS)


def build_parser(parser: argparse.ArgumentParser, *, command: str):
    if command not in METHODS:
        raise KeyError(command)
    if METHODS[command] == "fraction-growth":
        from .fraction_growth_workflow import build_parser as build_fraction_parser

        return build_fraction_parser(parser, command=command)
    parser.set_defaults(command=command)
    parser.description = (
        "Estimate domain-wall velocities from an existing hBN projected-polarity result folder. "
        "Reads CSV files only; never runs polarity or reads a trajectory. "
        "Positions and velocities use reference coordinates. Inspect diagnostics and spatial "
        "maps before interpreting fitted slopes as a coherent wall velocity."
    )
    parser.add_argument("--input", type=Path, required=True, help="Folder containing projected-polarity CSV results.")
    parser.add_argument("--axis", choices=list("xyz"), default="z", help="Spatial propagation direction, independent of dipole component.")
    parser.add_argument("--time-unit", choices=["fs", "ps", "ns", "s"], default="ps", help="Units of CSV time or --time-per-iteration; used for conversion to m/s.")
    parser.add_argument("--time-per-iteration", type=float, help="Physical time per iteration when CSV/summary time is absent.")
    parser.add_argument("--field-start-time", type=float, default=0.0, help="Subtract this physical onset time and exclude earlier samples.")
    parser.add_argument("--time-min", type=float, help="Fit-window start relative to field onset, in --time-unit.")
    parser.add_argument("--time-max", type=float, help="Fit-window end relative to field onset, in --time-unit.")
    parser.add_argument("--initial-sign", type=int, choices=[-1, 1], default=1, help="Initial polarity for fraction growth or crossing direction; default +1.")
    parser.add_argument("--output-dir", type=Path, help="Explicit destination; defaults to a unique workspace other/<command>/run_* folder.")
    if METHODS[command] != "tracking":
        parser.add_argument("--aggregate-transverse", action="store_true", help="Sum counts over the transverse bins instead of retaining separate slices.")
    if METHODS[command] == "crossing":
        parser.add_argument("--persistence", type=int, default=2, help="Consecutive saved samples on the target side, including the crossing sample.")
    else:
        parser.add_argument("--max-step", type=float, required=True, help="Maximum wall displacement in angstrom between consecutive saved frames; ambiguous matches start new tracks.")
        parser.add_argument("--periodic-length", type=float, help="Periodic length along --axis in angstrom; enables seam boundaries and track unwrapping.")
        parser.add_argument("--coordinate-decimals", type=int, default=4, help="Round transverse reference coordinates to this precision to identify local lines.")
    return parser


def run_main(command: str, args: argparse.Namespace) -> int:
    if command not in METHODS:
        raise KeyError(command)
    if METHODS[command] == "fraction-growth":
        from .fraction_growth_workflow import run_main as run_fraction

        return run_fraction(command, args)
    options = {name: getattr(args, name) for name in VelocityRequest.__dataclass_fields__ if hasattr(args, name)}
    request = VelocityRequest(method=METHODS[command], **options)
    result = calculate_domain_wall_velocity(args.input, request)
    destination = args.output_dir or default_project_root() / "other" / command / generate_run_id()
    destination = Path(destination).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    for name, table in (("samples", result.samples), ("velocities", result.velocities), ("diagnostics", result.diagnostics)):
        write_workflow_csv(table, destination / f"{name}.csv", index=False)
    if request.method != "tracking":
        write_workflow_csv(result.intervals, destination / "velocity_intervals.csv", index=False)
    settings = {"input": str(Path(args.input).resolve()), "command": command,
                "coordinate_system": "reference", **asdict(request)}
    (destination / "settings.json").write_text(json.dumps(settings, indent=2), encoding="utf-8")
    print(f"Wrote velocity samples, fits, diagnostics, and settings to {destination}")
    if result.velocities.empty or not result.velocities["status"].eq("ok").any():
        print("No valid velocity fit: inspect diagnostics and sampling/window choices.")
    return 0


__all__ = ["ALL_COMMANDS", "build_parser", "run_main"]
