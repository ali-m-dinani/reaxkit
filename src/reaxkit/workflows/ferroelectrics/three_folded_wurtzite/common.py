"""Shared CLI argument and storage helpers for three-folded wurtzite workflows."""

from __future__ import annotations

import argparse

from reaxkit.core.storage.storage_layout import add_storage_cli_arguments
from reaxkit.workflows.ferroelectrics.four_folded_wurtzite.common import (
    artifact_directory,
    parse_formal_charges,
    runtime_arguments,
    structural_request_kwargs,
    workspace_artifact_directory,
)


def add_input_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--engine", choices=["reaxff", "ams", "lammps"], default=None,
        help="Select the input adapter. Example: --engine reaxff, bypasses engine auto-detection.",
    )
    parser.add_argument(
        "--input", default=".",
        help="Set the path used for engine detection. Example: --input ./run, inspects ./run.",
    )
    parser.add_argument(
        "--run-dir", default=".",
        help="Set the fallback simulation directory. Example: --run-dir ./run, resolves default files there.",
    )
    parser.add_argument(
        "--fort7", default="fort.7",
        help="Select the ReaxFF charge file. Example: --fort7 ./run/fort.7, streams charges from that file.",
    )
    parser.add_argument(
        "--fort78", default="fort.78",
        help="Select the applied-field file. Example: --fort78 ./run/fort.78, reads field samples from that file.",
    )
    parser.add_argument(
        "--xmolout", default="xmolout",
        help="Select the coordinate trajectory. Example: --xmolout ./run/xmolout, reads atom names and positions there.",
    )
    parser.add_argument(
        "--summary", default=None,
        help="Select optional simulation metadata. Example: --summary ./run/summary.txt, reads timing and cell metadata there.",
    )


def restrict_native_charge_input(parser: argparse.ArgumentParser) -> None:
    """Limit projected-polarity input to supported engines and charge modes."""
    for action in parser._actions:
        if action.dest == "engine":
            action.choices = ("ams", "reaxff")
            action.help = (
                "Select AMS/KF or ReaxFF text input. If omitted, ReaxKit detects "
                "the engine from the input path."
            )
        elif action.dest == "charge_source":
            action.choices = ("auto", "formal")
            action.help = (
                "Use charges from the selected/detected engine, or use the values "
                "provided by --formal-charge. Default: auto."
            )


def add_structure_arguments(parser: argparse.ArgumentParser, *, include_polarity: bool = False) -> None:
    parser.add_argument(
        "--center", "--cation", dest="centers", nargs="+", default=["Al"],
        help="Choose center species. Example: --center Al Ga, analyzes Al- and Ga-centered sites.",
    )
    parser.add_argument(
        "--neighbor", "--anion", dest="neighbors", nargs="+", default=["N"],
        help="Choose neighbor species. Example: --neighbor N, searches nitrogen atoms for basal and apical roles.",
    )
    parser.add_argument(
        "--proton", dest="protons", nargs="+", default=["H"],
        help="Choose proton-like species. Example: --proton H, groups sites by nearby hydrogen atoms.",
    )
    parser.add_argument(
        "--charge-source", choices=["auto", "reaxff", "formal"], default="auto",
        help=(
            "Choose dynamic or formal charges. Example: --charge-source formal --formal-charge "
            "Al=3 N=-3, calculates charge-weighted values with those charges."
        ),
    )
    parser.add_argument(
        "--formal-charge", action="append", nargs="+", default=[], metavar="ELEMENT=CHARGE",
        help="Assign species formal charges. Example: --formal-charge Al=3 N=-3 H=1, uses charges in elementary-charge units.",
    )
    parser.add_argument(
        "--neighbor-cutoff", type=float, default=3.0,
        help="Set the candidate radius in angstrom. Example: --neighbor-cutoff 2.5, excludes neighbors farther than 2.5 angstrom.",
    )
    parser.add_argument(
        "--proton-cutoff", type=float, default=2.0,
        help="Set the proton-proximity radius in angstrom. Example: --proton-cutoff 1.8, marks sites within 1.8 angstrom of H.",
    )
    parser.add_argument(
        "--c-axis", nargs=3, type=float, default=(0.0, 0.0, 1.0),
        help="Set the Cartesian polar-axis direction. Example: --c-axis 0 0 1, interprets positive z as UP.",
    )
    parser.add_argument(
        "--periodic", choices=["none", "x", "y", "z", "xy", "xz", "yz", "xyz"],
        default="xyz",
        help="Choose periodic directions. Example: --periodic xy, wraps candidates across a and b while leaving the surface normal open.",
    )
    parser.add_argument(
        "--cell-lengths", nargs=3, type=float, default=None,
        help="Override cell lengths in angstrom. Example: --cell-lengths 10 10 16, uses that cell for every frame.",
    )
    parser.add_argument(
        "--cell-angles", nargs=3, type=float, default=(90.0, 90.0, 90.0),
        help="Set angles for --cell-lengths in degrees. Example: --cell-angles 90 90 120, defines a hexagonal cell.",
    )
    parser.add_argument(
        "--frames", nargs="*", default=None,
        help="Select zero-based source frames. Example: --frames 0:101:10, includes frames 0 through 100 every 10 frames.",
    )
    parser.add_argument(
        "--every", type=int, default=1,
        help="Stride the selected frames. Example: --every 5, keeps every fifth selected frame.",
    )
    if include_polarity:
        parser.add_argument(
            "--polarity-tolerance", type=float, default=1.0e-10,
            help="Set the zero-polarity tolerance in angstrom. Example: --polarity-tolerance 1e-6, treats smaller basal means as zero.",
        )
    parser.add_argument(
        "--log", choices=["verbose", "quiet"], default="quiet",
        help="Choose console logging detail. Example: --log verbose, prints diagnostic progress information.",
    )
    add_storage_cli_arguments(parser)


__all__ = [
    "add_input_arguments", "add_structure_arguments", "artifact_directory",
    "parse_formal_charges", "runtime_arguments", "structural_request_kwargs",
    "restrict_native_charge_input",
    "workspace_artifact_directory",
]
