"""Shared CLI helpers for four-fold wurtzite workflows."""

from __future__ import annotations

import argparse
from pathlib import Path

from reaxkit.core.storage.storage_layout import ReaxkitStorageLayout, add_storage_cli_arguments
from reaxkit.core.utils.frame_utils import parse_frame_indices


def add_input_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--engine",
        choices=["reaxff", "ams", "lammps"],
        default=None,
        help="Select the input adapter. Example: --engine reaxff, bypasses engine auto-detection.",
    )
    parser.add_argument(
        "--input",
        default=".",
        help="Set the path used for engine detection. Example: --input ./run, inspects ./run.",
    )
    parser.add_argument(
        "--run-dir",
        default=".",
        help="Set the fallback simulation directory. Example: --run-dir ./run, resolves default files there.",
    )
    parser.add_argument(
        "--fort7",
        default="fort.7",
        help="Select the ReaxFF charge file. Example: --fort7 ./run/fort.7, streams charges from that file.",
    )
    parser.add_argument(
        "--fort78",
        default="fort.78",
        help="Select the applied-field file. Example: --fort78 ./run/fort.78, reads field samples from that file.",
    )
    parser.add_argument(
        "--xmolout",
        default="xmolout",
        help="Select the coordinate and atom-name trajectory. Example: --xmolout ./run/xmolout, reads that trajectory.",
    )
    parser.add_argument(
        "--summary",
        default=None,
        help="Select optional simulation metadata. Example: --summary ./run/summary.txt, reads metadata from that file.",
    )


def add_structure_arguments(parser: argparse.ArgumentParser, *, include_polarity: bool = False) -> None:
    parser.add_argument(
        "--center", "--cation", dest="centers", nargs="+", default=["Al"],
        help="Choose center species. Example: --center Zn Mg, analyzes both Zn- and Mg-centered sites.",
    )
    parser.add_argument(
        "--neighbor", "--anion", dest="neighbors", nargs="+", default=["N"],
        help="Choose neighbor species. Example: --neighbor O, assigns the four nearest oxygen atoms.",
    )
    parser.add_argument(
        "--proton", dest="protons", nargs="+", default=["H"],
        help="Choose proton-like species for proximity grouping. Example: --proton H, tests nearby hydrogen atoms.",
    )
    parser.add_argument(
        "--charge-source", choices=["auto", "reaxff", "formal"], default="auto",
        help=(
            "Choose dynamic or formal charges. Formal mode has no implicit species fallback and "
            "requires --formal-charge for every center and neighbor. Example: --charge-source "
            "formal --formal-charge Al=3 N=-3 H=1."
        ),
    )
    parser.add_argument(
        "--formal-charge", action="append", nargs="+", default=[], metavar="ELEMENT=CHARGE",
        help=(
            "Assign species-specific formal charges. Example: --formal-charge Al=3 N=-3 H=1, "
            "uses +3, -3, and +1 elementary charges."
        ),
    )
    parser.add_argument(
        "--neighbor-cutoff", type=float, default=3.0,
        help=(
            "Set the maximum center-neighbor distance in angstrom. Example: "
            "--neighbor-cutoff 2.5, excludes candidates farther than 2.5 angstrom."
        ),
    )
    parser.add_argument(
        "--proton-cutoff",
        type=float,
        default=2.0,
        help="Set the proton-proximity radius in angstrom. Example: --proton-cutoff 1.8, marks centers within 1.8 angstrom of H.",
    )
    parser.add_argument(
        "--c-axis",
        nargs=3,
        type=float,
        default=(0.0, 0.0, 1.0),
        help="Set the Cartesian polar-axis direction. Example: --c-axis 0 0 1, projects bonds along z.",
    )
    parser.add_argument(
        "--periodic", choices=["none", "x", "y", "z", "xy", "xz", "yz", "xyz"],
        default="xyz",
        help="Choose periodic lattice directions. Example: --periodic xy, wraps neighbors across a and b only.",
    )
    parser.add_argument(
        "--cell-lengths",
        nargs=3,
        type=float,
        default=None,
        help="Override cell lengths in angstrom. Example: --cell-lengths 10 10 16, uses that cell for every frame.",
    )
    parser.add_argument(
        "--cell-angles",
        nargs=3,
        type=float,
        default=(90.0, 90.0, 90.0),
        help="Set angles for --cell-lengths in degrees. Example: --cell-angles 90 90 120, defines a hexagonal cell.",
    )
    parser.add_argument(
        "--frames",
        nargs="*",
        default=None,
        help="Select zero-based source frames. Example: --frames 0:101:10, includes frames 0 through 100 every 10 frames.",
    )
    parser.add_argument(
        "--every",
        type=int,
        default=1,
        help="Stride the selected frames. Example: --every 5, keeps every fifth selected frame.",
    )
    if include_polarity:
        parser.add_argument(
            "--polarity-tolerance",
            type=float,
            default=1.0e-10,
            help="Set the zero-polarity tolerance in angstrom. Example: --polarity-tolerance 1e-6, treats smaller delta values as zero.",
        )
    parser.add_argument(
        "--log",
        choices=["verbose", "quiet"],
        default="quiet",
        help="Choose console logging detail. Example: --log verbose, prints diagnostic progress information.",
    )
    add_storage_cli_arguments(parser)


def parse_formal_charges(values) -> dict[str, float]:
    result: dict[str, float] = {}
    flattened = [item for group in values for item in (group if isinstance(group, list) else [group])]
    for item in flattened:
        if "=" not in item:
            raise ValueError(f"Formal charge must use ELEMENT=CHARGE syntax: {item!r}.")
        label, raw_charge = item.split("=", 1)
        label = label.strip()
        if not label:
            raise ValueError("Formal charge element labels cannot be empty.")
        result[label] = float(raw_charge)
    return result


def structural_request_kwargs(args: argparse.Namespace) -> dict:
    formal_charges = parse_formal_charges(list(args.formal_charge))
    if str(args.charge_source) == "formal":
        configured = {label.casefold() for label in formal_charges}
        required = {str(value).casefold(): str(value) for value in (*args.centers, *args.neighbors)}
        missing = [label for key, label in required.items() if key not in configured]
        if missing:
            raise ValueError(
                "--charge-source formal requires explicit --formal-charge values for "
                f"every center and neighbor species; missing: {', '.join(sorted(missing, key=str.casefold))}."
            )
    return {
        "centers": tuple(args.centers),
        "neighbors": tuple(args.neighbors),
        "protons": tuple(args.protons),
        "charge_source": str(args.charge_source),
        "formal_charges": formal_charges,
        "neighbor_cutoff": None if float(args.neighbor_cutoff) == 0.0 else float(args.neighbor_cutoff),
        "proton_cutoff": float(args.proton_cutoff),
        "c_axis": tuple(float(value) for value in args.c_axis),
        "periodic": tuple(axis in str(args.periodic).lower() for axis in "xyz"),
        "cell_lengths": None if args.cell_lengths is None else tuple(args.cell_lengths),
        "cell_angles": tuple(args.cell_angles),
        "frames": parse_frame_indices(args.frames),
        "every": int(args.every),
    }


def artifact_directory(args: argparse.Namespace, command: str) -> Path:
    if getattr(args, "output_dir", None) is not None:
        return Path(args.output_dir).resolve()
    return workspace_artifact_directory(args, command)


def workspace_artifact_directory(args: argparse.Namespace, command: str) -> Path:
    """Return the canonical workspace directory, ignoring optional export destinations."""

    analysis_id = args.analysis_id or args.run_id or getattr(args, "_analysis_id", None) or "analysis"
    return ReaxkitStorageLayout(project_root=Path(args.project_root)).analysis_root / command / str(analysis_id)


def runtime_arguments(args: argparse.Namespace) -> dict:
    runtime = vars(args).copy()
    runtime["scope"] = "total"
    runtime["cache"] = False
    # ReaxFF routes this through engine.reaxff.quick_io.charges, avoiding connectivity tables.
    runtime["_quick_charge_only"] = True
    return runtime


__all__ = [
    "add_input_arguments",
    "add_structure_arguments",
    "artifact_directory",
    "parse_formal_charges",
    "runtime_arguments",
    "structural_request_kwargs",
    "workspace_artifact_directory",
]
