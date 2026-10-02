"""CLI for vector polarization relative to a hexagonal AlN reference."""

from __future__ import annotations

from reaxkit.presentation.workflow_artifacts import write_workflow_tables

import argparse
from pathlib import Path
from typing import cast

from reaxkit.analysis.ferroelectrics.hbn_reference.polarization import (
    ChargeSource,
    DEFAULT_FORMAL_CHARGES,
    HBNReferencePolarizationRequest,
    OrthogonalizeMode,
    REFERENCE_STRUCTURE_PATH,
    VolumeMethod,
    write_aligned_reference_xyz,
)
from reaxkit.core.registry.analysis_task_registry import TASK_REGISTRY
from reaxkit.core.resolve.command_alias_resolver import resolve_command_name
from reaxkit.core.runtime.analysis_executor import AnalysisExecutor
from reaxkit.core.storage.storage_layout import add_storage_cli_arguments
from reaxkit.core.utils.frame_utils import parse_frame_indices
from reaxkit.presentation.dispatcher import present_result
from reaxkit.workflows.ferroelectrics.three_folded_wurtzite.common import (
    add_input_arguments,
    artifact_directory,
    runtime_arguments,
)

COMMAND = "get-hbn-reference-polarization"
ALL_COMMANDS = (COMMAND,)
ALL_LEGACY_COMMANDS = (
    "get_hbn_reference_polarization",
    "hbn-reference-polarization",
)
TASK_KEY_BY_COMMAND = {COMMAND: COMMAND}


def _canonical_command(command: str) -> str:
    return resolve_command_name(command, ALL_COMMANDS, aliases={COMMAND: ALL_LEGACY_COMMANDS})


def _default_reference_path() -> Path:
    return REFERENCE_STRUCTURE_PATH


def _parse_assignments(values, *, option: str, value_type):
    result = {}
    flattened = [
        item
        for group in values
        for item in (group if isinstance(group, list) else [group])
    ]
    for item in flattened:
        if "=" not in str(item):
            raise ValueError(f"{option} must use ELEMENT=VALUE syntax: {item!r}.")
        label, raw = str(item).split("=", 1)
        if not label.strip():
            raise ValueError(f"{option} element labels cannot be empty.")
        result[label.strip()] = value_type(raw.strip())
    return result


def build_parser(parser: argparse.ArgumentParser, *, command: str) -> argparse.ArgumentParser:
    canonical = _canonical_command(command)
    parser.set_defaults(command=canonical, progress=True)
    parser.formatter_class = argparse.RawTextHelpFormatter
    parser.description = """Calculate polarization relative to replicated nonpolar hexagonal AlN.

The command optionally applies ReaxKit's hexagonal-to-orthogonal transform to
AlN_hbn.cif, repeats it by the explicitly supplied replication counts, maps it
into each instantaneous cell, preserves large vacuum gaps, and removes a
periodic rigid translation. It then matches atoms one-to-one and evaluates
mu = -q Delta u and P = e/Omega sum(mu). The output reports x, y, z, and
the selected c-axis projection. Omega is selected with --volume-method; hull
is the default and excludes slab vacuum.

Examples:
  reaxkit get-hbn-reference-polarization --orthogonalize-reference --replication 19 19 10

  reaxkit get-hbn-reference-polarization --replication 4 4 3 --charge-source formal --formal-charge Al=3 N=-3 B=3 --reference-species B=Al
"""
    add_input_arguments(parser)
    add_storage_cli_arguments(parser)
    parser.add_argument(
        "--reference", type=Path, default=_default_reference_path(),
        help="Select the nonpolar reference CIF. Default: bundled AlN_hbn.cif. Example: --reference AlN_hbn.cif, which reads that nonpolar reference structure.",
    )
    parser.add_argument(
        "--replication", nargs=3, type=int, required=True, metavar=("NX", "NY", "NZ"),
        help=(
            "Repeat the oriented reference explicitly. Example: --replication "
            "19 19 10 repeats an orthogonalized AlN_hbn cell by those counts."
        ),
    )
    parser.add_argument(
        "--charge-source", choices=["auto", "reaxff", "formal"], default="auto",
        help=(
            "Choose ReaxFF or formal charges. Example: --charge-source formal "
            "--formal-charge Al=3 N=-3, uses the configured species charges; auto "
            "uses fort.7 when available."
        ),
    )
    defaults = [f"{key}={value}" for key, value in DEFAULT_FORMAL_CHARGES.items()]
    parser.add_argument(
        "--formal-charge", action="append", nargs="+", default=[defaults], metavar="ELEMENT=CHARGE",
        help=(
            "Assign species formal charges in e. Defaults: Al=3 N=-3; provide values for additional trajectory species. Example: --formal-charge Al=3 N=-3, which assigns aluminum and nitrogen their specified charges in e."
        ),
    )
    parser.add_argument(
        "--reference-species", action="append", nargs="+", default=[["B=Al"]],
        metavar="TRAJECTORY=REFERENCE",
        help="Map substitutions onto reference sites. Default: B=Al. Example: --reference-species B=Al, which maps boron substitutions onto aluminum reference sites.",
    )
    parser.add_argument(
        "--c-axis", nargs=3, type=float, default=(0.0, 0.0, 1.0),
        help="Set the Cartesian longitudinal direction. Default: 0 0 1. Example: --c-axis 0 0 1, which uses the Cartesian z direction as the longitudinal axis.",
    )
    parser.add_argument(
        "--periodic", choices=["none", "x", "y", "z", "xy", "xz", "yz", "xyz"],
        default="xyz", help="Choose periodic directions used for matching and displacement. Example: --periodic xy, which applies periodic matching in x and y only.",
    )
    parser.add_argument(
        "--cell-lengths", nargs=3, type=float, default=None,
        help="Override trajectory cell lengths in angstrom for every frame. Example: --cell-lengths 30 30 60, which uses those cell lengths in angstrom for every frame.",
    )
    parser.add_argument(
        "--cell-angles", nargs=3, type=float, default=(90.0, 90.0, 90.0),
        help="Set angles for --cell-lengths in degrees. Example: --cell-angles 90 90 90, which uses an orthogonal cell with the supplied lengths.",
    )
    parser.add_argument(
        "--frames", nargs="*", default=None,
        help="Select zero-based source frames, for example --frames 0:101:10. Example: --frames 0:20:2, which includes source frames 0, 2, ..., 18.",
    )
    parser.add_argument("--every", type=int, default=1, help="Stride selected frames. Example: --every 5, which keeps every fifth selected frame.")
    parser.add_argument(
        "--reference-frame", type=int, default=0,
        help="Choose the frame used to size, assign, and write the reference lattice. Example: --reference-frame 0, which uses the initial frame to establish the reference assignment.",
    )
    parser.add_argument(
        "--orthogonalize-reference",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Apply ReaxKit's hexagonal-to-orthogonal transform before replication. Use --no-orthogonalize-reference to retain the CIF cell. If omitted, the cell angles select the better representation automatically. Example: --orthogonalize-reference, which converts the hexagonal reference cell before replication."
        ),
    )
    parser.add_argument(
        "--orthogonalize", choices=["auto", "always", "never"], default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--angle-tolerance", type=float, default=1.0,
        help="Set the cell-angle comparison tolerance in degrees. Example: --angle-tolerance 1, which allows a one-degree cell-angle mismatch.",
    )
    parser.add_argument(
        "--max-reference-strain", type=float, default=0.15,
        help=(
            "Scale a replicated reference axis to the trajectory box only when the relative length change is at most this value. Larger gaps are treated as vacuum. Default: 0.15. Example: --max-reference-strain 0.15, which allows at most 15 percent reference-axis scaling."
        ),
    )
    parser.add_argument(
        "--volume-method", choices=["hull", "bbox", "cell"], default="hull",
        help=(
            "Choose the polarization volume: occupied atomic convex hull (default), occupied bounding box, or full simulation cell. Example: --volume-method cell, which uses the full simulation-cell volume."
        ),
    )
    parser.add_argument(
        "--max-alignment-candidates", type=int, default=8,
        help="Limit periodic origin candidates considered during initial atom assignment. Example: --max-alignment-candidates 8, which tests at most eight periodic-origin candidates.",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="Choose the output directory for CSV and XYZ artifacts. Example: --output-dir analysis/charges, which writes generated artifacts beneath that directory.",
    )
    parser.add_argument(
        "--write-displacements",
        action="store_true",
        help=(
            "Write the per-atom displacement table. Disabled by default because the table can be very large. Example: --write-displacements, which writes the optional per-atom displacement table."
        ),
    )
    return parser


def build_request(args: argparse.Namespace) -> HBNReferencePolarizationRequest:
    formal_charges = _parse_assignments(
        args.formal_charge, option="--formal-charge", value_type=float
    )
    species = _parse_assignments(
        args.reference_species, option="--reference-species", value_type=str
    )
    explicit_orthogonalization = args.orthogonalize_reference
    if explicit_orthogonalization is None:
        orthogonalize = str(args.orthogonalize or "auto")
    else:
        orthogonalize = "always" if explicit_orthogonalization else "never"
    return HBNReferencePolarizationRequest(
        reference_path=Path(args.reference),
        replication=tuple(int(value) for value in args.replication),
        charge_source=cast(ChargeSource, str(args.charge_source)),
        formal_charges=formal_charges,
        reference_species=species,
        c_axis=tuple(float(value) for value in args.c_axis),
        periodic=tuple(axis in str(args.periodic).lower() for axis in "xyz"),
        cell_lengths=None if args.cell_lengths is None else tuple(args.cell_lengths),
        cell_angles=tuple(args.cell_angles),
        frames=parse_frame_indices(args.frames),
        every=int(args.every),
        reference_frame=int(args.reference_frame),
        orthogonalize=cast(OrthogonalizeMode, orthogonalize),
        angle_tolerance_degrees=float(args.angle_tolerance),
        max_reference_strain=float(args.max_reference_strain),
        max_alignment_candidates=int(args.max_alignment_candidates),
        include_displacements=bool(args.write_displacements),
        volume_method=cast(VolumeMethod, args.volume_method),
    )


REQUEST_BUILDERS = {COMMAND: build_request}


def run_main(command: str, args: argparse.Namespace) -> int:
    profile = getattr(args, "output_profile", None)
    if profile is None:
        args.output_profile = "legacy"
    if profile is not None:
        args.write_displacements = profile != "minimal" and (profile in {"full", "legacy"} or args.write_displacements)
    canonical = _canonical_command(command)
    result = AnalysisExecutor().run(
        TASK_REGISTRY[TASK_KEY_BY_COMMAND[canonical]](),
        REQUEST_BUILDERS[canonical](args),
        runtime_arguments(args),
    )
    output = artifact_directory(args, canonical)
    output.mkdir(parents=True, exist_ok=True)
    polarization_path = output / "hbn_reference_polarization.csv"
    detail_format = getattr(args, "detail_format", None) or ("csv" if args.output_profile == "legacy" else "parquet")
    displacement_path = output / f"hbn_reference_displacements.{detail_format}"
    mapping_path = output / "hbn_reference_mapping.csv"
    reference_path = output / "AlN_hbn_replicated_aligned.xyz"
    tables = {polarization_path: result.table, mapping_path: result.mapping}
    tables[displacement_path] = getattr(result, "table_chunks", {}).get("hbn_reference_displacements", result.displacements)
    write_workflow_tables(tables, args=args, details=(displacement_path.name,),
                          enabled_details=(displacement_path.name,) if args.write_displacements else ())
    write_aligned_reference_xyz(result, reference_path)
    args.suppress_table = True
    present_result(canonical, result, args)
    print(f"Wrote h-BN-reference polarization to {polarization_path}")
    if args.write_displacements:
        print(f"Wrote per-atom displacements to {displacement_path}")
    print(f"Wrote atom mapping to {mapping_path}")
    print(f"Wrote replicated aligned reference to {reference_path}")
    return 0


__all__ = [
    "ALL_COMMANDS", "ALL_LEGACY_COMMANDS", "COMMAND", "REQUEST_BUILDERS",
    "TASK_KEY_BY_COMMAND", "build_parser", "build_request", "run_main",
]
