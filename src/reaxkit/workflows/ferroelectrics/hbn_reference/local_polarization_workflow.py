"""CLI for local h-BN-reference dipole and polarization."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import cast

import numpy as np

from reaxkit.analysis.ferroelectrics.hbn_reference.local_polarization import (
    HBNReferenceLocalPolarizationRequest,
    HBNReferenceLocalPolarizationResult,
    LocalChargeTreatment,
    LocalGrouping,
    LocalVolumeMethod,
    _table_for_trajectory_frame,
    write_local_polarization_extxyz,
)
from reaxkit.core.platform.constants import const
from reaxkit.core.registry.analysis_task_registry import TASK_REGISTRY
from reaxkit.core.resolve.command_alias_resolver import resolve_command_name
from reaxkit.core.runtime.analysis_executor import AnalysisExecutor
from reaxkit.presentation.dispatcher import present_result
from reaxkit.workflows.ferroelectrics.hbn_reference import (
    polarization_workflow as global_workflow,
)
from reaxkit.workflows.ferroelectrics.three_folded_wurtzite.common import (
    artifact_directory,
    runtime_arguments,
)

COMMAND = "get-hbn-reference-local-polarization"
ALL_COMMANDS = (COMMAND,)
ALL_LEGACY_COMMANDS = (
    "get_hbn_reference_local_polarization",
    "hbn-reference-local-polarization",
)
TASK_KEY_BY_COMMAND = {COMMAND: COMMAND}


def _canonical_command(command: str) -> str:
    return resolve_command_name(command, ALL_COMMANDS, aliases={COMMAND: ALL_LEGACY_COMMANDS})


def build_parser(parser: argparse.ArgumentParser, *, command: str) -> argparse.ArgumentParser:
    canonical = _canonical_command(command)
    global_workflow.build_parser(parser, command=global_workflow.COMMAND)
    parser.set_defaults(command=canonical, progress=True)
    parser.description = """Calculate cell- and layer-resolved local dipole and polarization.

Each crystallographic cell is one Al2N2 reference cell. Each reference layer is
one coplanar AlN pair, so both spatial resolutions are written on every run.
--local-grouping selects the table used by plots and the generic local output.
Equal volume shares the selected frame volume among groups; deformation uses
normalized local affine-deformation weights. With the default hull volume,
vacuum is excluded before either assignment.

Examples:
  reaxkit get-hbn-reference-local-polarization --replication 19 19 10 --periodic xy --charge-source reaxff --volume-method hull --write-extxyz

  reaxkit get-hbn-reference-local-polarization --replication 19 19 10 --charge-source formal --formal-charge Al=3 N=-3 --local-volume-method deformation --plot-2d --plot-plane xz --plot-component c
"""
    parser.add_argument(
        "--local-grouping",
        choices=["cell", "layer"],
        default="cell",
        help=(
            "Use four-atom Al2N2 cells or two-atom AlN layers for plots and "
            "the generic local table. Both explicit tables are always written. Default: cell."
        ),
    )
    parser.add_argument(
        "--local-volume-method",
        choices=["equal", "deformation"],
        default="equal",
        help=(
            "Choose equal shares of the selected frame volume (default), or "
            "normalized local-deformation weights."
        ),
    )
    parser.add_argument(
        "--deformation-neighbors",
        type=int,
        default=12,
        help="Set the maximum neighboring reference cells used by local affine fits. Default: 12.",
    )
    parser.add_argument(
        "--local-charge-treatment",
        choices=["auto", "raw", "neutralize"],
        default="auto",
        help=(
            "Choose raw atomic charges, per-cell charge neutralization, or auto "
            "neutralization only when a cell is charged. Default: auto."
        ),
    )
    parser.add_argument(
        "--plot-2d",
        action="store_true",
        help="Write per-frame 2D maps after aggregating along the omitted coordinate.",
    )
    parser.add_argument(
        "--plot-3d",
        action="store_true",
        help="Write per-frame 3D cell-center scatter plots.",
    )
    parser.add_argument(
        "--plot-plane",
        choices=["xy", "xz", "yz"],
        default="xy",
        help="Choose the displayed plane for --plot-2d. Default: xy.",
    )
    parser.add_argument(
        "--plot-component",
        choices=["x", "y", "z", "c"],
        default="c",
        help="Choose the dipole or polarization component used as plot color. Default: c.",
    )
    parser.add_argument(
        "--plot-quantity",
        choices=["polarization", "dipole"],
        default="polarization",
        help="Plot local polarization (default) or local dipole.",
    )
    parser.add_argument(
        "--plot-bins",
        nargs=2,
        type=int,
        default=(40, 40),
        metavar=("NU", "NV"),
        help="Set the two in-plane bin counts for --plot-2d. Default: 40 40.",
    )
    parser.add_argument(
        "--global-scaling",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use shared symmetric color limits across frames.",
    )
    parser.add_argument(
        "--figure-dpi", type=int, default=180, help="Set PNG resolution. Default: 180."
    )
    parser.add_argument(
        "--write-extxyz",
        action="store_true",
        help="Write an OVITO-compatible trajectory with local vector properties.",
    )
    parser.add_argument(
        "--extxyz-precision",
        type=int,
        default=8,
        help="Set significant digits in Extended XYZ output. Default: 8.",
    )
    return parser


def build_request(args: argparse.Namespace) -> HBNReferenceLocalPolarizationRequest:
    base = global_workflow.build_request(args)
    base.include_displacements = True
    return HBNReferenceLocalPolarizationRequest(
        **vars(base),
        local_volume_method=cast(LocalVolumeMethod, str(args.local_volume_method)),
        deformation_neighbors=int(args.deformation_neighbors),
        local_charge_treatment=cast(
            LocalChargeTreatment, str(args.local_charge_treatment)
        ),
        local_grouping=cast(LocalGrouping, str(args.local_grouping)),
    )


REQUEST_BUILDERS = {COMMAND: build_request}


def _matplotlib():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.colors import Normalize
    except ImportError as exc:  # pragma: no cover
        raise ImportError("Plot generation requires matplotlib; install reaxkit[plot].") from exc
    return plt, Normalize


def _symmetric_limits(values: np.ndarray) -> tuple[float, float]:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    bound = float(np.max(np.abs(finite))) if finite.size else 1.0
    return (-bound, bound) if bound > 0.0 else (-1.0, 1.0)


def _edges(values: np.ndarray, count: int) -> np.ndarray:
    if int(count) < 1:
        raise ValueError("Each --plot-bins value must be at least 1.")
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return np.linspace(0.0, 1.0, int(count) + 1)
    lower, upper = float(np.min(finite)), float(np.max(finite))
    if lower == upper:
        lower -= 0.5
        upper += 0.5
    return np.linspace(lower, upper, int(count) + 1)


def aggregate_local_values_2d(
        result: HBNReferenceLocalPolarizationResult,
        frame: int,
        *,
        plane: str,
        component: str,
        quantity: str,
        bins: tuple[int, int],
        edges: tuple[np.ndarray, np.ndarray] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sum cell dipoles and volumes along the coordinate omitted from ``plane``."""

    group = result.table[result.table["frame_index"].astype(int).eq(int(frame))]
    coordinates = group[
        [f"center_{plane[0]} (angstrom)", f"center_{plane[1]} (angstrom)"]
    ].to_numpy(float)
    if edges is None:
        edges = (
            _edges(coordinates[:, 0], bins[0]),
            _edges(coordinates[:, 1], bins[1]),
        )
    dipole_values = group[f"dipole_{component} (e*angstrom)"].to_numpy(float)
    valid = np.isfinite(coordinates).all(axis=1) & np.isfinite(dipole_values)
    dipole, _, _ = np.histogram2d(
        coordinates[valid, 0],
        coordinates[valid, 1],
        bins=edges,
        weights=dipole_values[valid],
    )
    if quantity == "dipole":
        values = dipole
    elif quantity == "polarization":
        volume = group["local_volume (angstrom^3)"].to_numpy(float)
        valid &= np.isfinite(volume) & (volume > 0.0)
        dipole, _, _ = np.histogram2d(
            coordinates[valid, 0],
            coordinates[valid, 1],
            bins=edges,
            weights=dipole_values[valid],
        )
        volumes, _, _ = np.histogram2d(
            coordinates[valid, 0],
            coordinates[valid, 1],
            bins=edges,
            weights=volume[valid],
        )
        values = np.divide(
            dipole * float(const("ea3_to_uC_cm2")),
            volumes,
            out=np.full(dipole.shape, np.nan),
            where=volumes > 0.0,
        )
    else:
        raise ValueError("quantity must be 'polarization' or 'dipole'.")
    return edges[0], edges[1], values.T


def generate_local_2d_plots(
        result: HBNReferenceLocalPolarizationResult,
        output: Path,
        *,
        plane: str,
        component: str,
        quantity: str,
        bins: tuple[int, int],
        global_scaling: bool,
        dpi: int,
) -> list[Path]:
    plt, Normalize = _matplotlib()
    all_coordinates = result.table[
        [f"center_{plane[0]} (angstrom)", f"center_{plane[1]} (angstrom)"]
    ].to_numpy(float)
    shared_edges = (
        _edges(all_coordinates[:, 0], bins[0]),
        _edges(all_coordinates[:, 1], bins[1]),
    )
    maps = {
        source_frame: aggregate_local_values_2d(
            result,
            source_frame,
            plane=plane,
            component=component,
            quantity=quantity,
            bins=bins,
            edges=shared_edges,
        )[2]
        for frame in result.frame_indices
        for source_frame, _ in [_table_for_trajectory_frame(result, int(frame))]
    }
    shared_limits = (
        _symmetric_limits(np.concatenate([values.ravel() for values in maps.values()]))
        if maps
        else (-1.0, 1.0)
    )
    destination = Path(output) / "plots_2d" / f"{quantity}_{component}_{plane}"
    destination.mkdir(parents=True, exist_ok=True)
    units = "uC/cm^2" if quantity == "polarization" else "e*angstrom"
    written: list[Path] = []
    for frame, values in maps.items():
        limits = shared_limits if global_scaling else _symmetric_limits(values)
        figure, axis = plt.subplots(figsize=(7.2, 5.8))
        image = axis.pcolormesh(
            shared_edges[0],
            shared_edges[1],
            values,
            cmap="coolwarm",
            norm=Normalize(*limits),
            shading="flat",
        )
        axis.set_xlabel(f"{plane[0]} (angstrom)")
        axis.set_ylabel(f"{plane[1]} (angstrom)")
        axis.set_aspect("equal", adjustable="box")
        axis.set_title(f"h-BN-reference local {quantity} {component} | frame {frame}")
        figure.colorbar(image, ax=axis).set_label(f"{quantity}_{component} ({units})")
        figure.tight_layout()
        path = destination / f"frame_{frame:06d}.png"
        figure.savefig(path, dpi=int(dpi), bbox_inches="tight")
        plt.close(figure)
        written.append(path)
    return written


def generate_local_3d_plots(
        result: HBNReferenceLocalPolarizationResult,
        output: Path,
        *,
        component: str,
        quantity: str,
        global_scaling: bool,
        dpi: int,
) -> list[Path]:
    plt, Normalize = _matplotlib()
    column = (
        f"P_{component} (uC/cm^2)"
        if quantity == "polarization"
        else f"dipole_{component} (e*angstrom)"
    )
    shared_limits = _symmetric_limits(result.table[column].to_numpy(float))
    destination = Path(output) / "plots_3d" / f"{quantity}_{component}"
    destination.mkdir(parents=True, exist_ok=True)
    units = "uC/cm^2" if quantity == "polarization" else "e*angstrom"
    written: list[Path] = []
    for frame in np.asarray(result.frame_indices, dtype=int):
        source_frame, group = _table_for_trajectory_frame(result, int(frame))
        xyz = group[
            ["center_x (angstrom)", "center_y (angstrom)", "center_z (angstrom)"]
        ].to_numpy(float)
        values = group[column].to_numpy(float)
        valid = np.isfinite(xyz).all(axis=1) & np.isfinite(values)
        limits = shared_limits if global_scaling else _symmetric_limits(values[valid])
        figure = plt.figure(figsize=(8.0, 6.6))
        axis = figure.add_subplot(111, projection="3d")
        scatter = axis.scatter(
            xyz[valid, 0],
            xyz[valid, 1],
            xyz[valid, 2],
            c=values[valid],
            cmap="coolwarm",
            norm=Normalize(*limits),
            s=32,
        )
        axis.set_xlabel("x (angstrom)")
        axis.set_ylabel("y (angstrom)")
        axis.set_zlabel("z (angstrom)")
        axis.set_title(f"h-BN-reference local {quantity} {component} | frame {source_frame}")
        figure.colorbar(scatter, ax=axis, pad=0.1).set_label(
            f"{quantity}_{component} ({units})"
        )
        figure.tight_layout()
        path = destination / f"frame_{source_frame:06d}.png"
        figure.savefig(path, dpi=int(dpi), bbox_inches="tight")
        plt.close(figure)
        written.append(path)
    return written


def run_main(command: str, args: argparse.Namespace) -> int:
    canonical = _canonical_command(command)
    result = AnalysisExecutor().run(
        TASK_REGISTRY[TASK_KEY_BY_COMMAND[canonical]](),
        REQUEST_BUILDERS[canonical](args),
        runtime_arguments(args),
    )
    output = artifact_directory(args, canonical)
    output.mkdir(parents=True, exist_ok=True)
    local_path = output / "hbn_reference_local_polarization.csv"
    summary_path = output / "hbn_reference_local_polarization_summary.csv"
    cell_path = output / "hbn_reference_cell_polarization.csv"
    cell_summary_path = output / "hbn_reference_cell_polarization_summary.csv"
    layer_path = output / "hbn_reference_layer_polarization.csv"
    layer_summary_path = output / "hbn_reference_layer_polarization_summary.csv"
    displacement_path = output / "hbn_reference_displacements.csv"
    mapping_path = output / "hbn_reference_mapping.csv"
    result.table.to_csv(local_path, index=False)
    result.summary.to_csv(summary_path, index=False)
    result.cell_table.to_csv(cell_path, index=False)
    result.cell_summary.to_csv(cell_summary_path, index=False)
    result.layer_table.to_csv(layer_path, index=False)
    result.layer_summary.to_csv(layer_summary_path, index=False)
    if args.write_displacements:
        result.reference_result.displacements.to_csv(displacement_path, index=False)
    result.reference_result.mapping.to_csv(mapping_path, index=False)
    plots_2d = (
        generate_local_2d_plots(
            result,
            output,
            plane=str(args.plot_plane),
            component=str(args.plot_component),
            quantity=str(args.plot_quantity),
            bins=tuple(int(value) for value in args.plot_bins),
            global_scaling=bool(args.global_scaling),
            dpi=int(args.figure_dpi),
        )
        if args.plot_2d
        else []
    )
    plots_3d = (
        generate_local_3d_plots(
            result,
            output,
            component=str(args.plot_component),
            quantity=str(args.plot_quantity),
            global_scaling=bool(args.global_scaling),
            dpi=int(args.figure_dpi),
        )
        if args.plot_3d
        else []
    )
    extxyz = (
        write_local_polarization_extxyz(
            result,
            output / "hbn_reference_local_polarization.extxyz",
            precision=int(args.extxyz_precision),
        )
        if args.write_extxyz
        else None
    )
    args.suppress_table = True
    present_result(canonical, result, args)
    print(
        f"Wrote selected ({result.request.local_grouping}) h-BN-reference "
        f"polarization to {local_path}"
    )
    print(f"Wrote local/global closure summary to {summary_path}")
    print(f"Wrote crystallographic-cell polarization to {cell_path}")
    print(f"Wrote layer-resolved polarization to {layer_path}")
    if args.write_displacements:
        print(f"Wrote per-atom source displacements to {displacement_path}")
    print(f"Wrote atom-to-cell mapping to {mapping_path}")
    if plots_2d:
        print(f"Wrote {len(plots_2d):,} projected plot(s) under {output / 'plots_2d'}")
    if plots_3d:
        print(f"Wrote {len(plots_3d):,} 3D plot(s) under {output / 'plots_3d'}")
    if extxyz is not None:
        print(f"Wrote OVITO Extended XYZ trajectory to {extxyz}")
    return 0


__all__ = [
    "ALL_COMMANDS",
    "ALL_LEGACY_COMMANDS",
    "COMMAND",
    "REQUEST_BUILDERS",
    "TASK_KEY_BY_COMMAND",
    "aggregate_local_values_2d",
    "build_parser",
    "build_request",
    "generate_local_2d_plots",
    "generate_local_3d_plots",
    "run_main",
]
