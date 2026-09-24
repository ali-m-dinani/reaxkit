"""CLI for per-center basal-plane displacement polarization."""

from __future__ import annotations

from reaxkit.presentation.workflow_artifacts import write_workflow_tables

import argparse
from pathlib import Path
from typing import cast

import numpy as np

from reaxkit.analysis.ferroelectrics.basal_plane_displacement_for_dipole_moment.local_polarization import (
    BasalPlaneLocalPolarizationRequest,
    BasalPlaneLocalPolarizationResult,
    LocalVolumeMethod,
    _table_for_trajectory_frame,
    write_local_polarization_extxyz,
)
from reaxkit.analysis.ferroelectrics.three_folded_wurtzite.polarization import VolumeMethod
from reaxkit.core.platform.constants import const
from reaxkit.core.registry.analysis_task_registry import TASK_REGISTRY
from reaxkit.core.resolve.command_alias_resolver import resolve_command_name
from reaxkit.core.runtime.analysis_executor import AnalysisExecutor
from reaxkit.presentation.dispatcher import present_result
from reaxkit.workflows.ferroelectrics.three_folded_wurtzite.common import (
    add_input_arguments,
    add_structure_arguments,
    artifact_directory,
    runtime_arguments,
    structural_request_kwargs,
)

COMMAND = "get-basal-plane-displacement-local-polarization"
ALL_COMMANDS = (COMMAND,)
ALL_LEGACY_COMMANDS = (
    "get_basal_plane_displacement_local_polarization",
    "basal-plane-local-polarization",
)
TASK_KEY_BY_COMMAND = {COMMAND: COMMAND}


def _canonical_command(command: str) -> str:
    return resolve_command_name(command, ALL_COMMANDS, aliases={COMMAND: ALL_LEGACY_COMMANDS})


def build_parser(parser: argparse.ArgumentParser, *, command: str) -> argparse.ArgumentParser:
    canonical = _canonical_command(command)
    parser.set_defaults(command=canonical, progress=True)
    parser.formatter_class = argparse.RawTextHelpFormatter
    parser.description = """Calculate one local polarization vector per Al/B center.

Each center dipole is measured from the mean plane of its three basal N
neighbors. By default, every valid center receives an equal share of the frame
volume. The coordination method instead uses the tetrahedron formed by its
three basal and one apical N neighbors.

Examples:
  reaxkit get-basal-plane-displacement-local-polarization --periodic xy --charge-source formal --formal-charge Al=3 N=-3 --volume-method hull

  reaxkit get-basal-plane-displacement-local-polarization --local-volume-method coordination --periodic xy --charge-source reaxff
"""
    add_input_arguments(parser)
    add_structure_arguments(parser, include_polarity=True)
    parser.add_argument(
        "--local-volume-method",
        choices=["equal", "coordination"],
        default="equal",
        help=(
            "Choose the per-center volume: equal shares the selected frame volume "
            "among valid centers (default); coordination uses each N-neighbor tetrahedron."
        ),
    )
    parser.add_argument(
        "--volume-method",
        choices=["hull", "bbox", "cell"],
        default="hull",
        help=(
            "Choose the frame volume used by --local-volume-method equal: occupied "
            "convex hull (default), bounding box, or simulation cell."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Choose the output directory for local polarization and source dipole CSV files.",
    )
    parser.add_argument(
        "--plot-2d", action="store_true",
        help="Write per-frame 2D maps after aggregating along the coordinate omitted from --plot-plane.",
    )
    parser.add_argument(
        "--plot-3d", action="store_true",
        help="Write per-frame 3D center scatter plots colored by the selected local value.",
    )
    parser.add_argument(
        "--plot-plane", choices=["xy", "xz", "yz"], default="xy",
        help="Choose the displayed plane for --plot-2d. The remaining coordinate is aggregated.",
    )
    parser.add_argument(
        "--plot-component", choices=["x", "y", "z"], default="z",
        help="Choose the dipole or polarization component used as plot color.",
    )
    parser.add_argument(
        "--plot-quantity", choices=["polarization", "dipole"], default="polarization",
        help="Plot local polarization (default) or local dipole.",
    )
    parser.add_argument(
        "--plot-bins", nargs=2, type=int, default=(40, 40), metavar=("NU", "NV"),
        help="Set the two in-plane bin counts for --plot-2d. Default: 40 40.",
    )
    parser.add_argument(
        "--global-scaling", action=argparse.BooleanOptionalAction, default=False,
        help="Use shared symmetric color limits across frames for both 2D and 3D plots.",
    )
    parser.add_argument(
        "--figure-dpi", type=int, default=180,
        help="Set PNG resolution for 2D and 3D plots. Default: 180.",
    )
    parser.add_argument(
        "--write-extxyz", action="store_true",
        help="Write an OVITO-compatible Extended XYZ trajectory with local vector properties.",
    )
    parser.add_argument(
        "--include-electric-field", action="store_true",
        help=(
            "Add the iteration-aligned electric-field value from fort.78 to each "
            "Extended XYZ frame header."
        ),
    )
    parser.add_argument(
        "--field-direction", choices=["x", "y", "z"], default="z",
        help="Choose the electric-field component written to frame metadata. Default: z.",
    )
    parser.add_argument(
        "--extxyz-precision", type=int, default=8,
        help="Set significant digits in the Extended XYZ output. Default: 8.",
    )
    return parser


def build_request(args: argparse.Namespace) -> BasalPlaneLocalPolarizationRequest:
    return BasalPlaneLocalPolarizationRequest(
        **structural_request_kwargs(args),
        polarity_tolerance=float(args.polarity_tolerance),
        local_volume_method=cast(LocalVolumeMethod, str(args.local_volume_method)),
        volume_method=cast(VolumeMethod, str(args.volume_method)),
        include_electric_field=bool(args.include_electric_field),
        field_direction=str(args.field_direction),
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
    if not finite.size:
        return np.linspace(0.0, 1.0, int(count) + 1)
    lower, upper = float(np.min(finite)), float(np.max(finite))
    if lower == upper:
        lower -= 0.5
        upper += 0.5
    return np.linspace(lower, upper, int(count) + 1)


def aggregate_local_values_2d(
    result: BasalPlaneLocalPolarizationResult,
    frame: int,
    *,
    plane: str,
    component: str,
    quantity: str,
    bins: tuple[int, int],
    edges: tuple[np.ndarray, np.ndarray] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Aggregate along the coordinate omitted from ``plane`` into a 2D map."""

    group = result.table[result.table["frame_index"].astype(int).eq(int(frame))]
    coordinates = group[[
        f"site_{plane[0]} (angstrom)", f"site_{plane[1]} (angstrom)"
    ]].to_numpy(float)
    if edges is None:
        edges = (_edges(coordinates[:, 0], bins[0]), _edges(coordinates[:, 1], bins[1]))
    mu = group[f"mu_{component} (e*angstrom)"].to_numpy(float)
    valid = np.isfinite(coordinates).all(axis=1) & np.isfinite(mu)
    dipole, _, _ = np.histogram2d(
        coordinates[valid, 0], coordinates[valid, 1], bins=edges, weights=mu[valid]
    )
    if quantity == "dipole":
        values = dipole
    elif quantity == "polarization":
        volume = group["local_volume (angstrom^3)"].to_numpy(float)
        valid &= np.isfinite(volume) & (volume > 0.0)
        dipole, _, _ = np.histogram2d(
            coordinates[valid, 0], coordinates[valid, 1], bins=edges, weights=mu[valid]
        )
        volumes, _, _ = np.histogram2d(
            coordinates[valid, 0], coordinates[valid, 1], bins=edges, weights=volume[valid]
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
    result: BasalPlaneLocalPolarizationResult,
    output: Path,
    *,
    plane: str,
    component: str,
    quantity: str,
    bins: tuple[int, int],
    global_scaling: bool,
    dpi: int,
) -> list[Path]:
    """Write projected maps that aggregate dipole and volume along the omitted axis."""

    plt, Normalize = _matplotlib()
    all_coordinates = result.table[[
        f"site_{plane[0]} (angstrom)", f"site_{plane[1]} (angstrom)"
    ]].to_numpy(float)
    shared_edges = (
        _edges(all_coordinates[:, 0], bins[0]),
        _edges(all_coordinates[:, 1], bins[1]),
    )
    maps = {
        source_frame: aggregate_local_values_2d(
            result, source_frame, plane=plane, component=component,
            quantity=quantity, bins=bins, edges=shared_edges,
        )[2]
        for frame in result.frame_indices
        for source_frame, _ in [_table_for_trajectory_frame(result, int(frame))]
    }
    shared_limits = _symmetric_limits(
        np.concatenate([values.ravel() for values in maps.values()])
    ) if maps else (-1.0, 1.0)
    destination = Path(output) / "plots_2d" / f"{quantity}_{component}_{plane}"
    destination.mkdir(parents=True, exist_ok=True)
    units = "uC/cm^2" if quantity == "polarization" else "e*angstrom"
    written: list[Path] = []
    for frame, values in maps.items():
        limits = shared_limits if global_scaling else _symmetric_limits(values)
        figure, axis = plt.subplots(figsize=(7.2, 5.8))
        image = axis.pcolormesh(
            shared_edges[0], shared_edges[1], values,
            cmap="coolwarm", norm=Normalize(*limits), shading="flat",
        )
        axis.set_xlabel(f"{plane[0]} (angstrom)")
        axis.set_ylabel(f"{plane[1]} (angstrom)")
        axis.set_aspect("equal", adjustable="box")
        axis.set_title(
            f"Local {quantity} {component} on {plane} | frame {frame}"
        )
        figure.colorbar(image, ax=axis).set_label(f"{quantity}_{component} ({units})")
        figure.tight_layout()
        path = destination / f"frame_{frame:06d}.png"
        figure.savefig(path, dpi=int(dpi), bbox_inches="tight")
        plt.close(figure)
        written.append(path)
    return written


def generate_local_3d_plots(
    result: BasalPlaneLocalPolarizationResult,
    output: Path,
    *,
    component: str,
    quantity: str,
    global_scaling: bool,
    dpi: int,
) -> list[Path]:
    """Write per-frame 3D center scatter plots colored by one vector component."""

    plt, Normalize = _matplotlib()
    column = (
        f"P_{component} (uC/cm^2)"
        if quantity == "polarization"
        else f"mu_{component} (e*angstrom)"
    )
    shared_limits = _symmetric_limits(result.table[column].to_numpy(float))
    destination = Path(output) / "plots_3d" / f"{quantity}_{component}"
    destination.mkdir(parents=True, exist_ok=True)
    units = "uC/cm^2" if quantity == "polarization" else "e*angstrom"
    written: list[Path] = []
    for frame in np.asarray(result.frame_indices, dtype=int):
        source_frame, group = _table_for_trajectory_frame(result, int(frame))
        xyz = group[[
            "site_x (angstrom)", "site_y (angstrom)", "site_z (angstrom)"
        ]].to_numpy(float)
        values = group[column].to_numpy(float)
        valid = np.isfinite(xyz).all(axis=1) & np.isfinite(values)
        limits = shared_limits if global_scaling else _symmetric_limits(values[valid])
        figure = plt.figure(figsize=(8.0, 6.6))
        axis = figure.add_subplot(111, projection="3d")
        scatter = axis.scatter(
            xyz[valid, 0], xyz[valid, 1], xyz[valid, 2], c=values[valid],
            cmap="coolwarm", norm=Normalize(*limits), s=32,
        )
        axis.set_xlabel("x (angstrom)")
        axis.set_ylabel("y (angstrom)")
        axis.set_zlabel("z (angstrom)")
        axis.set_title(f"Local {quantity} {component} | frame {source_frame}")
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
    local = output / "basal_plane_local_polarization.csv"
    summary = output / "basal_plane_local_polarization_summary.csv"
    dipoles = output / "basal_plane_dipoles.csv"
    ions = output / "basal_plane_ions.csv"
    write_workflow_tables({local: result.table, summary: result.summary, dipoles: result.dipole_result.table, ions: result.dipole_result.ions}, args=args, summary=(summary.name,))
    plots_2d: list[Path] = []
    plots_3d: list[Path] = []
    if args.plot_2d:
        plots_2d = generate_local_2d_plots(
            result, output, plane=str(args.plot_plane), component=str(args.plot_component),
            quantity=str(args.plot_quantity), bins=tuple(int(value) for value in args.plot_bins),
            global_scaling=bool(args.global_scaling), dpi=int(args.figure_dpi),
        )
    if args.plot_3d:
        plots_3d = generate_local_3d_plots(
            result, output, component=str(args.plot_component),
            quantity=str(args.plot_quantity), global_scaling=bool(args.global_scaling),
            dpi=int(args.figure_dpi),
        )
    extxyz = None
    if args.write_extxyz:
        extxyz = write_local_polarization_extxyz(
            result, output / "basal_plane_local_polarization.extxyz",
            precision=int(args.extxyz_precision),
        )
    args.suppress_table = True
    present_result(canonical, result, args)
    print(f"Wrote per-center local polarization to {local}")
    if getattr(args, "output_profile", "standard") != "minimal":
        print(f"Wrote local polarization summary to {summary}")
    print(f"Wrote source dipoles to {dipoles}")
    print(f"Wrote unique-ion contributions to {ions}")
    if plots_2d:
        print(f"Wrote {len(plots_2d):,} projected 2D plot(s) under {output / 'plots_2d'}")
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
    "build_parser",
    "build_request",
    "aggregate_local_values_2d",
    "generate_local_2d_plots",
    "generate_local_3d_plots",
    "run_main",
]
