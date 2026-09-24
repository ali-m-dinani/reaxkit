"""CLI for TEM-like projected basal-plane polarity maps."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import cast

import numpy as np

from reaxkit.analysis.ferroelectrics.basal_plane_displacement_for_dipole_moment.projected_polarity import (
    BasalPlaneProjectedPolarityRequest,
    BasalPlaneProjectedPolarityResult,
    CartesianAxis,
    ProjectionPlane,
)
from reaxkit.core.registry.analysis_task_registry import TASK_REGISTRY
from reaxkit.core.resolve.command_alias_resolver import resolve_command_name
from reaxkit.core.runtime.analysis_executor import AnalysisExecutor
from reaxkit.core.runtime.artifacts import ArtifactSpec, ArtifactWriter
from reaxkit.presentation.dispatcher import present_result
from reaxkit.workflows.ferroelectrics.three_folded_wurtzite.common import (
    add_input_arguments,
    add_structure_arguments,
    artifact_directory,
    restrict_native_charge_input,
    runtime_arguments,
    structural_request_kwargs,
)

COMMAND = "get-basal-plane-displacement-projected-polarity"
ALL_COMMANDS = (COMMAND,)
ALL_LEGACY_COMMANDS = (
    "get_basal_plane_displacement_projected_polarity",
    "basal-plane-projected-polarity",
)
TASK_KEY_BY_COMMAND = {COMMAND: COMMAND}


def _canonical_command(command: str) -> str:
    return resolve_command_name(command, ALL_COMMANDS, aliases={COMMAND: ALL_LEGACY_COMMANDS})


def build_parser(parser: argparse.ArgumentParser, *, command: str) -> argparse.ArgumentParser:
    canonical = _canonical_command(command)
    parser.set_defaults(command=canonical, progress=True)
    parser.formatter_class = argparse.RawTextHelpFormatter
    parser.description = """Average local basal-plane polarity signs in spatial bins.

Each valid center contributes -1, 0, or +1 according to the selected dipole
component. Atom-to-bin assignments are fixed from the reference frame. The mean
therefore measures the signed polarity population without magnitude weighting.
Detailed per-center rows are omitted by default; --write-centers writes them
as Parquet unless CSV is explicitly requested.

Examples:
  reaxkit get-basal-plane-displacement-projected-polarity --periodic xy --charge-source formal --formal-charge Al=3 N=-3 --projection-plane xz --projection-bins 40 40 --plot-2d --plot-kymograph

  reaxkit get-basal-plane-displacement-projected-polarity --engine ams --input reaxout.kf --charge-source auto --frames 0:3200:50 --projection-plane xz --projection-bins 1 10 --plot-kymograph
"""
    add_input_arguments(parser)
    add_structure_arguments(parser, include_polarity=True)
    restrict_native_charge_input(parser)
    parser.add_argument(
        "--component", choices=["x", "y", "z"], default="z",
        help="Choose the local dipole component whose sign defines polarity. Default: z.",
    )
    parser.add_argument(
        "--projection-plane", choices=["xy", "xz", "yz"], default="xz",
        help="Choose the two spatial coordinates shown in each per-frame map. Default: xz.",
    )
    parser.add_argument(
        "--projection-bins", nargs=2, type=int, default=(40, 40),
        metavar=("NU", "NV"),
        help="Set bin counts along the two projection-plane axes. Default: 40 40.",
    )
    parser.add_argument(
        "--reference-frame", type=int, default=0,
        help=(
            "Assign each center to a fixed spatial bin using this source frame, then "
            "retain that atom-to-bin assignment for every analyzed frame. Default: 0."
        ),
    )
    parser.add_argument(
        "--profile-axis", choices=["x", "y", "z"], default=None,
        help=(
            "Choose the projection-plane axis on the vertical axis of the kymograph. "
            "It reuses that axis's --projection-bins count. Default: the second axis."
        ),
    )
    parser.add_argument(
        "--dipole-zero-tolerance", type=float, default=0.0,
        help=(
            "Map selected dipole magnitudes at or below this value to polarity 0. "
            "They remain included in bin means. Default: 0."
        ),
    )
    parser.add_argument(
        "--write-centers",
        action="store_true",
        help="Write the optional detailed per-center polarity table.",
    )
    parser.add_argument(
        "--centers-format",
        choices=["parquet", "csv"],
        default="parquet",
        help="Choose the detailed centers-table format. Default: parquet.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help=(
            "Override the automatically selected frame-worker count. "
            "Use 0 for automatic selection. Default: 0."
        ),
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=0,
        help=(
            "Override the maximum number of in-flight frames. "
            "Use 0 for the memory-aware automatic limit. Default: 0."
        ),
    )
    parser.add_argument(
        "--plot-2d", action="store_true",
        help="Write one projection-plane mean-polarity heatmap per selected frame.",
    )
    parser.add_argument(
        "--plot-kymograph", "--plot-evolution", dest="plot_kymograph",
        action="store_true",
        help=(
            "Write a kymograph: a frame-versus-position heatmap. "
            "--plot-evolution is retained as an alias."
        ),
    )
    parser.add_argument(
        "--figure-dpi", type=int, default=180,
        help="Set PNG resolution. Default: 180.",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="Choose the output directory for polarity tables and heatmaps.",
    )
    return parser


def build_request(args: argparse.Namespace) -> BasalPlaneProjectedPolarityRequest:
    projection_plane = cast(ProjectionPlane, str(args.projection_plane))
    profile_axis = cast(
        CartesianAxis,
        str(args.profile_axis) if args.profile_axis is not None else projection_plane[1],
    )
    return BasalPlaneProjectedPolarityRequest(
        **structural_request_kwargs(args),
        polarity_tolerance=float(args.polarity_tolerance),
        component=cast(CartesianAxis, str(args.component)),
        projection_plane=projection_plane,
        projection_bins=tuple(int(value) for value in args.projection_bins),
        profile_axis=profile_axis,
        reference_frame=int(args.reference_frame),
        dipole_zero_tolerance=float(args.dipole_zero_tolerance),
        include_centers=bool(args.write_centers),
        workers=int(args.workers),
        chunk_size=int(args.chunk_size),
    )


REQUEST_BUILDERS = {COMMAND: build_request}


def _matplotlib():
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("Polarity heatmaps require matplotlib.") from exc
    return plt


def _center_edges(values: np.ndarray) -> np.ndarray:
    centers = np.asarray(values, dtype=float)
    if centers.size == 1:
        return np.asarray([centers[0] - 0.5, centers[0] + 0.5])
    midpoints = (centers[:-1] + centers[1:]) / 2.0
    return np.concatenate((
        [centers[0] - (midpoints[0] - centers[0])],
        midpoints,
        [centers[-1] + (centers[-1] - midpoints[-1])],
    ))


def generate_projected_polarity_heatmaps(
        result: BasalPlaneProjectedPolarityResult,
        output: Path,
        *,
        dpi: int,
) -> list[Path]:
    """Write one fixed-scale 2D mean-polarity map per source frame."""

    plt = _matplotlib()
    u_edges, v_edges = result.projection_edges
    u_axis, v_axis = result.request.projection_plane
    nu, nv = (int(value) for value in result.request.projection_bins)
    destination = Path(output) / "projected_polarity_2d"
    destination.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for frame, group in result.projected_bins.groupby("frame_index", sort=True):
        values = (
            group.pivot(index="v_bin", columns="u_bin", values="mean_polarity")
            .reindex(index=range(nv), columns=range(nu))
            .to_numpy(float)
        )
        figure, axis = plt.subplots(figsize=(7.2, 5.8))
        image = axis.pcolormesh(
            u_edges, v_edges, values, cmap="coolwarm", vmin=-1.0, vmax=1.0,
            shading="flat",
        )
        axis.set_xlabel(f"{u_axis} (angstrom)")
        axis.set_ylabel(f"{v_axis} (angstrom)")
        axis.set_aspect("equal", adjustable="box")
        axis.set_title(
            f"Mean {result.request.component}-polarity on "
            f"{result.request.projection_plane} | frame {int(frame)}"
        )
        figure.colorbar(image, ax=axis).set_label("mean polarity")
        figure.tight_layout()
        path = destination / f"frame_{int(frame):06d}.png"
        figure.savefig(path, dpi=int(dpi), bbox_inches="tight")
        plt.close(figure)
        written.append(path)
    return written


def generate_polarity_kymograph(
        result: BasalPlaneProjectedPolarityResult,
        output: Path,
        *,
        dpi: int,
) -> Path:
    """Write a kymograph of mean polarity versus frame and position."""

    plt = _matplotlib()
    table = result.kymograph_bins
    frames = np.sort(table["frame_index"].astype(int).unique())
    profile_axis_index = result.request.projection_plane.index(result.request.profile_axis)
    profile_bins = np.arange(int(result.request.projection_bins[profile_axis_index]))
    values = (
        table.pivot(index="profile_bin", columns="frame_index", values="mean_polarity")
        .reindex(index=profile_bins, columns=frames)
        .to_numpy(float)
    )
    figure, axis = plt.subplots(figsize=(9.0, 5.8))
    image = axis.pcolormesh(
        _center_edges(frames), result.profile_edges, values,
        cmap="coolwarm", vmin=-1.0, vmax=1.0, shading="flat",
    )
    axis.set_xlabel("source frame")
    axis.set_ylabel(f"{result.request.profile_axis} (angstrom)")
    axis.set_title(
        f"Mean {result.request.component}-polarity kymograph along "
        f"{result.request.profile_axis}"
    )
    figure.colorbar(image, ax=axis).set_label("mean polarity")
    figure.tight_layout()
    destination = Path(output) / "projected_polarity_kymograph.png"
    figure.savefig(destination, dpi=int(dpi), bbox_inches="tight")
    plt.close(figure)
    return destination


generate_polarity_evolution_heatmap = generate_polarity_kymograph


def run_main(command: str, args: argparse.Namespace) -> int:
    canonical = _canonical_command(command)
    output = artifact_directory(args, canonical)
    output.mkdir(parents=True, exist_ok=True)
    profile = str(getattr(args, "output_profile", "standard"))
    args.write_centers = (bool(args.write_centers) or profile in {"full", "legacy"}) and profile != "minimal"
    centers_format = str(getattr(args, "detail_format", None) or ("csv" if profile == "legacy" else args.centers_format))
    centers = output / f"basal_plane_projected_polarity_centers.{centers_format}"
    projected = output / "basal_plane_projected_polarity_2d.csv"
    kymograph = output / "basal_plane_projected_polarity_kymograph.csv"
    specs = (
        ArtifactSpec("centers", centers.name, "detail", bool(args.write_centers), centers_format, True),
        ArtifactSpec("projected", projected.name, "core", True, "csv", True),
        ArtifactSpec("kymograph", kymograph.name, "core", True, "csv", True),
    )
    with ArtifactWriter(output, specs, profile=profile, overwrite=True) as writer:
        run_args = runtime_arguments(args)
        run_args["_artifact_writer"] = writer
        if args.write_centers:
            run_args["no_cache"] = True
        result = AnalysisExecutor().run(
            TASK_REGISTRY[TASK_KEY_BY_COMMAND[canonical]](),
            REQUEST_BUILDERS[canonical](args),
            run_args,
        )
        writer.metadata.update(execution_policy=run_args.get("_execution_policy", {}), source_frames=getattr(result.request, "frames", None))
        if args.write_centers and not result.centers.empty:
            writer.write_table("centers", result.centers)
        writer.write_table("projected", result.projected_bins)
        writer.write_table("kymograph", result.kymograph_bins)
        plots = (
            generate_projected_polarity_heatmaps(result, output, dpi=int(args.figure_dpi))
            if args.plot_2d else []
        )
        kymograph_plot = (
            generate_polarity_kymograph(result, output, dpi=int(args.figure_dpi))
            if args.plot_kymograph else None
        )
        args.suppress_table = True
        present_result(canonical, result, args)
    if args.write_centers:
        print(f"Wrote detailed per-center polarity signs to {centers}")
    print(f"Wrote projected mean-polarity bins to {projected}")
    print(f"Wrote kymograph bins to {kymograph}")
    if plots:
        print(f"Wrote {len(plots):,} projected heatmap(s) under {output / 'projected_polarity_2d'}")
    if kymograph_plot is not None:
        print(f"Wrote polarity kymograph to {kymograph_plot}")
    return 0


__all__ = [
    "ALL_COMMANDS",
    "ALL_LEGACY_COMMANDS",
    "COMMAND",
    "REQUEST_BUILDERS",
    "TASK_KEY_BY_COMMAND",
    "build_parser",
    "build_request",
    "generate_projected_polarity_heatmaps",
    "generate_polarity_evolution_heatmap",
    "generate_polarity_kymograph",
    "run_main",
]
