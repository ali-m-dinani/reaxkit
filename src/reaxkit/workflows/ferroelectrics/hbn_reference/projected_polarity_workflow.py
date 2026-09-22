"""CLI for projected h-BN-reference local-cell polarity maps."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import cast

import numpy as np

from reaxkit.analysis.ferroelectrics.hbn_reference.local_polarization import (
    LocalGrouping,
)
from reaxkit.analysis.ferroelectrics.hbn_reference.projected_polarity import (
    CartesianAxis,
    HBNReferenceProjectedPolarityRequest,
    HBNReferenceProjectedPolarityResult,
    PolarityComponent,
    ProjectionPlane,
)
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

COMMAND = "get-hbn-reference-projected-polarity"
ALL_COMMANDS = (COMMAND,)
ALL_LEGACY_COMMANDS = (
    "get_hbn_reference_projected_polarity",
    "hbn-reference-projected-polarity",
)
TASK_KEY_BY_COMMAND = {COMMAND: COMMAND}


def _canonical_command(command: str) -> str:
    return resolve_command_name(command, ALL_COMMANDS, aliases={COMMAND: ALL_LEGACY_COMMANDS})


def build_parser(parser: argparse.ArgumentParser, *, command: str) -> argparse.ArgumentParser:
    canonical = _canonical_command(command)
    global_workflow.build_parser(parser, command=global_workflow.COMMAND)
    parser.set_defaults(command=canonical, progress=True)
    parser.description = """Average h-BN-reference local-cell polarity in fixed spatial bins.

Each neutral reference cell contributes -1, 0, or +1 from the selected local
dipole component. Zero dipoles remain in the average. Spatial assignments are
made once from --reference-frame and remain fixed as atoms move. --plot-2d
writes one TEM-like projection per frame; --plot-kymograph writes the complete
frame-versus-position evolution map.

Example:
  reaxkit get-hbn-reference-projected-polarity --replication 19 19 10 --periodic xy --charge-source reaxff --component c --projection-plane xz --projection-bins 1 40 --profile-axis z --plot-2d --plot-kymograph
"""
    parser.add_argument(
        "--local-grouping",
        choices=["cell", "layer"],
        default="cell",
        help="Project four-atom crystallographic cells or two-atom AlN layers. Default: cell.",
    )
    parser.add_argument(
        "--local-volume-method",
        choices=["equal", "deformation"],
        default="equal",
        help="Set the local-volume convention retained in the cell table. Default: equal.",
    )
    parser.add_argument(
        "--deformation-neighbors",
        type=int,
        default=12,
        help="Set neighboring reference cells used by local affine fits. Default: 12.",
    )
    parser.add_argument(
        "--local-charge-treatment",
        choices=["auto", "raw", "neutralize"],
        default="auto",
        help=(
            "Choose raw atomic charges, per-cell charge neutralization, or auto "
            "neutralization for charged cells. Default: auto."
        ),
    )
    parser.add_argument(
        "--component",
        choices=["x", "y", "z", "c"],
        default="c",
        help="Choose the local dipole component whose sign defines polarity. Default: c.",
    )
    parser.add_argument(
        "--projection-plane",
        choices=["xy", "xz", "yz"],
        default="xz",
        help="Choose the two coordinates shown in each per-frame map. Default: xz.",
    )
    parser.add_argument(
        "--projection-bins",
        nargs=2,
        type=int,
        default=(40, 40),
        metavar=("NU", "NV"),
        help="Set bin counts along the projection-plane axes. Default: 40 40.",
    )
    parser.add_argument(
        "--profile-axis",
        choices=["x", "y", "z"],
        default=None,
        help=(
            "Choose the projection-plane axis on the kymograph vertical axis. "
            "Its --projection-bins count is reused. Default: the second axis."
        ),
    )
    parser.add_argument(
        "--dipole-zero-tolerance",
        type=float,
        default=0.0,
        help=(
            "Map local dipole magnitudes at or below this value to polarity 0; "
            "zero remains included in bin means. Default: 0."
        ),
    )
    parser.add_argument(
        "--plot-2d",
        action="store_true",
        help="Write one projection-plane mean-polarity heatmap per selected frame.",
    )
    parser.add_argument(
        "--plot-kymograph",
        "--plot-evolution",
        dest="plot_kymograph",
        action="store_true",
        help=(
            "Write a frame-versus-position polarity heatmap. "
            "--plot-evolution is an alias."
        ),
    )
    parser.add_argument(
        "--figure-dpi", type=int, default=180, help="Set PNG resolution. Default: 180."
    )
    return parser


def build_request(args: argparse.Namespace) -> HBNReferenceProjectedPolarityRequest:
    base = global_workflow.build_request(args)
    projection_plane = cast(ProjectionPlane, str(args.projection_plane))
    profile_axis = cast(
        CartesianAxis,
        str(args.profile_axis) if args.profile_axis is not None else projection_plane[1],
    )
    return HBNReferenceProjectedPolarityRequest(
        **vars(base),
        local_volume_method=str(args.local_volume_method),
        deformation_neighbors=int(args.deformation_neighbors),
        local_charge_treatment=str(args.local_charge_treatment),
        local_grouping=cast(LocalGrouping, str(args.local_grouping)),
        component=cast(PolarityComponent, str(args.component)),
        projection_plane=projection_plane,
        projection_bins=tuple(int(value) for value in args.projection_bins),
        profile_axis=profile_axis,
        dipole_zero_tolerance=float(args.dipole_zero_tolerance),
    )


REQUEST_BUILDERS = {COMMAND: build_request}


def _matplotlib():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover
        raise ImportError("Polarity heatmaps require matplotlib.") from exc
    return plt


def _center_edges(values: np.ndarray) -> np.ndarray:
    centers = np.asarray(values, dtype=float)
    if centers.size == 1:
        return np.asarray([centers[0] - 0.5, centers[0] + 0.5])
    midpoints = (centers[:-1] + centers[1:]) / 2.0
    return np.concatenate(
        (
            [centers[0] - (midpoints[0] - centers[0])],
            midpoints,
            [centers[-1] + (centers[-1] - midpoints[-1])],
        )
    )


def generate_projected_polarity_heatmaps(
        result: HBNReferenceProjectedPolarityResult,
        output: Path,
        *,
        dpi: int,
) -> list[Path]:
    """Write one fixed-scale mean-polarity map per source frame."""

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
            u_edges,
            v_edges,
            values,
            cmap="coolwarm",
            vmin=-1.0,
            vmax=1.0,
            shading="flat",
        )
        axis.set_xlabel(f"{u_axis} (angstrom)")
        axis.set_ylabel(f"{v_axis} (angstrom)")
        axis.set_aspect("equal", adjustable="box")
        axis.set_title(
            f"Mean h-BN-reference {result.request.component}-polarity | frame {int(frame)}"
        )
        figure.colorbar(image, ax=axis).set_label("mean polarity")
        figure.tight_layout()
        path = destination / f"frame_{int(frame):06d}.png"
        figure.savefig(path, dpi=int(dpi), bbox_inches="tight")
        plt.close(figure)
        written.append(path)
    return written


def generate_polarity_kymograph(
        result: HBNReferenceProjectedPolarityResult,
        output: Path,
        *,
        dpi: int,
) -> Path:
    """Write mean polarity versus source frame and reference position."""

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
        _center_edges(frames),
        result.profile_edges,
        values,
        cmap="coolwarm",
        vmin=-1.0,
        vmax=1.0,
        shading="flat",
    )
    axis.set_xlabel("source frame")
    axis.set_ylabel(f"{result.request.profile_axis} (angstrom)")
    axis.set_title(
        f"Mean h-BN-reference {result.request.component}-polarity kymograph"
    )
    figure.colorbar(image, ax=axis).set_label("mean polarity")
    figure.tight_layout()
    destination = Path(output) / "hbn_reference_projected_polarity_kymograph.png"
    figure.savefig(destination, dpi=int(dpi), bbox_inches="tight")
    plt.close(figure)
    return destination


generate_polarity_evolution_heatmap = generate_polarity_kymograph


def run_main(command: str, args: argparse.Namespace) -> int:
    canonical = _canonical_command(command)
    result = AnalysisExecutor().run(
        TASK_REGISTRY[TASK_KEY_BY_COMMAND[canonical]](),
        REQUEST_BUILDERS[canonical](args),
        runtime_arguments(args),
    )
    output = artifact_directory(args, canonical)
    output.mkdir(parents=True, exist_ok=True)
    cells_path = output / "hbn_reference_projected_polarity_cells.csv"
    projected_path = output / "hbn_reference_projected_polarity_2d.csv"
    kymograph_path = output / "hbn_reference_projected_polarity_kymograph.csv"
    result.centers.to_csv(cells_path, index=False)
    result.projected_bins.to_csv(projected_path, index=False)
    result.kymograph_bins.to_csv(kymograph_path, index=False)
    plots = (
        generate_projected_polarity_heatmaps(result, output, dpi=int(args.figure_dpi))
        if args.plot_2d
        else []
    )
    kymograph_plot = (
        generate_polarity_kymograph(result, output, dpi=int(args.figure_dpi))
        if args.plot_kymograph
        else None
    )
    args.suppress_table = True
    present_result(canonical, result, args)
    print(f"Wrote per-cell polarity signs to {cells_path}")
    print(f"Wrote projected mean-polarity bins to {projected_path}")
    print(f"Wrote kymograph bins to {kymograph_path}")
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
    "generate_polarity_evolution_heatmap",
    "generate_polarity_kymograph",
    "generate_projected_polarity_heatmaps",
    "run_main",
]
