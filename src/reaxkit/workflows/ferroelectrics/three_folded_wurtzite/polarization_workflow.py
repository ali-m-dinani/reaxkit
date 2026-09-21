"""CLI workflow for spatially binned three-folded wurtzite polarization."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from reaxkit.analysis.ferroelectrics.three_folded_wurtzite.polarization import (
    BinnedPolarizationRequest,
    BinnedPolarizationResult,
)
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

COMMAND = "get-three-folded-wurtzite-polarization"
ALL_COMMANDS = (COMMAND,)
ALL_LEGACY_COMMANDS = (
    "get_three_folded_wurtzite_polarization",
    "three-folded-wurtzite-polarization",
)
TASK_KEY_BY_COMMAND = {COMMAND: COMMAND}


def _canonical_command(command: str) -> str:
    return resolve_command_name(command, ALL_COMMANDS, aliases={COMMAND: ALL_LEGACY_COMMANDS})


def build_parser(parser: argparse.ArgumentParser, *, command: str) -> argparse.ArgumentParser:
    canonical = _canonical_command(command)
    parser.set_defaults(command=canonical, progress=True)
    parser.formatter_class = argparse.RawTextHelpFormatter
    parser.description = """Calculate spatially binned polarization from three-folded site dipoles.

The workflow first calculates each center's dipole using the three-folded polarity
analysis. It then sums those dipoles in a fixed three-dimensional grid and divides
by a hull, bounding-box, or simulation-cell bin volume. Optional 2D heatmaps
project the selected polarization component onto xy, xz, or yz.

Examples:
  1. Bin an AlN slab in x and y using occupied convex-hull volumes:
     reaxkit get-three-folded-wurtzite-polarization --bins-x 20 --bins-y 20 --bins-z 1 --volume-method hull --periodic xy --charge-source formal --formal-charge Al=3 N=-3

  2. Generate per-frame Pz heatmaps with one shared color scale:
     reaxkit get-three-folded-wurtzite-polarization --bins-x 30 --bins-y 30 --heatmaps --heatmap-plane xy --heatmap-component z --global-scaling

  3. Analyze selected frames with cell-normalized bins:
     reaxkit get-three-folded-wurtzite-polarization --frames 0:101:10 --bins-z 20 --volume-method cell
"""
    add_input_arguments(parser)
    add_structure_arguments(parser, include_polarity=True)
    parser.add_argument(
        "--bins-x", type=int, default=1,
        help="Set the number of x-direction bins. Example: --bins-x 20, divides the reference x extent into 20 intervals.",
    )
    parser.add_argument(
        "--bins-y", type=int, default=1,
        help="Set the number of y-direction bins. Example: --bins-y 20, divides the reference y extent into 20 intervals.",
    )
    parser.add_argument(
        "--bins-z", type=int, default=1,
        help="Set the number of z-direction bins. Example: --bins-z 10, divides the reference z extent into 10 intervals.",
    )
    parser.add_argument(
        "--volume-method", choices=["hull", "bbox", "cell"], default="hull",
        help="Choose the bin-volume estimator. Example: --volume-method bbox, uses occupied coordinate extents inside every bin.",
    )
    parser.add_argument(
        "--heatmaps", action="store_true",
        help="Write one 2D polarization heatmap per frame. Example: --heatmaps, writes PNG files under heatmaps.",
    )
    parser.add_argument(
        "--heatmap-plane", choices=["xy", "xz", "yz"], default="xy",
        help="Choose the heatmap projection plane. Example: --heatmap-plane xz, sums bins along y.",
    )
    parser.add_argument(
        "--heatmap-component", choices=["x", "y", "z"], default="z",
        help="Choose the plotted polarization component. Example: --heatmap-component z, colors the map by Pz.",
    )
    parser.add_argument(
        "--global-scaling", action=argparse.BooleanOptionalAction, default=False,
        help="Share color limits across frames. Example: --global-scaling, makes all frame colors directly comparable; --no-global-scaling rescales each frame.",
    )
    parser.add_argument(
        "--figure-dpi", type=int, default=180,
        help="Set heatmap resolution. Example: --figure-dpi 300, writes 300-DPI PNG files.",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="Choose the CSV and heatmap directory. Example: --output-dir ./polarization, writes artifacts there.",
    )
    return parser


def build_request(args: argparse.Namespace) -> BinnedPolarizationRequest:
    return BinnedPolarizationRequest(
        **structural_request_kwargs(args),
        polarity_tolerance=float(args.polarity_tolerance),
        bins_x=int(args.bins_x), bins_y=int(args.bins_y), bins_z=int(args.bins_z),
        volume_method=str(args.volume_method),
    )


REQUEST_BUILDERS = {COMMAND: build_request}


def _finite_symmetric_limit(values: np.ndarray) -> tuple[float, float]:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    bound = float(np.max(np.abs(finite))) if finite.size else 1.0
    return (-bound, bound) if bound > 0.0 else (-1.0, 1.0)


def _projected_frame(
    result: BinnedPolarizationResult, frame: int, plane: str, component: str
) -> tuple[np.ndarray, np.ndarray]:
    axes = ("xyz".index(plane[0]), "xyz".index(plane[1]))
    group = result.table[result.table["frame_index"].astype(int).eq(int(frame))]
    shape = (len(result.bin_edges[axes[0]]) - 1, len(result.bin_edges[axes[1]]) - 1)
    dipole_column = f"mu_{component} (e*angstrom)"
    projected = group.groupby(
        [f"bin_{plane[0]}", f"bin_{plane[1]}"], sort=True
    )[[dipole_column, "volume (angstrom^3)"]].sum(min_count=1)
    full_index = pd.MultiIndex.from_product(
        [range(shape[0]), range(shape[1])]
    )
    projected = projected.reindex(full_index)
    values = np.divide(
        projected[dipole_column].to_numpy(float) * float(const("ea3_to_uC_cm2")),
        projected["volume (angstrom^3)"].to_numpy(float),
        out=np.full(int(np.prod(shape)), np.nan),
        where=projected["volume (angstrom^3)"].to_numpy(float) > 0,
    ).reshape(shape).T
    return values, np.asarray([axes[0], axes[1]], dtype=int)


def generate_polarization_heatmaps(
    result: BinnedPolarizationResult,
    output: Path,
    *,
    plane: str = "xy",
    component: str = "z",
    global_scaling: bool = False,
    dpi: int = 180,
    title_prefix: str = "Three-folded",
) -> list[Path]:
    """Write projected polarization heatmaps with shared or per-frame scaling."""

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.colors import Normalize
    except ImportError as exc:  # pragma: no cover
        raise ImportError("Heatmap generation requires matplotlib; install reaxkit[plot].") from exc

    frames = [int(value) for value in result.frame_indices]
    maps = {frame: _projected_frame(result, frame, plane, component)[0] for frame in frames}
    shared = _finite_symmetric_limit(np.concatenate([value.ravel() for value in maps.values()])) if maps else (-1.0, 1.0)
    axes = ("xyz".index(plane[0]), "xyz".index(plane[1]))
    destination = Path(output) / "heatmaps" / f"P_{component}"
    destination.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for frame, values in maps.items():
        limits = shared if global_scaling else _finite_symmetric_limit(values)
        figure, axis = plt.subplots(figsize=(7.2, 5.8))
        image = axis.pcolormesh(
            result.bin_edges[axes[0]], result.bin_edges[axes[1]], values,
            cmap="coolwarm", norm=Normalize(*limits), shading="flat",
        )
        axis.set_xlabel(f"{plane[0]} (angstrom)")
        axis.set_ylabel(f"{plane[1]} (angstrom)")
        axis.set_aspect("equal", adjustable="box")
        axis.set_title(f"{title_prefix} P_{component} on {plane} | frame {frame}")
        figure.colorbar(image, ax=axis).set_label(f"P_{component} (uC/cm^2)")
        figure.tight_layout()
        path = destination / f"frame_{frame:06d}.png"
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
    bins_path = output / "binned_polarization.csv"
    summary_path = output / "polarization_summary.csv"
    result.table.to_csv(bins_path, index=False)
    result.summary.to_csv(summary_path, index=False)
    figures: list[Path] = []
    if args.heatmaps:
        figures = generate_polarization_heatmaps(
            result, output, plane=args.heatmap_plane,
            component=args.heatmap_component,
            global_scaling=bool(args.global_scaling), dpi=int(args.figure_dpi),
        )
    args.suppress_table = True
    present_result(canonical, result, args)
    print(f"Wrote binned polarization to {bins_path}")
    print(f"Wrote frame polarization summary to {summary_path}")
    if figures:
        scale = "global" if args.global_scaling else "per-frame"
        print(f"Wrote {len(figures):,} {scale}-scaled heatmap(s) under {output / 'heatmaps'}")
    return 0


__all__ = [
    "ALL_COMMANDS", "ALL_LEGACY_COMMANDS", "COMMAND", "REQUEST_BUILDERS",
    "TASK_KEY_BY_COMMAND", "build_parser", "build_request",
    "generate_polarization_heatmaps", "run_main",
]
