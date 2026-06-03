"""Direct command workflow for trajectory analyses.

This module implements CLI workflow orchestration for its command family, including argument parsing, request construction, execution dispatch, and result presentation handoff.

**Usage context**

- Command routing: Resolve CLI aliases and normalized command names.
- Task execution: Build request objects and invoke registered tasks.
- Output handling: Forward results to table, plot, export, or report flows.
"""

from __future__ import annotations

import argparse
import ast
import numpy as np
import pandas as pd
from typing import Callable, Sequence

from reaxkit.analysis import trajectory as _trajectory_tasks  # noqa: F401
from reaxkit.analysis.trajectory.dihedral import DihedralRequest
from reaxkit.analysis.trajectory.diffusivity import DiffusivityRequest
from reaxkit.analysis.trajectory.msd import MSDRequest
from reaxkit.analysis.trajectory.rdf import RDFPropertyRequest, RDFRequest
from reaxkit.analysis.trajectory.voronoi import VoronoiRequest
from reaxkit.core.runtime.analysis_executor import AnalysisExecutor
from reaxkit.core.registry.analysis_task_registry import TASK_REGISTRY
from reaxkit.core.resolve.command_alias_resolver import resolve_command_name
from reaxkit.core.utils.frame_utils import parse_frame_indices
from reaxkit.core.storage.storage_layout import add_storage_cli_arguments
from reaxkit.presentation.dispatcher import present_result
from reaxkit.presentation.convert import convert_xaxis

ALL_COMMANDS = ("get_dihedral", "get_diffusivity", "get_msd", "get_rdf", "get_rdf_property", "get_voronoi")
ALL_LEGACY_COMMANDS = (
    "dihedral",
    "diffusivity",
    "msd",
    "rdf",
    "rdf_property",
    "voronoi",
    "get-diffusivity",
    "get-msd",
    "get-rdf",
    "get-rdf-property",
    "get-voronoi",
    "get-dihedral",
)


def _add_runtime_arguments(parser: argparse.ArgumentParser) -> None:
    """Add runtime arguments."""
    parser.add_argument("--engine", choices=["reaxff", "ams", "lammps"], default=None, help="Engine override. Example: --engine reaxff, which applies ReaxFF-specific trajectory loading behavior.")
    parser.add_argument("--run-dir", "--dir", dest="run_dir", default=".", help="Run directory (fallback for detection). Example: --run-dir runs/job1, which sets backup path for file discovery.")
    parser.add_argument("--xmolout", "--file", dest="xmolout", default=None, help="Trajectory file path. Example: --xmolout runs/job1/xmolout, which provides coordinate trajectory input.")
    parser.add_argument("--log", choices=["verbose", "quiet"], default="quiet", help="Logging level. Example: --log verbose, which prints more runtime details.")
    add_storage_cli_arguments(parser)


def _add_presentation_arguments(parser: argparse.ArgumentParser) -> None:
    """Add presentation arguments."""
    parser.add_argument("--plot", choices=["single", "subplot"], default=None, help="Render a plot. Example: --plot single, which creates one combined figure.")
    parser.add_argument("--show", action="store_true", help="Show the generated plot window. Example: --show, which opens the figure interactively.")
    parser.add_argument("--save", default=None, help="Save the generated plot to a file path. Example: --save msd.png, which writes the figure image to disk.")
    parser.add_argument("--export", default=None, help="Write the result table to CSV. Example: --export rdf.csv, which saves tabular analysis output.")
    parser.add_argument("--grid", default=None, help="Subplot grid like 2x2 or 2*2. Example: --grid 2x2, which arranges subplot panels in a 2-by-2 layout.")
    parser.add_argument("--xaxis", default="frame", choices=["iter", "frame", "time"], help="Quantity on x-axis. Example: --xaxis time, which uses converted physical time when available.")


def _add_common_arguments(parser: argparse.ArgumentParser) -> None:
    """Add common arguments."""
    parser.add_argument(
        "--frames",
        nargs="*",
        default=None,
        help='Frame selector syntax. Example: --frames 0:20:2, which selects frames 0,2,4,...,20.',
    )
    parser.add_argument("--every", type=int, default=1, help="Frame stride. Example: --every 5, which keeps every fifth selected frame.")


def _build_msd_request(args: argparse.Namespace) -> MSDRequest:
    """Build msd request."""
    return MSDRequest(
        atom_ids=args.atom_ids,
        atom_types=args.atom_types,
        dims=tuple(args.dims),
        frames=parse_frame_indices(args.frames),
        every=args.every,
        unwrap=bool(args.unwrap),
        max_lag=args.max_lag,
        delta_t_ps=args.delta_t_ps,
    )


def _build_diffusivity_request(args: argparse.Namespace) -> DiffusivityRequest:
    """Build diffusivity request."""
    return DiffusivityRequest(
        atom_ids=args.atom_ids,
        atom_types=args.atom_types,
        dims=tuple(args.dims),
        origin=args.origin,
        frames=parse_frame_indices(args.frames),
        every=args.every,
        d=float(args.d),
        unwrap=bool(args.unwrap),
        max_lag=args.max_lag,
        delta_t_ps=args.delta_t_ps,
    )


def _build_dihedral_request(args: argparse.Namespace) -> DihedralRequest:
    """Build dihedral request."""
    return DihedralRequest(
        atom_ids=args.atom_ids,
        frames=parse_frame_indices(args.frames),
        every=args.every,
        units=args.units,
        backend=args.backend,
    )


def _build_rdf_request(args: argparse.Namespace) -> RDFRequest:
    """Build rdf request."""
    return RDFRequest(
        atom_ids_a=args.atom_ids_a,
        atom_ids_b=args.atom_ids_b,
        atom_types_a=args.atom_types_a,
        atom_types_b=args.atom_types_b,
        frames=parse_frame_indices(args.frames),
        every=args.every,
        bins=args.bins,
        r_max=args.r_max,
        backend=args.backend,
    )


def _build_rdf_property_request(args: argparse.Namespace) -> RDFPropertyRequest:
    """Build rdf property request."""
    return RDFPropertyRequest(
        property=args.property or args.prop or "first_peak",
        atom_ids_a=args.atom_ids_a,
        atom_ids_b=args.atom_ids_b,
        atom_types_a=args.atom_types_a,
        atom_types_b=args.atom_types_b,
        frames=parse_frame_indices(args.frames),
        every=args.every,
        bins=args.bins,
        r_max=args.r_max,
        backend=args.backend,
    )


def _build_voronoi_request(args: argparse.Namespace) -> VoronoiRequest:
    """Build voronoi request."""
    return VoronoiRequest(
        atom_ids=args.atom_ids,
        atom_types=args.atom_types,
        frames=parse_frame_indices(args.frames),
        every=args.every,
        backend=args.backend,
    )


REQUEST_BUILDERS: dict[str, Callable[[argparse.Namespace], object]] = {
    "get_dihedral": _build_dihedral_request,
    "get_diffusivity": _build_diffusivity_request,
    "get_msd": _build_msd_request,
    "get_rdf": _build_rdf_request,
    "get_rdf_property": _build_rdf_property_request,
    "get_voronoi": _build_voronoi_request,
}


def build_parser(parser: argparse.ArgumentParser, *, command: str) -> argparse.ArgumentParser:
    """Build the parser for a direct trajectory command."""
    canonical = resolve_command_name(command, task_names=ALL_COMMANDS)
    parser.set_defaults(command=canonical)
    parser.set_defaults(progress=True)
    parser.formatter_class = argparse.RawTextHelpFormatter

    _add_runtime_arguments(parser)
    _add_presentation_arguments(parser)
    _add_common_arguments(parser)

    if canonical == "get_dihedral":
        parser.description = (
            "Compute a dihedral angle (atom1-atom2-atom3-atom4) over selected frames.\n"
            "Use this command to track torsional evolution for a specific four-atom tuple.\n\n"
            "Examples:\n"
            "  1. Plot dihedral trajectory for one atom tuple:\n"
            "   reaxkit get_dihedral --atom-ids 1 2 3 4 --plot single\n\n"
            "  2. Export sampled frames in radians:\n"
            "   reaxkit get_dihedral --atom-ids 8 3 4 9 --frames 0:500:10 --units rad --export dih.csv\n\n"
            "  3. Save dihedral plot on iteration axis:\n"
            "   reaxkit get_dihedral --atom-ids 5 7 9 11 --xaxis iter --save dihedral_iter.png"
        )
        parser.add_argument("--atom-ids", type=int, nargs=4, required=True, help="Exactly four 1-based atom ids. Example: --atom-ids 1 2 3 4, which defines the torsion tuple order.")
        parser.add_argument("--units", choices=["deg", "rad"], default="deg", help="Output angle units. Example: --units rad, which reports dihedral values in radians.")
        parser.add_argument("--backend", choices=["numpy", "mdanalysis"], default="numpy", help="Dihedral backend. Example: --backend mdanalysis, which uses MDAnalysis implementation.")
    elif canonical == "get_msd":
        parser.description = (
            "Compute time-origin averaged mean-squared displacement (MSD) for selected atoms.\n"
            "The output is averaged over selected atoms and all valid time origins, giving MSD vs lag time.\n\n"
            "Examples:\n"
            "  1. Plot MSD for selected atom ids:\n"
            "   reaxkit get_msd --atom-ids 1 2 3 --max-lag 500 --delta-t-ps 0.25 --plot single\n\n"
            "  2. Save oxygen-only MSD vs lag time:\n"
            "   reaxkit get_msd --atom-types O --max-lag 800 --delta-t-ps 1.0 --save msd_oxygen.png\n\n"
            "  3. Export MSD using selected frames:\n"
            "   reaxkit get_msd --atom-ids 5 --frames 0:1000:10 --max-lag 80 --export msd_atom5.csv"
        )
        parser.add_argument("--atom-ids", type=int, nargs="*", default=None, help="1-based atom ids. Example: --atom-ids 1 2 3, which restricts MSD to those atoms.")
        parser.add_argument("--atom-types", nargs="*", default=None, help="Element symbols to include. Example: --atom-types O, which computes MSD for oxygen atoms only.")
        parser.add_argument("--dims", nargs="*", default=("x", "y", "z"), help="Coordinate dimensions to include. Example: --dims x y, which computes MSD using in-plane displacement.")
        parser.add_argument(
            "--max-lag",
            type=int,
            default=None,
            help=(
                "Maximum lag in number of selected frames. "
                "Example: --max-lag 800, which computes MSD from lag 0 to lag 799."
            ),
        )

        parser.add_argument(
            "--delta-t-ps",
            type=float,
            default=1.0,
            help=(
                "Time between selected trajectory frames in ps. "
                "Example: --delta-t-ps 0.25, which reports lag time in ps."
            ),
        )
        parser.add_argument(
            "--unwrap",
            action=argparse.BooleanOptionalAction,
            default=True,
            help="Unwrap coordinates across periodic boundaries when cell data exists. Example: --no-unwrap, which keeps wrapped coordinates.",
        )
    elif canonical == "get_diffusivity":
        parser.description = (
            "Estimate diffusivity from time-origin averaged MSD using the Einstein relation `MSD = 2*d*D*t`.\n"
            "This command first computes MSD averaged over selected atoms and valid time origins, then fits\n"
            "MSD versus lag time to estimate a diffusion coefficient for the selected atom group.\n\n"
            "Examples:\n"
            "  1. Plot diffusivity for selected atom ids:\n"
            "   reaxkit get_diffusivity --atom-ids 1 2 3 --max-lag 500 --delta-t-ps 1.0 --d 3 --plot single\n\n"
            "  2. Export oxygen diffusivity using 3D Einstein dimensionality:\n"
            "   reaxkit get_diffusivity --atom-types O --max-lag 800 --delta-t-ps 0.25 --d 3 --export diffusivity_oxygen.csv\n\n"
            "  3. Estimate diffusivity from sampled frames without PBC unwrapping:\n"
            "   reaxkit get_diffusivity --atom-ids 1 2 3 4 5 --frames 0:999:1 --max-lag 800 --delta-t-ps 1.0 --d 3 --no-unwrap --export diffusivity.csv"
        )
        parser.add_argument("--atom-ids", type=int, nargs="*", default=None, help="1-based atom ids. Example: --atom-ids 1 2 3, which restricts diffusivity estimates to those atoms.")
        parser.add_argument("--atom-types", nargs="*", default=None, help="Element symbols to include. Example: --atom-types O, which limits analysis to oxygen atoms.")
        parser.add_argument("--dims", nargs="*", default=("x", "y", "z"), help="Coordinate dimensions to include. Example: --dims x y z, which uses full 3D displacement.")
        parser.add_argument("--origin", default="first", help="Reference frame: 'first' or explicit index. Example: --origin first, which measures displacement from initial frame.")
        parser.add_argument("--d", type=float, default=3.0, help="Einstein dimensionality in MSD = 2*d*D*t. Example: --d 2, which applies 2D diffusivity relation.")
        parser.add_argument("--max-lag", type=int, default=None)
        parser.add_argument("--delta-t-ps", type=float, default=1.0)
        parser.add_argument(
            "--unwrap",
            action=argparse.BooleanOptionalAction,
            default=True,
            help="Unwrap coordinates across periodic boundaries when cell data exists. Example: --no-unwrap, which keeps wrapped coordinates.",
        )
    elif canonical == "get_rdf":
        parser.description = (
            "Compute radial distribution functions (RDF) for selected atom groups.\n"
            "Group A and B can be defined by atom ids or atom types, with configurable bins/radius.\n\n"
            "Examples:\n"
            "  1. Plot O-H RDF using type selectors:\n"
            "   reaxkit get_rdf --atom-types-a O --atom-types-b H --plot single\n\n"
            "  2. Save RDF for specific atom-id groups with higher resolution bins:\n"
            "   reaxkit get_rdf --atom-ids-a 1 2 --atom-ids-b 10 11 --bins 300 --save rdf_pairs.png\n\n"
            "  3. Plot frame-wise RDF subplots for Al-O:\n"
            "   reaxkit get_rdf --atom-types-a Al --atom-types-b O --frames 0 50 100 --plot subplot"
        )
        parser.add_argument("--atom-ids-a", type=int, nargs="*", default=None, help="1-based atom ids for group A. Example: --atom-ids-a 1 2, which defines source RDF group.")
        parser.add_argument("--atom-ids-b", type=int, nargs="*", default=None, help="1-based atom ids for group B. Example: --atom-ids-b 10 11, which defines target RDF group.")
        parser.add_argument("--atom-types-a", nargs="*", default=None, help="Element symbols for group A. Example: --atom-types-a O, which sets group A by atom type.")
        parser.add_argument("--atom-types-b", nargs="*", default=None, help="Element symbols for group B. Example: --atom-types-b H, which sets group B by atom type.")
        parser.add_argument("--bins", type=int, default=200, help="Number of RDF bins. Example: --bins 300, which increases radial histogram resolution.")
        parser.add_argument("--r-max", type=float, default=None, help="Maximum radius. Example: --r-max 8.0, which truncates RDF computation at radius 8.0.")
        parser.add_argument("--backend", choices=["freud", "ovito"], default="freud", help="RDF backend. Example: --backend ovito, which computes RDF using OVITO backend.")
    elif canonical == "get_rdf_property":
        parser.description = (
            "Compute RDF-derived properties across selected frames.\n"
            "Supported properties include first peak, dominant peak, area, and excess area.\n\n"
            "Examples:\n"
            "  1. Plot first-peak position for O-H pairs:\n"
            "   reaxkit get_rdf_property --property first_peak --atom-types-a O --atom-types-b H --plot single\n\n"
            "  2. Save RDF area series using legacy alias flag:\n"
            "   reaxkit get_rdf_property --prop area --atom-types-a Al --atom-types-b O --xaxis iter --save rdf_area.png\n\n"
            "  3. Export dominant-peak series on selected frames:\n"
            "   reaxkit get_rdf_property --property dominant_peak --frames 0 20 40 --export rdf_peak.csv"
        )
        parser.add_argument("--property", default=None, help="RDF property to extract. Example: --property first_peak, which returns first-peak position series.")
        parser.add_argument("--prop", choices=["first_peak", "dominant_peak", "area", "excess_area"], default=None, help="Legacy alias for --property. Example: --prop area, which requests integrated RDF area series.")
        parser.add_argument("--atom-ids-a", type=int, nargs="*", default=None, help="1-based atom ids for group A. Example: --atom-ids-a 1 2, which sets RDF group A atoms.")
        parser.add_argument("--atom-ids-b", type=int, nargs="*", default=None, help="1-based atom ids for group B. Example: --atom-ids-b 10 11, which sets RDF group B atoms.")
        parser.add_argument("--atom-types-a", nargs="*", default=None, help="Element symbols for group A. Example: --atom-types-a Al, which sets group A by type.")
        parser.add_argument("--atom-types-b", nargs="*", default=None, help="Element symbols for group B. Example: --atom-types-b O, which sets group B by type.")
        parser.add_argument("--bins", type=int, default=200, help="Number of RDF bins. Example: --bins 250, which refines radial resolution.")
        parser.add_argument("--r-max", type=float, default=None, help="Maximum radius. Example: --r-max 10.0, which sets RDF cutoff radius.")
        parser.add_argument("--backend", choices=["freud", "ovito"], default="freud", help="RDF backend. Example: --backend freud, which uses freud-based RDF implementation.")
    elif canonical == "get_voronoi":
        parser.description = (
            "Compute per-atom Voronoi metrics or Voronoi diagrams for selected frames.\n"
            "Use metrics mode for scalar series (e.g., volume) and diagram mode for cell geometry visualization.\n"
            "For a more discussion on voronoi plots, see the main documentaion at: https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.voronoi_plot_2d.html\n\n"
            "Examples:\n"
            "  1. Export Voronoi metrics table:\n"
            "   reaxkit get_voronoi --frames 0 10 20 --export voronoi.csv\n\n"
            "  2. Plot Voronoi metric series for selected atom types:\n"
            "   reaxkit get_voronoi --plot single --plot-target metrics --atom-types O --frames 0:100:5\n\n"
            "  3. Plot a 2D Voronoi diagram projection:\n"
            "   reaxkit get_voronoi --plot single --plot-target diagram --diagram-dim 2d --projection xy --frames 10\n\n"
        )
        parser.add_argument("--atom-ids", type=int, nargs="*", default=None, help="1-based atom ids. Example: --atom-ids 1 2 3, which limits Voronoi analysis to those atoms.")
        parser.add_argument("--atom-types", nargs="*", default=None, help="Element symbols to include. Example: --atom-types O, which restricts analysis to oxygen atoms.")
        parser.add_argument("--backend", choices=["scipy", "pyvoro"], default="scipy", help="Voronoi backend. Example: --backend pyvoro, which uses pyvoro-based computation.")
        parser.add_argument("--plot-target", choices=["metrics", "diagram"], default="metrics", help="Plot Voronoi metrics or cell diagrams. Example: --plot-target diagram, which switches to geometry visualization mode.")
        parser.add_argument("--diagram-dim", choices=["2d", "3d"], default="2d", help="Diagram dimensionality. Example: --diagram-dim 3d, which renders 3D Voronoi cell wireframes.")
        parser.add_argument("--projection", choices=["xy", "xz", "yz"], default="xy", help="Projection plane for 2D diagram mode. Example: --projection xz, which projects 2D diagram onto XZ plane.")
    else:
        raise KeyError(f"Unsupported trajectory command '{canonical}'.")

    return parser


def _safe_obj(value):
    """Safe obj."""
    if isinstance(value, (list, tuple, dict)):
        return value
    if isinstance(value, str):
        txt = value.strip()
        if txt.startswith("[") or txt.startswith("{") or txt.startswith("("):
            try:
                return ast.literal_eval(txt)
            except Exception:
                return value
    return value


def _project_xyz(point: Sequence[float], plane: str) -> tuple[float, float]:
    """Project xyz."""
    x, y, z = float(point[0]), float(point[1]), float(point[2])
    if plane == "xz":
        return x, z
    if plane == "yz":
        return y, z
    return x, y


def _collect_cell_edges(vertices, faces) -> list[tuple[int, int]]:
    """Collect cell edges."""
    verts = _safe_obj(vertices)
    face_rows = _safe_obj(faces)
    if not isinstance(verts, list) or not isinstance(face_rows, list):
        return []
    n_vertices = len(verts)
    edges: set[tuple[int, int]] = set()
    for face in face_rows:
        if isinstance(face, dict):
            idxs = face.get("vertex_indices", [])
        else:
            idxs = []
        idxs = _safe_obj(idxs)
        if not isinstance(idxs, list):
            continue
        clean = [int(v) for v in idxs if isinstance(v, (int, np.integer)) or (isinstance(v, str) and v.isdigit())]
        if len(clean) < 2:
            continue
        for i in range(len(clean)):
            a = int(clean[i])
            b = int(clean[(i + 1) % len(clean)])
            if a < 0 or b < 0 or a >= n_vertices or b >= n_vertices or a == b:
                continue
            edge = (a, b) if a < b else (b, a)
            edges.add(edge)
    return sorted(edges)


def _clip_segment_to_bbox(
    p0: tuple[float, float],
    p1: tuple[float, float],
    *,
    xmin: float,
    xmax: float,
    ymin: float,
    ymax: float,
) -> tuple[tuple[float, float], tuple[float, float]] | None:
    """Clip a segment to an axis-aligned bounding box (Liang-Barsky)."""
    x0, y0 = float(p0[0]), float(p0[1])
    x1, y1 = float(p1[0]), float(p1[1])
    dx = x1 - x0
    dy = y1 - y0

    t0, t1 = 0.0, 1.0
    checks = (
        (-dx, x0 - xmin),
        (dx, xmax - x0),
        (-dy, y0 - ymin),
        (dy, ymax - y0),
    )
    for p, q in checks:
        if abs(p) < 1e-12:
            if q < 0.0:
                return None
            continue
        r = q / p
        if p < 0.0:
            if r > t1:
                return None
            t0 = max(t0, r)
        else:
            if r < t0:
                return None
            t1 = min(t1, r)
    if t0 > t1:
        return None
    a = (x0 + t0 * dx, y0 + t0 * dy)
    b = (x0 + t1 * dx, y0 + t1 * dy)
    return a, b


def _ray_to_bbox_endpoint(
    origin: np.ndarray,
    direction: np.ndarray,
    *,
    xmin: float,
    xmax: float,
    ymin: float,
    ymax: float,
) -> tuple[float, float] | None:
    """Intersect a forward ray with the bounding box and return the nearest hit."""
    ox, oy = float(origin[0]), float(origin[1])
    dx, dy = float(direction[0]), float(direction[1])
    candidates: list[tuple[float, float, float]] = []

    if abs(dx) > 1e-12:
        for x in (xmin, xmax):
            t = (x - ox) / dx
            if t <= 0.0:
                continue
            y = oy + t * dy
            if ymin - 1e-9 <= y <= ymax + 1e-9:
                candidates.append((t, x, y))
    if abs(dy) > 1e-12:
        for y in (ymin, ymax):
            t = (y - oy) / dy
            if t <= 0.0:
                continue
            x = ox + t * dx
            if xmin - 1e-9 <= x <= xmax + 1e-9:
                candidates.append((t, x, y))

    if not candidates:
        return None
    _, xh, yh = min(candidates, key=lambda v: v[0])
    return float(xh), float(yh)


def _voronoi_diagram_payload_2d(table: pd.DataFrame, args: argparse.Namespace) -> dict[str, object] | None:
    """Voronoi diagram payload 2d."""
    frame_groups = list(table.groupby("frame_index", sort=True))
    if not frame_groups:
        return None
    plane = str(getattr(args, "projection", "xy")).strip().lower()

    def frame_series(dfi: pd.DataFrame) -> tuple[list[dict[str, object]], tuple[float, float, float, float] | None]:
        """Frame series.

        Execute the workflow function for this command path and return the
        computed result for downstream CLI handling.

        Parameters
        -----
        dfi : Any
            Function argument.

        Returns
        -----
        list[dict[str, object]]
            Function return value.

        Examples
        -----
        >>> # See workflow CLI usage for concrete examples.
        """
        points_2d: list[tuple[float, float]] = []
        for _, row in dfi.iterrows():
            site = _safe_obj(row.get("site_position"))
            if not isinstance(site, (list, tuple)) or len(site) < 3:
                continue
            px, py = _project_xyz(site, plane)
            if np.isfinite(px) and np.isfinite(py):
                points_2d.append((float(px), float(py)))
        if len(points_2d) < 3:
            return [], None

        pts = np.asarray(points_2d, dtype=float)
        xmin, ymin = np.min(pts, axis=0)
        xmax, ymax = np.max(pts, axis=0)
        span_x = float(xmax - xmin)
        span_y = float(ymax - ymin)
        pad_x = max(1e-6, 0.08 * span_x) if span_x > 0 else 1.0
        pad_y = max(1e-6, 0.08 * span_y) if span_y > 0 else 1.0
        xmin -= pad_x
        xmax += pad_x
        ymin -= pad_y
        ymax += pad_y

        try:
            from scipy.spatial import Voronoi as ScipyVoronoi

            # QJ helps with near-collinear projected slabs.
            vor2d = ScipyVoronoi(pts, qhull_options="Qbb Qc QJ")
        except Exception:
            return [], (xmin, xmax, ymin, ymax)

        center = np.mean(vor2d.points, axis=0)
        xs: list[float | None] = []
        ys: list[float | None] = []
        for (p1, p2), rv in zip(vor2d.ridge_points, vor2d.ridge_vertices):
            if len(rv) != 2:
                continue
            a, b = int(rv[0]), int(rv[1])
            if a >= 0 and b >= 0:
                v0 = vor2d.vertices[a]
                v1 = vor2d.vertices[b]
                clipped = _clip_segment_to_bbox(
                    (float(v0[0]), float(v0[1])),
                    (float(v1[0]), float(v1[1])),
                    xmin=xmin,
                    xmax=xmax,
                    ymin=ymin,
                    ymax=ymax,
                )
                if clipped is None:
                    continue
                (cx0, cy0), (cx1, cy1) = clipped
                xs.extend([cx0, cx1, None])
                ys.extend([cy0, cy1, None])
                continue

            finite_idx = a if a >= 0 else b
            if finite_idx < 0:
                continue
            v = np.asarray(vor2d.vertices[finite_idx], dtype=float)
            t = vor2d.points[int(p2)] - vor2d.points[int(p1)]
            nrm = np.linalg.norm(t)
            if not np.isfinite(nrm) or nrm <= 0.0:
                continue
            t /= nrm
            n = np.array([-t[1], t[0]], dtype=float)
            midpoint = vor2d.points[[int(p1), int(p2)]].mean(axis=0)
            sign = np.sign(np.dot(midpoint - center, n))
            if sign == 0:
                sign = 1.0
            direction = sign * n
            endpoint = _ray_to_bbox_endpoint(v, direction, xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax)
            if endpoint is None:
                continue
            clipped = _clip_segment_to_bbox(
                (float(v[0]), float(v[1])),
                endpoint,
                xmin=xmin,
                xmax=xmax,
                ymin=ymin,
                ymax=ymax,
            )
            if clipped is None:
                continue
            (cx0, cy0), (cx1, cy1) = clipped
            xs.extend([cx0, cx1, None])
            ys.extend([cy0, cy1, None])

        if not xs:
            return [], (xmin, xmax, ymin, ymax)

        series: list[dict[str, object]] = [
            {
                "x": xs,
                "y": ys,
                "label": "Voronoi edges",
                "linewidth": 1.0,
                "alpha": 0.9,
                "marker": None,
            }
        ]
        series.append(
            {
                "x": pts[:, 0].tolist(),
                "y": pts[:, 1].tolist(),
                "label": "sites",
                "linewidth": 0.0,
                "marker": ".",
                "markersize": 3.0,
                "alpha": 0.6,
            }
        )
        return series, (xmin, xmax, ymin, ymax)

    if getattr(args, "plot", None) == "subplot":
        subplots: list[list[dict[str, object]]] = []
        titles: list[str] = []
        for frame_index, dfi in frame_groups:
            ser, _bbox = frame_series(dfi)
            if not ser:
                continue
            iter_vals = dfi["iter"].dropna().unique().tolist() if "iter" in dfi.columns else []
            iter_txt = f", iter {int(iter_vals[0])}" if iter_vals else ""
            titles.append(f"frame {int(frame_index)}{iter_txt}")
            subplots.append(ser)
        if not subplots:
            return None
        return {
            "plot_type": "multi_subplots",
            "subplots": subplots,
            "title": titles,
            "xlabel": f"{plane[0]} (A)",
            "ylabel": f"{plane[1]} (A)",
            "legend": False,
            "grid": getattr(args, "grid", None),
        }

    frame_index, dfi = frame_groups[0]
    ser, bbox = frame_series(dfi)
    if not ser:
        return None
    iter_vals = dfi["iter"].dropna().unique().tolist() if "iter" in dfi.columns else []
    iter_txt = f", iter {int(iter_vals[0])}" if iter_vals else ""
    payload = {
        "plot_type": "single_plot",
        "series": ser,
        "title": f"Voronoi Diagram 2D ({plane}), frame {int(frame_index)}{iter_txt}",
        "xlabel": f"{plane[0]} (A)",
        "ylabel": f"{plane[1]} (A)",
        "legend": True,
        "aspect": "equal",
    }
    if bbox is not None:
        xmin, xmax, ymin, ymax = bbox
        payload["xlim"] = [float(xmin), float(xmax)]
        payload["ylim"] = [float(ymin), float(ymax)]
    return payload


def _voronoi_diagram_payload_3d(table: pd.DataFrame, args: argparse.Namespace) -> dict[str, object] | None:
    """Voronoi diagram payload 3d."""
    frame_groups = list(table.groupby("frame_index", sort=True))
    if not frame_groups:
        return None

    def frame_payload(frame_index: int, dfi: pd.DataFrame) -> dict[str, object] | None:
        """Frame payload.

        Execute the workflow function for this command path and return the
        computed result for downstream CLI handling.

        Parameters
        -----
        frame_index : Any
            Function argument.
        dfi : Any
            Function argument.

        Returns
        -----
        dict[str, object] | None
            Function return value.

        Examples
        -----
        >>> # See workflow CLI usage for concrete examples.
        """
        segments: list[list[list[float]]] = []
        segment_keys: set[tuple[tuple[float, float, float], tuple[float, float, float]]] = set()
        points: list[list[float]] = []
        values: list[float] = []

        # Unbounded cells tend to produce extreme geometry artifacts; exclude them for plotting.
        dfi_cells = dfi
        if "is_bounded" in dfi.columns:
            bounded = dfi[dfi["is_bounded"] == True].copy()
            if not bounded.empty:
                dfi_cells = bounded

        def _add_segment(p0: np.ndarray, p1: np.ndarray) -> None:
            if p0.shape != (3,) or p1.shape != (3,):
                return
            if not (np.isfinite(p0).all() and np.isfinite(p1).all()):
                return
            a = (float(p0[0]), float(p0[1]), float(p0[2]))
            b = (float(p1[0]), float(p1[1]), float(p1[2]))
            key = tuple(sorted((tuple(round(v, 8) for v in a), tuple(round(v, 8) for v in b))))
            if key in segment_keys:
                return
            segment_keys.add(key)
            segments.append([[a[0], a[1], a[2]], [b[0], b[1], b[2]]])

        def _hull_edges(vertices_obj) -> list[tuple[int, int]]:
            verts = _safe_obj(vertices_obj)
            if not isinstance(verts, list) or len(verts) < 4:
                return []
            arr = np.asarray(verts, dtype=float)
            if arr.ndim != 2 or arr.shape[1] < 3:
                return []
            arr = arr[:, :3]
            if not np.isfinite(arr).all():
                return []
            try:
                from scipy.spatial import ConvexHull

                hull = ConvexHull(arr, qhull_options="QJ")
            except Exception:
                return []
            edges: set[tuple[int, int]] = set()
            for simplex in np.asarray(hull.simplices, dtype=int):
                if simplex.ndim != 1 or simplex.size < 2:
                    continue
                for i in range(simplex.size):
                    a = int(simplex[i])
                    b = int(simplex[(i + 1) % simplex.size])
                    if a == b:
                        continue
                    edge = (a, b) if a < b else (b, a)
                    edges.add(edge)
            return sorted(edges)

        for _, row in dfi_cells.iterrows():
            vertices = _safe_obj(row.get("vertices"))
            edges = _hull_edges(vertices)
            if not edges:
                continue
            arr = np.asarray(vertices, dtype=float)
            if arr.ndim != 2 or arr.shape[1] < 3 or not np.isfinite(arr[:, :3]).all():
                continue
            for a, b in edges:
                if a < 0 or b < 0 or a >= arr.shape[0] or b >= arr.shape[0]:
                    continue
                _add_segment(arr[a, :3], arr[b, :3])

        for _, row in dfi.iterrows():
            site = _safe_obj(row.get("site_position"))
            if isinstance(site, (list, tuple)) and len(site) >= 3:
                p = np.asarray([site[0], site[1], site[2]], dtype=float)
                if np.isfinite(p).all():
                    points.append([float(p[0]), float(p[1]), float(p[2])])
                    values.append(float(pd.to_numeric(row.get("voronoi_volume"), errors="coerce")))

        if not segments and not points:
            return None

        xlim = ylim = zlim = None
        box_aspect = None
        if points:
            parr = np.asarray(points, dtype=float)
            mins = np.nanmin(parr, axis=0)
            maxs = np.nanmax(parr, axis=0)
            spans = maxs - mins
            pads = np.where(spans > 0.0, 0.08 * spans, 1.0)
            lo = mins - pads
            hi = maxs + pads
            xlim = [float(lo[0]), float(hi[0])]
            ylim = [float(lo[1]), float(hi[1])]
            zlim = [float(lo[2]), float(hi[2])]
            safe_spans = np.where(spans > 1e-12, spans, 1.0)
            box_aspect = [float(safe_spans[0]), float(safe_spans[1]), float(safe_spans[2])]

        iter_vals = dfi["iter"].dropna().unique().tolist() if "iter" in dfi.columns else []
        iter_txt = f", iter {int(iter_vals[0])}" if iter_vals else ""
        return {
            "segments": segments,
            "points": points,
            "values": values,
            "title": f"frame {int(frame_index)}{iter_txt}",
            "xlim": xlim,
            "ylim": ylim,
            "zlim": zlim,
            "box_aspect": box_aspect,
        }

    payloads: list[dict[str, object]] = []
    for frame_index, dfi in frame_groups:
        p = frame_payload(int(frame_index), dfi)
        if p is not None:
            payloads.append(p)
    if not payloads:
        return None

    if getattr(args, "plot", None) == "subplot":
        return {
            "plot_type": "wireframe3d_subplots",
            "subplots": payloads,
            "title": "Voronoi Diagram 3D",
            "grid": getattr(args, "grid", None),
            "show_colorbar": False,
        }
    one = payloads[0]
    return {
        "plot_type": "wireframe3d_plot",
        "segments": one["segments"],
        "points": one["points"],
        "values": one["values"],
        "title": f"Voronoi Diagram 3D ({one['title']})",
        "show_colorbar": True,
        "colorbar_label": "Voronoi volume",
    }


def _plot_payload(command: str, result, args: argparse.Namespace) -> dict[str, object] | None:
    """Plot payload."""
    table = result.table
    if table.empty:
        return None

    if command == "get_msd":
        if "msd" not in table.columns:
            return None

        work = table.copy()

        if "time_ps" in work.columns:
            x_col = "time_ps"
            xlabel = "Time lag (ps)"
        elif "lag_frame" in work.columns:
            x_col = "lag_frame"
            xlabel = "Lag (frames)"
        else:
            return None

        work = work.sort_values(x_col)

        payload = {
            "x": pd.to_numeric(work[x_col], errors="coerce").tolist(),
            "y": pd.to_numeric(work["msd"], errors="coerce").tolist(),
            "label": "MSD",
        }

        return {
            "plot_type": "single_plot",
            "series": [payload],
            "xlabel": xlabel,
            "ylabel": "MSD (A^2)",
            "title": "Time-Origin Averaged MSD",
            "legend": False,
        }

    if command == "get_diffusivity":
        if "atom_id" not in table.columns or "get_diffusivity" not in table.columns:
            return None
        work = table.sort_values("atom_id")
        return {
            "plot_type": "single_plot",
            "x": pd.to_numeric(work["atom_id"], errors="coerce").tolist(),
            "y": pd.to_numeric(work["diffusivity"], errors="coerce").tolist(),
            "xlabel": "Atom ID",
            "ylabel": "Diffusivity",
            "title": "Diffusivity by Atom",
        }

    if command == "get_dihedral":
        x_col = "frame_index"
        xlabel = "Frame Index"
        if getattr(args, "xaxis", "frame") == "iter" and "iter" in table.columns:
            x_col = "iter"
            xlabel = "Iteration"
        elif getattr(args, "xaxis", "frame") == "time":
            source = "iter" if "iter" in table.columns else "frame_index"
            converted, xlabel = convert_xaxis(table[source].to_numpy(dtype=int), "time")
            table = table.copy()
            table["x_axis"] = np.asarray(converted, dtype=float)
            x_col = "x_axis"

        units = str(table["units"].iloc[0]) if "units" in table.columns and not table.empty else ""
        return {
            "plot_type": "single_plot",
            "x": table[x_col].tolist(),
            "y": table["dihedral"].tolist(),
            "xlabel": xlabel,
            "ylabel": f"Dihedral ({units})" if units else "Dihedral",
            "title": "Dihedral Angle",
        }

    if command == "get_rdf":
        if "frame_index" not in table.columns or "iter" not in table.columns:
            return {
                "plot_type": "single_plot",
                "x": table["r"].tolist(),
                "y": table["g"].tolist(),
                "xlabel": "r (A)",
                "ylabel": "g(r)",
                "title": "Radial Distribution Function",
            }

        grouped = table.groupby(["frame_index", "iter"], sort=True)
        series = []
        subplots = []
        for (frame_index, iter_value), dfi in grouped:
            dfi = dfi.sort_values("r")
            payload = {
                "x": dfi["r"].tolist(),
                "y": dfi["g"].tolist(),
                "label": f"frame {frame_index} (iter {iter_value})",
            }
            series.append(payload)
            subplots.append([payload])

        if getattr(args, "plot", None) == "subplot":
            return {
                "plot_type": "multi_subplots",
                "subplots": subplots,
                "xlabel": "r (A)",
                "ylabel": "g(r)",
                "title": "Radial Distribution Function",
                "legend": False,
                "grid": getattr(args, "grid", None),
            }

        return {
            "plot_type": "single_plot",
            "series": series,
            "xlabel": "r (A)",
            "ylabel": "g(r)",
            "title": "Radial Distribution Function",
            "legend": True,
        }

    if command == "get_rdf_property":
        property_name = getattr(args, "property", None) or getattr(args, "prop", None) or "first_peak"
        property_name = str(property_name).strip().lower()
        x_col = "frame_index"
        xlabel = "Frame Index"
        if getattr(args, "xaxis", "frame") == "iter" and "iter" in table.columns:
            x_col = "iter"
            xlabel = "Iteration"

        property_columns = {
            "first_peak": ("r_first_peak", "First Peak Position"),
            "dominant_peak": ("r_peak", "Dominant Peak Position"),
            "area": ("area", "RDF Area"),
            "excess_area": ("excess_area", "RDF Excess Area"),
        }
        y_col, title = property_columns.get(property_name, (None, None))
        if y_col is None or y_col not in table.columns:
            return None
        return {
            "plot_type": "single_plot",
            "x": table[x_col].tolist(),
            "y": table[y_col].tolist(),
            "xlabel": xlabel,
            "ylabel": y_col,
            "title": title,
        }

    if command == "get_voronoi":
        if getattr(args, "plot_target", "metrics") == "diagram":
            if "vertices" not in table.columns or "faces" not in table.columns:
                return None
            if getattr(args, "diagram_dim", "2d") == "3d":
                return _voronoi_diagram_payload_3d(table, args)
            return _voronoi_diagram_payload_2d(table, args)

        if "voronoi_volume" not in table.columns:
            return None
        work = table[table["is_bounded"] == True].copy() if "is_bounded" in table.columns else table.copy()
        if work.empty:
            return None

        if getattr(args, "xaxis", "frame") == "iter" and "iter" in work.columns:
            x_col = "iter"
            xlabel = "Iteration"
        elif getattr(args, "xaxis", "frame") == "time" and "iter" in work.columns:
            converted, xlabel = convert_xaxis(work["iter"].to_numpy(dtype=int), "time")
            work["x_axis"] = np.asarray(converted, dtype=float)
            x_col = "x_axis"
        else:
            x_col = "frame_index"
            xlabel = "Frame Index"

        series = []
        subplots = []
        if "atom_id" in work.columns:
            for atom_id, dfi in work.groupby("atom_id", sort=True):
                dfi = dfi.sort_values("frame_index")
                payload = {
                    "x": dfi[x_col].tolist(),
                    "y": pd.to_numeric(dfi["voronoi_volume"], errors="coerce").tolist(),
                    "label": f"atom {atom_id}",
                }
                series.append(payload)
                subplots.append([payload])
        else:
            dfi = work.sort_values("frame_index")
            series.append(
                {
                    "x": dfi[x_col].tolist(),
                    "y": pd.to_numeric(dfi["voronoi_volume"], errors="coerce").tolist(),
                    "label": "voronoi_volume",
                }
            )
            subplots.append(series)

        if getattr(args, "plot", None) == "subplot":
            return {
                "plot_type": "multi_subplots",
                "subplots": subplots,
                "xlabel": xlabel,
                "ylabel": "Voronoi Volume",
                "title": "Voronoi Volume",
                "legend": False,
                "grid": getattr(args, "grid", None),
            }
        return {
            "plot_type": "single_plot",
            "series": series,
            "xlabel": xlabel,
            "ylabel": "Voronoi Volume",
            "title": "Voronoi Volume",
            "legend": len(series) > 1,
        }

    value_columns = [c for c in table.columns if c not in {"frame_index", "iter"}]
    if not value_columns:
        return None
    y_col = value_columns[0]
    x_col = "frame_index"
    if getattr(args, "xaxis", "frame") == "iter" and "iter" in table.columns:
        x_col = "iter"
    return {
        "plot_type": "single_plot",
        "x": table[x_col].tolist(),
        "y": table[y_col].tolist(),
        "xlabel": "Iteration" if x_col == "iter" else "Frame Index",
        "ylabel": y_col,
        "title": f"{command.replace('_', ' ').title()}",
    }


def run_main(command: str, args: argparse.Namespace) -> int:
    """Run a direct trajectory command."""
    canonical = resolve_command_name(command, task_names=ALL_COMMANDS)
    task_key = canonical
    if canonical == "get_voronoi":
        backend = str(getattr(args, "backend", "scipy")).strip().lower()
        plot_target = str(getattr(args, "plot_target", "metrics")).strip().lower()
        if plot_target == "diagram":
            backend_to_task = {"scipy": "get_voronoi_geometry_scipy", "pyvoro": "get_voronoi_geometry_pyvoro"}
        else:
            backend_to_task = {"scipy": "get_voronoi_scipy", "pyvoro": "get_voronoi_pyvoro"}
        if backend not in backend_to_task:
            raise ValueError("voronoi backend must be 'scipy' or 'pyvoro'")
        task_key = backend_to_task[backend]
    task_cls = TASK_REGISTRY[task_key]
    request = REQUEST_BUILDERS[canonical](args)

    executor = AnalysisExecutor()
    result = executor.run(task_cls(), request, vars(args))
    present_result(canonical, result, args, plot_payload_builder=_plot_payload)
    return 0
