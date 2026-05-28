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
from reaxkit.core.analysis_executor import AnalysisExecutor
from reaxkit.core.analysis_task_registry import TASK_REGISTRY
from reaxkit.core.command_alias_resolver import resolve_command_name
from reaxkit.core.frame_utils import parse_frame_indices
from reaxkit.core.storage_layout import add_storage_cli_arguments
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
        origin=args.origin,
        frames=parse_frame_indices(args.frames),
        every=args.every,
        unwrap=bool(args.unwrap),
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
            "Compute mean-squared displacement (MSD) for selected atoms.\n"
            "MSD is computed over selected frames and can be filtered by atom ids or atom types.\n\n"
            "Examples:\n"
            "  1. Plot MSD for selected atom ids:\n"
            "   reaxkit get_msd --atom-ids 1 2 3 --plot single\n\n"
            "  2. Save oxygen-only MSD on time axis:\n"
            "   reaxkit get_msd --atom-types O --xaxis time --save msd_oxygen.png\n\n"
            "  3. Export MSD for one atom on selected frames:\n"
            "   reaxkit get_msd --atom-ids 5 --frames 0 10 20 --export msd_atom5.csv"
        )
        parser.add_argument("--atom-ids", type=int, nargs="*", default=None, help="1-based atom ids. Example: --atom-ids 1 2 3, which restricts MSD to those atoms.")
        parser.add_argument("--atom-types", nargs="*", default=None, help="Element symbols to include. Example: --atom-types O, which computes MSD for oxygen atoms only.")
        parser.add_argument("--dims", nargs="*", default=("x", "y", "z"), help="Coordinate dimensions to include. Example: --dims x y, which computes MSD using in-plane displacement.")
        parser.add_argument("--origin", default="first", help="Reference frame: 'first' or explicit index. Example: --origin first, which measures displacement from initial frame.")
        parser.add_argument(
            "--unwrap",
            action=argparse.BooleanOptionalAction,
            default=True,
            help="Unwrap coordinates across periodic boundaries when cell data exists. Example: --no-unwrap, which keeps wrapped coordinates.",
        )
    elif canonical == "get_diffusivity":
        parser.description = (
            "Estimate per-atom diffusivity from Einstein relation `MSD = 2*d*D*t`.\n"
            "This command fits/derives diffusivity using selected dimensions, atoms, and frame windows.\n\n"
            "Examples:\n"
            "  1. Plot diffusivity for selected atom ids:\n"
            "   reaxkit get_diffusivity --atom-ids 1 2 3 --plot single\n\n"
            "  2. Export oxygen diffusivity using 3D Einstein dimensionality:\n"
            "   reaxkit get_diffusivity --atom-types O --d 3 --export diffusivity_oxygen.csv\n\n"
            "  3. Save atom-specific diffusivity with frame sampling:\n"
            "   reaxkit get_diffusivity --atom-ids 5 --frames 0:100:5 --d 2 --save diffusivity_atom5.png"
        )
        parser.add_argument("--atom-ids", type=int, nargs="*", default=None, help="1-based atom ids. Example: --atom-ids 1 2 3, which restricts diffusivity estimates to those atoms.")
        parser.add_argument("--atom-types", nargs="*", default=None, help="Element symbols to include. Example: --atom-types O, which limits analysis to oxygen atoms.")
        parser.add_argument("--dims", nargs="*", default=("x", "y", "z"), help="Coordinate dimensions to include. Example: --dims x y z, which uses full 3D displacement.")
        parser.add_argument("--origin", default="first", help="Reference frame: 'first' or explicit index. Example: --origin first, which measures displacement from initial frame.")
        parser.add_argument("--d", type=float, default=3.0, help="Einstein dimensionality in MSD = 2*d*D*t. Example: --d 2, which applies 2D diffusivity relation.")
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
            "Use metrics mode for scalar series (e.g., volume) and diagram mode for cell geometry visualization.\n\n"
            "Examples:\n"
            "  1. Export Voronoi metrics table:\n"
            "   reaxkit get_voronoi --frames 0 10 20 --export voronoi.csv\n\n"
            "  2. Plot Voronoi metric series for selected atom types:\n"
            "   reaxkit get_voronoi --plot single --plot-target metrics --atom-types O --frames 0:100:5\n\n"
            "  3. Plot a 2D Voronoi diagram projection:\n"
            "   reaxkit get_voronoi --plot single --plot-target diagram --diagram-dim 2d --projection xy --frames 10\n\n"
            "  4. Plot 3D Voronoi diagrams as subplots over frame samples:\n"
            "   reaxkit get_voronoi --plot subplot --plot-target diagram --diagram-dim 3d --frames 0:20:5"
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


def _voronoi_diagram_payload_2d(table: pd.DataFrame, args: argparse.Namespace) -> dict[str, object] | None:
    """Voronoi diagram payload 2d."""
    frame_groups = list(table.groupby("frame_index", sort=True))
    if not frame_groups:
        return None
    plane = str(getattr(args, "projection", "xy")).strip().lower()

    def frame_series(dfi: pd.DataFrame) -> list[dict[str, object]]:
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
        series: list[dict[str, object]] = []
        for _, row in dfi.iterrows():
            vertices = _safe_obj(row.get("vertices"))
            faces = _safe_obj(row.get("faces"))
            if not isinstance(vertices, list) or not isinstance(faces, list):
                continue
            edges = _collect_cell_edges(vertices, faces)
            if not edges:
                continue
            xs: list[float | None] = []
            ys: list[float | None] = []
            for a, b in edges:
                va = _safe_obj(vertices[a])
                vb = _safe_obj(vertices[b])
                if not isinstance(va, (list, tuple)) or not isinstance(vb, (list, tuple)) or len(va) < 3 or len(vb) < 3:
                    continue
                x1, y1 = _project_xyz(va, plane)
                x2, y2 = _project_xyz(vb, plane)
                xs.extend([x1, x2, None])
                ys.extend([y1, y2, None])
            if xs:
                series.append(
                    {
                        "x": xs,
                        "y": ys,
                        "label": f"atom {int(row['atom_id'])}",
                        "linewidth": 0.8,
                        "alpha": 0.75,
                    }
                )
        return series

    if getattr(args, "plot", None) == "subplot":
        subplots: list[list[dict[str, object]]] = []
        titles: list[str] = []
        for frame_index, dfi in frame_groups:
            ser = frame_series(dfi)
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
    ser = frame_series(dfi)
    if not ser:
        return None
    iter_vals = dfi["iter"].dropna().unique().tolist() if "iter" in dfi.columns else []
    iter_txt = f", iter {int(iter_vals[0])}" if iter_vals else ""
    return {
        "plot_type": "single_plot",
        "series": ser,
        "title": f"Voronoi Diagram 2D ({plane}), frame {int(frame_index)}{iter_txt}",
        "xlabel": f"{plane[0]} (A)",
        "ylabel": f"{plane[1]} (A)",
        "legend": len(ser) <= 20,
    }


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
        points: list[list[float]] = []
        values: list[float] = []
        for _, row in dfi.iterrows():
            vertices = _safe_obj(row.get("vertices"))
            faces = _safe_obj(row.get("faces"))
            if isinstance(vertices, list) and isinstance(faces, list):
                edges = _collect_cell_edges(vertices, faces)
                for a, b in edges:
                    va = _safe_obj(vertices[a])
                    vb = _safe_obj(vertices[b])
                    if not isinstance(va, (list, tuple)) or not isinstance(vb, (list, tuple)) or len(va) < 3 or len(vb) < 3:
                        continue
                    segments.append([[float(va[0]), float(va[1]), float(va[2])], [float(vb[0]), float(vb[1]), float(vb[2])]])

            site = _safe_obj(row.get("site_position"))
            if isinstance(site, (list, tuple)) and len(site) >= 3:
                points.append([float(site[0]), float(site[1]), float(site[2])])
                values.append(float(pd.to_numeric(row.get("voronoi_volume"), errors="coerce")))
        if not segments and not points:
            return None
        iter_vals = dfi["iter"].dropna().unique().tolist() if "iter" in dfi.columns else []
        iter_txt = f", iter {int(iter_vals[0])}" if iter_vals else ""
        return {
            "segments": segments,
            "points": points,
            "values": values,
            "title": f"frame {int(frame_index)}{iter_txt}",
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
        grouped = table.groupby(["frame_index", "iter", "atom_id"], as_index=False)["msd"].mean()
        frame_values = np.sort(grouped["frame_index"].unique())
        xvals, xlabel = convert_xaxis(frame_values, getattr(args, "xaxis", "frame"))
        frame_to_x = dict(zip(frame_values, xvals))

        series = []
        subplots = []
        for atom_id in sorted(grouped["atom_id"].unique()):
            dfi = grouped[grouped["atom_id"] == atom_id].sort_values("frame_index")
            payload = {
                "x": [frame_to_x[idx] for idx in dfi["frame_index"].to_numpy()],
                "y": dfi["msd"].tolist(),
                "label": f"atom {atom_id}",
            }
            series.append(payload)
            subplots.append([payload])

        title = f"MSD of atoms: {', '.join(str(v) for v in sorted(grouped['atom_id'].unique()))}"
        if getattr(args, "plot", None) == "subplot":
            return {
                "plot_type": "multi_subplots",
                "subplots": subplots,
                "xlabel": xlabel,
                "ylabel": "A^2",
                "title": title,
                "legend": True,
                "grid": getattr(args, "grid", None),
            }
        return {
            "plot_type": "single_plot",
            "series": series,
            "xlabel": xlabel,
            "ylabel": "A^2",
            "title": title,
            "legend": True,
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
