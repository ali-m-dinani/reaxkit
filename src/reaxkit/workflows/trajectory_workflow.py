"""Direct command workflow for trajectory analyses."""

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

TRAJECTORY_COMMANDS = ("dihedral", "diffusivity", "msd", "rdf", "rdf_property", "voronoi")


def _add_runtime_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--engine", choices=["reaxff", "ams", "lammps"], default=None)
    parser.add_argument("--run-dir", "--dir", dest="run_dir", default=".", help="Run directory (fallback for detection)")
    parser.add_argument("--xmolout", "--file", dest="xmolout", default=None, help="Trajectory file path")
    parser.add_argument("--log", choices=["verbose", "quiet"], default="quiet", help="Logging level")
    add_storage_cli_arguments(parser)


def _add_presentation_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--plot", choices=["single", "subplot"], default=None, help="Render a plot")
    parser.add_argument("--show", action="store_true", help="Show the generated plot window")
    parser.add_argument("--save", default=None, help="Save the generated plot to a file path")
    parser.add_argument("--export", default=None, help="Write the result table to CSV")
    parser.add_argument("--grid", default=None, help="Subplot grid like 2x2 or 2*2")
    parser.add_argument("--xaxis", default="frame", choices=["iter", "frame", "time"], help="Quantity on x-axis")


def _add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--frames",
        nargs="*",
        default=None,
        help='Frames: "0,10,20", "0 10 20", "0:20", "0-20", or "0:20:2"',
    )
    parser.add_argument("--every", type=int, default=1, help="Frame stride")


def _build_msd_request(args: argparse.Namespace) -> MSDRequest:
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
    return DihedralRequest(
        atom_ids=args.atom_ids,
        frames=parse_frame_indices(args.frames),
        every=args.every,
        units=args.units,
        backend=args.backend,
    )


def _build_rdf_request(args: argparse.Namespace) -> RDFRequest:
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
    return VoronoiRequest(
        atom_ids=args.atom_ids,
        atom_types=args.atom_types,
        frames=parse_frame_indices(args.frames),
        every=args.every,
        backend=args.backend,
    )


REQUEST_BUILDERS: dict[str, Callable[[argparse.Namespace], object]] = {
    "dihedral": _build_dihedral_request,
    "diffusivity": _build_diffusivity_request,
    "msd": _build_msd_request,
    "rdf": _build_rdf_request,
    "rdf_property": _build_rdf_property_request,
    "voronoi": _build_voronoi_request,
}


def build_parser(parser: argparse.ArgumentParser, *, command: str) -> argparse.ArgumentParser:
    """Build the parser for a direct trajectory command."""
    canonical = resolve_command_name(command, task_names=TRAJECTORY_COMMANDS)
    parser.set_defaults(command=canonical)
    parser.set_defaults(progress=True)
    parser.formatter_class = argparse.RawTextHelpFormatter

    _add_runtime_arguments(parser)
    _add_presentation_arguments(parser)
    _add_common_arguments(parser)

    if canonical == "dihedral":
        parser.description = (
            "Compute a dihedral angle (atom1-atom2-atom3-atom4) over selected frames.\n\n"
            "Examples:\n"
            "  reaxkit dihedral --atom-ids 1 2 3 4 --plot single\n"
            "  reaxkit dihedral --atom-ids 8 3 4 9 --frames 0:500:10 --units rad --export dih.csv\n"
            "  reaxkit dihedral --atom-ids 5 7 9 11 --xaxis iter --save dihedral_iter.png"
        )
        parser.add_argument("--atom-ids", type=int, nargs=4, required=True, help="Exactly four 1-based atom ids")
        parser.add_argument("--units", choices=["deg", "rad"], default="deg", help="Output angle units")
        parser.add_argument("--backend", choices=["numpy", "mdanalysis"], default="numpy", help="Dihedral backend")
    elif canonical == "msd":
        parser.description = (
            "Compute mean-squared displacement for selected atoms.\n\n"
            "Examples:\n"
            "  reaxkit msd --atom-ids 1 2 3 --plot single\n"
            "  reaxkit msd --atom-types O --xaxis time --save msd_oxygen.png\n"
            "  reaxkit msd --atom-ids 5 --frames 0 10 20 --export msd_atom5.csv"
        )
        parser.add_argument("--atom-ids", type=int, nargs="*", default=None, help="1-based atom ids")
        parser.add_argument("--atom-types", nargs="*", default=None, help="Element symbols to include")
        parser.add_argument("--dims", nargs="*", default=("x", "y", "z"), help="Coordinate dimensions to include")
        parser.add_argument("--origin", default="first", help="Reference frame: 'first' or an explicit index")
        parser.add_argument(
            "--unwrap",
            action=argparse.BooleanOptionalAction,
            default=True,
            help="Unwrap coordinates across periodic boundaries when cell data exists",
        )
    elif canonical == "diffusivity":
        parser.description = (
            "Estimate per-atom diffusivity from Einstein relation MSD = 2*d*D*t.\n\n"
            "Examples:\n"
            "  reaxkit diffusivity --atom-ids 1 2 3 --plot single\n"
            "  reaxkit diffusivity --atom-types O --d 3 --export diffusivity_oxygen.csv\n"
            "  reaxkit diffusivity --atom-ids 5 --frames 0:100:5 --d 2 --save diffusivity_atom5.png"
        )
        parser.add_argument("--atom-ids", type=int, nargs="*", default=None, help="1-based atom ids")
        parser.add_argument("--atom-types", nargs="*", default=None, help="Element symbols to include")
        parser.add_argument("--dims", nargs="*", default=("x", "y", "z"), help="Coordinate dimensions to include")
        parser.add_argument("--origin", default="first", help="Reference frame: 'first' or an explicit index")
        parser.add_argument("--d", type=float, default=3.0, help="Einstein dimensionality in MSD = 2*d*D*t")
        parser.add_argument(
            "--unwrap",
            action=argparse.BooleanOptionalAction,
            default=True,
            help="Unwrap coordinates across periodic boundaries when cell data exists",
        )
    elif canonical == "rdf":
        parser.description = (
            "Compute radial distribution functions for selected atoms.\n\n"
            "Examples:\n"
            "  reaxkit rdf --atom-types-a O --atom-types-b H --plot single\n"
            "  reaxkit rdf --atom-ids-a 1 2 --atom-ids-b 10 11 --bins 300 --save rdf_pairs.png\n"
            "  reaxkit rdf --atom-types-a Al --atom-types-b O --frames 0 50 100 --plot subplot"
        )
        parser.add_argument("--atom-ids-a", type=int, nargs="*", default=None, help="1-based atom ids for group A")
        parser.add_argument("--atom-ids-b", type=int, nargs="*", default=None, help="1-based atom ids for group B")
        parser.add_argument("--atom-types-a", nargs="*", default=None, help="Element symbols for group A")
        parser.add_argument("--atom-types-b", nargs="*", default=None, help="Element symbols for group B")
        parser.add_argument("--bins", type=int, default=200, help="Number of RDF bins")
        parser.add_argument("--r-max", type=float, default=None, help="Maximum radius")
        parser.add_argument("--backend", choices=["freud", "ovito"], default="freud")
    elif canonical == "rdf_property":
        parser.description = (
            "Compute RDF-derived properties across selected frames.\n\n"
            "Examples:\n"
            "  reaxkit rdf_property --property first_peak --atom-types-a O --atom-types-b H --plot single\n"
            "  reaxkit rdf_property --prop area --atom-types-a Al --atom-types-b O --xaxis iter --save rdf_area.png\n"
            "  reaxkit rdf_property --property dominant_peak --frames 0 20 40 --export rdf_peak.csv"
        )
        parser.add_argument("--property", default=None, help="RDF property to extract")
        parser.add_argument("--prop", choices=["first_peak", "dominant_peak", "area", "excess_area"], default=None, help="Legacy alias for --property")
        parser.add_argument("--atom-ids-a", type=int, nargs="*", default=None, help="1-based atom ids for group A")
        parser.add_argument("--atom-ids-b", type=int, nargs="*", default=None, help="1-based atom ids for group B")
        parser.add_argument("--atom-types-a", nargs="*", default=None, help="Element symbols for group A")
        parser.add_argument("--atom-types-b", nargs="*", default=None, help="Element symbols for group B")
        parser.add_argument("--bins", type=int, default=200, help="Number of RDF bins")
        parser.add_argument("--r-max", type=float, default=None, help="Maximum radius")
        parser.add_argument("--backend", choices=["freud", "ovito"], default="freud")
    elif canonical == "voronoi":
        parser.description = (
            "Compute per-atom Voronoi metrics or Voronoi diagrams for selected frames.\n\n"
            "Examples:\n"
            "  reaxkit voronoi --frames 0 10 20 --export voronoi.csv\n"
            "  reaxkit voronoi --plot single --plot-target metrics --atom-types O --frames 0:100:5\n"
            "  reaxkit voronoi --plot single --plot-target diagram --diagram-dim 2d --projection xy --frames 10\n"
            "  reaxkit voronoi --plot subplot --plot-target diagram --diagram-dim 3d --frames 0:20:5"
        )
        parser.add_argument("--atom-ids", type=int, nargs="*", default=None, help="1-based atom ids")
        parser.add_argument("--atom-types", nargs="*", default=None, help="Element symbols to include")
        parser.add_argument("--backend", choices=["scipy", "pyvoro"], default="scipy", help="Voronoi backend")
        parser.add_argument("--plot-target", choices=["metrics", "diagram"], default="metrics", help="Plot Voronoi metrics or cell diagrams")
        parser.add_argument("--diagram-dim", choices=["2d", "3d"], default="2d", help="Diagram dimensionality")
        parser.add_argument("--projection", choices=["xy", "xz", "yz"], default="xy", help="Projection plane for 2D diagram mode")
    else:
        raise KeyError(f"Unsupported trajectory command '{canonical}'.")

    return parser


def _safe_obj(value):
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
    x, y, z = float(point[0]), float(point[1]), float(point[2])
    if plane == "xz":
        return x, z
    if plane == "yz":
        return y, z
    return x, y


def _collect_cell_edges(vertices, faces) -> list[tuple[int, int]]:
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
    frame_groups = list(table.groupby("frame_index", sort=True))
    if not frame_groups:
        return None
    plane = str(getattr(args, "projection", "xy")).strip().lower()

    def frame_series(dfi: pd.DataFrame) -> list[dict[str, object]]:
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
    frame_groups = list(table.groupby("frame_index", sort=True))
    if not frame_groups:
        return None

    def frame_payload(frame_index: int, dfi: pd.DataFrame) -> dict[str, object] | None:
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
    table = result.table
    if table.empty:
        return None

    if command == "msd":
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

    if command == "diffusivity":
        if "atom_id" not in table.columns or "diffusivity" not in table.columns:
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

    if command == "dihedral":
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

    if command == "rdf":
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

    if command == "rdf_property":
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

    if command == "voronoi":
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
    canonical = resolve_command_name(command, task_names=TRAJECTORY_COMMANDS)
    task_key = canonical
    if canonical == "voronoi":
        backend = str(getattr(args, "backend", "scipy")).strip().lower()
        plot_target = str(getattr(args, "plot_target", "metrics")).strip().lower()
        if plot_target == "diagram":
            backend_to_task = {"scipy": "voronoi_geometry_scipy", "pyvoro": "voronoi_geometry_pyvoro"}
        else:
            backend_to_task = {"scipy": "voronoi_scipy", "pyvoro": "voronoi_pyvoro"}
        if backend not in backend_to_task:
            raise ValueError("voronoi backend must be 'scipy' or 'pyvoro'")
        task_key = backend_to_task[backend]
    task_cls = TASK_REGISTRY[task_key]
    request = REQUEST_BUILDERS[canonical](args)

    executor = AnalysisExecutor()
    result = executor.run(task_cls(), request, vars(args))
    present_result(canonical, result, args, plot_payload_builder=_plot_payload)
    return 0
