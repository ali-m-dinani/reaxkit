"""Compatibility RDF helpers for the former handler-based API."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

from reaxkit.analysis.trajectory.rdf import _dominant_peak, _first_local_max
from reaxkit.engine.reaxff.io.xmolout_handler import XmoloutHandler


def _rdf_freud_frame(
    handler: XmoloutHandler,
    frame_index: int,
    *,
    types_a: Iterable[str] | None = None,
    types_b: Iterable[str] | None = None,
    r_max: float | None = None,
    bins: int = 200,
) -> tuple[np.ndarray, np.ndarray]:
    try:
        import freud
    except ImportError as exc:
        raise ImportError("freud backend is not available; install freud-analysis") from exc

    frame = handler.frame(int(frame_index))
    row = handler.dataframe().iloc[int(frame_index)]
    atom_types = np.asarray(frame["atom_types"], dtype=str)
    a, b, c = (float(row[name]) for name in ("a", "b", "c"))
    alpha = float(row.get("alpha", 90.0))
    beta = float(row.get("beta", 90.0))
    gamma = float(row.get("gamma", 90.0))
    box = freud.box.Box.from_box_lengths_and_angles(
        a, b, c, np.radians(alpha), np.radians(beta), np.radians(gamma)
    )
    mask_a = np.ones(len(atom_types), dtype=bool) if types_a is None else np.isin(atom_types, list(types_a))
    mask_b = np.ones(len(atom_types), dtype=bool) if types_b is None else np.isin(atom_types, list(types_b))
    half_box = 0.5 * min(a, b, c)
    cutoff = min(float(r_max), half_box - 1.0e-6) if r_max is not None else half_box - 1.0e-6
    rdf = freud.density.RDF(bins=int(bins), r_max=cutoff)
    coords = np.asarray(frame["coords"], dtype=float)
    rdf.compute((box, coords[mask_b]), query_points=coords[mask_a])
    r = np.asarray(rdf.bin_centers, dtype=float)
    g = np.asarray(rdf.rdf, dtype=float)
    return (r[1:], g[1:]) if len(r) > 1 else (r, g)


def _write_xyz(frame: dict) -> str:
    import os
    import tempfile

    fd, path = tempfile.mkstemp(suffix=".xyz", prefix="reaxkit_rdf_")
    os.close(fd)
    with open(path, "w", encoding="utf-8") as stream:
        stream.write(f"{len(frame['atom_types'])}\ncompatibility RDF frame\n")
        for symbol, xyz in zip(frame["atom_types"], frame["coords"]):
            stream.write(f"{symbol} {xyz[0]:.9f} {xyz[1]:.9f} {xyz[2]:.9f}\n")
    return path


def _rdf_ovito_frame(
    handler: XmoloutHandler,
    frame_index: int,
    *,
    types_a: Iterable[str] | None,
    types_b: Iterable[str] | None,
    r_max: float,
    bins: int,
) -> tuple[np.ndarray, np.ndarray]:
    import os

    try:
        from ovito.io import import_file
        from ovito.modifiers import CoordinationAnalysisModifier
    except ImportError as exc:
        raise ImportError("ovito backend is not available; install ovito") from exc

    frame = handler.frame(int(frame_index))
    path = _write_xyz(frame)
    partial = types_a is not None and types_b is not None
    try:
        pipeline = import_file(path)
        pipeline.modifiers.append(
            CoordinationAnalysisModifier(cutoff=float(r_max), number_of_bins=int(bins), partial=partial)
        )
        table = pipeline.compute().tables["coordination-rdf"]
        values = np.asarray(table.xy())
        if not partial:
            return values[:, 0], values[:, 1]
        type_a = str(next(iter(types_a or ())))
        type_b = str(next(iter(types_b or ())))
        names = list(table.y.component_names)
        pair = f"{type_a}-{type_b}"
        if pair not in names:
            pair = f"{type_b}-{type_a}"
        return values[:, 0], values[:, 1 + names.index(pair)]
    finally:
        try:
            os.remove(path)
        except OSError:
            pass


def rdf_property_over_frames(
    handler: XmoloutHandler,
    *,
    backend: str = "ovito",
    frames: Iterable[int] | None = None,
    property: str = "first_peak",
    r_max: float | None = None,
    bins: int = 200,
    types_a: Iterable[str] | None = None,
    types_b: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Compute one RDF-derived property per selected handler frame."""
    prop = str(property).strip().lower()
    allowed = {"first_peak", "dominant_peak", "area", "excess_area"}
    if prop not in allowed:
        raise ValueError(f"property must be one of {sorted(allowed)}")
    backend_name = str(backend).strip().lower()
    if backend_name not in {"freud", "ovito"}:
        raise ValueError("backend must be 'freud' or 'ovito'")

    simulation = handler.dataframe()
    selected = list(range(len(simulation))) if frames is None else [int(i) for i in frames]
    rows: list[dict[str, float | int]] = []
    for frame_index in selected:
        if backend_name == "freud":
            r, g = _rdf_freud_frame(
                handler,
                frame_index,
                types_a=types_a,
                types_b=types_b,
                r_max=r_max,
                bins=bins,
            )
        else:
            r, g = _rdf_ovito_frame(
                handler,
                frame_index,
                types_a=types_a,
                types_b=types_b,
                r_max=float(r_max or 4.0),
                bins=bins,
            )

        if prop == "first_peak":
            peak_r, peak_g = _first_local_max(r, g)
            values = {"r_first_peak": peak_r, "g_first_peak": peak_g}
        elif prop == "dominant_peak":
            peak_r, peak_g = _dominant_peak(r, g)
            values = {"r_peak": peak_r, "g_peak": peak_g}
        elif prop == "area":
            values = {"area": float(np.trapezoid(g, r)) if len(r) else np.nan}
        else:
            values = {"excess_area": float(np.trapezoid(g - 1.0, r)) if len(r) else np.nan}
        iteration = int(simulation.iloc[frame_index]["iter"]) if "iter" in simulation.columns else frame_index
        rows.append({"frame_index": frame_index, "iter": iteration, **values})
    return pd.DataFrame(rows).sort_values("frame_index").reset_index(drop=True)


__all__ = ["_dominant_peak", "_first_local_max", "rdf_property_over_frames"]
