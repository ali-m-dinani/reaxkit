"""Dynamic-trajectory and geometry loaders for the ReaxFF adapter.

This module groups load routines that operate on trajectory-like time evolution
and geometry/connectivity assembly for ReaxFF outputs.

**Usage context**

- Trajectory ingest: Load xmolout-driven trajectory/simulation data.
- Geometry ingest: Load initial/final geometry records and metadata.
- Connectivity ingest: Load fort.7 connectivity and merged trajectory bundles.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from reaxkit.domain.data_models import (
    ConnectivityData,
    ConnectivityTrajectoryData,
    CoordinationStatusBundleData,
    ForceFieldParametersData,
    GeometryData,
    SimulationData,
    TrajectoryData,
)
from reaxkit.engine.reaxff.adapter_parts.normalizers import (
    _connectivity_from_fort7_handler,
    _connectivity_trajectory_from_handlers,
    _geometry_from_geo_handler,
    _merge_simulation_data,
    _simulation_from_summary_handler,
    _trajectory_from_xmolout_handler,
)

if TYPE_CHECKING:
    from reaxkit.engine.reaxff.adapter import ReaxFFAdapter


def load_trajectory(adapter: ReaxFFAdapter, args: dict, reporter=None) -> TrajectoryData:
    """Load trajectory data for ReaxFF runs.

    Resolves the trajectory source (typically `xmolout`), builds the
    corresponding handler, and normalizes it into `TrajectoryData`. Simulation
    metadata from `summary.txt` is merged when available.

    Parameters
    ----------
    adapter : ReaxFFAdapter
        Adapter instance used to resolve paths and construct handlers.
    args : dict
        Loader arguments, including optional path overrides.
    reporter : Any, optional
        Optional reporter passed to handler constructors.

    Returns
    -------
    TrajectoryData
        Normalized trajectory data with merged simulation metadata when present.

    Examples
    --------
    >>> data = adapter.load_trajectory({"xmolout": "run/xmolout"})
    """
    from reaxkit.engine.reaxff.io.xmolout_handler import XmoloutHandler

    xmol_path = adapter._resolve_reaxff_path(args, "xmolout", default="xmolout")
    handler = adapter._build_handler(
        args,
        handler_name="XmoloutHandler",
        source_path=xmol_path,
        factory=lambda: XmoloutHandler(xmol_path, reporter=reporter),
    )
    trj = adapter._time_source(
        args,
        handler_name="XmoloutHandler",
        source_path=xmol_path,
        loader=lambda: _trajectory_from_xmolout_handler(handler),
    )
    trj.simulation = _merge_simulation_data(
        trj.simulation,
        adapter._load_simulation_from_summary(args, reporter=reporter),
    )
    return trj


def load_geometry(adapter: ReaxFFAdapter, args: dict, reporter=None) -> GeometryData:
    """Load initial geometry data for ReaxFF runs.

    Resolves an initial-geometry source and parses it through `GeoHandler`.
    When `geometry_role="final"` or a final-geometry file is provided, this
    delegates to `load_final_geometry`.

    Parameters
    ----------
    adapter : ReaxFFAdapter
        Adapter instance used for path resolution and handler lifecycle.
    args : dict
        Loader arguments with optional `geo`, `geometry`, or `input`.
    reporter : Any, optional
        Optional reporter passed to handler constructors.

    Returns
    -------
    GeometryData
        Normalized initial-geometry record (or final geometry when delegated).

    Examples
    --------
    >>> geom = adapter.load_geometry({"geo": "run/geo"})
    """
    from reaxkit.engine.reaxff.io.geo_handler import GeoHandler

    if str(args.get("geometry_role") or "").strip().lower() == "final":
        return adapter.load_final_geometry(args, reporter=reporter)

    raw = args.get("geo") or args.get("geometry") or args.get("input") or "geo"
    p = Path(raw)
    geo_path = p / "geo" if p.is_dir() else p
    geo_path = adapter._resolve_against_run_dir(args, geo_path)
    if geo_path.name.lower() == "fort.90":
        return adapter.load_final_geometry({**args, "final_geometry": str(geo_path)}, reporter=reporter)
    handler = adapter._build_handler(
        args,
        handler_name="GeoHandler",
        source_path=geo_path,
        factory=lambda: GeoHandler(geo_path, reporter=reporter),
    )
    return adapter._time_source(
        args,
        handler_name="GeoHandler",
        source_path=geo_path,
        loader=lambda: _geometry_from_geo_handler(
            handler,
            source_file=geo_path.name or "geo",
            geometry_role="initial",
        ),
    )


def load_final_geometry(adapter: ReaxFFAdapter, args: dict, reporter=None) -> GeometryData:
    """Load final (optimized) geometry data for ReaxFF runs.

    Resolves a final-geometry source and parses it through `GeoHandler`. If
    the ReaxFF engine is used, then this file would usually be `fort.90`.

    Parameters
    ----------
    adapter : ReaxFFAdapter
        Adapter instance used for path resolution and handler lifecycle.
    args : dict
        Loader arguments with optional `final_geometry`, `fort90`, or `input`.
    reporter : Any, optional
        Optional reporter passed to handler constructors.

    Returns
    -------
    GeometryData
        Normalized final-geometry record.

    Examples
    --------
    >>> final_geom = adapter.load_final_geometry({"final_geometry": "run/fort.90"})
    """
    from reaxkit.engine.reaxff.io.geo_handler import GeoHandler

    raw = args.get("final_geometry") or args.get("fort90") or args.get("input") or "fort.90"
    p = Path(raw)
    fort90_path = p / "fort.90" if p.is_dir() else p
    fort90_path = adapter._resolve_against_run_dir(args, fort90_path)
    handler = adapter._build_handler(
        args,
        handler_name="GeoHandler",
        source_path=fort90_path,
        factory=lambda: GeoHandler(fort90_path, reporter=reporter),
    )
    return adapter._time_source(
        args,
        handler_name="GeoHandler",
        source_path=fort90_path,
        loader=lambda: _geometry_from_geo_handler(
            handler,
            source_file=fort90_path.name or "fort.90",
            geometry_role="final",
        ),
    )


def load_simulation(adapter: ReaxFFAdapter, args: dict, reporter=None) -> SimulationData:
    """Load merged simulation metadata for ReaxFF runs.

    Attempts to load simulation metadata from trajectory and summary sources,
    then merges both records. Raises when neither source is available.

    Parameters
    ----------
    adapter : ReaxFFAdapter
        Adapter instance that provides simulation sub-loaders.
    args : dict
        Loader arguments containing optional trajectory/summary paths.
    reporter : Any, optional
        Optional reporter passed to underlying handlers.

    Returns
    -------
    SimulationData
        Merged simulation metadata.

    Examples
    --------
    >>> sim = adapter.load_simulation({"run_dir": "run"})
    """
    sim = adapter._load_simulation_from_xmolout(args, reporter=reporter)
    sim = _merge_simulation_data(sim, adapter._load_simulation_from_summary(args, reporter=reporter))
    if sim is None:
        raise FileNotFoundError("SimulationData for reaxff currently requires xmolout or summary.txt.")
    return sim


def _load_simulation_from_xmolout(adapter_cls: type[ReaxFFAdapter], args: dict, reporter=None) -> SimulationData | None:
    """Load simulation metadata from xmolout, if available."""
    from reaxkit.engine.reaxff.io.xmolout_handler import XmoloutHandler

    xmol_path = adapter_cls._resolve_reaxff_path(args, "xmolout", default="xmolout")
    if not xmol_path.exists():
        return None
    handler = adapter_cls._build_handler(
        args,
        handler_name="XmoloutHandler",
        source_path=xmol_path,
        factory=lambda: XmoloutHandler(xmol_path, reporter=reporter),
    )
    trj = adapter_cls._time_source(
        args,
        handler_name="XmoloutHandler",
        source_path=xmol_path,
        loader=lambda: _trajectory_from_xmolout_handler(handler),
    )
    return trj.simulation


def _load_simulation_from_summary(adapter_cls: type[ReaxFFAdapter], args: dict, reporter=None) -> SimulationData | None:
    """Load simulation metadata from summary.txt, if available."""
    from reaxkit.engine.reaxff.io.summary_handler import SummaryHandler

    candidates = [args.get("summary"), args.get("xmolout"), args.get("run_dir"), args.get("input")]
    summary_path = None
    for raw in candidates:
        if not raw:
            continue
        p = Path(raw)
        if p.is_dir():
            candidate = p / "summary.txt"
        elif p.name == "summary.txt":
            candidate = p
        else:
            candidate = p.parent / "summary.txt"
        if candidate.exists() and candidate.name == "summary.txt":
            summary_path = candidate
            break
    if summary_path is None:
        return None
    handler = adapter_cls._build_handler(
        args,
        handler_name="SummaryHandler",
        source_path=summary_path,
        factory=lambda: SummaryHandler(summary_path, reporter=reporter),
    )
    return adapter_cls._time_source(
        args,
        handler_name="SummaryHandler",
        source_path=summary_path,
        loader=lambda: _simulation_from_summary_handler(handler),
    )


def load_connectivity(adapter: ReaxFFAdapter, args: dict, reporter=None) -> ConnectivityData:
    """Load connectivity matrices and merged simulation metadata.

    Resolves connectivity input and parses it via `Fort7Handler`, then merges
    simulation metadata from trajectory/summary sources into the returned model.
    If the ReaxFF engine is used, then the connectivity file would usually be
    `fort.7`.

    Parameters
    ----------
    adapter : ReaxFFAdapter
        Adapter instance used to resolve paths and build handlers.
    args : dict
        Loader arguments with optional `fort7`, `connectivity`, or `input`.
    reporter : Any, optional
        Optional reporter passed to handler constructors.

    Returns
    -------
    ConnectivityData
        Connectivity record with simulation-derived atom metadata when present.

    Examples
    --------
    >>> conn = adapter.load_connectivity({"fort7": "run/fort.7"})
    """
    from reaxkit.engine.reaxff.io.fort7_handler import Fort7Handler

    raw = args.get("fort7") or args.get("connectivity") or args.get("input") or "fort.7"
    p = Path(raw)
    fort7_path = p / "fort.7" if p.is_dir() else p
    handler = adapter._build_handler(
        args,
        handler_name="Fort7Handler",
        source_path=fort7_path,
        factory=lambda: Fort7Handler(fort7_path, reporter=reporter),
    )
    conn = adapter._time_source(
        args,
        handler_name="Fort7Handler",
        source_path=fort7_path,
        loader=lambda: _connectivity_from_fort7_handler(handler, reporter=reporter),
    )
    sim = _merge_simulation_data(
        adapter._load_simulation_from_xmolout(args, reporter=reporter),
        adapter._load_simulation_from_summary(args, reporter=reporter),
    )
    if sim is not None:
        conn.simulation = _merge_simulation_data(sim, conn.simulation)
        conn.elements = conn.simulation.elements
        conn.atom_ids = conn.simulation.atom_ids
    return conn


def load_coordination_status_bundle(adapter: ReaxFFAdapter, args: dict, reporter=None) -> CoordinationStatusBundleData:
    """Load a coordination-status bundle from connectivity and force-field data.

    Composes a bundle by invoking connectivity and force-field loaders and
    packaging their normalized outputs into one data model.

    Parameters
    ----------
    adapter : ReaxFFAdapter
        Adapter instance that exposes dependent load methods.
    args : dict
        Loader arguments forwarded to dependent loaders.
    reporter : Any, optional
        Optional reporter forwarded to dependent loaders.

    Returns
    -------
    CoordinationStatusBundleData
        Bundle containing connectivity and force-field parameter records.

    Examples
    --------
    >>> bundle = adapter.load_coordination_status_bundle({"run_dir": "run"})
    """
    return CoordinationStatusBundleData(
        connectivity=adapter.load_connectivity(args, reporter=reporter),
        force_field_parameters=adapter.load_force_field(args, reporter=reporter),
    )


def load_connectivity_trajectory(adapter: ReaxFFAdapter, args: dict, reporter=None) -> ConnectivityTrajectoryData:
    """Load combined connectivity and trajectory data.

    Builds connectivity and trajectory handlers, optionally includes summary
    simulation metadata and force-field parameters, then constructs a unified
    `ConnectivityTrajectoryData` object. If the ReaxFF engine is used, then
    the inputs would usually be `fort.7` and `xmolout`.

    Parameters
    ----------
    adapter : ReaxFFAdapter
        Adapter instance used for path resolution and handler construction.
    args : dict
        Loader arguments with optional connectivity/trajectory path overrides.
    reporter : Any, optional
        Optional reporter passed to handler constructors.

    Returns
    -------
    ConnectivityTrajectoryData
        Combined connectivity and trajectory dataset with optional supplements.

    Examples
    --------
    >>> ctd = adapter.load_connectivity_trajectory({"run_dir": "run"})
    """
    from reaxkit.engine.reaxff.io.fort7_handler import Fort7Handler
    from reaxkit.engine.reaxff.io.xmolout_handler import XmoloutHandler

    fort7_raw = args.get("fort7") or args.get("connectivity") or args.get("input") or "fort.7"
    fort7_path = Path(fort7_raw)
    fort7_path = fort7_path / "fort.7" if fort7_path.is_dir() else fort7_path

    xmol_path = adapter._resolve_reaxff_path(args, "xmolout", default="xmolout")
    fort7_handler = adapter._build_handler(
        args,
        handler_name="Fort7Handler",
        source_path=fort7_path,
        factory=lambda: Fort7Handler(fort7_path, reporter=reporter),
    )
    xmol_handler = adapter._build_handler(
        args,
        handler_name="XmoloutHandler",
        source_path=xmol_path,
        factory=lambda: XmoloutHandler(xmol_path, reporter=reporter),
    )
    summary_simulation = adapter._load_simulation_from_summary(args, reporter=reporter)
    force_field_parameters: ForceFieldParametersData | None = None
    try:
        ff_args = dict(args)
        if not ff_args.get("ffield"):
            ff_args["ffield"] = str(
                adapter._resolve_reaxff_path(
                    args,
                    "ffield",
                    "force_field",
                    "atom_reference",
                    default="ffield",
                )
            )
        force_field_parameters = adapter.load_force_field(ff_args, reporter=reporter)
    except Exception:
        force_field_parameters = None
    return _connectivity_trajectory_from_handlers(
        fort7_handler,
        xmol_handler,
        summary_simulation=summary_simulation,
        force_field_parameters=force_field_parameters,
        reporter=reporter,
    )
