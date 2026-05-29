"""Property and auxiliary loaders for the ReaxFF adapter.

This module groups load routines for energies, controls, electrostatics, and
other analysis-oriented ReaxFF artifacts.

**Usage context**

- Property ingest: Load partial energies, electric fields, and restraints.
- Run metadata ingest: Load control/eregime/geometry optimization streams.
- Aggregate assembly: Compose electrostatics and molecular-analysis bundles.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from reaxkit.domain.data_models import (
    AtomicKinematicsData,
    ChargeData,
    ControlParametersData,
    ElectrostaticsData,
    EregimeData,
    ElectricFieldData,
    GeometryOptimizationProgressData,
    GeometrySummaryData,
    MolecularAnalysisData,
    PartialEnergyData,
    RestraintData,
)
from reaxkit.engine.reaxff.adapter_parts.normalizers import (
    _atomic_kinematics_from_vels_handler,
    _charges_from_fort7_handler,
    _control_parameters_from_control_handler,
    _electric_field_from_fort78_handler,
    _eregime_from_handler,
    _geometry_optimization_from_fort57_handler,
    _merge_simulation_data,
    _molecular_analysis_from_molfra_handler,
    _partial_energy_from_energy_log_handler,
    _restraint_from_fort76_handler,
    _structure_summary_from_fort74_handler,
)

if TYPE_CHECKING:
    from reaxkit.engine.reaxff.adapter import ReaxFFAdapter


def load_structure_summary(adapter: ReaxFFAdapter, args: dict, reporter=None) -> GeometrySummaryData:
    """Load structure-summary data.

    Resolves a structure-summary source and parses it through `Fort74Handler`
    into a normalized `GeometrySummaryData` model. If the ReaxFF engine is
    used, then this file would usually be `fort.74`.

    Parameters
    ----------
    adapter : ReaxFFAdapter
        Adapter instance used for source resolution and handler creation.
    args : dict
        Loader arguments with optional `fort74`, alias, or `input`.
    reporter : Any, optional
        Optional reporter passed to handler constructors.

    Returns
    -------
    GeometrySummaryData
        Normalized structure-summary record.

    Examples
    --------
    >>> summary = adapter.load_structure_summary({"fort74": "run/fort.74"})
    """
    from reaxkit.engine.reaxff.io.fort74_handler import Fort74Handler

    raw = args.get("fort74") or args.get("structure_summary") or args.get("input") or "fort.74"
    p = Path(raw)
    fort74_path = p / "fort.74" if p.is_dir() else p
    handler = adapter._build_handler(
        args,
        handler_name="Fort74Handler",
        source_path=fort74_path,
        factory=lambda: Fort74Handler(fort74_path, reporter=reporter),
    )
    return adapter._time_source(
        args,
        handler_name="Fort74Handler",
        source_path=fort74_path,
        loader=lambda: _structure_summary_from_fort74_handler(handler),
    )


def load_partial_energy(adapter: ReaxFFAdapter, args: dict, reporter=None) -> PartialEnergyData:
    """Load partial-energy trajectories from energy output files.

    Resolves partial-energy sources, preferring common run-directory candidates,
    then parses data through `Fort73Handler`. If the ReaxFF engine is used,
    then files would commonly include `fort.73`, `energylog`, or `fort.58`.

    Parameters
    ----------
    adapter : ReaxFFAdapter
        Adapter instance used for path resolution and handler lifecycle.
    args : dict
        Loader arguments with optional `fort73`, alias, or `input`.
    reporter : Any, optional
        Optional reporter passed to handler constructors.

    Returns
    -------
    PartialEnergyData
        Normalized partial-energy stream data.

    Examples
    --------
    >>> pe = adapter.load_partial_energy({"fort73": "run/fort.73"})
    """
    from reaxkit.engine.reaxff.io.fort73_handler import Fort73Handler

    raw = args.get("fort73") or args.get("partial_energy") or args.get("input") or "fort.73"
    p = Path(raw)
    if p.is_dir():
        candidates = [p / "fort.73", p / "energylog", p / "fort.58"]
        partial_energy_path = next((candidate for candidate in candidates if candidate.exists()), candidates[0])
    else:
        partial_energy_path = p
    handler = adapter._build_handler(
        args,
        handler_name="Fort73Handler",
        source_path=partial_energy_path,
        factory=lambda: Fort73Handler(partial_energy_path, reporter=reporter),
    )
    return adapter._time_source(
        args,
        handler_name="Fort73Handler",
        source_path=partial_energy_path,
        loader=lambda: _partial_energy_from_energy_log_handler(handler),
    )


def load_restraints(adapter: ReaxFFAdapter, args: dict, reporter=None) -> RestraintData:
    """Load restraint-trajectory data.

    Resolves a restraint source path and parses it through `Fort76Handler`.
    If the ReaxFF engine is used, then this file would usually be `fort.76`.

    Parameters
    ----------
    adapter : ReaxFFAdapter
        Adapter instance used for path resolution and handler creation.
    args : dict
        Loader arguments with optional `fort76`, `restraints`, or `input`.
    reporter : Any, optional
        Optional reporter passed to handler constructors.

    Returns
    -------
    RestraintData
        Normalized restraint trajectory model.

    Examples
    --------
    >>> r = adapter.load_restraints({"fort76": "run/fort.76"})
    """
    from reaxkit.engine.reaxff.io.fort76_handler import Fort76Handler

    raw = args.get("fort76") or args.get("restraints") or args.get("input") or "fort.76"
    p = Path(raw)
    fort76_path = p / "fort.76" if p.is_dir() else p
    handler = adapter._build_handler(
        args,
        handler_name="Fort76Handler",
        source_path=fort76_path,
        factory=lambda: Fort76Handler(fort76_path, reporter=reporter),
    )
    return adapter._time_source(
        args,
        handler_name="Fort76Handler",
        source_path=fort76_path,
        loader=lambda: _restraint_from_fort76_handler(handler),
    )


def load_geometry_optimization(adapter: ReaxFFAdapter, args: dict, reporter=None) -> GeometryOptimizationProgressData:
    """Load geometry-optimization progress data.

    Resolves a geometry-optimization source and parses it through
    `Fort57Handler`. If the ReaxFF engine is used, then this file would usually
    be `fort.57`.

    Parameters
    ----------
    adapter : ReaxFFAdapter
        Adapter instance used for path resolution and handler lifecycle.
    args : dict
        Loader arguments with optional `fort57`, alias, or `input`.
    reporter : Any, optional
        Optional reporter passed to handler constructors.

    Returns
    -------
    GeometryOptimizationProgressData
        Normalized geometry-optimization progress model.

    Examples
    --------
    >>> go = adapter.load_geometry_optimization({"fort57": "run/fort.57"})
    """
    from reaxkit.engine.reaxff.io.fort57_handler import Fort57Handler

    raw = args.get("fort57") or args.get("geometry_optimization") or args.get("input") or "fort.57"
    p = Path(raw)
    fort57_path = p / "fort.57" if p.is_dir() else p
    handler = adapter._build_handler(
        args,
        handler_name="Fort57Handler",
        source_path=fort57_path,
        factory=lambda: Fort57Handler(fort57_path, reporter=reporter),
    )
    return adapter._time_source(
        args,
        handler_name="Fort57Handler",
        source_path=fort57_path,
        loader=lambda: _geometry_optimization_from_fort57_handler(handler),
    )


def load_control_parameters(adapter: ReaxFFAdapter, args: dict, reporter=None) -> ControlParametersData:
    """Load control-parameter data from a control input file.

    Resolves a control source and parses it through `ControlHandler` into a
    normalized control-parameter model. If the ReaxFF engine is used, then this
    file would usually be named `control`.

    Parameters
    ----------
    adapter : ReaxFFAdapter
        Adapter instance used for path resolution and handler creation.
    args : dict
        Loader arguments with optional `control`, `control_file`, or `input`.
    reporter : Any, optional
        Optional reporter passed to handler constructors.

    Returns
    -------
    ControlParametersData
        Normalized control-parameter record.

    Examples
    --------
    >>> ctrl = adapter.load_control_parameters({"control": "run/control"})
    """
    from reaxkit.engine.reaxff.io.control_handler import ControlHandler

    raw = args.get("control") or args.get("control_file") or args.get("input") or "control"
    p = Path(raw)
    control_path = p / "control" if p.is_dir() else p
    handler = adapter._build_handler(
        args,
        handler_name="ControlHandler",
        source_path=control_path,
        factory=lambda: ControlHandler(control_path, reporter=reporter),
    )
    return adapter._time_source(
        args,
        handler_name="ControlHandler",
        source_path=control_path,
        loader=lambda: _control_parameters_from_control_handler(handler),
    )


def load_eregime(adapter: ReaxFFAdapter, args: dict, reporter=None) -> EregimeData:
    """Load electric-regime schedule data.

    Resolves an electric-regime source and parses it through `EregimeHandler`
    into a normalized `EregimeData` model. If the ReaxFF engine is used, then
    this file would usually be `eregime.in`.

    Parameters
    ----------
    adapter : ReaxFFAdapter
        Adapter instance used for source resolution and handler lifecycle.
    args : dict
        Loader arguments with optional `eregime`, `eregime_file`, or `input`.
    reporter : Any, optional
        Optional reporter passed to handler constructors.

    Returns
    -------
    EregimeData
        Normalized electric-regime schedule model.

    Examples
    --------
    >>> er = adapter.load_eregime({"eregime": "run/eregime.in"})
    """
    from reaxkit.engine.reaxff.io.eregime_handler import EregimeHandler

    raw = args.get("eregime") or args.get("eregime_file") or args.get("input") or "eregime.in"
    p = Path(raw)
    eregime_path = p / "eregime.in" if p.is_dir() else p
    handler = adapter._build_handler(
        args,
        handler_name="EregimeHandler",
        source_path=eregime_path,
        factory=lambda: EregimeHandler(eregime_path, reporter=reporter),
    )
    return adapter._time_source(
        args,
        handler_name="EregimeHandler",
        source_path=eregime_path,
        loader=lambda: _eregime_from_handler(handler),
    )


def load_charges(adapter: ReaxFFAdapter, args: dict, reporter=None) -> ChargeData:
    """Load atomic-charge trajectories.

    Resolves a charge source, parses it through `Fort7Handler`, and enriches
    the output with merged simulation metadata when available. If the ReaxFF
    engine is used, then this file would usually be `fort.7`.

    Parameters
    ----------
    adapter : ReaxFFAdapter
        Adapter instance used for source resolution and handler creation.
    args : dict
        Loader arguments with optional `fort7`, `charges`, or `input`.
    reporter : Any, optional
        Optional reporter passed to handler constructors.

    Returns
    -------
    ChargeData
        Normalized charge trajectory data.

    Examples
    --------
    >>> charges = adapter.load_charges({"fort7": "run/fort.7"})
    """
    from reaxkit.engine.reaxff.io.fort7_handler import Fort7Handler

    raw = args.get("fort7") or args.get("charges") or args.get("input") or "fort.7"
    p = Path(raw)
    fort7_path = p / "fort.7" if p.is_dir() else p
    handler = adapter._build_handler(
        args,
        handler_name="Fort7Handler",
        source_path=fort7_path,
        factory=lambda: Fort7Handler(fort7_path, reporter=reporter),
    )
    sim = _merge_simulation_data(
        adapter._load_simulation_from_xmolout(args, reporter=reporter),
        adapter._load_simulation_from_summary(args, reporter=reporter),
    )
    return adapter._time_source(
        args,
        handler_name="Fort7Handler",
        source_path=fort7_path,
        loader=lambda: _charges_from_fort7_handler(handler, simulation=sim, reporter=reporter),
    )


def load_electrostatics(adapter: ReaxFFAdapter, args: dict, reporter=None) -> ElectrostaticsData:
    """Load a combined electrostatics bundle.

    Assembles trajectory, charge, and connectivity data, and conditionally
    includes electric-field data when requested by command mode or available
    sources.

    Parameters
    ----------
    adapter : ReaxFFAdapter
        Adapter instance that provides dependent loader methods.
    args : dict
        Loader arguments forwarded to dependent loaders.
    reporter : Any, optional
        Optional reporter forwarded to dependent loaders.

    Returns
    -------
    ElectrostaticsData
        Bundle containing electrostatics-relevant trajectory data.

    Examples
    --------
    >>> es = adapter.load_electrostatics({"run_dir": "run", "command": "hyst"})
    """
    trajectory = adapter.load_trajectory(args, reporter=reporter)
    charges = adapter.load_charges(args, reporter=reporter)
    connectivity = adapter.load_connectivity(args, reporter=reporter)
    electric_field = None
    command = str(args.get("command") or "").strip().lower()
    fort78_path = adapter._resolve_reaxff_path(args, "fort78", default="fort.78")
    if command == "hyst" or fort78_path.exists():
        try:
            electric_field = adapter.load_electric_field(args, reporter=reporter)
        except FileNotFoundError:
            if command == "hyst":
                raise
    return ElectrostaticsData(
        trajectory=trajectory,
        charges=charges,
        connectivity=connectivity,
        electric_field=electric_field,
    )


def load_atomic_kinematics(adapter: ReaxFFAdapter, args: dict, reporter=None) -> AtomicKinematicsData:
    """Load atomic-kinematics trajectories from velocity-like outputs.

    Resolves velocity-source candidates and parses the selected file through
    `VelsHandler` into a normalized kinematics model. If the ReaxFF engine is
    used, then files can include `vels`, `moldyn.vel`, or `molsav`.

    Parameters
    ----------
    adapter : ReaxFFAdapter
        Adapter instance used for source resolution and handler construction.
    args : dict
        Loader arguments with optional `vels`, `kinematics`, or `input`.
    reporter : Any, optional
        Optional reporter passed to handler constructors.

    Returns
    -------
    AtomicKinematicsData
        Normalized atomic kinematics trajectory model.

    Examples
    --------
    >>> kin = adapter.load_atomic_kinematics({"vels": "run/vels"})
    """
    from reaxkit.engine.reaxff.io.vels_handler import VelsHandler

    raw = args.get("vels") or args.get("kinematics") or args.get("input") or "vels"
    p = Path(raw)
    if p.is_dir():
        if (p / "vels").exists():
            vels_path = p / "vels"
        elif (p / "moldyn.vel").exists():
            vels_path = p / "moldyn.vel"
        elif (p / "molsav").exists():
            vels_path = p / "molsav"
        else:
            vels_path = p / "vels"
    else:
        vels_path = p
    handler = adapter._build_handler(
        args,
        handler_name="VelsHandler",
        source_path=vels_path,
        factory=lambda: VelsHandler(vels_path, reporter=reporter),
    )
    return adapter._time_source(
        args,
        handler_name="VelsHandler",
        source_path=vels_path,
        loader=lambda: _atomic_kinematics_from_vels_handler(handler),
    )


def load_electric_field(adapter: ReaxFFAdapter, args: dict, reporter=None) -> ElectricFieldData:
    """Load electric-field trajectory data.

    Resolves an electric-field source and parses it through `Fort78Handler`.
    If the ReaxFF engine is used, then this file would usually be `fort.78`.

    Parameters
    ----------
    adapter : ReaxFFAdapter
        Adapter instance used for source resolution and handler creation.
    args : dict
        Loader arguments with optional `fort78`, `electric_field`, or `input`.
    reporter : Any, optional
        Optional reporter passed to handler constructors.

    Returns
    -------
    ElectricFieldData
        Normalized electric-field trajectory model.

    Examples
    --------
    >>> ef = adapter.load_electric_field({"fort78": "run/fort.78"})
    """
    from reaxkit.engine.reaxff.io.fort78_handler import Fort78Handler

    raw = args.get("fort78") or args.get("electric_field") or args.get("input") or "fort.78"
    p = Path(raw)
    fort78_path = p / "fort.78" if p.is_dir() else p
    handler = adapter._build_handler(
        args,
        handler_name="Fort78Handler",
        source_path=fort78_path,
        factory=lambda: Fort78Handler(fort78_path, reporter=reporter),
    )
    return adapter._time_source(
        args,
        handler_name="Fort78Handler",
        source_path=fort78_path,
        loader=lambda: _electric_field_from_fort78_handler(handler),
    )


def load_molecular_analysis(adapter: ReaxFFAdapter, args: dict, reporter=None) -> MolecularAnalysisData:
    """Load molecular-species analysis data.

    Resolves molecular-analysis output candidates and parses the selected source
    through `MolFraHandler` into a normalized molecular-analysis model. If the
    ReaxFF engine is used, then files can include `molfra.out` or
    `molfra_ig.out`.

    Parameters
    ----------
    adapter : ReaxFFAdapter
        Adapter instance used for path resolution and handler creation.
    args : dict
        Loader arguments with optional `molfra`, alias, or `input`.
    reporter : Any, optional
        Optional reporter passed to handler constructors.

    Returns
    -------
    MolecularAnalysisData
        Normalized molecular-analysis output model.

    Examples
    --------
    >>> ma = adapter.load_molecular_analysis({"molfra": "run/molfra.out"})
    """
    from reaxkit.engine.reaxff.io.molfra_handler import MolFraHandler

    raw = args.get("molfra") or args.get("molecular_analysis") or args.get("input") or "molfra.out"
    p = Path(raw)
    if p.is_dir():
        if (p / "molfra.out").exists():
            molfra_path = p / "molfra.out"
        elif (p / "molfra_ig.out").exists():
            molfra_path = p / "molfra_ig.out"
        else:
            molfra_path = p / "molfra.out"
    else:
        molfra_path = p
    handler = adapter._build_handler(
        args,
        handler_name="MolFraHandler",
        source_path=molfra_path,
        factory=lambda: MolFraHandler(molfra_path, reporter=reporter),
    )
    return adapter._time_source(
        args,
        handler_name="MolFraHandler",
        source_path=molfra_path,
        loader=lambda: _molecular_analysis_from_molfra_handler(handler),
    )
