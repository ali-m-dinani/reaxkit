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
    """Load geometry summary from fort.74."""
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
    """Load partial energy stream from fort.73/energylog/fort.58."""
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
    """Load restraint trajectory from fort.76."""
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
    """Load geometry-optimization progress from fort.57."""
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
    """Load control parameters from control input."""
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
    """Load electric-regime schedule data from eregime.in."""
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
    """Load charge trajectories from fort.7."""
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
    """Load electrostatics bundle for trajectory, charges, and field data."""
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
    """Load atomic kinematics from vels-like outputs."""
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
    """Load electric-field trajectory from fort.78."""
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
    """Load molecular-species analysis from molfra outputs."""
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
