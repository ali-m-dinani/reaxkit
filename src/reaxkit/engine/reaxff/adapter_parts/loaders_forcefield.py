"""Force-field and optimization loaders for the ReaxFF adapter.

This module groups load routines for ReaxFF force-field parameters and related
optimization artifacts.

**Usage context**

- Parameter ingest: Load ffield and params representations.
- Optimization ingest: Load fort.13/79/99 and trainset artifacts.
- Bundle assembly: Compose higher-level optimization bundle models.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from reaxkit.domain.data_models import (
    ForceFieldOptimizationData,
    ForceFieldOptimizationDiagnosticBundleData,
    ForceFieldOptimizationDiagnosticData,
    ForceFieldOptimizationParameterBundleData,
    ForceFieldOptimizationParameterData,
    ForceFieldOptimizationProgressData,
    ForceFieldOptimizationReportData,
    ForceFieldOptimizationReportEOSBundleData,
    ForceFieldOptimizationTrainingSetData,
    ForceFieldParametersData,
)
from reaxkit.engine.reaxff.adapter_parts.normalizers import (
    _force_field_from_ffield_handler,
    _force_field_optimization_from_fort13_handler,
    _force_field_optimization_parameters_from_params_handler,
    _force_field_optimization_report_from_fort99_handler,
    _force_field_optimization_training_set_from_trainset_handler,
    _parameter_optimization_diagnostic_from_fort79_handler,
)

if TYPE_CHECKING:
    from reaxkit.engine.reaxff.adapter import ReaxFFAdapter


def load_force_field(adapter: ReaxFFAdapter, args: dict, reporter=None) -> ForceFieldParametersData:
    """Load ReaxFF force-field parameters from ffield."""
    from reaxkit.engine.common.io.ffield_handler import FFieldHandler

    raw = args.get("ffield") or args.get("force_field") or args.get("atom_reference") or args.get("input") or "ffield"
    p = Path(raw)
    ffield_path = p / "ffield" if p.is_dir() else p
    handler = adapter._build_handler(
        args,
        handler_name="FFieldHandler",
        source_path=ffield_path,
        factory=lambda: FFieldHandler(ffield_path, reporter=reporter),
    )
    return adapter._time_source(
        args,
        handler_name="FFieldHandler",
        source_path=ffield_path,
        loader=lambda: _force_field_from_ffield_handler(handler),
    )


def load_force_field_optimization(adapter: ReaxFFAdapter, args: dict, reporter=None) -> ForceFieldOptimizationProgressData:
    """Load force-field optimization progress from fort.13."""
    from reaxkit.engine.reaxff.io.fort13_handler import Fort13Handler

    raw = args.get("fort13") or args.get("force_field_optimization") or args.get("input") or "fort.13"
    p = Path(raw)
    fort13_path = p / "fort.13" if p.is_dir() else p
    handler = adapter._build_handler(
        args,
        handler_name="Fort13Handler",
        source_path=fort13_path,
        factory=lambda: Fort13Handler(fort13_path, reporter=reporter),
    )
    return adapter._time_source(
        args,
        handler_name="Fort13Handler",
        source_path=fort13_path,
        loader=lambda: _force_field_optimization_from_fort13_handler(handler),
    )


def load_force_field_optimization_report(adapter: ReaxFFAdapter, args: dict, reporter=None) -> ForceFieldOptimizationReportData:
    """Load force-field optimization report data from fort.99."""
    from reaxkit.engine.reaxff.io.fort99_handler import Fort99Handler

    raw = args.get("fort99") or args.get("force_field_optimization_report") or args.get("input") or "fort.99"
    p = Path(raw)
    fort99_path = p / "fort.99" if p.is_dir() else p
    handler = adapter._build_handler(
        args,
        handler_name="Fort99Handler",
        source_path=fort99_path,
        factory=lambda: Fort99Handler(fort99_path, reporter=reporter),
    )
    return adapter._time_source(
        args,
        handler_name="Fort99Handler",
        source_path=fort99_path,
        loader=lambda: _force_field_optimization_report_from_fort99_handler(handler),
    )


def load_force_field_optimization_training_set(
    adapter: ReaxFFAdapter,
    args: dict,
    reporter=None,
) -> ForceFieldOptimizationTrainingSetData:
    """Load force-field optimization training-set tables from trainset.in."""
    from reaxkit.engine.reaxff.io.trainset_handler import TrainsetHandler

    raw = args.get("trainset") or args.get("force_field_optimization_training_set") or args.get("input") or "trainset.in"
    p = Path(raw)
    trainset_path = p / "trainset.in" if p.is_dir() else p
    handler = adapter._build_handler(
        args,
        handler_name="TrainsetHandler",
        source_path=trainset_path,
        factory=lambda: TrainsetHandler(trainset_path, reporter=reporter),
    )
    return adapter._time_source(
        args,
        handler_name="TrainsetHandler",
        source_path=trainset_path,
        loader=lambda: _force_field_optimization_training_set_from_trainset_handler(handler),
    )


def load_force_field_optimization_parameters(
    adapter: ReaxFFAdapter,
    args: dict,
    reporter=None,
) -> ForceFieldOptimizationParameterData:
    """Load force-field optimization parameter search space from params."""
    from reaxkit.engine.reaxff.io.params_handler import ParamsHandler

    raw = args.get("params") or args.get("force_field_optimization_parameters") or args.get("input") or "params"
    p = Path(raw)
    params_path = p / "params" if p.is_dir() else p
    handler = adapter._build_handler(
        args,
        handler_name="ParamsHandler",
        source_path=params_path,
        factory=lambda: ParamsHandler(params_path, reporter=reporter),
    )
    return adapter._time_source(
        args,
        handler_name="ParamsHandler",
        source_path=params_path,
        loader=lambda: _force_field_optimization_parameters_from_params_handler(handler),
    )


def load_force_field_optimization_data(adapter: ReaxFFAdapter, args: dict, reporter=None) -> ForceFieldOptimizationData:
    """Load combined force-field parameters and optimization parameters."""
    return ForceFieldOptimizationData(
        force_field_parameters=adapter.load_force_field(args, reporter=reporter),
        optimization_parameters=adapter.load_force_field_optimization_parameters(args, reporter=reporter),
    )


def load_force_field_optimization_parameter_bundle(
    adapter: ReaxFFAdapter,
    args: dict,
    reporter=None,
) -> ForceFieldOptimizationParameterBundleData:
    """Load force-field optimization parameter bundle."""
    return ForceFieldOptimizationParameterBundleData(
        optimization_parameters=adapter.load_force_field_optimization_parameters(args, reporter=reporter),
        force_field_parameters=adapter.load_force_field(args, reporter=reporter),
    )


def load_parameter_optimization_diagnostic(
    adapter: ReaxFFAdapter,
    args: dict,
    reporter=None,
) -> ForceFieldOptimizationDiagnosticData:
    """Load parameter optimization diagnostics from fort.79."""
    from reaxkit.engine.reaxff.io.fort79_handler import Fort79Handler

    raw = args.get("fort79") or args.get("parameter_optimization_diagnostic") or args.get("input") or "fort.79"
    p = Path(raw)
    fort79_path = p / "fort.79" if p.is_dir() else p
    handler = adapter._build_handler(
        args,
        handler_name="Fort79Handler",
        source_path=fort79_path,
        factory=lambda: Fort79Handler(fort79_path, reporter=reporter),
    )
    return adapter._time_source(
        args,
        handler_name="Fort79Handler",
        source_path=fort79_path,
        loader=lambda: _parameter_optimization_diagnostic_from_fort79_handler(handler),
    )


def load_parameter_optimization_diagnostic_bundle(
    adapter: ReaxFFAdapter,
    args: dict,
    reporter=None,
) -> ForceFieldOptimizationDiagnosticBundleData:
    """Load diagnostic bundle with diagnostics and force-field parameters."""
    return ForceFieldOptimizationDiagnosticBundleData(
        diagnostics=adapter.load_parameter_optimization_diagnostic(args, reporter=reporter),
        force_field_parameters=adapter.load_force_field(args, reporter=reporter),
    )


def load_force_field_optimization_report_eos_bundle(
    adapter: ReaxFFAdapter,
    args: dict,
    reporter=None,
) -> ForceFieldOptimizationReportEOSBundleData:
    """Load EOS bundle with optimization report and structure summary."""
    return ForceFieldOptimizationReportEOSBundleData(
        report=adapter.load_force_field_optimization_report(args, reporter=reporter),
        geometry_summary=adapter.load_structure_summary(args, reporter=reporter),
    )
