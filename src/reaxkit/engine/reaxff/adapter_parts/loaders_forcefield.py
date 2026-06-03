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
    """Load force-field parameters from a force-field definition file.

    Resolves a force-field source path, builds the corresponding handler, and
    normalizes parameter content into `ForceFieldParametersData`. If the ReaxFF
    engine is used, then this file would usually be `ffield`.

    Parameters
    ----------
    adapter : ReaxFFAdapter
        Adapter instance used for path resolution and handler creation.
    args : dict
        Loader arguments with optional `ffield`, `force_field`, or `input`.
    reporter : Any, optional
        Optional reporter passed to handler constructors.

    Returns
    -------
    ForceFieldParametersData
        Normalized force-field parameter model.

    Examples
    --------
    >>> ff = adapter.load_force_field({"ffield": "run/ffield"})
    """
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
    """Load force-field optimization progress data.

    Resolves an optimization-progress source, parses it through `Fort13Handler`,
    and returns normalized optimization history. If the ReaxFF engine is used,
    then this file would usually be `fort.13`.

    Parameters
    ----------
    adapter : ReaxFFAdapter
        Adapter instance used for source resolution and handler lifecycle.
    args : dict
        Loader arguments with optional `fort13`, optimization alias, or `input`.
    reporter : Any, optional
        Optional reporter passed to handler constructors.

    Returns
    -------
    ForceFieldOptimizationProgressData
        Optimization-progress record parsed from the selected source.

    Examples
    --------
    >>> progress = adapter.load_force_field_optimization({"fort13": "run/fort.13"})
    """
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
    """Load force-field optimization report data.

    Resolves a report source and parses it through `Fort99Handler` into a
    normalized optimization report. If the ReaxFF engine is used, then this
    file would usually be `fort.99`.

    Parameters
    ----------
    adapter : ReaxFFAdapter
        Adapter instance used to resolve paths and build handlers.
    args : dict
        Loader arguments with optional `fort99`, report alias, or `input`.
    reporter : Any, optional
        Optional reporter passed to handler constructors.

    Returns
    -------
    ForceFieldOptimizationReportData
        Normalized optimization report model.

    Examples
    --------
    >>> report = adapter.load_force_field_optimization_report({"fort99": "run/fort.99"})
    """
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
    """Load force-field optimization training-set tables.

    Resolves a training-set source and parses it through `TrainsetHandler` into
    a normalized training-set representation. If the ReaxFF engine is used,
    then this file would usually be `trainset.in`.

    Parameters
    ----------
    adapter : ReaxFFAdapter
        Adapter instance used for path resolution and handler creation.
    args : dict
        Loader arguments with optional `trainset`, alias, or `input`.
    reporter : Any, optional
        Optional reporter passed to handler constructors.

    Returns
    -------
    ForceFieldOptimizationTrainingSetData
        Normalized optimization training-set record.

    Examples
    --------
    >>> ts = adapter.load_force_field_optimization_training_set({"trainset": "run/trainset.in"})
    """
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
    """Load optimization-parameter search-space data.

    Resolves a parameter-definition source and parses it through `ParamsHandler`
    into a normalized parameter-search model. If the ReaxFF engine is used,
    then this file would usually be `params`.

    Parameters
    ----------
    adapter : ReaxFFAdapter
        Adapter instance used for source resolution and handler lifecycle.
    args : dict
        Loader arguments with optional `params`, alias, or `input`.
    reporter : Any, optional
        Optional reporter passed to handler constructors.

    Returns
    -------
    ForceFieldOptimizationParameterData
        Normalized optimization-parameter definition model.

    Examples
    --------
    >>> pdef = adapter.load_force_field_optimization_parameters({"params": "run/params"})
    """
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
    """Load combined force-field and optimization-parameter data.

    Composes `ForceFieldOptimizationData` by calling the force-field loader and
    optimization-parameter loader with shared arguments.

    Parameters
    ----------
    adapter : ReaxFFAdapter
        Adapter instance that provides dependent load methods.
    args : dict
        Loader arguments forwarded to dependent loaders.
    reporter : Any, optional
        Optional reporter forwarded to dependent loaders.

    Returns
    -------
    ForceFieldOptimizationData
        Bundle containing force-field parameters and optimization parameters.

    Examples
    --------
    >>> data = adapter.load_force_field_optimization_data({"run_dir": "run"})
    """
    return ForceFieldOptimizationData(
        force_field_parameters=adapter.load_force_field(args, reporter=reporter),
        optimization_parameters=adapter.load_force_field_optimization_parameters(args, reporter=reporter),
    )


def load_force_field_optimization_parameter_bundle(
    adapter: ReaxFFAdapter,
    args: dict,
    reporter=None,
) -> ForceFieldOptimizationParameterBundleData:
    """Load a force-field optimization parameter bundle.

    Builds a bundle model that pairs optimization-parameter definitions with
    force-field parameters for downstream analysis steps.

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
    ForceFieldOptimizationParameterBundleData
        Bundle containing optimization parameters and force-field parameters.

    Examples
    --------
    >>> bundle = adapter.load_force_field_optimization_parameter_bundle({"run_dir": "run"})
    """
    return ForceFieldOptimizationParameterBundleData(
        optimization_parameters=adapter.load_force_field_optimization_parameters(args, reporter=reporter),
        force_field_parameters=adapter.load_force_field(args, reporter=reporter),
    )


def load_parameter_optimization_diagnostic(
    adapter: ReaxFFAdapter,
    args: dict,
    reporter=None,
) -> ForceFieldOptimizationDiagnosticData:
    """Load parameter-optimization diagnostic data.

    Resolves a diagnostic source path and parses it through `Fort79Handler`
    into a normalized diagnostics model. If the ReaxFF engine is used, then
    this file would usually be `fort.79`.

    Parameters
    ----------
    adapter : ReaxFFAdapter
        Adapter instance used for source resolution and handler construction.
    args : dict
        Loader arguments with optional `fort79`, alias, or `input`.
    reporter : Any, optional
        Optional reporter passed to handler constructors.

    Returns
    -------
    ForceFieldOptimizationDiagnosticData
        Normalized parameter-optimization diagnostics.

    Examples
    --------
    >>> diag = adapter.load_parameter_optimization_diagnostic({"fort79": "run/fort.79"})
    """
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
    """Load a diagnostic bundle with diagnostics and force-field parameters.

    Composes a bundle by combining parameter-optimization diagnostics with the
    currently resolved force-field parameter set.

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
    ForceFieldOptimizationDiagnosticBundleData
        Bundle containing diagnostics and force-field parameters.

    Examples
    --------
    >>> bundle = adapter.load_parameter_optimization_diagnostic_bundle({"run_dir": "run"})
    """
    return ForceFieldOptimizationDiagnosticBundleData(
        diagnostics=adapter.load_parameter_optimization_diagnostic(args, reporter=reporter),
        force_field_parameters=adapter.load_force_field(args, reporter=reporter),
    )


def load_force_field_optimization_report_eos_bundle(
    adapter: ReaxFFAdapter,
    args: dict,
    reporter=None,
) -> ForceFieldOptimizationReportEOSBundleData:
    """Load an EOS-oriented bundle with report and structure summary.

    Composes a combined model from optimization report output and structure
    summary output for EOS-focused post-processing workflows.

    Parameters
    ----------
    adapter : ReaxFFAdapter
        Adapter instance that provides report and structure loaders.
    args : dict
        Loader arguments forwarded to dependent loaders.
    reporter : Any, optional
        Optional reporter forwarded to dependent loaders.

    Returns
    -------
    ForceFieldOptimizationReportEOSBundleData
        Bundle containing optimization report and structure summary data.

    Examples
    --------
    >>> eos = adapter.load_force_field_optimization_report_eos_bundle({"run_dir": "run"})
    """
    return ForceFieldOptimizationReportEOSBundleData(
        report=adapter.load_force_field_optimization_report(args, reporter=reporter),
        geometry_summary=adapter.load_structure_summary(args, reporter=reporter),
    )
