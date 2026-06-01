"""ReaxFF engine adapter.

**Usage context**

- Engine dispatch: Route typed load/write requests to engine-specific implementations.
- Data normalization: Convert raw engine files into canonical domain models.
- Workflow support: Provide reusable adapter entry points for CLI/workflow layers.
"""

from __future__ import annotations

from pathlib import Path

from reaxkit.core.platform.engine_resolver import register_engine
from reaxkit.engine.reaxff.adapter_parts import (
    _build_handler as _build_handler_helper,
    _emit_load_timing as _emit_load_timing_helper,
    _quick_n_frames as _quick_n_frames_helper,
    _quick_n_frames_from_control as _quick_n_frames_from_control_helper,
    _quick_n_frames_from_geo_xmol as _quick_n_frames_from_geo_xmol_helper,
    _resolve_against_run_dir as _resolve_against_run_dir_helper,
    _resolve_reaxff_path as _resolve_reaxff_path_helper,
    _time_source as _time_source_helper,
    _write_control_data,
    _write_trajectory_data,
)
from reaxkit.domain.data_models import (
    AtomicKinematicsData,
    ChargeData,
    ConnectivityTrajectoryData,
    CoordinationStatusBundleData,
    ConnectivityData,
    ControlParametersData,
    ElectrostaticsData,
    EregimeData,
    ElectricFieldData,
    ForceFieldParametersData,
    ForceFieldOptimizationProgressData,
    ForceFieldOptimizationTrainingSetData,
    ForceFieldOptimizationData,
    ForceFieldOptimizationParameterBundleData,
    ForceFieldOptimizationDiagnosticBundleData,
    ForceFieldOptimizationReportEOSBundleData,
    ForceFieldOptimizationParameterData,
    ForceFieldOptimizationReportData,
    GeometryData,
    GeometryOptimizationProgressData,
    MolecularAnalysisData,
    ForceFieldOptimizationDiagnosticData,
    PartialEnergyData,
    RestraintData,
    GeometrySummaryData,
    SimulationData,
    TrajectoryData,
)
from reaxkit.engine.base import EngineAdapter

from reaxkit.engine.reaxff.adapter_parts.loaders_dynamics import (
    _load_simulation_from_summary as _load_simulation_from_summary_impl,
    _load_simulation_from_xmolout as _load_simulation_from_xmolout_impl,
    load_connectivity as _load_connectivity_impl,
    load_connectivity_trajectory as _load_connectivity_trajectory_impl,
    load_coordination_status_bundle as _load_coordination_status_bundle_impl,
    load_final_geometry as _load_final_geometry_impl,
    load_geometry as _load_geometry_impl,
    load_simulation as _load_simulation_impl,
    load_trajectory as _load_trajectory_impl,
)
from reaxkit.engine.reaxff.adapter_parts.loaders_forcefield import (
    load_force_field as _load_force_field_impl,
    load_force_field_optimization as _load_force_field_optimization_impl,
    load_force_field_optimization_data as _load_force_field_optimization_data_impl,
    load_force_field_optimization_parameter_bundle as _load_force_field_optimization_parameter_bundle_impl,
    load_force_field_optimization_parameters as _load_force_field_optimization_parameters_impl,
    load_force_field_optimization_report as _load_force_field_optimization_report_impl,
    load_force_field_optimization_report_eos_bundle as _load_force_field_optimization_report_eos_bundle_impl,
    load_force_field_optimization_training_set as _load_force_field_optimization_training_set_impl,
    load_parameter_optimization_diagnostic as _load_parameter_optimization_diagnostic_impl,
    load_parameter_optimization_diagnostic_bundle as _load_parameter_optimization_diagnostic_bundle_impl,
)
from reaxkit.engine.reaxff.adapter_parts.loaders_properties import (
    load_atomic_kinematics as _load_atomic_kinematics_impl,
    load_charges as _load_charges_impl,
    load_control_parameters as _load_control_parameters_impl,
    load_electric_field as _load_electric_field_impl,
    load_electrostatics as _load_electrostatics_impl,
    load_eregime as _load_eregime_impl,
    load_geometry_optimization as _load_geometry_optimization_impl,
    load_molecular_analysis as _load_molecular_analysis_impl,
    load_partial_energy as _load_partial_energy_impl,
    load_restraints as _load_restraints_impl,
    load_structure_summary as _load_structure_summary_impl,
)
from reaxkit.engine.reaxff.adapter_parts.normalizers import (
    charges_from_fort7_handler,
    connectivity_from_fort7_handler,
    trajectory_from_xmolout_handler,
)

# Backward-compatible aliases for older imports.
_charges_from_fort7_handler = charges_from_fort7_handler
_connectivity_from_fort7_handler = connectivity_from_fort7_handler
_trajectory_from_xmolout_handler = trajectory_from_xmolout_handler


@register_engine("reaxff")
class ReaxFFAdapter(EngineAdapter):
    """Adapter that loads ReaxFF outputs into domain models."""

    @staticmethod
    def _resolve_reaxff_path(args: dict, *keys: str, default: str) -> Path:
        """Resolve reaxff path."""
        return _resolve_reaxff_path_helper(args, *keys, default=default)

    @staticmethod
    def _resolve_against_run_dir(args: dict, path: Path) -> Path:
        """Resolve against run dir."""
        return _resolve_against_run_dir_helper(args, path)

    @staticmethod
    def _quick_n_frames_from_control(control_path: Path) -> int | None:
        """Quick n frames from control."""
        return _quick_n_frames_from_control_helper(control_path)

    @staticmethod
    def _quick_n_frames_from_geo_xmol(geo_path: Path, xmol_path: Path) -> int | None:
        """Quick n frames from geo xmol."""
        return _quick_n_frames_from_geo_xmol_helper(geo_path, xmol_path)

    @classmethod
    def quick_n_frames(cls, args: dict) -> int | None:
        """Fast frame-count probe for Web UI metadata updates."""
        return _quick_n_frames_helper(args)

    def detect(self, path: str | Path) -> float:
        """Detect.

        Parameters
        ----------
        path : str | Path
            Input parameter.

        Returns
        -------
        float
            Return value.

        Examples
        --------
        ```python
        # Example
        detect(...)
        ```
        """
        p = Path(path)
        if p.is_dir():
            if (p / "xmolout").exists() or (p / "fort.7").exists() or (p / "summary.txt").exists():
                return 0.95
            return 0.0
        if p.is_file():
            lower_name = p.name.lower()
            if "xmolout" in lower_name or p.name == "fort.7":
                return 0.95
        return 0.0

    def required_input_files(self, data_type, args: dict) -> tuple[str, ...] | None:
        """Required input files.

        Parameters
        ----------
        data_type : Any
            Input parameter.
        args : dict
            Input parameter.

        Returns
        -------
        tuple[str, ...] | None
            Return value.

        Examples
        --------
        ```python
        # Example
        required_input_files(...)
        ```
        """
        mapping: dict[object, tuple[str, ...]] = {
            TrajectoryData: ("xmolout", "summary.txt"),
            GeometryData: ("geo", "fort.90"),
            SimulationData: ("xmolout", "summary.txt"),
            ConnectivityData: ("fort.7", "xmolout", "summary.txt"),
            ConnectivityTrajectoryData: ("fort.7", "xmolout", "summary.txt", "ffield"),
            CoordinationStatusBundleData: ("fort.7", "xmolout", "summary.txt", "ffield"),
            ChargeData: ("fort.7", "xmolout", "summary.txt"),
            ElectrostaticsData: ("xmolout", "fort.7", "summary.txt"),
            AtomicKinematicsData: ("vels",),
            ElectricFieldData: ("fort.78",),
            EregimeData: ("eregime.in",),
            ForceFieldParametersData: ("ffield",),
            ForceFieldOptimizationProgressData: ("fort.13",),
            ForceFieldOptimizationTrainingSetData: ("trainset.in",),
            ForceFieldOptimizationData: ("ffield", "params"),
            ForceFieldOptimizationParameterBundleData: ("params", "ffield"),
            ForceFieldOptimizationDiagnosticBundleData: ("fort.79", "ffield"),
            ForceFieldOptimizationReportEOSBundleData: ("fort.99", "fort.74"),
            ForceFieldOptimizationParameterData: ("params",),
            ForceFieldOptimizationReportData: ("fort.99",),
            ForceFieldOptimizationDiagnosticData: ("fort.79",),
            GeometrySummaryData: ("fort.74",),
            PartialEnergyData: ("fort.73", "energylog", "fort.58"),
            RestraintData: ("fort.76",),
            GeometryOptimizationProgressData: ("fort.57",),
            ControlParametersData: ("control",),
            MolecularAnalysisData: ("molfra.out", "molfra_ig.out"),
        }
        if data_type is ElectrostaticsData and str(args.get("command") or "").strip().lower() == "hyst":
            return ("xmolout", "fort.7", "fort.78", "summary.txt")
        return mapping.get(data_type)

    @staticmethod
    def _emit_load_timing(
        args: dict,
        *,
        handler: str,
        source_path: Path | str | None,
        seconds: float,
    ) -> None:
        """Emit load timing."""
        _emit_load_timing_helper(args, handler=handler, source_path=source_path, seconds=seconds)

    @classmethod
    def _build_handler(
        cls,
        args: dict,
        *,
        handler_name: str,
        source_path: Path | str | None,
        factory,
    ):
        """Build handler."""
        return _build_handler_helper(
            args,
            handler_name=handler_name,
            source_path=source_path,
            factory=factory,
        )

    @classmethod
    def _time_source(
        cls,
        args: dict,
        *,
        handler_name: str,
        source_path: Path | str | None,
        loader,
    ):
        """Time source."""
        return _time_source_helper(
            args,
            handler_name=handler_name,
            source_path=source_path,
            loader=loader,
        )

    def load_trajectory(self, args: dict, reporter=None) -> TrajectoryData:
        """Load trajectory data from ReaxFF outputs."""
        return _load_trajectory_impl(self, args, reporter=reporter)

    def load_geometry(self, args: dict, reporter=None) -> GeometryData:
        """Load initial geometry data from ReaxFF outputs."""
        return _load_geometry_impl(self, args, reporter=reporter)

    def load_final_geometry(self, args: dict, reporter=None) -> GeometryData:
        """Load final geometry data from ReaxFF outputs."""
        return _load_final_geometry_impl(self, args, reporter=reporter)

    def load_simulation(self, args: dict, reporter=None) -> SimulationData:
        """Load merged simulation metadata from ReaxFF outputs."""
        return _load_simulation_impl(self, args, reporter=reporter)

    @classmethod
    def _load_simulation_from_xmolout(cls, args: dict, reporter=None) -> SimulationData | None:
        """Load simulation from xmolout."""
        return _load_simulation_from_xmolout_impl(cls, args, reporter=reporter)

    @classmethod
    def _load_simulation_from_summary(cls, args: dict, reporter=None) -> SimulationData | None:
        """Load simulation from summary."""
        return _load_simulation_from_summary_impl(cls, args, reporter=reporter)

    def load_connectivity(self, args: dict, reporter=None) -> ConnectivityData:
        """Load connectivity data from ReaxFF outputs."""
        return _load_connectivity_impl(self, args, reporter=reporter)

    def load_coordination_status_bundle(self, args: dict, reporter=None) -> CoordinationStatusBundleData:
        """Load a coordination-status bundle for ReaxFF outputs."""
        return _load_coordination_status_bundle_impl(self, args, reporter=reporter)

    def load_connectivity_trajectory(self, args: dict, reporter=None) -> ConnectivityTrajectoryData:
        """Load combined connectivity and trajectory data."""
        return _load_connectivity_trajectory_impl(self, args, reporter=reporter)

    def load_force_field(self, args: dict, reporter=None) -> ForceFieldParametersData:
        """Load ReaxFF force-field parameters."""
        return _load_force_field_impl(self, args, reporter=reporter)

    def load_force_field_optimization(self, args: dict, reporter=None) -> ForceFieldOptimizationProgressData:
        """Load force-field optimization progress data."""
        return _load_force_field_optimization_impl(self, args, reporter=reporter)

    def load_force_field_optimization_report(self, args: dict, reporter=None) -> ForceFieldOptimizationReportData:
        """Load force-field optimization report data."""
        return _load_force_field_optimization_report_impl(self, args, reporter=reporter)

    def load_force_field_optimization_training_set(
        self,
        args: dict,
        reporter=None,
    ) -> ForceFieldOptimizationTrainingSetData:
        """Load force-field optimization training-set data."""
        return _load_force_field_optimization_training_set_impl(self, args, reporter=reporter)

    def load_force_field_optimization_parameters(
        self,
        args: dict,
        reporter=None,
    ) -> ForceFieldOptimizationParameterData:
        """Load force-field optimization parameter data."""
        return _load_force_field_optimization_parameters_impl(self, args, reporter=reporter)

    def load_force_field_optimization_data(
        self,
        args: dict,
        reporter=None,
    ) -> ForceFieldOptimizationData:
        """Load combined force-field optimization data."""
        return _load_force_field_optimization_data_impl(self, args, reporter=reporter)

    def load_force_field_optimization_parameter_bundle(
        self,
        args: dict,
        reporter=None,
    ) -> ForceFieldOptimizationParameterBundleData:
        """Load force-field optimization parameter bundle data."""
        return _load_force_field_optimization_parameter_bundle_impl(self, args, reporter=reporter)

    def load_parameter_optimization_diagnostic(
        self,
        args: dict,
        reporter=None,
    ) -> ForceFieldOptimizationDiagnosticData:
        """Load parameter-optimization diagnostic data."""
        return _load_parameter_optimization_diagnostic_impl(self, args, reporter=reporter)

    def load_parameter_optimization_diagnostic_bundle(
        self,
        args: dict,
        reporter=None,
    ) -> ForceFieldOptimizationDiagnosticBundleData:
        """Load parameter-optimization diagnostic bundle data."""
        return _load_parameter_optimization_diagnostic_bundle_impl(self, args, reporter=reporter)

    def load_force_field_optimization_report_eos_bundle(
        self,
        args: dict,
        reporter=None,
    ) -> ForceFieldOptimizationReportEOSBundleData:
        """Load force-field optimization EOS bundle data."""
        return _load_force_field_optimization_report_eos_bundle_impl(self, args, reporter=reporter)

    def load_structure_summary(self, args: dict, reporter=None) -> GeometrySummaryData:
        """Load structure-summary data."""
        return _load_structure_summary_impl(self, args, reporter=reporter)

    def load_partial_energy(self, args: dict, reporter=None) -> PartialEnergyData:
        """Load partial-energy data."""
        return _load_partial_energy_impl(self, args, reporter=reporter)

    def load_restraints(self, args: dict, reporter=None) -> RestraintData:
        """Load restraint data."""
        return _load_restraints_impl(self, args, reporter=reporter)

    def load_geometry_optimization(self, args: dict, reporter=None) -> GeometryOptimizationProgressData:
        """Load geometry-optimization progress data."""
        return _load_geometry_optimization_impl(self, args, reporter=reporter)

    def load_control_parameters(self, args: dict, reporter=None) -> ControlParametersData:
        """Load control-parameter data."""
        return _load_control_parameters_impl(self, args, reporter=reporter)

    def load_eregime(self, args: dict, reporter=None) -> EregimeData:
        """Load electric-regime schedule data."""
        return _load_eregime_impl(self, args, reporter=reporter)

    def load_charges(self, args: dict, reporter=None) -> ChargeData:
        """Load charge data."""
        return _load_charges_impl(self, args, reporter=reporter)

    def load_electrostatics(self, args: dict, reporter=None) -> ElectrostaticsData:
        """Load electrostatics bundle data."""
        return _load_electrostatics_impl(self, args, reporter=reporter)

    def load_atomic_kinematics(self, args: dict, reporter=None) -> AtomicKinematicsData:
        """Load atomic-kinematics data."""
        return _load_atomic_kinematics_impl(self, args, reporter=reporter)

    def load_electric_field(self, args: dict, reporter=None) -> ElectricFieldData:
        """Load electric-field data."""
        return _load_electric_field_impl(self, args, reporter=reporter)

    def load_molecular_analysis(self, args: dict, reporter=None) -> MolecularAnalysisData:
        """Load molecular-analysis data."""
        return _load_molecular_analysis_impl(self, args, reporter=reporter)

    def write_control(
        self,
        data: ControlParametersData,
        out_path: str | Path,
        args: dict | None = None,
    ):
        """Write control data in ReaxFF control-file format."""
        return _write_control_data(data=data, out_path=out_path, args=args)

    def write_trajectory(self, data: TrajectoryData, out_path: str | Path, args: dict | None = None):
        """Write trajectory data in ReaxFF xmolout format."""
        return _write_trajectory_data(data=data, out_path=out_path, args=args)
