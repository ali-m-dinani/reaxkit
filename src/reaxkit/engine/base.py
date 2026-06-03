"""Generic engine adapter API.

**Usage context**

- Engine dispatch: Route typed load/write requests to engine-specific implementations.
- Data normalization: Convert raw engine files into canonical domain models.
- Workflow support: Provide reusable adapter entry points for CLI/workflow layers.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
import inspect
from pathlib import Path

from reaxkit.domain.data_models import (
    AtomicKinematicsData,
    AtomStrainEnergyData,
    AtomTemperatureData,
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
    EnergyMinimizationSummaryData,
    SimulationData,
    TrajectoryData,
)


class EngineAdapter(ABC):
    """Engine adapter interface for detection + typed data loading."""

    name: str = "base"

    @abstractmethod
    def detect(self, path: str | Path) -> float:
        """Return confidence score [0, 1]."""

    def required_input_files(self, data_type, args: dict) -> tuple[str, ...] | None:
        """Return minimal source filenames needed to load ``data_type``.

        Returning ``None`` means "use the storage layer's default broad snapshot list".
        """
        _ = (data_type, args)
        return None

    def load(self, data_type, args: dict, reporter=None):
        """Load requested domain data type from engine-specific sources."""
        if data_type is TrajectoryData:
            return self._invoke_loader("load_trajectory", args, reporter=reporter)
        if data_type is GeometryData:
            return self._invoke_loader("load_geometry", args, reporter=reporter)
        if data_type is SimulationData:
            return self._invoke_loader("load_simulation", args, reporter=reporter)
        if data_type is ConnectivityData:
            return self._invoke_loader("load_connectivity", args, reporter=reporter)
        if data_type is ConnectivityTrajectoryData:
            return self._invoke_loader("load_connectivity_trajectory", args, reporter=reporter)
        if data_type is CoordinationStatusBundleData:
            return self._invoke_loader("load_coordination_status_bundle", args, reporter=reporter)
        if data_type is ChargeData:
            return self._invoke_loader("load_charges", args, reporter=reporter)
        if data_type is AtomicKinematicsData:
            return self._invoke_loader("load_atomic_kinematics", args, reporter=reporter)
        if data_type is AtomTemperatureData:
            return self._invoke_loader("load_atom_temperature", args, reporter=reporter)
        if data_type is AtomStrainEnergyData:
            return self._invoke_loader("load_atom_strain_energy", args, reporter=reporter)
        if data_type is ElectricFieldData:
            return self._invoke_loader("load_electric_field", args, reporter=reporter)
        if data_type is EregimeData:
            return self._invoke_loader("load_eregime", args, reporter=reporter)
        if data_type is ForceFieldParametersData:
            return self._invoke_loader("load_force_field", args, reporter=reporter)
        if data_type is ForceFieldOptimizationProgressData:
            return self._invoke_loader("load_force_field_optimization", args, reporter=reporter)
        if data_type is ForceFieldOptimizationTrainingSetData:
            return self._invoke_loader("load_force_field_optimization_training_set", args, reporter=reporter)
        if data_type is ForceFieldOptimizationData:
            return self._invoke_loader("load_force_field_optimization_data", args, reporter=reporter)
        if data_type is ForceFieldOptimizationParameterBundleData:
            return self._invoke_loader("load_force_field_optimization_parameter_bundle", args, reporter=reporter)
        if data_type is ForceFieldOptimizationDiagnosticBundleData:
            return self._invoke_loader("load_parameter_optimization_diagnostic_bundle", args, reporter=reporter)
        if data_type is ForceFieldOptimizationReportEOSBundleData:
            return self._invoke_loader("load_force_field_optimization_report_eos_bundle", args, reporter=reporter)
        if data_type is ForceFieldOptimizationParameterData:
            return self._invoke_loader("load_force_field_optimization_parameters", args, reporter=reporter)
        if data_type is ForceFieldOptimizationReportData:
            return self._invoke_loader("load_force_field_optimization_report", args, reporter=reporter)
        if data_type is ForceFieldOptimizationDiagnosticData:
            return self._invoke_loader("load_parameter_optimization_diagnostic", args, reporter=reporter)
        if data_type is EnergyMinimizationSummaryData:
            return self._invoke_loader("load_structure_summary", args, reporter=reporter)
        if data_type is PartialEnergyData:
            return self._invoke_loader("load_partial_energy", args, reporter=reporter)
        if data_type is RestraintData:
            return self._invoke_loader("load_restraints", args, reporter=reporter)
        if data_type is GeometryOptimizationProgressData:
            return self._invoke_loader("load_geometry_optimization", args, reporter=reporter)
        if data_type is ControlParametersData:
            return self._invoke_loader("load_control_parameters", args, reporter=reporter)
        if data_type is ElectrostaticsData:
            return self._invoke_loader("load_electrostatics", args, reporter=reporter)
        if data_type is MolecularAnalysisData:
            return self._invoke_loader("load_molecular_analysis", args, reporter=reporter)
        raise ValueError(f"{self.name} cannot load data type: {data_type}")

    def write(self, data, out_path, args: dict | None = None):
        """Write a domain data object using an engine-appropriate writer."""
        if isinstance(data, TrajectoryData):
            return self._invoke_writer("write_trajectory", data, out_path, args or {})
        if isinstance(data, ControlParametersData):
            return self._invoke_writer("write_control", data, out_path, args or {})
        raise ValueError(f"{self.name} cannot write data object of type: {type(data).__name__}")

    def _invoke_loader(self, method_name: str, args: dict, reporter=None):
        """Invoke loader."""
        method = getattr(self, method_name, None)
        if method is None:
            raise ValueError(f"{self.name} cannot load data type via missing method: {method_name}")
        params = inspect.signature(method).parameters
        if "reporter" in params:
            return method(args, reporter=reporter)
        return method(args)

    def _invoke_writer(self, method_name: str, data, out_path, args: dict):
        """Invoke writer."""
        method = getattr(self, method_name, None)
        if method is None:
            raise ValueError(f"{self.name} cannot write data via missing method: {method_name}")
        params = inspect.signature(method).parameters
        if "args" in params:
            return method(data, out_path, args=args)
        return method(data, out_path)
