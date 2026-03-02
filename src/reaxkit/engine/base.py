"""Generic engine adapter API."""

from __future__ import annotations

from abc import ABC, abstractmethod
import inspect
from pathlib import Path

from reaxkit.domain.data_models import (
    AtomicKinematicsData,
    ChargeData,
    ConnectivityData,
    ControlParametersData,
    EregimeData,
    ElectricFieldData,
    ForceFieldParametersData,
    ForceFieldOptimizationProgressData,
    ForceFieldOptimizationTrainingSetData,
    ForceFieldOptimizationParameterData,
    ForceFieldOptimizationReportData,
    GeometryOptimizationProgressData,
    MolecularAnalysisData,
    ForceFieldOptimizationDiagnosticData,
    PartialEnergyData,
    RestraintData,
    GeometrySummaryData,
    SimulationData,
    TrajectoryData,
)


class EngineAdapter(ABC):
    """Engine adapter interface for detection + typed data loading."""

    name: str = "base"

    @abstractmethod
    def detect(self, path: str | Path) -> float:
        """Return confidence score [0, 1]."""

    def load(self, data_type, args: dict, reporter=None):
        """Load requested domain data type from engine-specific sources."""
        if data_type is TrajectoryData:
            return self._invoke_loader("load_trajectory", args, reporter=reporter)
        if data_type is SimulationData:
            return self._invoke_loader("load_simulation", args, reporter=reporter)
        if data_type is ConnectivityData:
            return self._invoke_loader("load_connectivity", args, reporter=reporter)
        if data_type is ChargeData:
            return self._invoke_loader("load_charges", args, reporter=reporter)
        if data_type is AtomicKinematicsData:
            return self._invoke_loader("load_atomic_kinematics", args, reporter=reporter)
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
        if data_type is ForceFieldOptimizationParameterData:
            return self._invoke_loader("load_force_field_optimization_parameters", args, reporter=reporter)
        if data_type is ForceFieldOptimizationReportData:
            return self._invoke_loader("load_force_field_optimization_report", args, reporter=reporter)
        if data_type is ForceFieldOptimizationDiagnosticData:
            return self._invoke_loader("load_parameter_optimization_diagnostic", args, reporter=reporter)
        if data_type is GeometrySummaryData:
            return self._invoke_loader("load_structure_summary", args, reporter=reporter)
        if data_type is PartialEnergyData:
            return self._invoke_loader("load_partial_energy", args, reporter=reporter)
        if data_type is RestraintData:
            return self._invoke_loader("load_restraints", args, reporter=reporter)
        if data_type is GeometryOptimizationProgressData:
            return self._invoke_loader("load_geometry_optimization", args, reporter=reporter)
        if data_type is ControlParametersData:
            return self._invoke_loader("load_control_parameters", args, reporter=reporter)
        if data_type is MolecularAnalysisData:
            return self._invoke_loader("load_molecular_analysis", args, reporter=reporter)
        raise ValueError(f"{self.name} cannot load data type: {data_type}")

    def _invoke_loader(self, method_name: str, args: dict, reporter=None):
        method = getattr(self, method_name, None)
        if method is None:
            raise ValueError(f"{self.name} cannot load data type via missing method: {method_name}")
        params = inspect.signature(method).parameters
        if "reporter" in params:
            return method(args, reporter=reporter)
        return method(args)
