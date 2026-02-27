"""Generic engine adapter API."""

from __future__ import annotations

from abc import ABC, abstractmethod
import inspect
from pathlib import Path

from reaxkit.domain.data_models import (
    ChargeData,
    ConnectivityData,
    ElectricFieldData,
    ForceFieldData,
    MolecularAnalysisData,
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
        if data_type is ElectricFieldData:
            return self._invoke_loader("load_electric_field", args, reporter=reporter)
        if data_type is ForceFieldData:
            return self._invoke_loader("load_force_field", args, reporter=reporter)
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
