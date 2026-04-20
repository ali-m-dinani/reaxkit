"""Compatibility analyzers namespace."""

from analyzers.dihedral import (
    DihedralRequest,
    DihedralResult,
    DihedralTask,
    calculate_dihedral_mdanalysis,
    calculate_dihedral_numpy,
)

__all__ = [
    "calculate_dihedral_numpy",
    "calculate_dihedral_mdanalysis",
    "DihedralRequest",
    "DihedralResult",
    "DihedralTask",
]
