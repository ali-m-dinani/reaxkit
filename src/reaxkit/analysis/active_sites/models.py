"""Shared request/result models for active-site analysis tasks."""

from __future__ import annotations

from dataclasses import dataclass, field as dc_field
from typing import Any, Optional, Sequence

import numpy as np
import pandas as pd

from reaxkit.domain.base_request import BaseRequest
from reaxkit.domain.base_result import BaseResult


@dataclass
class ActiveSiteStructuralRequest(BaseRequest):
    """Request for frame-level active-site structural descriptors."""

    frame: int = dc_field(
        default=0,
        metadata={
            "label": "Frame",
            "help": "Frame index for structural analysis.",
            "units": "frame_index",
            "min": 0,
        },
    )
    bo_threshold: float = dc_field(
        default=0.3,
        metadata={
            "label": "BO Threshold",
            "help": "Bond-order threshold used to construct the connectivity graph.",
            "min": 0.0,
        },
    )
    bond_mode: str = dc_field(
        default="bo",
        metadata={
            "label": "Bond Mode",
            "help": "Bond graph mode: bo (ConnectivityData.bond_orders) or distance (TRACT-style geometric cutoffs).",
            "choices": ["bo", "distance"],
        },
    )
    bond_scale: float = dc_field(
        default=1.2,
        metadata={
            "label": "Bond Scale",
            "help": "Scale factor on covalent radii for distance-based bond detection.",
            "min": 0.0,
        },
    )
    alpha_radius: float = dc_field(
        default=0.0,
        metadata={
            "label": "Alpha Radius",
            "help": "Alpha-shape radius for non-periodic boundary detection; <=0 disables alpha-shape.",
            "units": "angstrom",
        },
    )
    gap_deg: float = dc_field(
        default=220.0,
        metadata={
            "label": "Gap Deg",
            "help": "Angular-gap threshold used in boundary detection fallback.",
            "units": "degree",
        },
    )
    carbon_element: str = dc_field(
        default="C",
        metadata={
            "label": "Carbon Element",
            "help": "Element symbol used for carbon network analysis.",
        },
    )
    include_noncarbon: bool = dc_field(
        default=True,
        metadata={
            "label": "Include Noncarbon",
            "help": "Include non-carbon atoms in the output table.",
            "choices": [True, False],
        },
    )
    strict_tract: bool = dc_field(
        default=False,
        metadata={
            "label": "Strict TRACT",
            "help": "Raise when canonical structural fields are too incomplete for TRACT compatibility.",
            "choices": [True, False],
        },
    )
    soap: bool = dc_field(
        default=False,
        metadata={
            "label": "SOAP",
            "help": "Compute optional SOAP descriptors for carbon atoms.",
            "choices": [True, False],
        },
    )
    soap_ref_path: Optional[str] = dc_field(
        default=None,
        metadata={
            "label": "SOAP Ref Path",
            "help": "Optional .npy path of reactive-site SOAP vectors used for soap_score.",
        },
    )
    soap_r_cut: float = dc_field(
        default=5.0,
        metadata={
            "label": "SOAP r_cut",
            "help": "SOAP cutoff radius in angstrom.",
            "units": "angstrom",
            "min": 0.0,
        },
    )
    soap_n_max: int = dc_field(
        default=9,
        metadata={
            "label": "SOAP n_max",
            "help": "SOAP radial basis size.",
            "min": 1,
        },
    )
    soap_l_max: int = dc_field(
        default=9,
        metadata={
            "label": "SOAP l_max",
            "help": "SOAP angular basis size.",
            "min": 1,
        },
    )
    soap_zeta: int = dc_field(
        default=2,
        metadata={
            "label": "SOAP zeta",
            "help": "Exponent used for SOAP kernel similarity scoring when reference vectors are provided.",
            "min": 1,
        },
    )


@dataclass
class ActiveSiteStructuralResult(BaseResult):
    """Frame-level active-site structural descriptors."""

    table: pd.DataFrame
    tract_table: pd.DataFrame
    summary: dict[str, Any]
    request: ActiveSiteStructuralRequest
    soap_descriptors: Optional[np.ndarray] = None


@dataclass
class ActiveSiteEventsRequest(BaseRequest):
    """Request for trajectory-level active-site event extraction."""

    frames: Optional[Sequence[int]] = dc_field(
        default=None,
        metadata={
            "label": "Frames",
            "help": "Optional frame indices to evaluate.",
            "units": "frame_index",
        },
    )
    every: int = dc_field(
        default=1,
        metadata={
            "label": "Every",
            "help": "Frame stride for selected frames.",
            "min": 1,
            "units": "frames",
        },
    )
    mode: str = dc_field(
        default="auto",
        metadata={
            "label": "Mode",
            "help": "Event mode: auto | bo | dist.",
            "choices": ["auto", "bo", "dist"],
        },
    )
    bo_threshold: float = dc_field(
        default=0.8,
        metadata={
            "label": "BO Threshold",
            "help": "Bond-order threshold for event detection in bo mode.",
            "min": 0.0,
        },
    )
    r_CO: float = dc_field(
        default=1.65,
        metadata={
            "label": "r_CO",
            "help": "C-O cutoff in angstrom for distance mode.",
            "min": 0.0,
            "units": "angstrom",
        },
    )
    r_CSi: float = dc_field(
        default=2.10,
        metadata={
            "label": "r_CSi",
            "help": "C-Si cutoff in angstrom for distance mode.",
            "min": 0.0,
            "units": "angstrom",
        },
    )
    persist: int = dc_field(
        default=50,
        metadata={
            "label": "Persist",
            "help": "Required consecutive analyzed frames for confirmed binding.",
            "min": 1,
        },
    )
    carbon_element: str = dc_field(
        default="C",
        metadata={
            "label": "Carbon Element",
            "help": "Element symbol used as reactive substrate atom type.",
        },
    )
    oxygen_element: str = dc_field(
        default="O",
        metadata={
            "label": "Oxygen Element",
            "help": "Element symbol treated as oxygen target.",
        },
    )
    silicon_element: str = dc_field(
        default="Si",
        metadata={
            "label": "Silicon Element",
            "help": "Element symbol treated as silicon target.",
        },
    )
    strict_tract: bool = dc_field(
        default=False,
        metadata={
            "label": "Strict TRACT",
            "help": "Raise when canonical events fields are too incomplete for TRACT compatibility.",
            "choices": [True, False],
        },
    )


@dataclass
class ActiveSiteEventsResult(BaseResult):
    """Per-carbon event table from trajectory-level active-site extraction."""

    table: pd.DataFrame
    tract_table: pd.DataFrame
    summary: dict[str, Any]
    request: ActiveSiteEventsRequest
