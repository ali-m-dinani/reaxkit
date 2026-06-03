"""Define shared request/result dataclasses for active-site analyzer tasks.

This module centralizes task request and result schemas for active-site
structural and event extraction analyzers. It is scoped to typed payload models
and does not implement analysis computation.

**Usage context**

- Task contracts: Provide stable input/output shapes for active-site tasks.
- Serialization support: Keep result payloads consistent for presentation/reporting.
- Cross-module reuse: Share request/result models across structural/event modules.
"""

from __future__ import annotations

from dataclasses import dataclass, field as dc_field
from typing import Any, Optional, Sequence

import numpy as np
import pandas as pd

from reaxkit.domain.base_request import BaseRequest
from reaxkit.domain.base_result import BaseResult


@dataclass
class ActiveSiteStructuralRequest(BaseRequest):
    """Request payload for frame-level active-site structural analysis.

    Carries frame selection and descriptor configuration for structural active-
    site characterization, including bonding mode and optional SOAP settings.

    Fields
    -----
    frame : int
        Frame index to analyze. Must be non-negative.
    bo_threshold : float
        Bond-order threshold used when `bond_mode="bo"`.
    bond_mode : str
        Bond graph source mode: `"bo"` or `"distance"`.
    bond_scale : float
        Covalent-radii scale factor for distance-mode bond detection.
    alpha_radius : float
        Alpha-shape radius used for non-periodic boundary detection.
    gap_deg : float
        Angular-gap threshold for boundary fallback heuristics.
    carbon_element : str
        Element symbol treated as carbon substrate.
    include_noncarbon : bool
        Whether non-carbon atoms are retained in output table.
    strict_tract : bool
        If `True`, enforce strict TRACT required-column compatibility.
    soap : bool
        Enable optional SOAP descriptor computation for carbon atoms.
    soap_ref_path : Optional[str]
        Optional `.npy` path containing SOAP reference vectors.
    soap_r_cut : float
        SOAP cutoff radius in angstrom.
    soap_n_max : int
        SOAP radial basis size.
    soap_l_max : int
        SOAP angular basis size.
    soap_zeta : int
        Exponent used for SOAP-kernel scoring against reference vectors.

    Examples
    -----
    ```python
    req = ActiveSiteStructuralRequest(
        frame=0,
        bond_mode="bo",
        bo_threshold=0.3,
        soap=False,
    )
    ```
    Sample output:
    `ActiveSiteStructuralRequest(...)`
    Meaning:
    The request configures one-frame structural descriptor extraction.
    """

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
    """Result payload for frame-level active-site structural analysis.

    Stores raw structural descriptors, TRACT-compatible projection, summary
    statistics, and optional SOAP descriptor matrix.

    Fields
    -----
    table : pd.DataFrame
        Full per-atom descriptor table for analyzed frame.
    tract_table : pd.DataFrame
        TRACT-compatible structural table view.
    summary : dict[str, Any]
        Aggregate metrics and counts for report generation.
    request : ActiveSiteStructuralRequest
        Request used to produce this result.
    soap_descriptors : Optional[np.ndarray]
        Optional SOAP descriptor matrix for selected carbon atoms.

    Examples
    -----
    ```python
    result = ActiveSiteStructuralResult(table=df, tract_table=tract_df, summary={}, request=req)
    ```
    Sample output:
    `ActiveSiteStructuralResult(...)`
    Meaning:
    The result packages detailed and report-ready structural outputs.
    """

    table: pd.DataFrame
    tract_table: pd.DataFrame
    summary: dict[str, Any]
    request: ActiveSiteStructuralRequest
    soap_descriptors: Optional[np.ndarray] = None


@dataclass
class ActiveSiteEventsRequest(BaseRequest):
    """Request payload for trajectory-level active-site event extraction.

    Configures frame sampling, detection mode, species definitions, and event
    persistence criteria for extracting C-O and C-Si event summaries.

    Fields
    -----
    frames : Optional[Sequence[int]]
        Frame indices to evaluate. `None` means all frames.
    every : int
        Stride over selected frames. Must be `>= 1`.
    mode : str
        Event mode selector: `"auto"`, `"bo"`, or `"dist"`.
    bo_threshold : float
        BO threshold used to define bound contacts in `"bo"` mode.
    r_CO : float
        Carbon-oxygen cutoff distance (angstrom) for `"dist"` mode.
    r_CSi : float
        Carbon-silicon cutoff distance (angstrom) for `"dist"` mode.
    persist : int
        Minimum consecutive analyzed frames to confirm an event.
    carbon_element : str
        Element symbol used as substrate carbon type.
    oxygen_element : str
        Element symbol treated as oxygen target type.
    silicon_element : str
        Element symbol treated as silicon target type.
    strict_tract : bool
        If `True`, enforce strict TRACT required-column compatibility.

    Examples
    -----
    ```python
    req = ActiveSiteEventsRequest(mode="auto", every=10, persist=50)
    ```
    Sample output:
    `ActiveSiteEventsRequest(...)`
    Meaning:
    The request controls trajectory windowing and event confirmation logic.
    """

    frames: Optional[Sequence[int]] = dc_field(
        default=None,
        metadata={
            "label": "Frames",
            "help": "Optional frame indices to evaluate.",
            "units": "frame_index",
        },
    )
    every: int = dc_field(
        default=10,
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
    """Result payload for trajectory-level active-site event extraction.

    Stores per-carbon event descriptors, TRACT-compatible table projection, and
    run-level event summary metadata.

    Fields
    -----
    table : pd.DataFrame
        Per-carbon event table with counts, reactivity flags, and contact stats.
    tract_table : pd.DataFrame
        TRACT-compatible events table view.
    summary : dict[str, Any]
        Aggregate metrics such as reactive counts and analyzed frame window.
    request : ActiveSiteEventsRequest
        Request used to produce this result.

    Examples
    -----
    ```python
    result = ActiveSiteEventsResult(table=df, tract_table=tract_df, summary={}, request=req)
    ```
    Sample output:
    `ActiveSiteEventsResult(...)`
    Meaning:
    The result bundles event rows and summary/report metadata.
    """

    table: pd.DataFrame
    tract_table: pd.DataFrame
    summary: dict[str, Any]
    request: ActiveSiteEventsRequest


@dataclass
class ActiveSiteEventDiagnosticsRequest(BaseRequest):
    """Request payload for active-site event cutoff diagnostics.

    Samples trajectory frames to characterize nearest C-O and C-Si distance
    distributions before full event extraction. The diagnostic helps choose
    `r_CO`, `r_CSi`, and persistence thresholds for distance-mode runs.
    """

    frames: Optional[Sequence[int]] = dc_field(
        default=None,
        metadata={
            "label": "Frames",
            "help": "Optional frame indices to sample. Empty means all frames.",
            "units": "frame_index",
        },
    )
    every: int = dc_field(
        default=10,
        metadata={
            "label": "Every",
            "help": "Frame stride for diagnostic sampling.",
            "min": 1,
            "units": "frames",
        },
    )
    r_probe: float = dc_field(
        default=2.5,
        metadata={
            "label": "r_probe",
            "help": "Generous C-X distance cutoff used to detect close-approach episodes.",
            "min": 0.0,
            "units": "angstrom",
        },
    )
    max_diag_frames: int = dc_field(
        default=500,
        metadata={
            "label": "Max Diagnostic Frames",
            "help": "Maximum number of sampled frames to analyze.",
            "min": 1,
            "units": "frames",
        },
    )
    timestep_fs: float = dc_field(
        default=10.0,
        metadata={
            "label": "Timestep",
            "help": "Raw trajectory timestep used to convert episode lengths to ps.",
            "min": 0.0,
            "units": "fs",
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


@dataclass
class ActiveSiteEventDiagnosticsResult(BaseResult):
    """Result payload for active-site event cutoff diagnostics."""

    table: pd.DataFrame
    distance_table: pd.DataFrame
    episode_table: pd.DataFrame
    summary: dict[str, Any]
    request: ActiveSiteEventDiagnosticsRequest
