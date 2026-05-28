"""Provide engine-agnostic connectivity and bond-event analyzer tasks.

This module extracts connection lists, connection matrices, aggregated
connectivity statistics, and bond formation/breakage events from bond-order
trajectories. It is limited to connectivity-domain signals and does not compute
geometric descriptors outside bond-order-derived relationships.

**Usage context**

- Bond-network inspection: Build edge tables and per-frame connectivity matrices.
- Aggregation workflows: Compute pairwise connectivity statistics over time.
- Reaction-event tracking: Detect bond formation/breakage with hysteresis logic.
"""

from __future__ import annotations

from dataclasses import dataclass, field as dc_field
from typing import Any, Literal, Optional, Sequence

import numpy as np
import pandas as pd

from reaxkit.analysis.base import AnalysisTask
from reaxkit.core.analysis_task_registry import register_task
from reaxkit.domain.base_request import BaseRequest
from reaxkit.domain.base_result import BaseResult
from reaxkit.domain.data_models import ConnectivityData
from reaxkit.presentation.specs import PresentationSpec
from reaxkit.utils.numerical.moving_average import exponential_moving_average, simple_moving_average
from reaxkit.utils.numerical.signal_ops import clean_flicker, schmitt_hysteresis


def _as_dense_frame(fr) -> np.ndarray:
    if isinstance(fr, np.ndarray):
        return fr.astype(float)
    if hasattr(fr, "toarray"):
        return np.asarray(fr.toarray(), dtype=float)
    if hasattr(fr, "todense"):
        return np.asarray(fr.todense(), dtype=float)
    return np.asarray(fr, dtype=float)


def _bond_order_frames(data: ConnectivityData) -> list[np.ndarray]:
    bo = data.bond_orders
    if bo is None:
        raise ValueError("ConnectivityData.bond_orders is required for connectivity graph/event tasks.")
    if isinstance(bo, np.ndarray):
        if bo.ndim != 3:
            raise ValueError("bond_orders ndarray must have shape (n_frames, n_atoms, n_atoms).")
        return [bo[i].astype(float) for i in range(bo.shape[0])]
    if isinstance(bo, (list, tuple)):
        frames = [_as_dense_frame(fr) for fr in bo]
        if not frames:
            return []
        return frames
    raise TypeError("Unsupported bond_orders type; use ndarray or list/tuple of per-frame matrices.")


def _atom_ids(data: ConnectivityData, n_atoms: int) -> np.ndarray:
    if data.atom_ids is None:
        return np.arange(1, n_atoms + 1, dtype=int)
    ids = np.asarray(data.atom_ids, dtype=int).reshape(-1)
    if ids.shape[0] != n_atoms:
        raise ValueError(f"atom_ids length ({ids.shape[0]}) must match n_atoms ({n_atoms}).")
    return ids


def _atom_types(data: ConnectivityData, n_atoms: int) -> np.ndarray:
    if data.elements is None:
        return np.asarray(["X"] * n_atoms, dtype=object)
    types = np.asarray(list(data.elements), dtype=object).reshape(-1)
    if types.shape[0] != n_atoms:
        raise ValueError(f"elements length ({types.shape[0]}) must match n_atoms ({n_atoms}).")
    return types


def _iterations(data: ConnectivityData, n_frames: int) -> np.ndarray:
    if data.iterations is None:
        return np.arange(n_frames, dtype=int)
    it = np.asarray(data.iterations, dtype=int).reshape(-1)
    if it.shape[0] != n_frames:
        raise ValueError(f"iterations length ({it.shape[0]}) must match n_frames ({n_frames}).")
    return it


def _frame_indices(n_frames: int, frames: Optional[Sequence[int]], every: int) -> list[int]:
    idx = list(range(n_frames)) if frames is None else [int(i) for i in frames]
    idx = [i for i in idx if 0 <= i < n_frames][:: max(1, int(every))]
    return idx


@dataclass
class ConnectionListRequest(BaseRequest):
    """Request for extracting frame-resolved bond connections.

    Fields
    -----
    frames : Optional[Sequence[int]]
        Optional frame indices to include. `None` means all frames.
    every : int
        Frame stride after selection. Must be `>= 1`.
    min_bo : float
        Minimum bond-order threshold for including an edge.
    undirected : bool
        If `True`, treat `i-j` and `j-i` as one undirected edge.
    include_self : bool
        If `True`, include self-edges (`i == j`).

    Examples
    -----
    ```python
    req = ConnectionListRequest(frames=[0, 10, 20], min_bo=0.3, undirected=True)
    ```
    Sample output:
    `ConnectionListRequest(...)`
    Meaning:
    The request configures frame sampling and edge-threshold extraction rules.
    """

    frames: Optional[Sequence[int]] = dc_field(
        default=None,
        metadata={
            'label': 'Frames',
            'help': "Optional frame indices to include. Example: [0, 10, 20].",
            'units': 'frame_index',
        },
    )
    every: int = dc_field(
        default=1,
        metadata={
            'label': 'Every',
            'help': "Stride for selected frames. Example: every=5.",
            'min': 1,
            'units': 'frames',
        },
    )
    min_bo: float = dc_field(
        default=0.0,
        metadata={
            'label': 'Min Bo',
            'help': "Minimum bond-order threshold for connection inclusion. Example: 0.3.",
            'min': 0.0,
        },
    )
    undirected: bool = dc_field(
        default=True,
        metadata={
            'label': 'Undirected',
            'help': "Merge i-j and j-i into one edge when true.",
            'choices': [True, False],
        },
    )
    include_self: bool = dc_field(
        default=False,
        metadata={
            'label': 'Include Self',
            'help': "Include self-edges (src == dst) when true.",
            'choices': [True, False],
        },
    )


@dataclass
class ConnectionListResult(BaseResult):
    """Connection-list extraction result.

    Fields
    -----
    table : pd.DataFrame
        Edge table with columns `frame_idx`, `iteration`, `source`,
        `source_type`, `destination`, `destination_type`, `BO`.
    request : ConnectionListRequest
        Request object used for this analysis run.

    Examples
    -----
    ```python
    result = ConnectionListTask().run(data, req)
    result.table.head()
    ```
    Sample output:
    DataFrame rows for each selected edge occurrence.
    Meaning:
    Each row is one bond edge record at one frame.
    """

    table: pd.DataFrame
    request: ConnectionListRequest


@register_task("get_connection_list", label="Connection List")
class ConnectionListTask(AnalysisTask):
    required_data = ConnectivityData

    @staticmethod
    def recommended_presentations(
        _result: ConnectionListResult,
        payload: dict[str, Any],
    ) -> list[PresentationSpec]:
        """Build default table/plot presentations for connection-list outputs.

        Works on
        -----
        Analyzer task output payloads

        Parameters
        -----
        _result : ConnectionListResult
            Analysis result object for the executed task.
        payload : dict[str, Any]
            Serialized result payload used by presentation dispatch.

        Returns
        -----
        list[PresentationSpec]
            Recommended renderer specs for table and BO trend plot.

        Examples
        -----
        ```python
        specs = ConnectionListTask.recommended_presentations(result, payload)
        ```
        Sample output:
        Table view plus `BO vs frame_idx` plot view.
        Meaning:
        Connection-list outputs can be rendered with default mappings.
        """
        rows = payload.get("table")
        if not isinstance(rows, list) or not rows:
            return [PresentationSpec(renderer="table", label="Table", view_type="table")]
        sample = rows[0] if isinstance(rows[0], dict) else {}
        if "frame_idx" not in sample or "BO" not in sample:
            return [PresentationSpec(renderer="table", label="Table", view_type="table")]
        return [
            PresentationSpec(renderer="table", label="Table", view_type="table"),
            PresentationSpec(
                renderer="single_plot",
                label="BO vs frame_idx",
                mapping={"x_col": "frame_idx", "y_col": "BO", "group_by_col": "source"},
                options={
                    "title": "Connection BO vs Frame",
                    "xlabel": "frame_idx",
                    "ylabel": "BO",
                    "legend": True,
                },
                view_type="plot2d",
            ),
        ]

    def run(self, data: ConnectivityData, request: ConnectionListRequest, reporter=None) -> ConnectionListResult:
        """Extract frame-resolved connectivity edges from bond-order matrices.

        Works on
        -----
        `ConnectivityData` plus `ConnectionListRequest` analyzer inputs

        Parameters
        -----
        data : ConnectivityData
            Connectivity series containing bond-order matrices and metadata.
        request : ConnectionListRequest
            Selection and threshold configuration for edge extraction.
        reporter : Any, optional
            Optional progress callback invoked during frame processing.

        Returns
        -----
        ConnectionListResult
            Edge table with one row per qualifying bond occurrence.

        Examples
        -----
        ```python
        result = ConnectionListTask().run(data, ConnectionListRequest(min_bo=0.2))
        ```
        Sample output:
        `result.table` with columns including `source`, `destination`, and `BO`.
        Meaning:
        Qualifying bond-order edges are emitted per selected frame.
        """
        frames = _bond_order_frames(data)
        if not frames:
            return ConnectionListResult(
                table=pd.DataFrame(
                    columns=[
                        "frame_idx",
                        "iteration",
                        "source",
                        "source_type",
                        "destination",
                        "destination_type",
                        "BO",
                    ]
                ),
                request=request,
            )

        n_frames = len(frames)
        n_atoms = int(frames[0].shape[0])
        atom_ids = _atom_ids(data, n_atoms)
        atom_types = _atom_types(data, n_atoms)
        iters = _iterations(data, n_frames)
        idx = _frame_indices(n_frames, request.frames, request.every)

        rows: list[dict] = []
        min_bo = float(request.min_bo)
        total = len(idx)
        for step_i, fi in enumerate(idx, start=1):
            mat = _as_dense_frame(frames[fi])
            if mat.shape != (n_atoms, n_atoms):
                raise ValueError("Each bond-order frame must be square with consistent atom dimension.")

            for i in range(n_atoms):
                for j in range(n_atoms):
                    if not request.include_self and i == j:
                        continue
                    bo = float(mat[i, j])
                    if bo < min_bo:
                        continue
                    rows.append(
                        {
                            "frame_idx": int(fi),
                            "iteration": int(iters[fi]),
                            "source": int(atom_ids[i]),
                            "source_type": str(atom_types[i]),
                            "destination": int(atom_ids[j]),
                            "destination_type": str(atom_types[j]),
                            "BO": bo,
                        }
                    )
            if reporter:
                reporter("analyze", step_i, total, "Building connection list")

        if not rows:
            return ConnectionListResult(
                table=pd.DataFrame(
                    columns=[
                        "frame_idx",
                        "iteration",
                        "source",
                        "source_type",
                        "destination",
                        "destination_type",
                        "BO",
                    ]
                ),
                request=request,
            )

        out = pd.DataFrame(rows)
        if request.undirected:
            swap_mask = out["source"] > out["destination"]
            if swap_mask.any():
                src_vals = out.loc[swap_mask, "source"].copy()
                src_type_vals = out.loc[swap_mask, "source_type"].copy()
                out.loc[swap_mask, "source"] = out.loc[swap_mask, "destination"].to_numpy()
                out.loc[swap_mask, "source_type"] = out.loc[swap_mask, "destination_type"].to_numpy()
                out.loc[swap_mask, "destination"] = src_vals.to_numpy()
                out.loc[swap_mask, "destination_type"] = src_type_vals.to_numpy()

            by = ["frame_idx", "iteration", "source", "source_type", "destination", "destination_type"]
            out = out.groupby(by, as_index=False)["BO"].max()

        out = out.sort_values(["frame_idx", "source", "destination"], kind="stable").reset_index(drop=True)
        return ConnectionListResult(table=out, request=request)


@dataclass
class ConnectionTableRequest(BaseRequest):
    """Request for a single-frame connectivity matrix.

    Fields
    -----
    frame : int
        Frame index to extract from connectivity trajectory.
    min_bo : float
        Minimum bond-order threshold for retaining matrix edges.
    undirected : bool
        If `True`, combine directional edges `i->j` and `j->i`.
    fill_value : float
        Fill value for missing pivoted matrix entries.

    Examples
    -----
    ```python
    req = ConnectionTableRequest(frame=0, min_bo=0.3, undirected=True)
    ```
    Sample output:
    `ConnectionTableRequest(...)`
    Meaning:
    The request configures one-frame connectivity-matrix extraction.
    """

    frame: int = dc_field(
        default=0,
        metadata={
            'label': 'Frame',
            'help': "Frame index to extract from the connectivity series. Example: 0.",
        },
    )
    min_bo: float = dc_field(
        default=0.0,
        metadata={
            'label': 'Min Bo',
            'help': "Minimum bond-order threshold for including edges. Example: 0.3.",
            'min': 0.0,
        },
    )
    undirected: bool = dc_field(
        default=True,
        metadata={
            'label': 'Undirected',
            'help': "Merge i-j and j-i into one edge when true.",
            'choices': [True, False],
        },
    )
    fill_value: float = dc_field(
        default=0.0,
        metadata={
            'label': 'Fill Value',
            'help': "Fill value for missing matrix cells after pivot. Example: 0.0.",
        },
    )


@dataclass
class ConnectionTableResult(BaseResult):
    """Connection-table extraction result.

    Fields
    -----
    table : pd.DataFrame
        One-frame connectivity matrix with source IDs on rows and destination
        IDs on columns; values are bond orders.
    request : ConnectionTableRequest
        Request object used for this analysis run.

    Examples
    -----
    ```python
    result = ConnectionTableTask().run(data, req)
    ```
    Sample output:
    Matrix-like DataFrame where `table.loc[src, dst]` is bond order.
    Meaning:
    Missing edges are filled using `request.fill_value`.
    """

    table: pd.DataFrame
    request: ConnectionTableRequest


@register_task("get_connection_table", label="Connection Table")
class ConnectionTableTask(AnalysisTask):
    required_data = ConnectivityData

    @staticmethod
    def recommended_presentations(
        _result: ConnectionTableResult,
        payload: dict[str, Any],
    ) -> list[PresentationSpec]:
        """Build default table/plot presentations for connection-table outputs.

        Works on
        -----
        Analyzer task output payloads

        Parameters
        -----
        _result : ConnectionTableResult
            Analysis result object for the executed task.
        payload : dict[str, Any]
            Serialized result payload used by presentation dispatch.

        Returns
        -----
        list[PresentationSpec]
            Recommended renderer specs for table and optional numeric projection.

        Examples
        -----
        ```python
        specs = ConnectionTableTask.recommended_presentations(result, payload)
        ```
        Sample output:
        Table view and optionally one simple plot mapping.
        Meaning:
        Matrix outputs get at least a tabular default view.
        """
        rows = payload.get("table")
        if not isinstance(rows, list) or not rows:
            return [PresentationSpec(renderer="table", label="Table", view_type="table")]
        sample = rows[0] if isinstance(rows[0], dict) else {}
        if not isinstance(sample, dict):
            return [PresentationSpec(renderer="table", label="Table", view_type="table")]

        x_col = "src" if "src" in sample else next(iter(sample.keys()), "")
        y_col = ""
        for k, v in sample.items():
            if k == x_col:
                continue
            if isinstance(v, (int, float)):
                y_col = k
                break
        if not x_col or not y_col:
            return [PresentationSpec(renderer="table", label="Table", view_type="table")]

        return [
            PresentationSpec(renderer="table", label="Table", view_type="table"),
            PresentationSpec(
                renderer="single_plot",
                label=f"{y_col} vs {x_col}",
                mapping={"x_col": x_col, "y_col": y_col, "group_by_col": ""},
                options={
                    "title": f"Connection Table: {y_col} vs {x_col}",
                    "xlabel": x_col,
                    "ylabel": y_col,
                    "legend": False,
                },
                view_type="plot2d",
            ),
        ]

    def run(self, data: ConnectivityData, request: ConnectionTableRequest, reporter=None) -> ConnectionTableResult:
        """Build a one-frame connectivity matrix from extracted edge list.

        Works on
        -----
        `ConnectivityData` plus `ConnectionTableRequest` analyzer inputs

        Parameters
        -----
        data : ConnectivityData
            Connectivity series containing bond-order matrices and metadata.
        request : ConnectionTableRequest
            One-frame matrix extraction configuration.
        reporter : Any, optional
            Optional progress callback forwarded to edge extraction.

        Returns
        -----
        ConnectionTableResult
            Pivoted connectivity matrix result for selected frame.

        Examples
        -----
        ```python
        result = ConnectionTableTask().run(data, ConnectionTableRequest(frame=0))
        ```
        Sample output:
        `result.table` as a source-by-destination bond-order matrix.
        Meaning:
        Edges are thresholded/normalized before matrix pivot.
        """
        edges = ConnectionListTask().run(
            data,
            ConnectionListRequest(
                frames=[int(request.frame)],
                min_bo=float(request.min_bo),
                undirected=bool(request.undirected),
            ),
            reporter=reporter,
        ).table
        edges = edges.rename(
            columns={
                "iteration": "iter",
                "source": "src",
                "destination": "dst",
                "BO": "bo",
            }
        )
        if edges.empty:
            return ConnectionTableResult(table=pd.DataFrame(), request=request)
        tbl = edges.pivot_table(index="src", columns="dst", values="bo", aggfunc="max", fill_value=float(request.fill_value))
        tbl = tbl.sort_index(axis=0).sort_index(axis=1)
        return ConnectionTableResult(table=tbl, request=request)


@dataclass
class ConnectionStatsRequest(BaseRequest):
    """Request for aggregated connectivity statistics across frames.

    Fields
    -----
    frames : Optional[Sequence[int]]
        Optional frame indices to include. `None` means all frames.
    every : int
        Frame stride after selection. Must be `>= 1`.
    min_bo : float
        Minimum bond-order threshold applied before aggregation.
    undirected : bool
        If `True`, treat `i-j` and `j-i` as one edge pair.
    how : Literal["mean", "max", "count"]
        Aggregation method for pairwise edge statistics.

    Examples
    -----
    ```python
    req = ConnectionStatsRequest(how="mean", min_bo=0.3)
    ```
    Sample output:
    `ConnectionStatsRequest(...)`
    Meaning:
    The request configures pairwise connectivity aggregation behavior.
    """

    frames: Optional[Sequence[int]] = dc_field(
        default=None,
        metadata={
            'label': 'Frames',
            'help': "Optional frame indices to include. Example: [0, 10, 20].",
            'units': 'frame_index',
        },
    )
    every: int = dc_field(
        default=1,
        metadata={
            'label': 'Every',
            'help': "Stride for selected frames. Example: every=5.",
            'min': 1,
            'units': 'frames',
        },
    )
    min_bo: float = dc_field(
        default=0.0,
        metadata={
            'label': 'Min Bo',
            'help': "Minimum BO threshold before aggregation. Example: 0.3.",
            'min': 0.0,
        },
    )
    undirected: bool = dc_field(
        default=True,
        metadata={
            'label': 'Undirected',
            'help': "Merge i-j and j-i into one pair when true.",
            'choices': [True, False],
        },
    )
    how: Literal["mean", "max", "count"] = dc_field(
        default="mean",
        metadata={
            'label': 'How',
            'help': "Aggregation statistic for each source/destination pair across selected frames.",
            'choices': ['mean', 'max', 'count'],
        },
    )


@dataclass
class ConnectionStatsResult(BaseResult):
    """Connection-statistics aggregation result.

    Fields
    -----
    table : pd.DataFrame
        Aggregated pair table with columns `source`, `source_type`,
        `destination`, `destination_type`, and `value`.
    request : ConnectionStatsRequest
        Request object used for this analysis run.

    Notes
    -----
    `value` semantics depend on `request.how` (`mean`, `max`, or `count`).

    Examples
    -----
    ```python
    result = ConnectionStatsTask().run(data, req)
    ```
    Sample output:
    Aggregated rows per atom pair with statistic in `value`.
    Meaning:
    Pairwise connectivity behavior is condensed across selected frames.
    """

    table: pd.DataFrame
    request: ConnectionStatsRequest


@register_task("get_connection_stats", label="Connection Stats")
class ConnectionStatsTask(AnalysisTask):
    required_data = ConnectivityData

    @staticmethod
    def recommended_presentations(
        _result: ConnectionStatsResult,
        payload: dict[str, Any],
    ) -> list[PresentationSpec]:
        """Build default table/plot presentations for connection-stats outputs.

        Works on
        -----
        Analyzer task output payloads

        Parameters
        -----
        _result : ConnectionStatsResult
            Analysis result object for the executed task.
        payload : dict[str, Any]
            Serialized result payload used by presentation dispatch.

        Returns
        -----
        list[PresentationSpec]
            Recommended renderer specs for table and stats plot view.

        Examples
        -----
        ```python
        specs = ConnectionStatsTask.recommended_presentations(result, payload)
        ```
        Sample output:
        Table view plus `value vs source` grouped plot.
        Meaning:
        Aggregated connectivity stats have default visual mappings.
        """
        rows = payload.get("table")
        if not isinstance(rows, list) or not rows:
            return [PresentationSpec(renderer="table", label="Table", view_type="table")]
        sample = rows[0] if isinstance(rows[0], dict) else {}
        if "source" not in sample or "value" not in sample:
            return [PresentationSpec(renderer="table", label="Table", view_type="table")]
        return [
            PresentationSpec(renderer="table", label="Table", view_type="table"),
            PresentationSpec(
                renderer="single_plot",
                label="value vs source",
                mapping={"x_col": "source", "y_col": "value", "group_by_col": "destination"},
                options={
                    "title": "Connection Stats",
                    "xlabel": "source",
                    "ylabel": "value",
                    "legend": True,
                },
                view_type="plot2d",
            ),
        ]

    def run(self, data: ConnectivityData, request: ConnectionStatsRequest, reporter=None) -> ConnectionStatsResult:
        """Aggregate connectivity edges across frames into pairwise statistics.

        Works on
        -----
        `ConnectivityData` plus `ConnectionStatsRequest` analyzer inputs

        Parameters
        -----
        data : ConnectivityData
            Connectivity series containing bond-order matrices and metadata.
        request : ConnectionStatsRequest
            Aggregation configuration (`how`, thresholds, and frame selection).
        reporter : Any, optional
            Optional progress callback forwarded to edge extraction.

        Returns
        -----
        ConnectionStatsResult
            Pairwise aggregated connectivity statistic table.

        Examples
        -----
        ```python
        result = ConnectionStatsTask().run(data, ConnectionStatsRequest(how="count"))
        ```
        Sample output:
        `result.table` with one row per source-destination pair.
        Meaning:
        Edge behavior is summarized according to the selected aggregation method.
        """
        edges = ConnectionListTask().run(
            data,
            ConnectionListRequest(
                frames=request.frames,
                every=request.every,
                min_bo=request.min_bo,
                undirected=request.undirected,
            ),
            reporter=reporter,
        ).table
        if edges.empty:
            return ConnectionStatsResult(
                table=pd.DataFrame(
                    columns=[
                        "source",
                        "source_type",
                        "destination",
                        "destination_type",
                        "value",
                    ]
                ),
                request=request,
            )

        by = ["source", "source_type", "destination", "destination_type"]
        if request.how == "count":
            out = edges.groupby(by, as_index=False).size().rename(columns={"size": "value"})
        elif request.how == "max":
            out = edges.groupby(by, as_index=False)["BO"].max().rename(columns={"BO": "value"})
        else:
            out = edges.groupby(by, as_index=False)["BO"].mean().rename(columns={"BO": "value"})
        out = out.sort_values(["source", "destination"], kind="stable").reset_index(drop=True)
        return ConnectionStatsResult(table=out, request=request)


@dataclass
class BondEventsRequest(BaseRequest):
    """Request for bond formation/breakage event detection.

    Fields
    -----
    frames : Optional[Sequence[int]]
        Optional frame indices to include. `None` means all frames.
    every : int
        Frame stride after selection. Must be `>= 1`.
    src : Optional[int]
        Optional source atom-id filter.
    dst : Optional[int]
        Optional destination atom-id filter.
    threshold : float
        Schmitt-trigger center threshold for bonded/unbonded transitions.
    hysteresis : float
        Schmitt-trigger hysteresis half-width around `threshold`.
    smooth : Optional[Literal["ma", "ema"]]
        Optional prefilter for BO traces before transition detection.
    window : int
        Smoothing window size used by moving-average or EMA heuristics.
    ema_alpha : Optional[float]
        Optional explicit EMA alpha (otherwise inferred from window).
    min_run : int
        Minimum consecutive-state run length after hysteresis cleanup.
    undirected : bool
        If `True`, treat `i-j` and `j-i` as one bond.

    Notes
    -----
    Upstream `min_bo` controls which BO entries exist, while `threshold` and
    `hysteresis` govern event transition detection on those traces.

    Examples
    -----
    ```python
    req = BondEventsRequest(threshold=0.35, hysteresis=0.05, smooth="ma")
    ```
    Sample output:
    `BondEventsRequest(...)`
    Meaning:
    The request configures bond event detection and optional smoothing logic.

    Parameters
    ----------
    frames
        Optional frame indices to include. If omitted, all frames are used.
        Example: ``[0, 10, 20]``.
    every
        Frame stride after selection. Example: ``every=5``.
    src
        Optional source atom-id filter. Example: ``src=12``.
    dst
        Optional destination atom-id filter. Example: ``dst=27``.
    threshold
        Central threshold for Schmitt trigger transition logic.
        Example: ``threshold=0.35``.
        NOTE: min_bo filters which bond-order entries exist in the input data,
        while threshold (with hysteresis) decides when a bond’s time-series
        crosses between unbonded and bonded states to mark breakage/formation events.
    hysteresis
        Hysteresis half-width around ``threshold``.
        Example: ``hysteresis=0.05``.
    smooth
        Optional pre-filter on BO traces before event detection:
        ``ma`` for moving average, ``ema`` for exponential moving average.
    window
        Smoothing window size used by ``smooth`` method. Example: ``window=9``.
    ema_alpha
        Optional explicit EMA alpha; if omitted, alpha is inferred from window.
    min_run
        Minimum consecutive-state run length after hysteresis, used to clean
        flicker. Example: ``min_run=3``.
    undirected
        If ``True``, treat ``i-j`` and ``j-i`` as the same bond.
    """

    frames: Optional[Sequence[int]] = dc_field(
        default=None,
        metadata={
            'label': 'Frames',
            'help': "Optional frame indices to include. Example: [0, 10, 20].",
            'units': 'frame_index',
        },
    )
    every: int = dc_field(
        default=1,
        metadata={
            'label': 'Every',
            'help': "Stride for selected frames. Example: every=5.",
            'min': 1,
            'units': 'frames',
        },
    )
    src: Optional[int] = dc_field(
        default=None,
        metadata={
            'label': 'Source',
            'help': "Optional source atom id filter. Example: 12.",
            'units': 'index',
        },
    )
    dst: Optional[int] = dc_field(
        default=None,
        metadata={
            'label': 'Destination',
            'help': "Optional destination atom id filter. Example: 27.",
            'units': 'index',
        },
    )
    threshold: float = dc_field(
        default=0.35,
        metadata={
            'label': 'Threshold',
            'help': "Schmitt trigger threshold for switching bonded/unbonded states. Example: 0.35.",
            'min': 0.0,
        },
    )
    hysteresis: float = dc_field(
        default=0.05,
        metadata={
            'label': 'Hysteresis',
            'help': "Hysteresis half-width around threshold for robust transitions. Example: 0.05.",
            'min': 0.0,
        },
    )
    smooth: Optional[Literal["ma", "ema"]] = dc_field(
        default="ma",
        metadata={
            'label': 'Smooth',
            'help': "Optional smoothing method applied before event detection.",
            'choices': ['ma', 'ema'],
        },
    )
    window: int = dc_field(
        default=7,
        metadata={
            'label': 'Window',
            'help': "Smoothing window size for moving-average based preprocessing. Example: 7.",
            'min': 1,
        },
    )
    ema_alpha: Optional[float] = dc_field(
        default=None,
        metadata={
            'label': 'Ema Alpha',
            'help': "Optional EMA alpha override. Example: 0.2.",
        },
    )
    min_run: int = dc_field(
        default=3,
        metadata={
            'label': 'Min Run',
            'help': "Minimum consecutive state length after hysteresis cleanup. Example: 3.",
            'min': 1,
        },
    )
    undirected: bool = dc_field(
        default=True,
        metadata={
            'label': 'Undirected',
            'help': "Merge i-j and j-i into one bond when true.",
            'choices': [True, False],
        },
    )


@dataclass
class BondEventsResult(BaseResult):
    """Bond-event detection result.

    Fields
    -----
    table : pd.DataFrame
        Event table with columns `source`, `source_type`, `destination`,
        `destination_type`, `event`, `frame_idx`, `iter`, `bo_at_event`,
        `threshold`, and `hysteresis`.
    request : BondEventsRequest
        Request object used for this analysis run.

    Examples
    -----
    ```python
    result = BondEventsTask().run(data, req)
    result.table.head()
    ```
    Sample output:
    Event rows labeled as `formation` or `breakage`.
    Meaning:
    Each row marks one detected bond-state transition.

    Output structure
    ----------------
    - ``request``: the :class:`BondEventsRequest` used for event detection.
    - ``table``: pandas.DataFrame with columns:
      ``['source', 'source_type', 'destination', 'destination_type', 'event', 'frame_idx', 'iter',
      'bo_at_event', 'threshold', 'hysteresis']``.

    Column meanings
    ---------------
    - ``source`` / ``destination``: atom-id pair where the event was detected.
    - ``source_type`` / ``destination_type``: atom element/type labels.
    - ``event``: event class, one of ``formation`` or ``breakage``.
    - ``frame_idx`` / ``iter``: frame and iteration location for the event.
    - ``bo_at_event``: BO value (possibly smoothed) at detection point.
    - ``threshold`` / ``hysteresis``: parameters used for detection.

    Example
    -------
    A row like ``source=3, source_type='C', destination=7, destination_type='O',
    event='formation', frame_idx=12, iter=1200, bo_at_event=0.41`` indicates
    bond 3-7 formed at that point.
    """

    table: pd.DataFrame
    request: BondEventsRequest


@register_task("get_bond_events", label="Bond Events")
class BondEventsTask(AnalysisTask):
    required_data = ConnectivityData

    @staticmethod
    def recommended_presentations(
        _result: BondEventsResult,
        payload: dict[str, Any],
    ) -> list[PresentationSpec]:
        """Build default table/plot presentations for bond-event outputs.

        Works on
        -----
        Analyzer task output payloads

        Parameters
        -----
        _result : BondEventsResult
            Analysis result object for the executed task.
        payload : dict[str, Any]
            Serialized result payload used by presentation dispatch.

        Returns
        -----
        list[PresentationSpec]
            Recommended renderer specs for event table and event trace plot.

        Examples
        -----
        ```python
        specs = BondEventsTask.recommended_presentations(result, payload)
        ```
        Sample output:
        Table view plus `bo_at_event vs iter` grouped by event type.
        Meaning:
        Detected events can be visualized with default mappings.
        """
        rows = payload.get("table")
        if not isinstance(rows, list) or not rows:
            return [PresentationSpec(renderer="table", label="Table", view_type="table")]
        sample = rows[0] if isinstance(rows[0], dict) else {}
        if "iter" not in sample or "bo_at_event" not in sample:
            return [PresentationSpec(renderer="table", label="Table", view_type="table")]
        return [
            PresentationSpec(renderer="table", label="Table", view_type="table"),
            PresentationSpec(
                renderer="single_plot",
                label="bo_at_event vs iter",
                mapping={"x_col": "iter", "y_col": "bo_at_event", "group_by_col": "event"},
                options={
                    "title": "Bond Events",
                    "xlabel": "iter",
                    "ylabel": "bo_at_event",
                    "legend": True,
                },
                view_type="plot2d",
            ),
        ]

    def run(self, data: ConnectivityData, request: BondEventsRequest, reporter=None) -> BondEventsResult:
        """Detect bond formation and breakage transitions from BO time series.

        Works on
        -----
        `ConnectivityData` plus `BondEventsRequest` analyzer inputs

        Parameters
        -----
        data : ConnectivityData
            Connectivity series containing bond-order matrices and metadata.
        request : BondEventsRequest
            Event detection, smoothing, and filtering configuration.
        reporter : Any, optional
            Optional progress callback invoked during grouped event detection.

        Returns
        -----
        BondEventsResult
            Event transition table for selected bond pairs.

        Examples
        -----
        ```python
        result = BondEventsTask().run(data, BondEventsRequest(src=1, dst=2))
        ```
        Sample output:
        `result.table` rows with `event`, `frame_idx`, and `bo_at_event`.
        Meaning:
        Bond-state transitions are extracted using hysteresis logic.
        """
        ts = ConnectionListTask().run(
            data,
            ConnectionListRequest(
                frames=request.frames,
                every=request.every,
                min_bo=0.0,
                undirected=request.undirected,
                include_self=False,
            ),
            reporter=reporter,
        ).table
        ts = ts.rename(
            columns={
                "iteration": "iter",
                "BO": "bo",
            }
        )
        if ts.empty:
            return BondEventsResult(
                table=pd.DataFrame(
                    columns=[
                        "source",
                        "source_type",
                        "destination",
                        "destination_type",
                        "event",
                        "frame_idx",
                        "iter",
                        "bo_at_event",
                        "threshold",
                        "hysteresis",
                    ]
                ),
                request=request,
            )

        if request.src is not None and request.dst is not None:
            a, b = (int(request.src), int(request.dst))
            if request.undirected and a > b:
                a, b = b, a
            ts = ts[(ts["source"] == a) & (ts["destination"] == b)].copy()
            if ts.empty:
                return BondEventsResult(
                    table=pd.DataFrame(
                        columns=[
                            "source",
                            "source_type",
                            "destination",
                            "destination_type",
                            "event",
                            "frame_idx",
                            "iter",
                            "bo_at_event",
                            "threshold",
                            "hysteresis",
                        ]
                    ),
                    request=request,
                )

        groups = ts.groupby(
            ["source", "source_type", "destination", "destination_type"],
            sort=False,
        )
        out_rows: list[pd.DataFrame] = []

        total_groups = len(groups)
        for group_i, ((source, source_type, destination, destination_type), g) in enumerate(groups, start=1):
            g = g.sort_values(["frame_idx"]).reset_index(drop=True)
            bo = g["bo"].to_numpy(dtype=float)

            if request.smooth is None:
                bo_s = bo
            elif request.smooth == "ema":
                bo_s = exponential_moving_average(pd.Series(bo), window=int(request.window), alpha=request.ema_alpha, adjust=False).to_numpy()
            else:
                bo_s = simple_moving_average(pd.Series(bo), window=int(request.window), center=True, min_periods=1).to_numpy()

            st = schmitt_hysteresis(bo_s, th=float(request.threshold), hys=float(request.hysteresis))
            st = clean_flicker(st, min_run=int(request.min_run))

            prev = np.r_[st[0], st[:-1]]
            rising = (~prev) & st
            falling = prev & (~st)
            mask = rising | falling
            if not mask.any():
                continue

            ev = g.loc[mask, ["frame_idx", "iter"]].reset_index(drop=True)
            ev.insert(0, "destination_type", str(destination_type))
            ev.insert(0, "destination", int(destination))
            ev.insert(0, "source_type", str(source_type))
            ev.insert(0, "source", int(source))
            ev["event"] = np.where(rising[mask], "formation", "breakage")
            ev["bo_at_event"] = bo_s[mask]
            ev["threshold"] = float(request.threshold)
            ev["hysteresis"] = float(request.hysteresis)
            out_rows.append(
                ev[
                    [
                        "source",
                        "source_type",
                        "destination",
                        "destination_type",
                        "event",
                        "frame_idx",
                        "iter",
                        "bo_at_event",
                        "threshold",
                        "hysteresis",
                    ]
                ]
            )
            if reporter:
                reporter("analyze", group_i, total_groups, "Detecting bond events")

        if not out_rows:
            return BondEventsResult(
                table=pd.DataFrame(
                    columns=[
                        "source",
                        "source_type",
                        "destination",
                        "destination_type",
                        "event",
                        "frame_idx",
                        "iter",
                        "bo_at_event",
                        "threshold",
                        "hysteresis",
                    ]
                ),
                request=request,
            )
        out = pd.concat(out_rows, ignore_index=True)
        out = out.sort_values(["source", "destination", "iter", "event"], kind="stable").reset_index(drop=True)
        return BondEventsResult(table=out, request=request)


__all__ = [
    "ConnectionListRequest",
    "ConnectionListResult",
    "ConnectionListTask",
    "ConnectionTableRequest",
    "ConnectionTableResult",
    "ConnectionTableTask",
    "ConnectionStatsRequest",
    "ConnectionStatsResult",
    "ConnectionStatsTask",
    "BondEventsRequest",
    "BondEventsResult",
    "BondEventsTask",
]
