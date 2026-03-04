"""Engine-agnostic connectivity and bond-event analysis tasks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Sequence

import numpy as np
import pandas as pd

from reaxkit.analysis.base import AnalysisTask
from reaxkit.core.analysis_task_registry import register_task
from reaxkit.domain.base_request import BaseRequest
from reaxkit.domain.base_result import BaseResult
from reaxkit.domain.data_models import ConnectivityData
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
    frames: Optional[Sequence[int]] = None
    every: int = 1
    min_bo: float = 0.0
    undirected: bool = True
    aggregate: Literal["max", "mean"] = "max"
    include_self: bool = False


@dataclass
class ConnectionListResult(BaseResult):
    table: pd.DataFrame


@register_task("connection_list")
class ConnectionListTask(AnalysisTask):
    required_data = ConnectivityData

    def run(self, data: ConnectivityData, request: ConnectionListRequest) -> ConnectionListResult:
        frames = _bond_order_frames(data)
        if not frames:
            return ConnectionListResult(table=pd.DataFrame(columns=["frame_idx", "iter", "src", "dst", "bo"]))

        n_frames = len(frames)
        n_atoms = int(frames[0].shape[0])
        atom_ids = _atom_ids(data, n_atoms)
        iters = _iterations(data, n_frames)
        idx = _frame_indices(n_frames, request.frames, request.every)

        rows: list[dict] = []
        min_bo = float(request.min_bo)
        for fi in idx:
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
                            "iter": int(iters[fi]),
                            "src": int(atom_ids[i]),
                            "dst": int(atom_ids[j]),
                            "bo": bo,
                        }
                    )

        if not rows:
            return ConnectionListResult(table=pd.DataFrame(columns=["frame_idx", "iter", "src", "dst", "bo"]))

        out = pd.DataFrame(rows)
        if request.undirected:
            src_min = out[["src", "dst"]].min(axis=1)
            dst_max = out[["src", "dst"]].max(axis=1)
            out["src"], out["dst"] = src_min, dst_max
            by = ["frame_idx", "iter", "src", "dst"]
            if request.aggregate == "mean":
                out = out.groupby(by, as_index=False)["bo"].mean()
            else:
                out = out.groupby(by, as_index=False)["bo"].max()

        out = out.sort_values(["frame_idx", "src", "dst"], kind="stable").reset_index(drop=True)
        return ConnectionListResult(table=out)


@dataclass
class ConnectionTableRequest(BaseRequest):
    frame: int = 0
    min_bo: float = 0.0
    undirected: bool = True
    fill_value: float = 0.0


@dataclass
class ConnectionTableResult(BaseResult):
    table: pd.DataFrame


@register_task("connection_table")
class ConnectionTableTask(AnalysisTask):
    required_data = ConnectivityData

    def run(self, data: ConnectivityData, request: ConnectionTableRequest) -> ConnectionTableResult:
        edges = ConnectionListTask().run(
            data,
            ConnectionListRequest(
                frames=[int(request.frame)],
                min_bo=float(request.min_bo),
                undirected=bool(request.undirected),
            ),
        ).table
        if edges.empty:
            return ConnectionTableResult(table=pd.DataFrame())
        tbl = edges.pivot_table(index="src", columns="dst", values="bo", aggfunc="max", fill_value=float(request.fill_value))
        tbl = tbl.sort_index(axis=0).sort_index(axis=1)
        return ConnectionTableResult(table=tbl)


@dataclass
class ConnectionStatsRequest(BaseRequest):
    frames: Optional[Sequence[int]] = None
    every: int = 1
    min_bo: float = 0.0
    undirected: bool = True
    how: Literal["mean", "max", "count"] = "mean"


@dataclass
class ConnectionStatsResult(BaseResult):
    table: pd.DataFrame


@register_task("connection_stats")
class ConnectionStatsTask(AnalysisTask):
    required_data = ConnectivityData

    def run(self, data: ConnectivityData, request: ConnectionStatsRequest) -> ConnectionStatsResult:
        edges = ConnectionListTask().run(
            data,
            ConnectionListRequest(
                frames=request.frames,
                every=request.every,
                min_bo=request.min_bo,
                undirected=request.undirected,
            ),
        ).table
        if edges.empty:
            return ConnectionStatsResult(table=pd.DataFrame(columns=["src", "dst", "value"]))

        by = ["src", "dst"]
        if request.how == "count":
            out = edges.groupby(by, as_index=False).size().rename(columns={"size": "value"})
        elif request.how == "max":
            out = edges.groupby(by, as_index=False)["bo"].max().rename(columns={"bo": "value"})
        else:
            out = edges.groupby(by, as_index=False)["bo"].mean().rename(columns={"bo": "value"})
        return ConnectionStatsResult(table=out.sort_values(["src", "dst"], kind="stable").reset_index(drop=True))


@dataclass
class BondTimeseriesRequest(BaseRequest):
    frames: Optional[Sequence[int]] = None
    every: int = 1
    undirected: bool = True
    bo_threshold: float = 0.0


@dataclass
class BondTimeseriesResult(BaseResult):
    table: pd.DataFrame


@register_task("bond_timeseries")
class BondTimeseriesTask(AnalysisTask):
    required_data = ConnectivityData

    def run(self, data: ConnectivityData, request: BondTimeseriesRequest) -> BondTimeseriesResult:
        edges = ConnectionListTask().run(
            data,
            ConnectionListRequest(
                frames=request.frames,
                every=request.every,
                min_bo=0.0,
                undirected=request.undirected,
            ),
        ).table
        if edges.empty:
            return BondTimeseriesResult(table=pd.DataFrame(columns=["frame_idx", "iter", "src", "dst", "bo"]))

        # one row per frame/pair with max BO
        edges = edges.groupby(["frame_idx", "iter", "src", "dst"], as_index=False)["bo"].max()

        frames = _bond_order_frames(data)
        n_frames = len(frames)
        idx = _frame_indices(n_frames, request.frames, request.every)
        iters = _iterations(data, n_frames)
        frame_meta = pd.DataFrame({"frame_idx": idx, "iter": [int(iters[i]) for i in idx]})

        all_bonds = sorted(set(zip(edges["src"], edges["dst"])))
        pivot = edges.pivot_table(index=["frame_idx", "iter"], columns=["src", "dst"], values="bo", aggfunc="max")
        pivot = pivot.reindex(index=pd.MultiIndex.from_frame(frame_meta[["frame_idx", "iter"]]), fill_value=0.0)
        pivot = pivot.reindex(columns=pd.MultiIndex.from_tuples(all_bonds, names=["src", "dst"]), fill_value=0.0)
        pivot = pivot.fillna(0.0)

        if request.bo_threshold > 0.0:
            pivot = pivot.mask(pivot < float(request.bo_threshold), 0.0)

        # tidy
        try:
            stacked = pivot.stack(future_stack=True)
        except TypeError:
            stacked = pivot.stack()
        if isinstance(stacked, pd.Series):
            tidy = stacked.rename("bo").reset_index()
        else:
            if stacked.shape[1] == 1:
                tidy = stacked.rename(columns={stacked.columns[0]: "bo"}).reset_index()
            else:
                rows = []
                for (fi, it), row in pivot.iterrows():
                    for (src, dst), bo in row.items():
                        rows.append([fi, it, src, dst, bo])
                tidy = pd.DataFrame(rows, columns=["frame_idx", "iter", "src", "dst", "bo"])
                return BondTimeseriesResult(
                    table=tidy.sort_values(["frame_idx", "src", "dst"], kind="stable").reset_index(drop=True)
                )

        if "src" not in tidy.columns or "dst" not in tidy.columns:
            names = pivot.columns.names or ["src", "dst"]
            rename_map = {}
            if names[0] in tidy.columns:
                rename_map[names[0]] = "src"
            if names[1] in tidy.columns:
                rename_map[names[1]] = "dst"
            tidy = tidy.rename(columns=rename_map)
            if "src" not in tidy.columns or "dst" not in tidy.columns:
                lvl_cols = [c for c in tidy.columns if str(c).startswith("level_")]
                if "src" not in tidy.columns and lvl_cols:
                    tidy = tidy.rename(columns={lvl_cols[0]: "src"})
                if "dst" not in tidy.columns and len(lvl_cols) > 1:
                    tidy = tidy.rename(columns={lvl_cols[1]: "dst"})

        tidy = tidy.sort_values(["frame_idx", "src", "dst"], kind="stable").reset_index(drop=True)
        return BondTimeseriesResult(table=tidy)


@dataclass
class BondEventsRequest(BaseRequest):
    frames: Optional[Sequence[int]] = None
    every: int = 1
    src: Optional[int] = None
    dst: Optional[int] = None
    threshold: float = 0.35
    hysteresis: float = 0.05
    smooth: Optional[Literal["ma", "ema"]] = "ma"
    window: int = 7
    ema_alpha: Optional[float] = None
    min_run: int = 3
    xaxis: Literal["iter", "frame"] = "iter"
    undirected: bool = True


@dataclass
class BondEventsResult(BaseResult):
    table: pd.DataFrame


@register_task("bond_events")
class BondEventsTask(AnalysisTask):
    required_data = ConnectivityData

    def run(self, data: ConnectivityData, request: BondEventsRequest) -> BondEventsResult:
        ts = BondTimeseriesTask().run(
            data,
            BondTimeseriesRequest(
                frames=request.frames,
                every=request.every,
                undirected=request.undirected,
                bo_threshold=0.0,
                as_wide=False,
            ),
        ).table
        if ts.empty:
            return BondEventsResult(
                table=pd.DataFrame(
                    columns=["src", "dst", "event", "frame_idx", "iter", "x_axis", "bo_at_event", "threshold", "hysteresis"]
                )
            )

        if request.src is not None and request.dst is not None:
            a, b = (int(request.src), int(request.dst))
            if request.undirected and a > b:
                a, b = b, a
            ts = ts[(ts["src"] == a) & (ts["dst"] == b)].copy()
            if ts.empty:
                return BondEventsResult(
                    table=pd.DataFrame(
                        columns=["src", "dst", "event", "frame_idx", "iter", "x_axis", "bo_at_event", "threshold", "hysteresis"]
                    )
                )

        groups = ts.groupby(["src", "dst"], sort=False)
        xcol = "iter" if request.xaxis == "iter" else "frame_idx"
        out_rows: list[pd.DataFrame] = []

        for (a, b), g in groups:
            g = g.sort_values(["frame_idx"]).reset_index(drop=True)
            x = g[xcol].to_numpy()
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

            ev = pd.DataFrame({xcol: x[mask], "event": np.where(rising[mask], "formation", "breakage"), "bo_at_event": bo_s[mask]})
            ev.insert(0, "dst", b)
            ev.insert(0, "src", a)

            merge_cols = ["frame_idx", "iter"]
            if xcol not in merge_cols:
                merge_cols.append(xcol)
            meta = g[merge_cols].drop_duplicates(subset=[xcol])
            ev = ev.merge(meta, on=xcol, how="left")
            ev["x_axis"] = ev[xcol]
            ev["threshold"] = float(request.threshold)
            ev["hysteresis"] = float(request.hysteresis)
            out_rows.append(ev[["src", "dst", "event", "frame_idx", "iter", "x_axis", "bo_at_event", "threshold", "hysteresis"]])

        if not out_rows:
            return BondEventsResult(
                table=pd.DataFrame(
                    columns=["src", "dst", "event", "frame_idx", "iter", "x_axis", "bo_at_event", "threshold", "hysteresis"]
                )
            )
        out = pd.concat(out_rows, ignore_index=True)
        out = out.sort_values(["src", "dst", "x_axis", "event"], kind="stable").reset_index(drop=True)
        return BondEventsResult(table=out)


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
    "BondTimeseriesRequest",
    "BondTimeseriesResult",
    "BondTimeseriesTask",
    "BondEventsRequest",
    "BondEventsResult",
    "BondEventsTask",
]
