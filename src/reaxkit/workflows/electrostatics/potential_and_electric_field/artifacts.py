"""CSV and plot artifacts for local electrostatics results."""

from __future__ import annotations

from reaxkit.presentation.workflow_artifacts import write_workflow_csv

import re
from pathlib import Path

import numpy as np
import pandas as pd

from reaxkit.analysis.electrostatics.potential_and_electric_field.spatial import (
    bin_probe_table, global_edges, plot_binned_frame, plot_binned_kymograph,
)

KYMOGRAPH_VALUES = (
    "internal-potential", "external-potential", "total-local-potential",
    "internal-field", "external-field", "total-local-field",
)


def _safe(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip()) or "probe"


def write_tables(result, output: Path) -> dict[str, Path]:
    output.mkdir(parents=True, exist_ok=True)
    fields = output / "voltages_and_electric_fields"; fields.mkdir(exist_ok=True)
    paths = {"coulomb_per_atom": output / "coulomb_per_atom.csv",
             "coulomb_totals": output / "coulomb_totals.csv",
             "probe_average": fields / "probe_average_per_atom.csv"}
    write_workflow_csv(result.coulomb_table, paths["coulomb_per_atom"], index=False)
    write_workflow_csv(result.totals, paths["coulomb_totals"], index=False)
    write_workflow_csv(result.table, paths["probe_average"], index=False)
    for label, table in result.probe_tables.items():
        path = fields / f"probe_{_safe(label)}_per_atom.csv"; write_workflow_csv(table, path, index=False); paths[f"probe_{label}"] = path
    result.prewritten_csvs = tuple(paths.values())
    result.skip_automatic_csv_persistence = True
    return paths


def _kymograph_column(value: str, component: str, units: str) -> str:
    value = str(value).strip().lower()
    if value not in KYMOGRAPH_VALUES:
        raise ValueError(f"Unsupported kymograph value {value!r}.")
    label, kind = value.rsplit("-", 1)
    label = label.replace("-", "_")
    if kind == "potential":
        return f"{label}_potential (V)"
    suffix = "MV/cm" if units == "mv/cm" else "V/angstrom"
    prefix = (f"magnitude_of_average_{label}_electric_field" if component == "magnitude" else
              f"{label}_electric_field_magnitude" if component == "mean-magnitude" else
              f"{label}_electric_field_{component}")
    return f"{prefix} ({suffix})"


def _write_kymographs(binned: dict[tuple[str, int], pd.DataFrame], directory: Path,
                      *, axis: str, spatial_edges: np.ndarray, values,
                      component: str, units: str, time_axis: str, dpi: int) -> list[Path]:
    time_column = "frame_index" if time_axis == "frame" else "iter"
    requested = tuple(dict.fromkeys(str(value).strip().lower() for value in values))
    unsupported = [value for value in requested if value not in KYMOGRAPH_VALUES]
    if unsupported:
        raise ValueError(f"Unsupported kymograph value(s): {unsupported}.")
    by_probe: dict[str, list[pd.DataFrame]] = {}
    for (label, _frame), table in binned.items():
        by_probe.setdefault(label, []).append(table)
    combined = {label: pd.concat(tables, ignore_index=True) for label, tables in by_probe.items()}
    output = directory / "kymographs"; output.mkdir(exist_ok=True)
    paths: list[Path] = []
    for value in requested:
        column = _kymograph_column(value, component, units)
        finite = np.concatenate([table[column].to_numpy(dtype=float) for table in combined.values()])
        finite = finite[np.isfinite(finite)]
        signed = value.endswith("-potential") or component not in {"magnitude", "mean-magnitude"}
        if signed:
            limit = float(np.max(np.abs(finite))) if len(finite) else None
            vmin, vmax, cmap = (-limit if limit is not None else None,
                                 limit if limit is not None else None, "coolwarm_r")
        else:
            vmin, vmax, cmap = (0.0, float(np.max(finite)) if len(finite) else None, "viridis")
        for label, table in combined.items():
            suffix = f"_{component}" if value.endswith("-field") else ""
            path = output / f"probe_{_safe(label)}_kymograph_{axis}_{value}{suffix}_by_{time_axis}.png"
            title = f"{label} probe: {column} along {axis}"
            paths.append(plot_binned_kymograph(
                table, axis, spatial_edges, path, value_column=column,
                time_column=time_column, title=title, cmap=cmap,
                vmin=vmin, vmax=vmax, dpi=dpi,
            ))
    return paths


def write_binning(result, output: Path, *, axes: str, bins, plot: bool,
                  component: str, units: str, dpi: int = 180,
                  kymograph: bool = False,
                  kymograph_values=("total-local-potential", "total-local-field"),
                  kymograph_time_axis: str = "frame") -> list[Path]:
    directory = output / "binned_data_and_plots"; directory.mkdir(parents=True, exist_ok=True)
    datasets = {**result.probe_tables, "average": result.table}
    edges = global_edges(list(datasets.values()), axes, bins)
    from contextlib import ExitStack
    from reaxkit.core.runtime.artifacts import TableChunks, TableSpool
    from reaxkit.core.runtime.result_store import table_chunks
    paths: list[Path] = []
    with ExitStack() as stack:
        spools = {}
        for label, table in datasets.items():
            spool = spools[label] = TableSpool()
            stack.callback(spool.close)
            groups = (item for chunk in table_chunks(table) for item in chunk.groupby("frame_index", sort=True))
            for frame, frame_data in groups:
                values = bin_probe_table(frame_data, axes, edges)
                values.insert(0, "probe_element", label)
                values.insert(0, "iter", int(frame_data["iter"].iloc[0]))
                values.insert(0, "frame_index", int(frame))
                spool.append(values)
            path = directory / f"probe_{_safe(label)}_bins_{axes}.csv"
            write_workflow_csv(TableChunks(spool), path, index=False)
            paths.append(path)
        if plot:
            suffix = "MV/cm" if units == "mv/cm" else "V/angstrom"
            prefix = ("magnitude_of_average_total_local_electric_field" if component == "magnitude" else
                      "total_local_electric_field_magnitude" if component == "mean-magnitude" else
                      f"total_local_electric_field_{component}")
            column = f"{prefix} ({suffix})"
            lower, upper = np.inf, -np.inf
            for spool in spools.values():
                for table in spool:
                    finite = table[column].to_numpy(dtype=float)
                    finite = finite[np.isfinite(finite)]
                    if len(finite):
                        lower, upper = min(lower, finite.min()), max(upper, finite.max())
            for label, spool in spools.items():
                for table in spool:
                    frame = int(table["frame_index"].iloc[0])
                    path = directory / f"probe_{_safe(label)}_bins_{axes}_frame_{frame}.png"
                    plot_binned_frame(table, axes, path, component=component, units=units,
                        vmin=float(lower) if np.isfinite(lower) else None,
                        vmax=float(upper) if np.isfinite(upper) else None, dpi=dpi)
                    paths.append(path)
        if kymograph:
            if len(edges) != 1:
                raise ValueError("Kymographs require exactly one bin axis: x, y, or z.")
            if kymograph_time_axis not in {"frame", "iteration"}:
                raise ValueError("Kymograph time axis must be frame or iteration.")
            binned = {(label, int(table["frame_index"].iloc[0])): table for label, spool in spools.items() for table in spool}
            paths.extend(_write_kymographs(binned, directory, axis=str(axes), spatial_edges=np.asarray(edges[0]),
                values=kymograph_values, component=component, units=units, time_axis=kymograph_time_axis, dpi=dpi))
    return paths


__all__ = ["KYMOGRAPH_VALUES", "write_binning", "write_tables"]
