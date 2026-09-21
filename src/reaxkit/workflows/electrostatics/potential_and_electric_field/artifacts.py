"""CSV and plot artifacts for local electrostatics results."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from reaxkit.analysis.electrostatics.potential_and_electric_field.spatial import (
    bin_probe_table, global_edges, plot_binned_frame,
)


def _safe(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip()) or "probe"


def write_tables(result, output: Path) -> dict[str, Path]:
    output.mkdir(parents=True, exist_ok=True)
    fields = output / "voltages_and_electric_fields"; fields.mkdir(exist_ok=True)
    paths = {"coulomb_per_atom": output / "coulomb_per_atom.csv",
             "coulomb_totals": output / "coulomb_totals.csv",
             "probe_average": fields / "probe_average_per_atom.csv"}
    result.coulomb_table.to_csv(paths["coulomb_per_atom"], index=False)
    result.totals.to_csv(paths["coulomb_totals"], index=False)
    result.table.to_csv(paths["probe_average"], index=False)
    for label, table in result.probe_tables.items():
        path = fields / f"probe_{_safe(label)}_per_atom.csv"; table.to_csv(path, index=False); paths[f"probe_{label}"] = path
    return paths


def write_binning(result, output: Path, *, axes: str, bins, plot: bool,
                  component: str, units: str, dpi: int = 180) -> list[Path]:
    directory = output / "binned_data_and_plots"; directory.mkdir(parents=True, exist_ok=True)
    datasets = {**result.probe_tables, "average": result.table}
    edges = global_edges(list(datasets.values()), axes, bins)
    binned: dict[tuple[str, int], pd.DataFrame] = {}
    paths: list[Path] = []
    for label, table in datasets.items():
        chunks = []
        for frame, frame_table in table.groupby("frame_index", sort=True):
            values = bin_probe_table(frame_table, axes, edges)
            values.insert(0, "probe_element", label); values.insert(0, "iter", int(frame_table["iter"].iloc[0])); values.insert(0, "frame_index", int(frame))
            binned[(label, int(frame))] = values; chunks.append(values)
        path = directory / f"probe_{_safe(label)}_bins_{axes}.csv"
        pd.concat(chunks, ignore_index=True).to_csv(path, index=False); paths.append(path)
    if plot:
        suffix = "MV/cm" if units == "mv/cm" else "V/angstrom"
        prefix = "magnitude_of_average_field" if component == "magnitude" else "electric_field_magnitude" if component == "mean-magnitude" else f"electric_field_{component}"
        column = f"{prefix} ({suffix})"
        finite = np.concatenate([table[column].to_numpy(dtype=float) for table in binned.values()])
        finite = finite[np.isfinite(finite)]; vmin = float(finite.min()) if len(finite) else None; vmax = float(finite.max()) if len(finite) else None
        for (label, frame), table in binned.items():
            path = directory / f"probe_{_safe(label)}_bins_{axes}_frame_{frame}.png"
            plot_binned_frame(table, axes, path, component=component, units=units, vmin=vmin, vmax=vmax, dpi=dpi); paths.append(path)
    return paths


__all__ = ["write_binning", "write_tables"]
