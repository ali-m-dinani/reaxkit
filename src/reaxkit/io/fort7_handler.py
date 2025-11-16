"""handler for parsing and cleaning data in fort.7 file"""
from __future__ import annotations
from pathlib import Path
from typing import List, Dict, Any, Optional
import pandas as pd

from reaxkit.io.file_handler import FileHandler


class Fort7Handler(FileHandler):
    """
    Parse ReaxFF fort.7 into:
      • Summary DataFrame (one row per iteration) with columns:
        ["iter","num_of_atoms","num_of_bonds","total_BO","total_LP",
         "total_BO_uncorrected","total_charge"]
      • Per-iteration atom tables in `self._frames`:
        columns = ["atom_num","atom_type_num","atom_cnn1..nb","molecule_num",
                   "BO1..nb","sum_BOs","num_LPs","partial_charge", ...]
    Notes:
      - Header line (6 tokens): n_atoms, sim_name, kw, iter, kw, nbonds
      - Totals line: <6 tokens (floats)
      - Atom lines: ints (nbonds+3) then floats
    """


    def __init__(self, file_path: str | Path = "fort.7"):
        super().__init__(file_path)
        self._frames: List[pd.DataFrame] = []
        self._sim_name: Optional[str] = None

    def _parse(self) -> tuple[pd.DataFrame, dict[str, Any]]:
        sim_rows: List[List[Any]] = []
        frames: List[pd.DataFrame] = []
        totals: List[List[float]] = []

        cur_atoms_rows: List[List[float | int]] = []
        cur_totals: List[float] = []
        cur_num_particles: Optional[int] = None
        cur_nbonds: Optional[int] = None
        sim_name: str = ""

        def _finalize_iteration() -> None:
            if cur_num_particles is None or cur_nbonds is None or not cur_atoms_rows:
                return
            nb = int(cur_nbonds)
            atom_cols = (
                ["atom_num", "atom_type_num"]
                + [f"atom_cnn{i}" for i in range(1, nb + 1)]
                + ["molecule_num"]
                + [f"BO{i}" for i in range(1, nb + 1)]
                + ["sum_BOs", "num_LPs", "partial_charge"]
            )
            extra = max(0, len(cur_atoms_rows[0]) - len(atom_cols))
            if extra > 0:
                atom_cols += [f"unknown{i}" for i in range(1, extra + 1)]
            frames.append(pd.DataFrame(cur_atoms_rows, columns=atom_cols))
            totals.append(cur_totals[:] if cur_totals else [float("nan")] * 4)

        with open(self.path, "r") as fh:
            for raw in fh:
                values = raw.split()
                if not values:
                    continue

                # Header
                if len(values) == 6:
                    if cur_atoms_rows:
                        _finalize_iteration()
                        cur_atoms_rows.clear()
                        cur_totals.clear()

                    cur_num_particles = int(values[0])
                    sim_name = values[1]
                    iteration = int(values[3])
                    cur_nbonds = int(values[5])
                    sim_rows.append([iteration, cur_num_particles, cur_nbonds])

                # Totals
                elif len(values) < 6:
                    cur_totals.extend(map(float, values))

                # Atom line
                else:
                    nb = int(cur_nbonds)
                    int_part = list(map(int, values[0: nb + 3]))
                    float_part = list(map(float, values[nb + 3:]))
                    cur_atoms_rows.append(int_part + float_part)

        # Final iter
        if cur_atoms_rows:
            _finalize_iteration()

        # Summary dataframe
        sim_df = pd.DataFrame(sim_rows, columns=["iter", "num_of_atoms", "num_of_bonds"])
        totals_df = pd.DataFrame(
            totals,
            columns=["total_BO", "total_LP", "total_BO_uncorrected", "total_charge"]
            if totals and len(totals[0]) == 4
            else [f"total_val{i}" for i in range(1, (len(totals[0]) if totals else 0) + 1)]
        )
        if not totals_df.empty:
            totals_df = totals_df.iloc[: len(sim_df)].reset_index(drop=True)
            sim_df = pd.concat([sim_df.reset_index(drop=True), totals_df], axis=1)

        # Deduplicate
        if not sim_df.empty and "iter" in sim_df.columns:
            keep_idx = sim_df.drop_duplicates("iter", keep="last").index
            frames = [frames[i] for i in keep_idx]
            sim_df = sim_df.loc[keep_idx].reset_index(drop=True)

        self._frames = frames
        self._sim_name = sim_name

        meta: Dict[str, Any] = {
            "n_frames": len(frames),
            "n_records": len(sim_df),
            "simulation_name": sim_name,
        }

        return sim_df, meta
