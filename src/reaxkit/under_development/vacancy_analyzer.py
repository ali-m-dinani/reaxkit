# reaxkit/analysis/vacancy_analyzer.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


"""
Vacancy / under-coordination analysis (data-driven; no file parsing here)

This module expects *analyzers* to provide already-cleaned data:

Required inputs (per frame):
- atom_frames: List[pd.DataFrame] with columns at least ['atom_type','x','y','z'].
- sim_table:   pd.DataFrame with per-frame metadata and at least columns:
               ['iter','energy','a','b','c','alpha','beta','gamma'].
- fort7_frames: List[pd.DataFrame] with columns including:
               ['atom_num','sum_BOs', 'atom_cnn1','atom_cnn2', ...] (neighbor indices).
               (Names may vary; adapt the adapter in the workflow if needed.)

Optional inputs:
- atom_types_per_frame: Optional[List[Sequence[str]]] — if available per frame;
  otherwise we’ll derive from atom_frames[i]['atom_type'].

You can construct this analyzer with raw data or via the helper classmethod
`from_analyzers(xmol_analyzer, fort7_analyzer)` if your analyzers expose:
- xmol_analyzer.atom_frames() -> List[pd.DataFrame]
- xmol_analyzer.summary() -> pd.DataFrame
- fort7_analyzer.atom_frames() -> List[pd.DataFrame]
Adjust in the workflow if your analyzer method names differ.

Unified vacancy “criteria”:
- A criterion is a Callable[[pd.Series], bool] evaluated per atom-row in the
  *combined* per-frame table (atoms + sum_BOs + optional mapped neighbor types).
- You can compose multiple criteria with logical ANY or ALL.

Built-in criteria:
- undercoordination(valency_map: Dict[str, float], threshold: float)
- al_underconnected(min_n_neighbors_of_N=3)  # requires neighbor type mapping
- bo_cutoff(botype: Tuple[str,str], cutoff: float)  # optional, if you add per-bond BO cols

Outputs:
- Boolean masks, index lists per frame, filtered frames, or xmolout-style writes.
"""


# -------------------------- Data structures --------------------------

@dataclass
class VacancyResult:
    frame_index: int
    iteration: int
    indices: List[int]  # row indices in the per-frame DataFrame


# -------------------------- Utilities --------------------------

def _require_cols(df: pd.DataFrame, cols: Iterable[str], where: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns {missing} in {where}.")


def _neighbor_columns(df: pd.DataFrame) -> List[str]:
    return [c for c in df.columns if c.startswith("atom_cnn")]


# -------------------------- Analyzer --------------------------

class VacancyAnalyzer:
    """
    Core vacancy/under-coordination analyzer that operates on analyzer-provided data.
    """

    def __init__(
        self,
        *,
        atom_frames: List[pd.DataFrame],
        sim_table: pd.DataFrame,
        fort7_frames: List[pd.DataFrame],
        atom_types_per_frame: Optional[List[Sequence[str]]] = None,
    ):
        if len(atom_frames) != len(fort7_frames):
            raise ValueError("atom_frames and fort7_frames must have the same length (per-frame alignment).")

        _require_cols(sim_table, ["iter", "energy", "a", "b", "c", "alpha", "beta", "gamma"], "sim_table")

        self.atom_frames = atom_frames
        self.sim_table = sim_table.reset_index(drop=True)
        self.fort7_frames = fort7_frames
        self.atom_types_per_frame = atom_types_per_frame  # optional

        # Lazies
        self._combined_frames: Optional[List[pd.DataFrame]] = None
        self._neighbor_types_frames: Optional[List[pd.DataFrame]] = None

    # ---------- Construction from analyzers (adapter) ----------

    @classmethod
    def from_analyzers(
        cls,
        xmol_analyzer,
        fort7_analyzer,
        *,
        atom_type_source: str = "atoms",  # "atoms" or "explicit"; use "explicit" if fort7 analyzer already maps types
    ) -> "VacancyAnalyzer":
        """
        Assumes the provided analyzers expose these methods:
          - xmol_analyzer.atom_frames() -> List[pd.DataFrame]
          - xmol_analyzer.summary() -> pd.DataFrame
          - fort7_analyzer.atom_frames() -> List[pd.DataFrame]
        Adjust in the workflow if your method names differ.
        """
        atom_frames = xmol_analyzer.atom_frames()
        sim_table = xmol_analyzer.summary()
        fort7_frames = fort7_analyzer.atom_frames()

        # If you already have atom types per frame elsewhere, pass atom_types_per_frame explicitly.
        atom_types_per_frame = None if atom_type_source == "atoms" else None

        return cls(
            atom_frames=atom_frames,
            sim_table=sim_table,
            fort7_frames=fort7_frames,
            atom_types_per_frame=atom_types_per_frame,
        )

    # ---------- Data preparation ----------

    def combined_frames(self) -> List[pd.DataFrame]:
        """
        Merge per-frame atom coords/types with fort7 per-atom metrics (sum_BOs, neighbors).
        Returns a list of DataFrames with at least:
          ['atom_type','x','y','z','sum_BOs','atom_num','atom_cnn*'...]
        """
        if self._combined_frames is not None:
            return self._combined_frames

        combined = []
        for i, (atoms_df, f7_df) in enumerate(zip(self.atom_frames, self.fort7_frames)):
            _require_cols(atoms_df, ["atom_type", "x", "y", "z"], f"atom_frames[{i}]")
            _require_cols(f7_df, ["atom_num", "sum_BOs"], f"fort7_frames[{i}]")

            # Ensure indexing alignment: both should represent the same atoms in the same order
            # If not, you may need to align by 'atom_num' externally.
            df = atoms_df.copy()
            for col in f7_df.columns:
                if col not in df.columns:
                    df[col] = f7_df[col].values
            combined.append(df)

        self._combined_frames = combined
        return combined

    def map_neighbor_types(self) -> List[pd.DataFrame]:
        """
        Map neighbor indices (atom_cnn*) in fort7 to atom types per frame.
        Adds new columns 'type_cnn1', 'type_cnn2', ... to each combined frame.

        Requires that for each frame we know types of *all atoms in that frame*,
        which we take from self.atom_frames[i]['atom_type'] (or atom_types_per_frame[i]).
        """
        if self._neighbor_types_frames is not None:
            return self._neighbor_types_frames

        frames = []
        combined = self.combined_frames()

        for i, df in enumerate(combined):
            types = (
                list(self.atom_types_per_frame[i])
                if self.atom_types_per_frame is not None
                else df["atom_type"].tolist()
            )
            cnns = _neighbor_columns(df)
            mapped = df.copy()

            for c in cnns:
                # neighbor indices are 1-based in most ReaxFF outputs; 0/neg/None means no neighbor
                def _idx_to_type(idx: int) -> Optional[str]:
                    try:
                        return types[idx - 1] if int(idx) > 0 else None
                    except Exception:
                        return None

                mapped[f"type_{c}"] = df[c].apply(_idx_to_type)

            frames.append(mapped)

        self._neighbor_types_frames = frames
        return frames

    # ---------- Built-in criteria factories ----------

    @staticmethod
    def criterion_undercoordination(
        valency_map: Dict[str, float],
        threshold: float = 1.1,
    ) -> Callable[[pd.Series], bool]:
        """
        Marks an atom "vacant/under-coordinated" if (valency - sum_BOs) > threshold.
        """
        def _crit(row: pd.Series) -> bool:
            at = row.get("atom_type", "X")
            if at == "X" or pd.isna(at):
                return False
            sum_bo = row.get("sum_BOs", np.nan)
            if pd.isna(sum_bo):
                return False
            valency = valency_map.get(at, 0.0)
            return (valency - float(sum_bo)) > float(threshold)
        return _crit

    @staticmethod
    def criterion_al_underconnected(min_n_neighbors_of_N: int = 3) -> Callable[[pd.Series], bool]:
        """
        Requires neighbor types mapped via map_neighbor_types().
        Flags an Al atom if it has fewer than `min_n_neighbors_of_N` neighbors of type 'N'.
        """
        def _crit(row: pd.Series) -> bool:
            if row.get("atom_type") != "Al":
                return False
            nn = 0
            for col, val in row.items():
                if col.startswith("type_atom_cnn") and val == "N":
                    nn += 1
            return nn < int(min_n_neighbors_of_N)
        return _crit

    @staticmethod
    def criterion_true() -> Callable[[pd.Series], bool]:
        """Select everything (useful for testing or piping a pre-filter elsewhere)."""
        return lambda row: True

    # ---------- Composition helpers ----------

    @staticmethod
    def any_of(*criteria: Callable[[pd.Series], bool]) -> Callable[[pd.Series], bool]:
        return lambda row: any(c(row) for c in criteria)

    @staticmethod
    def all_of(*criteria: Callable[[pd.Series], bool]) -> Callable[[pd.Series], bool]:
        return lambda row: all(c(row) for c in criteria)

    # ---------- Evaluation ----------

    def evaluate(
        self,
        criterion: Callable[[pd.Series], bool],
        *,
        require_neighbor_types: bool = False,
    ) -> Tuple[List[VacancyResult], List[pd.DataFrame]]:
        """
        Apply a single (possibly composed) criterion to all frames.

        Returns:
          - results: list of VacancyResult with indices of rows that matched
          - filtered_frames: list of per-frame DataFrames containing only matched rows
        """
        frames = self.map_neighbor_types() if require_neighbor_types else self.combined_frames()

        results: List[VacancyResult] = []
        filtered: List[pd.DataFrame] = []

        for i, df in enumerate(frames):
            mask = df.apply(criterion, axis=1)
            idxs = np.flatnonzero(mask.to_numpy(dtype=bool)).tolist()

            iter_val = int(self.sim_table.iloc[i]["iter"]) if i < len(self.sim_table) else i
            results.append(VacancyResult(frame_index=i, iteration=iter_val, indices=idxs))
            filtered.append(df.iloc[idxs].copy())

        return results, filtered

    # ---------- Writers ----------

    def write_xmolout_subset(
        self,
        filtered_frames: List[pd.DataFrame],
        out_path: str | Path,
        *,
        label: str = "undercoordinated_frame",
        float_fmt: str = ".6f",
        skip_empty_frames: bool = True,
    ) -> Path:
        """
        Write a new xmolout-style file containing only atoms from `filtered_frames` per frame.
        """
        out_path = Path(out_path)
        with out_path.open("w") as f:
            for i, df in enumerate(filtered_frames):
                if (len(df) == 0) and skip_empty_frames:
                    continue

                n = int(len(df))
                f.write(f"{n}\n")

                # metadata
                srow = self.sim_table.iloc[i]
                meta = (
                    int(srow["iter"]),
                    float(srow["energy"]),
                    float(srow["a"]),
                    float(srow["b"]),
                    float(srow["c"]),
                    float(srow["alpha"]),
                    float(srow["beta"]),
                    float(srow["gamma"]),
                )
                f.write(
                    f"{label} {meta[0]} {meta[1]:{float_fmt}} {meta[2]:{float_fmt}} {meta[3]:{float_fmt}} {meta[4]:{float_fmt}} "
                    f"{meta[5]:{float_fmt}} {meta[6]:{float_fmt}} {meta[7]:{float_fmt}}\n"
                )

                # atoms
                for _, r in df.iterrows():
                    f.write(
                        f"{r['atom_type']} {r['x']:{float_fmt}} {r['y']:{float_fmt}} {r['z']:{float_fmt}}\n"
                    )
        return out_path
