"""Identify molecular fragments (connected components) in each ReaxFF frame using fort.7 connectivity data.

A molecule is defined as any set of atoms connected through non-zero bond-order edges. The module supports:

- Per-frame molecule detection via graph connected components.
- Optional `exclude` filtering to remove certain atoms (by element symbol
  or atom number), causing molecules to split accordingly.
- Output as a tidy DataFrame listing (frame_idx, iter, mol_id, atom_num, label).
- Pretty-print utility to display molecules in "C - H - O" / "12 - 3 - 6" form.

Useful for analyzing fragmentation, adsorption/desorption, ligand breaking,
and identifying dynamic molecular species during ReaxFF simulations.

"""


from __future__ import annotations
from typing import Iterable, Union, Optional, List, Dict
import pandas as pd

from reaxkit.utils.frame_utils import resolve_indices
from reaxkit.analysis.connectivity_analyzer import connection_list

Indexish = Union[int, Iterable[int], None]
Excludeish = Optional[Iterable[Union[str, int]]]


def _build_exclude_mask(frame_df: pd.DataFrame, exclude: Excludeish) -> pd.Series:
    """Return boolean mask of atoms to exclude for this frame."""
    if not exclude:
        return pd.Series(False, index=frame_df.index)

    exclude_set = set(exclude)
    mask = pd.Series(False, index=frame_df.index)

    # Strings: element symbols or atom names
    str_vals = {x for x in exclude_set if isinstance(x, str)}
    if str_vals:
        for col in ("elem", "element", "atom_name", "name", "symbol"):
            if col in frame_df.columns:
                mask |= frame_df[col].astype(str).isin(str_vals)

    # Ints: atom_type or atom_num
    int_vals = {x for x in exclude_set if isinstance(x, int)}
    if int_vals:
        for col in ("atom_type", "type", "atom_num"):
            if col in frame_df.columns:
                mask |= frame_df[col].astype(int).isin(int_vals)

    return mask


def identify_molecules(
    handler,
    frames: Indexish = None,
    iterations: Indexish = None,
    *,
    min_bo: float = 0.0,
    exclude: Excludeish = None,
) -> pd.DataFrame:
    """
    Identify molecules (connected components) for selected frames.

    Returns a tidy DataFrame with columns:
      ["frame_idx", "iter", "mol_id", "atom_num", "label"]

    - `label` is element/atom name if available, otherwise atom_type or atom_num.
    - `exclude` can be element symbols ("Pt") and/or integers (atom_num / atom_type).
      Excluded atoms are removed from the graph, so they split molecules.
    """
    # Resolve which frames we’ll work with
    idx_list = resolve_indices(handler, frames=frames, iterations=iterations)
    if not idx_list:
        return pd.DataFrame(columns=["frame_idx", "iter", "mol_id", "atom_num", "label"])

    # Get bond edges for those frames
    edges = connection_list(
        handler,
        frames=idx_list,
        iterations=None,        # already filtered by resolve_indices
        min_bo=min_bo,
        undirected=True,
        include_self=False,
    )

    sim_df = handler.dataframe()
    rows: List[Dict] = []

    for fi in idx_list:
        # Per-frame metadata and atom table
        frame_meta = sim_df.iloc[fi]
        iter_val = int(frame_meta["iter"])
        frame_df = handler._frames[fi]  # same pattern as connectivity_analyzer

        # Atom numbers for this frame
        atom_nums = frame_df["atom_num"].astype(int).tolist()

        # Exclusion
        excl_mask = _build_exclude_mask(frame_df, exclude)
        excluded_nums = set(frame_df.loc[excl_mask, "atom_num"].astype(int).tolist())
        included_nums = [a for a in atom_nums if a not in excluded_nums]

        if not included_nums:
            continue

        # Build adjacency for included atoms
        adj: Dict[int, set] = {a: set() for a in included_nums}

        if not edges.empty:
            e_f = edges[edges["frame_idx"] == fi].copy()
            if not e_f.empty:
                # Drop edges touching excluded atoms
                e_f = e_f[
                    ~e_f["src"].isin(excluded_nums)
                    & ~e_f["dst"].isin(excluded_nums)
                ]
                for _, r in e_f.iterrows():
                    s, d = int(r["src"]), int(r["dst"])
                    if s in adj and d in adj:
                        adj[s].add(d)
                        adj[d].add(s)

        # Map atom_num → label (element, name, or type/num)
        if "elem" in frame_df.columns:
            label_col = "elem"
        elif "element" in frame_df.columns:
            label_col = "element"
        elif "atom_name" in frame_df.columns:
            label_col = "atom_name"
        elif "atom_type" in frame_df.columns:
            label_col = "atom_type"
        else:
            label_col = None

        meta = frame_df.set_index("atom_num")

        def _label(a: int) -> str:
            if label_col and a in meta.index:
                return str(meta.at[a, label_col])
            return str(a)

        # Find connected components (DFS/BFS)
        visited = set()
        mol_id = 0

        for start in sorted(included_nums):
            if start in visited:
                continue
            mol_id += 1
            stack = [start]
            comp = []

            while stack:
                v = stack.pop()
                if v in visited:
                    continue
                visited.add(v)
                comp.append(v)
                for nbr in adj.get(v, []):
                    if nbr not in visited:
                        stack.append(nbr)

            comp = sorted(comp)
            for a in comp:
                rows.append(
                    {
                        "frame_idx": fi,
                        "iter": iter_val,
                        "mol_id": mol_id,
                        "atom_num": a,
                        "label": _label(a),
                    }
                )

    if not rows:
        return pd.DataFrame(columns=["frame_idx", "iter", "mol_id", "atom_num", "label"])

    out = pd.DataFrame(rows)
    return out.sort_values(["frame_idx", "mol_id", "atom_num"], kind="stable").reset_index(drop=True)


def pretty_print_molecules(
    handler,
    frame: int,
    *,
    min_bo: float = 0.0,
    exclude: Excludeish = None,
) -> None:
    """
    Pretty-print molecules for a single frame like:

    C - H - O - Pt
    12 - 3 - 6 - 22
    """
    df = identify_molecules(
        handler,
        frames=[frame],
        iterations=None,
        min_bo=min_bo,
        exclude=exclude,
    )

    if df.empty:
        print(f"[frame {frame}] no molecules found.")
        return

    for mol_id, g in df.groupby("mol_id"):
        g = g.sort_values("atom_num")
        labels = " - ".join(str(x) for x in g["label"].tolist())
        nums   = " - ".join(str(int(x)) for x in g["atom_num"].tolist())
        print(f"Molecule {mol_id} (frame {frame}):")
        print(labels)
        print(nums)
        print()
