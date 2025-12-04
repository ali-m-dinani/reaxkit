"""analyzer for molfra.out or molfra_ig.out file"""
from __future__ import annotations
import re
import pandas as pd
from typing import Optional, Iterable, Dict, Sequence

from reaxkit.io.molfra_handler import MolFraHandler
from reaxkit.utils.convert import convert_xaxis

# =======================
# Molecule-level analysis
# =======================
def get_occurrences_wide(
    handler: MolFraHandler,
    *,
    molecules: Optional[Iterable[str]] = None,
    iters: Optional[Sequence[int]] = None,
    by_index: bool = False,
    fill_value: int = 0,
) -> pd.DataFrame:
    """Return freqs (counts) for specific molecules across all or selected frames (wide table).

    Output columns:
      - iter
      - one column per requested molecule_type (count; filled with `fill_value` if missing)
    """
    df = handler.dataframe().copy()
    if df.empty:
        cols = ["iter"] + (list(molecules) if molecules else [])
        return pd.DataFrame(columns=cols)

    # Filter molecules if requested
    if molecules is not None:
        df = df[df["molecular_formula"].isin(set(molecules))]

    # Frame selection
    if iters is not None:
        if by_index:
            uniq = sorted(df["iter"].unique().tolist())
            chosen = [uniq[i] for i in iters if 0 <= i < len(uniq)]
            df = df[df["iter"].isin(set(chosen))]
        else:
            df = df[df["iter"].isin(set(iters))]

    # Pivot
    pivot = (
        df.pivot_table(
            index="iter",
            columns="molecular_formula",
            values="freq",
            aggfunc="max",
            fill_value=fill_value,
        )
        .sort_index()
        .reset_index()
    )

    # Ensure requested molecules exist
    if molecules is not None:
        for m in molecules:
            if m not in pivot.columns:
                pivot[m] = fill_value
        pivot = pivot[["iter"] + list(molecules)]
    else:
        pivot = pivot[["iter"] + [c for c in pivot.columns if c != "iter"]]

    return pivot


def get_occurrences_long(
    handler: MolFraHandler,
    *,
    molecules: Optional[Iterable[str]] = None,
    iters: Optional[Sequence[int]] = None,
    by_index: bool = False,
    fill_value: int = 0,
) -> pd.DataFrame:
    """Return freqs (counts) for specific molecules across frames (long format).
    """
    wide = get_occurrences_wide(
        handler,
        molecules=molecules,
        iters=iters,
        by_index=by_index,
        fill_value=fill_value,
    )
    if wide.empty:
        return pd.DataFrame(columns=["iter", "molecular_formula", "freq"])

    long_df = (
        wide.melt(id_vars="iter", var_name="molecular_formula", value_name="freq")
        .sort_values(["iter", "molecular_formula"])
        .reset_index(drop=True)
    )
    return long_df


def _qualifying_types(
    handler: MolFraHandler,
    *,
    threshold: int = 3,
    exclude_types: Optional[Iterable[str]] = ("Pt",),
) -> list[str]:
    """Return molecule types whose maximum count >= threshold. This filters out molecules with low appearance.
    """
    df = handler.dataframe()
    if df.empty:
        return []
    if exclude_types:
        df = df[~df["molecular_formula"].isin(set(exclude_types))]
    grp = df.groupby("molecular_formula")["freq"].max()
    return sorted(grp[grp >= threshold].index.tolist())


# ====================
# Totals-level analysis
# ====================
def get_molfra_totals_vs_axis(
    handler: MolFraHandler,
    *,
    xaxis: str = "iter",
    control_file: str = "control",
    quantities: Optional[Iterable[str]] = ("total_molecules", "total_atoms", "total_molecular_mass"),
) -> pd.DataFrame:
    """Return totals (molecules, atoms, mass) vs requested x-axis.

    Output columns:
      - <xcol> (e.g., iter, frame, time_ps/ns/fs)
      - one column per quantity in 'quantities'
    """
    if not hasattr(handler, "_df_totals"):
        raise AttributeError("Totals dataframe not found. Parse handler with updated version first.")

    df = handler._df_totals.copy()
    if df.empty:
        return pd.DataFrame()

    iters = df["iter"].to_numpy()
    x_vals, xlabel = convert_xaxis(iters, xaxis, control_file=control_file)

    # Prepare output
    xcol = (xlabel.strip().lower()
            .replace(" ", "_")
            .replace("(", "")
            .replace(")", ""))  # e.g., "time_ps"
    out_cols = [c for c in (quantities or []) if c in df.columns]
    out = df[["iter"] + out_cols].copy()
    if xaxis != "iter":
        out.insert(0, xcol, x_vals)
    else:
        out.rename(columns={"iter": xcol}, inplace=True)
    return out


# ============================================================================================
# the molecule type whose individual molecular mass is the highest = main slab
# ============================================================================================
def largest_molecule_by_individual_mass(
    handler: MolFraHandler,
) -> pd.DataFrame:
    """For each iter, find the molecule type with the largest individual molecular mass, which mostly is the 'slab.'

    This uses the 'mass' column directly (not multiplied by count).

    Returns
    -------
    pd.DataFrame
        Columns:
          - iter : int
          - molecule_type : str
          - mass : float

    Examples:
    --------
       iter molecular_formula  molecular_mass
    0     0          Al48N48          3425.6
    1     1          Al48N48          3425.6
    2     2          Al48N48          3425.6

    """
    df = handler.dataframe().copy()
    if df.empty:
        return pd.DataFrame(columns=["iter", "molecular_formula", "freq"])

    # For each iter, select the molecule with the highest molecular mass
    idx = df.groupby("iter")["molecular_mass"].idxmax()
    df_max = df.loc[idx, ["iter", "molecular_formula", "molecular_mass"]].reset_index(drop=True)

    return df_max.sort_values("iter").reset_index(drop=True)


def atoms_in_the_largest_molecule_wide_format(handler: MolFraHandler) -> pd.DataFrame:
    """Return a stable, element-keyed wide table of atom counts for the largest (by individual mass) molecule at each iter.

    Columns:
      - iter
      - one column per element symbol (e.g., N, Al, O, H), values = counts (int)

    Examples
    --------
       iter  Al   N  O  H
    0     0  48  48  0  0
    1     1  48  48  0  0
    2     2  48  48  0  0
    """
    # Get largest molecule per iter
    df_largest = largest_molecule_by_individual_mass(handler)
    if df_largest.empty:
        return pd.DataFrame(columns=["iter"])

    rows = []
    all_elems = set()

    for _, r in df_largest.iterrows():
        it = int(r["iter"])
        formula = str(r["molecular_formula"])
        pairs = re.findall(r"([A-Z][a-z]*)(\d+)", formula)

        # per-iter element->count
        elem_counts: Dict[str, int] = {"iter": it}
        for elem, cnt in pairs:
            cnt_i = int(cnt)
            elem_counts[elem] = elem_counts.get(elem, 0) + cnt_i
            all_elems.add(elem)

        rows.append(elem_counts)

    # Build wide, ensure all elements present, fill missing with 0
    wide = pd.DataFrame(rows).sort_values("iter").reset_index(drop=True)
    for elem in sorted(all_elems):
        if elem not in wide.columns:
            wide[elem] = 0

    # Order columns: iter first, then alphabetical elements
    cols = ["iter"] + sorted([c for c in wide.columns if c != "iter"])
    return wide[cols]


def atoms_in_the_largest_molecule_long_format(handler: MolFraHandler) -> pd.DataFrame:
    """Long-form (iter, element, count) from the stable wide table.
    """
    wide = atoms_in_the_largest_molecule_wide_format(handler)
    if wide.empty:
        return pd.DataFrame(columns=["iter", "element", "freq"])
    return wide.melt(id_vars="iter", var_name="element", value_name="freq") \
               .sort_values(["iter", "element"]).reset_index(drop=True)

