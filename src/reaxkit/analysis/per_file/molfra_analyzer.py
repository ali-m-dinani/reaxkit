"""
molfra (molecular fragment) analysis utilities.

This module provides molecule-level and system-level analysis tools
for ReaxFF ``molfra.out`` and ``molfra_ig.out`` files via ``MolFraHandler``.

Typical use cases include:

- tracking molecular species counts over time
- converting molecule occurrence tables between wide and long formats
- extracting system totals (molecules, atoms, mass) versus iteration or time
- identifying and characterizing the largest (slab) molecule in the system
"""


from __future__ import annotations
import re
import pandas as pd
from typing import Optional, Iterable, Dict, Sequence

from reaxkit.io.handlers.molfra_handler import MolFraHandler
from reaxkit.utils.media.convert import convert_xaxis

# =======================
# Molecule-level analysis
# =======================
def get_molfra_data_wide_format(
    handler: MolFraHandler,
    *,
    molecules: Optional[Iterable[str]] = None,
    iters: Optional[Sequence[int]] = None,
    by_index: bool = False,
    fill_value: int = 0,
) -> pd.DataFrame:
    """
    Return molecule occurrence counts across iterations (wide format).

    Works on
    --------
    MolFraHandler — ``molfra.out`` / ``molfra_ig.out``

    Parameters
    ----------
    handler : MolFraHandler
        Parsed molecular fragment handler.
    molecules : iterable of str, optional
        Molecular formulas to include (e.g. ``"H2O"``, ``"CO2"``).
        If None, all detected molecules are included.
    iters : sequence of int, optional
        Iteration numbers to include.
    by_index : bool, default=False
        If True, interpret ``iters`` as indices into the unique iteration list.
    fill_value : int, default=0
        Value used when a requested molecule is absent at an iteration.

    Returns
    -------
    pandas.DataFrame
        Wide table with columns:
        ``iter`` and one column per molecule containing occurrence counts.

    Examples
    --------
    >>> df = get_molfra_data_wide_format(h, molecules=["H2O", "OH"], iters=[0, 100])
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


def get_molfra_data_long_format(
    handler: MolFraHandler,
    *,
    molecules: Optional[Iterable[str]] = None,
    iters: Optional[Sequence[int]] = None,
    by_index: bool = False,
    fill_value: int = 0,
) -> pd.DataFrame:
    """Return molecule occurrence counts across iterations (long format).

    Works on
    --------
    MolFraHandler — ``molfra.out`` / ``molfra_ig.out``

    Parameters
    ----------
    handler : MolFraHandler
        Parsed molecular fragment handler.
    molecules, iters, by_index, fill_value
        Same meaning as in :func:`get_occurrences_wide`.

    Returns
    -------
    pandas.DataFrame
        Long-form table with columns:
        ``iter``, ``molecular_formula``, ``freq``.

    Examples
    --------
    >>> df = get_molfra_data_long_format(h, molecules=["H2O"])
    """
    wide = get_molfra_data_wide_format(
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
    """Return system-level totals versus a chosen x-axis.

    Works on
    --------
    MolFraHandler — ``molfra.out`` / ``molfra_ig.out``

    Parameters
    ----------
    handler : MolFraHandler
        Parsed molecular fragment handler with totals data available.
    xaxis : {"iter", "frame", "time"}, default="iter"
        X-axis to use. ``time`` conversion uses the control file.
    control_file : str, default="control"
        Path to the ReaxFF control file for time conversion.
    quantities : iterable of str, optional
        Totals to include (e.g. ``total_molecules``, ``total_atoms``,
        ``total_molecular_mass``).

    Returns
    -------
    pandas.DataFrame
        Table with one column for the x-axis and one column per requested quantity.

    Examples
    --------
    >>> df = get_molfra_totals_vs_axis(h, xaxis="time")
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
    """Identify the molecule type with the largest individual mass at each iteration.

    This is typically the main slab or backbone molecule.

    Works on
    --------
    MolFraHandler — ``molfra.out`` / ``molfra_ig.out``

    Parameters
    ----------
    handler : MolFraHandler
        Parsed molecular fragment handler.

    Returns
    -------
    pandas.DataFrame
        Table with columns:
        ``iter``, ``molecular_formula``, ``molecular_mass``.

    Examples
    --------
    >>> df = largest_molecule_by_individual_mass(h)
    """
    df = handler.dataframe().copy()
    if df.empty:
        return pd.DataFrame(columns=["iter", "molecular_formula", "freq"])

    # For each iter, select the molecule with the highest molecular mass
    idx = df.groupby("iter")["molecular_mass"].idxmax()
    df_max = df.loc[idx, ["iter", "molecular_formula", "molecular_mass"]].reset_index(drop=True)

    return df_max.sort_values("iter").reset_index(drop=True)


def atoms_in_the_largest_molecule_wide_format(handler: MolFraHandler) -> pd.DataFrame:
    """Return per-element atom counts for the largest molecule at each iteration (wide format).

    Works on
    --------
    MolFraHandler — ``molfra.out`` / ``molfra_ig.out``

    Parameters
    ----------
    handler : MolFraHandler
        Parsed molecular fragment handler.

    Returns
    -------
    pandas.DataFrame
        Wide table with columns:
        ``iter`` and one column per element symbol (e.g. ``Al``, ``N``, ``O``),
        containing atom counts.

    Examples
    --------
    >>> df = atoms_in_the_largest_molecule_wide_format(h)
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
    """Return per-element atom counts for the largest molecule at each iteration (long format).

    Works on
    --------
    MolFraHandler — ``molfra.out`` / ``molfra_ig.out``

    Parameters
    ----------
    handler : MolFraHandler
        Parsed molecular fragment handler.

    Returns
    -------
    pandas.DataFrame
        Long-form table with columns:
        ``iter``, ``element``, ``freq``.

    Examples
    --------
    >>> df = atoms_in_the_largest_molecule_long_format(h)
    """
    wide = atoms_in_the_largest_molecule_wide_format(handler)
    if wide.empty:
        return pd.DataFrame(columns=["iter", "element", "freq"])
    return wide.melt(id_vars="iter", var_name="element", value_name="freq") \
               .sort_values(["iter", "element"]).reset_index(drop=True)

