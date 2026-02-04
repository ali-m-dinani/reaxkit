"""
fort.99 (training-set and EOS) analysis utilities.

This module provides helpers for analyzing ReaxFF training-set output
stored in ``fort.99`` files, including error inspection, two-body ENERGY
term parsing, and equation-of-state (EOS) analysis for bulk modulus
extraction.

Typical use cases include:

- inspecting QM–FF energy differences for training targets
- parsing pairwise ENERGY terms from fort.99 titles
- constructing energy–volume datasets using fort.74
- computing bulk modulus via a Vinet equation-of-state fit
"""


from __future__ import annotations
import pandas as pd
import numpy as np

from reaxkit.io.base_handler import FileHandler
from reaxkit.utils.constants import const
from reaxkit.utils.equation_of_states import vinet_energy_ev

def get_fort99_data(
    handler: FileHandler,
    *,
    sortby: str = "lineno",
    ascending: bool = True
) -> pd.DataFrame:
    """
    Retrieve fort.99 data and compute QM–FF energy differences.

    A new column ``qm_ff_difference`` is added as:
    ``qm_value - ffield_value``.

    Works on
    --------
    Fort99Handler — ``fort.99``

    Parameters
    ----------
    handler : TemplateHandler
        Parsed ``fort.99`` handler.
    sortby : str, default="lineno"
        Column name to sort by (e.g. ``error``, ``lineno``).
    ascending : bool, default=True
        Sort order.

    Returns
    -------
    pandas.DataFrame
        fort.99 table including the additional ``qm_ff_difference`` column.

    Examples
    --------
    >>> df = get_fort99_data(h, sortby="error", ascending=False)
    """
    df = handler.dataframe().copy()

    # Add qm_ff_difference column
    df["qm_ff_difference"] = df["qm_value"] - df["ffield_value"]

    # Validate requested sort column
    if sortby not in df.columns:
        raise ValueError(
            f"Invalid sort key: '{sortby}'. "
            f"Available columns: {list(df.columns)}"
        )

    # Sort based on user choice
    df = df.sort_values(sortby, ascending=ascending)

    return df

# --------------------------------------------------------------------------------------------------
# getting the EOS data (i.e., rows where section = ENERGY and have 2 identifiers in their title
# --------------------------------------------------------------------------------------------------

def parse_fort99_two_body_energy_terms(handler: FileHandler) -> pd.DataFrame:
    """
    Parse pairwise ENERGY terms from the ``ENERGY`` section of fort.99.

    Only ENERGY titles containing exactly two ``/`` separators are kept,
    corresponding to two-body interaction terms.

    Works on
    --------
    Fort99Handler — ``fort.99``

    Parameters
    ----------
    handler : TemplateHandler
        Parsed ``fort.99`` handler.

    Returns
    -------
    pandas.DataFrame
        ENERGY rows augmented with parsed columns:
        ``opt1``, ``iden1``, ``n1``, ``opt2``, ``iden2``, ``n2``.

    Examples
    --------
    >>> df = parse_fort99_two_body_energy_terms(h)
    """
    import re
    import numpy as np
    import pandas as pd

    df = handler.dataframe().copy()

    # 1) Basic sanity check
    if "section" not in df.columns or "title" not in df.columns:
        raise KeyError("Expected 'section' and 'title' columns in fort.99 DataFrame.")

    # 2) Restrict to ENERGY section (case-insensitive)
    energy_df = df[df["section"].astype(str).str.upper() == "ENERGY"].copy()

    # 3) Keep rows with *at least* 2 "/" (some triple-body lines may be truncated)
    energy_df = energy_df[energy_df["title"].astype(str).str.count("/") == 2].copy()

    # Regex:
    #   Energy +Zn_h2o-1_P2/1.00 -Zn_oh-1_P1/1.00 ...
    #
    #  - allow any non-"/" chars in identifiers: [^/]+?
    #  - allow ints or floats for n1, n2: \d+(?:\.\d+)?
    #  - ignore case on "Energy"
    pattern = re.compile(
        r"^Energy\s+"
        r"(?P<sign1>[+-])(?P<iden1>[^/]+?)\s*/\s*(?P<n1>\d+(?:\.\d+)?)\s+"
        r"(?P<sign2>[+-])(?P<iden2>[^/]+?)\s*/\s*(?P<n2>\d+(?:\.\d+)?)",
        flags=re.IGNORECASE,
    )

    def _parse_title(title: str) -> dict:
        """Return parsed components or NaNs if it doesn't match."""
        m = pattern.search(title)
        if not m:
            # Return NaNs so we can drop these rows later instead of raising
            return {
                "opt1": np.nan,
                "iden1": np.nan,
                "n1": np.nan,
                "opt2": np.nan,
                "iden2": np.nan,
                "n2": np.nan,
            }

        g = m.groupdict()
        return {
            "opt1": 1 if g["sign1"] == "+" else -1,
            "iden1": g["iden1"].strip(),
            "n1": float(g["n1"]),
            "opt2": 1 if g["sign2"] == "+" else -1,
            "iden2": g["iden2"].strip(),
            "n2": float(g["n2"]),
        }

    # Apply parser row-wise
    parsed = energy_df["title"].astype(str).apply(_parse_title)
    parsed_df = pd.DataFrame(list(parsed))  # list-of-dicts -> DataFrame

    # Merge parsed columns into energy_df
    energy_df = pd.concat([energy_df.reset_index(drop=True), parsed_df], axis=1)

    # Drop rows where parsing failed (NaNs in iden1/iden2)
    energy_df = energy_df.dropna(subset=["iden1", "iden2"])

    # (Optional) if you no longer need the raw 'title' column, remove it:
    # energy_df = energy_df.drop(columns=["title"])

    return energy_df


def fort99_energy_vs_volume(
    fort99_handler: FileHandler,
    fort74_handler: FileHandler,
) -> pd.DataFrame:
    """
    Construct an energy–volume table from fort.99 and fort.74 data.

    Pairwise ENERGY terms from fort.99 are matched with corresponding
    volumes extracted from fort.74 based on shared identifiers.

    Works on
    --------
    Fort99Handler + Fort74Handler — ``fort.99`` + ``fort.74``

    Parameters
    ----------
    fort99_handler : TemplateHandler
        Parsed ``fort.99`` handler.
    fort74_handler : TemplateHandler
        Parsed ``fort.74`` handler providing volumes.

    Returns
    -------
    pandas.DataFrame
        Table with columns:
        ``iden1``, ``iden2``, ``ffield_value``, ``qm_value``, ``V_iden2``.

    Examples
    --------
    >>> df = fort99_energy_vs_volume(f99, f74)
    """
    from reaxkit.analysis.per_file import fort74_analyzer

    # 1) Build ENERGY section with parsed two-body terms
    energy_df = parse_fort99_two_body_energy_terms(fort99_handler)

    if energy_df.empty:
        return pd.DataFrame(
            columns=["iden1", "iden2", "ffield_value", "qm_value", "V_iden2"]
        )

    # 2) Keep only iden1 values appearing more than once
    repeated = energy_df.groupby("iden1").filter(lambda g: len(g) > 1).copy()

    if repeated.empty:
        return pd.DataFrame(
            columns=["iden1", "iden2", "ffield_value", "qm_value", "V_iden2"]
        )

    # 🔥 FIX: remove rows where iden1 == iden2
    repeated = repeated[repeated["iden1"] != repeated["iden2"]]

    if repeated.empty:
        return pd.DataFrame(
            columns=["iden1", "iden2", "ffield_value", "qm_value", "V_iden2"]
        )

    # 3) Load fort.74 data and extract identifier → volume
    fort74_df = fort74_analyzer.get_fort74_data(fort74_handler)

    if (
        fort74_df.empty
        or "identifier" not in fort74_df.columns
        or "V" not in fort74_df.columns
    ):
        return pd.DataFrame(
            columns=["iden1", "iden2", "ffield_value", "qm_value", "V_iden2"]
        )

    vol_df = fort74_df[["identifier", "V"]].drop_duplicates()

    # 4) Attach volume of iden2
    merged = repeated.merge(
        vol_df,
        left_on="iden2",
        right_on="identifier",
        how="left",
    )

    # 5) Final output
    out = merged[["iden1", "iden2", "ffield_value", "qm_value", "V"]].rename(
        columns={"V": "V_iden2"}
    )

    return out.sort_values(["iden1", "iden2"]).reset_index(drop=True)


#################################################################################
# Finding the bulk modulus of a system using Rose–Vinet equation of state
#################################################################################

def get_fort99_bulk_modulus(
    fort99_handler,
    fort74_handler,
    *,
    iden: str,
    source: str = "ffield",          # "ffield" or "qm" which defines which source of energy data should be used
    shift_min_to_zero: bool = True,  # helps conditioning; doesn't change K0
    flip_sign: bool = False,
    dropna: bool = True,
) -> dict:
    """
    Compute the bulk modulus using a Vinet EOS fit to E(V) data.

    Energy–volume points are obtained from pairwise ENERGY terms in
    ``fort.99`` and corresponding volumes from ``fort.74``.

    Works on
    --------
    Fort99Handler + Fort74Handler — ``fort.99`` + ``fort.74``

    Parameters
    ----------
    fort99_handler : TemplateHandler
        Parsed ``fort.99`` handler.
    fort74_handler : TemplateHandler
        Parsed ``fort.74`` handler.
    iden : str
        Identifier (``iden1``) for which the EOS fit is performed.
    source : {"ffield", "qm"}, default="ffield"
        Energy source used for fitting.
    shift_min_to_zero : bool, default=True
        Shift minimum energy to zero to improve numerical conditioning.
    flip_sign : bool, default=False
        Flip the sign of energies before fitting.
    dropna : bool, default=True
        Drop rows with NaN energy or volume values.

    Returns
    -------
    dict
        Dictionary containing:
        ``iden``, ``source``, ``n_points``, ``V0_A3``, ``K0_eV_A3``,
        ``K0_GPa``, ``E0_eV``, ``C``, ``success``.

    Examples
    --------
    >>> res = get_fort99_bulk_modulus(
    ...     f99, f74, iden="bulk_0", source="ffield"
    ... )
    >>> res["K0_GPa"]
    """
    from scipy.optimize import curve_fit # local import to avoid overload

    df = fort99_energy_vs_volume(
        fort99_handler=fort99_handler,
        fort74_handler=fort74_handler,
    )

    if df.empty:
        raise ValueError("No ENERGY vs volume data found (fort99_energy_vs_volume returned empty).")

    g = df[df["iden1"] == iden].copy()
    if g.empty:
        raise ValueError(f"No rows found for iden1 == {iden!r}.")

    # Choose energy column
    src = (source or "").strip().lower()
    if src in {"ffield", "ff", "forcefield", "force-field"}:
        e_col = "ffield_value"
        src_name = "ffield"
    elif src in {"qm", "dft", "reference"}:
        e_col = "qm_value"
        src_name = "qm"
    else:
        raise ValueError("source must be one of {'ffield','qm'}.")

    # Pull V and E
    V = g["V_iden2"].to_numpy(dtype=float)
    E_kcal = g[e_col].to_numpy(dtype=float)

    if dropna:
        m = np.isfinite(V) & np.isfinite(E_kcal)
        V = V[m]
        E_kcal = E_kcal[m]

    if len(V) < 6:
        raise ValueError(
            f"Need at least ~6 E(V) points for a stable EOS fit; got {len(V)} for iden={iden!r}."
        )

    # Sort by V for stability
    order = np.argsort(V)
    V = V[order]
    E_kcal = E_kcal[order]

    # Optional sign flip (kept for parity with plotting)
    if flip_sign:
        E_kcal = -E_kcal

    # Convert kcal/mol -> eV
    E = E_kcal * const("energy_kcalmol_to_eV")

    # Optional energy shift (doesn't affect K0)
    if shift_min_to_zero:
        E = E - np.nanmin(E)

    # Initial guesses
    V0_guess = float(V[np.nanargmin(E)])
    E0_guess = float(np.nanmin(E))
    K0_guess = 0.5 / const("eV_per_A3_to_GPa")
    C_guess = 4.0

    p0 = [E0_guess, K0_guess, V0_guess, C_guess]

    # Basic bounds to prevent nonsense fits
    # K0 > 0, V0 > 0, C > 0
    bounds = (
        [-np.inf, 1e-12, 1e-9, 1e-6],
        [ np.inf, 1e3,   np.inf, 1e3],
    )

    popt, pcov = curve_fit(
        vinet_energy_ev,
        V,
        E,
        p0=p0,
        bounds=bounds,
        maxfev=20000,
    )

    E0_fit, K0_fit, V0_fit, C_fit = popt
    K0_GPa = float(K0_fit * const("eV_per_A3_to_GPa"))

    return {
        "iden": iden,
        "source": src_name,
        "n_points": int(len(V)),
        "V0_A3": float(V0_fit),
        "K0_eV_A3": float(K0_fit),
        "K0_GPa": K0_GPa,
        "E0_eV": float(E0_fit),
        "C": float(C_fit),
        "success": True,
    }

