"""analyzer for fort.99 file"""

from __future__ import annotations
import pandas as pd
import numpy as np

from reaxkit.io.template_handler import TemplateHandler
from reaxkit.utils.constants import *

def get_fort99(
    handler: TemplateHandler,
    *,
    sortby: str = "lineno",
    ascending: bool = True
) -> pd.DataFrame:
    """
    Compute:
        qm_ff_difference = qm_value - ffield_value
    Then sort by a user-chosen column ('sortby').

    Parameters
    ----------
    handler : TemplateHandler
        Usually Fort99Handler instance.
    sortby : str, optional
        Column name to sort by. Default: 'error'.
    ascending : bool, optional
        Sort in ascending order. Default: False (largest first).

    Returns
    -------
    pd.DataFrame
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

def parse_fort99_two_body_energy_terms(handler: TemplateHandler) -> pd.DataFrame:
    """
    Parse ENERGY section titles of fort.99 into structured columns, but only
    for titles that contain (at least) two "/" symbols (i.e. pairwise terms).

    Output columns:
        opt1  = +1 / -1
        iden1 = first identifier
        n1    = float stoichiometry
        opt2  = +1 / -1
        iden2 = second identifier
        n2    = float stoichiometry
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
    fort99_handler: TemplateHandler,
    fort74_handler: TemplateHandler,
) -> pd.DataFrame:
    """
    Using the two-body ENERGY terms from fort.99, find entries where `iden1`
    is repeated more than once (e.g., many rows with iden1 == 'bulk_0') and,
    for each corresponding `iden2`, attach its volume from fort.74.

    Returns a DataFrame with columns:
        iden1, iden2, ffield_value, qm_value, V_iden2
    """
    from reaxkit.analysis import fort74_analyzer

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
    fort74_df = fort74_analyzer.get_fort74(fort74_handler)

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
# according to doi:10.1029/JB092iB09p09319
#################################################################################
def _vinet_energy(V: np.ndarray, E0: float, K0: float, V0: float, C: float) -> np.ndarray:
    """
    Vinet equation of state (EOS) — energy–volume form.

    This function models the total energy of a solid as a function of volume
    using the Vinet EOS, which is well-suited for describing both compression
    and moderate expansion around equilibrium.

    Parameters
    ----------
    V : np.ndarray [Å³]
        System volume(s) at which the energy is evaluated. Typically obtained
        from scaled-cell calculations around the equilibrium volume.

    E0 : float [eV]
        Equilibrium energy at the minimum of the EOS curve.
        This is the total energy of the system at the equilibrium volume V0.
        (An arbitrary constant shift in energy does not affect the bulk modulus.)

    K0 : float [eV/Å³]
        Bulk modulus at equilibrium volume V0.
        Physically, K0 measures the resistance of the material to uniform
        (hydrostatic) compression:
            K0 = V * (∂²E / ∂V²) |_{V = V0}
        This parameter is converted to GPa after fitting.

    V0 : float [Å³]
        Equilibrium volume corresponding to the minimum of the energy–volume curve.
        This represents the relaxed unit-cell (or supercell) volume of the system.

    C : float [dimensionless]
        Shape parameter of the Vinet EOS, related to the pressure derivative
        of the bulk modulus (K′).
        It controls the asymmetry of the E(V) curve under compression vs expansion.
        Typical values are ~3–6 for many solids.

    Returns
    -------
    E : np.ndarray [eV]
        Predicted total energy of the system at volume(s) V according to
        the Vinet equation of state.
    """
    nu = V / V0
    eta = nu ** (1.0 / 3.0)
    term = 1.0 - (1.0 + C * (eta - 1.0)) * np.exp(C * (1.0 - eta))
    return E0 + 9.0 * K0 * V0 / (C ** 2) * term


def fort99_bulk_modulus(
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
    Compute bulk modulus K0 (GPa) for a selected fort.99 ENERGY identifier (iden1),
    using E(V) data from fort99_energy_vs_volume(...) and a Vinet EOS fit.

    Notes
    -----
    - fort99_energy_vs_volume currently returns energies in kcal/mol and volumes in Å^3
      (consistent with fort99_workflow plotting). We convert kcal/mol -> eV before fitting.
    - Fit returns K0 in eV/Å^3, then converted to GPa via 160.21766208.

    Returns
    -------
    dict with:
      iden, source, n_points, V0_A3, K0_eV_A3, K0_GPa, E0_eV, C, success
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
    E = E_kcal * CONSTANTS["energy_kcalmol_to_eV"]

    # Optional energy shift (doesn't affect K0)
    if shift_min_to_zero:
        E = E - np.nanmin(E)

    # Initial guesses
    V0_guess = float(V[np.nanargmin(E)])
    E0_guess = float(np.nanmin(E))
    K0_guess = 0.5 / CONSTANTS['eV_per_A3_to_GPa']
    C_guess = 4.0

    p0 = [E0_guess, K0_guess, V0_guess, C_guess]

    # Basic bounds to prevent nonsense fits
    # K0 > 0, V0 > 0, C > 0
    bounds = (
        [-np.inf, 1e-12, 1e-9, 1e-6],
        [ np.inf, 1e3,   np.inf, 1e3],
    )

    popt, pcov = curve_fit(
        _vinet_energy,
        V,
        E,
        p0=p0,
        bounds=bounds,
        maxfev=20000,
    )

    E0_fit, K0_fit, V0_fit, C_fit = popt
    K0_GPa = float(K0_fit * CONSTANTS['eV_per_A3_to_GPa'])

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

