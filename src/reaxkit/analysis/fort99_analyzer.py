# reaxkit/analysis/fort99_analyzer.py
from __future__ import annotations
import pandas as pd
from reaxkit.io.template_handler import TemplateHandler

def get(
    handler: TemplateHandler,
    *,
    sortby: str = "lineno",
    ascending: bool = True
) -> pd.DataFrame:
    """
    Compute:
        difference = qm_value - ffield_value
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

    # Add difference column
    df["difference"] = df["qm_value"] - df["ffield_value"]

    # Validate requested sort column
    if sortby not in df.columns:
        raise ValueError(
            f"Invalid sort key: '{sortby}'. "
            f"Available columns: {list(df.columns)}"
        )

    # Sort based on user choice
    df = df.sort_values(sortby, ascending=ascending)

    return df

#------------------------------------------------------------------------------------------
# getting the EOS data (i.e., rows where section = ENERGY and have 2 identifiers in their title
#------------------------------------------------------------------------------------------

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
    fort74_df = fort74_analyzer.get(fort74_handler)

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



