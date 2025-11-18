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
    for titles that contain two "/" symbols (i.e. pairwise terms).

    Filtering rules:
      1. Keep rows where section == "ENERGY" (case-insensitive).
      2. From those, keep only titles that contain exactly two "/" characters.

    Example title:
        "Energy +bulk_0/  1 -bulk_c5/  1"

    Output columns:
        opt1  = +1
        iden1 = bulk_0
        n1    = 1
        opt2  = -1
        iden2 = bulk_c5
        n2    = 1

    Returns
    -------
    energy_section_df : pd.DataFrame
        ENERGY rows (with two "/") with parsed columns added and 'title' removed.
    """
    import re

    df = handler.dataframe().copy()

    # 1) Restrict to ENERGY section (case-insensitive)
    if "section" not in df.columns or "title" not in df.columns:
        raise KeyError("Expected 'section' and 'title' columns in fort.99 DataFrame.")

    energy_df = df[df["section"].astype(str).str.upper() == "ENERGY"].copy()

    # 2) Keep only titles that contain exactly two "/" symbols
    energy_df = energy_df[
        energy_df["title"].astype(str).str.count("/") == 2
    ].copy()

    # Regex for entries like:
    #   Energy +bulk_0/  1 -bulk_c5/  1
    pattern = re.compile(
        r"^Energy\s+"
        r"(?P<sign1>[+-])(?P<iden1>[A-Za-z0-9_]+)\s*/\s*(?P<n1>\d+)\s+"
        r"(?P<sign2>[+-])(?P<iden2>[A-Za-z0-9_]+)\s*/\s*(?P<n2>\d+)"
    )

    def _parse_title(title: str) -> pd.Series:
        m = pattern.match(title.strip())
        if not m:
            raise ValueError(f"Could not parse ENERGY title: {title!r}")

        sign1 = m.group("sign1")
        iden1 = m.group("iden1")
        n1 = int(m.group("n1"))

        sign2 = m.group("sign2")
        iden2 = m.group("iden2")
        n2 = int(m.group("n2"))

        return pd.Series(
            {
                "opt1": 1 if sign1 == "+" else -1,
                "iden1": iden1,
                "n1": n1,
                "opt2": 1 if sign2 == "+" else -1,
                "iden2": iden2,
                "n2": n2,
            }
        )

    parsed = energy_df["title"].astype(str).apply(_parse_title)

    # Replace title column with parsed columns
    energy_section_df = (
        energy_df
        .drop(columns=["title"])
        .join(parsed)
    )

    return energy_section_df


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



