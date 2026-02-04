"""Eregime (eregime.in) analysis utilities.

This module provides helpers for extracting and plotting electric-field regime
data from a parsed ``eregime.in`` file via ``EregimeHandler``.

Typical use cases include:

- selecting a column (with alias support) such as field magnitude or direction
- converting the x-axis from iteration to frame index or physical time
- exporting a clean two-column table for plotting
"""


from __future__ import annotations
import pandas as pd

from reaxkit.io.handlers.eregime_handler import EregimeHandler
from reaxkit.utils.alias import resolve_alias_from_columns, normalize_choice  # uses shared alias map
from reaxkit.utils.media.convert import convert_xaxis  # converts iter → time or frame


def _resolve_with_fallback(df: pd.DataFrame, name: str) -> str:
    """
    Resolve a requested column name using the shared alias map.
    Falls back to enumerated columns (field1/field_dir1, etc.) if generic keys are absent.
    """
    if not name:
        raise ValueError("Column name is empty.")

    # Normalize common variants (case-insensitive) to a canonical key
    canonical = normalize_choice(name)

    # Try direct/aliased match
    hit = resolve_alias_from_columns(df.columns, canonical)
    if hit:
        return hit

    # Fallbacks for generic single-zone keys in multi-zone files
    if canonical in ("field", "field_dir"):
        # Prefer the first enumerated pair if present
        for i in range(1, 16):  # generous upper bound
            cand = f"{canonical}{i}" if canonical != "field_dir" else f"field_dir{i}"
            hit = resolve_alias_from_columns(df.columns, cand)
            if hit:
                return hit

    raise ValueError(
        f"Column '{name}' not found (after alias resolution). "
        f"Available columns: {list(df.columns)}"
    )


def get_eregime_data(
    handler: EregimeHandler,
    y: str,
    xaxis: str = "iter",
    control_file: str = "control",
) -> pd.DataFrame:
    """Extract a two-column table of a selected ``eregime.in`` quantity versus an x-axis.

    Works on
    --------
    EregimeHandler — ``eregime.in``

    Parameters
    ----------
    handler : EregimeHandler
        Parsed ``eregime.in`` handler.
    y : str
        Name of the y-column to extract. Aliases are supported (e.g., ``E``, ``Ef``,
        ``Magnitude(V/A)`` → field; ``direction``/``dir`` → field direction).
        For multi-zone files, you may request enumerated keys like ``field2`` or
        ``field_dir3``.
    xaxis : {"iter", "frame", "time"}, default="iter"
        X-axis to use:
        - ``iter``: raw iteration index from the file
        - ``frame``: 0..N-1
        - ``time``: converts iteration → time using ``control_file`` and auto-scales fs/ps/ns
    control_file : str, default="control"
        Path to the ReaxFF control file used for ``xaxis="time"`` conversion.

    Returns
    -------
    pandas.DataFrame
        Two-column table with columns: ``[x_label, y]`` where ``x_label`` is one of
        ``iter``, ``Frame``, or ``Time (fs/ps/ns)``.

    Examples
    --------
    >>> from reaxkit.io.handlers.eregime_handler import EregimeHandler
    >>> from reaxkit.analysis.per_file.eregime_analyzer import get_eregime_data
    >>> h = EregimeHandler("eregime.in")
    >>> df = get_eregime_data(h, y="E", xaxis="time", control_file="control")
    """
    df = handler.dataframe()

    # Resolve iteration column (needed for x conversion)
    iter_col = _resolve_with_fallback(df, "iter")

    # Resolve Y column with aliases + fallbacks
    y_col = _resolve_with_fallback(df, y)

    # Convert x-axis
    x_vals, x_label = convert_xaxis(df[iter_col].to_numpy(), xaxis=xaxis, control_file=control_file)

    out = pd.DataFrame({x_label: x_vals, y: df[y_col].to_numpy()})
    out = out.sort_values(x_label).reset_index(drop=True)
    return out
