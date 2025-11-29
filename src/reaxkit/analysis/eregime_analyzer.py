"""analyzer for eregime.in file"""
from __future__ import annotations
import pandas as pd

from reaxkit.io.eregime_handler import EregimeHandler
from reaxkit.utils.alias import resolve_alias_from_columns, normalize_choice  # uses shared alias map
from reaxkit.utils.convert import convert_xaxis  # converts iter → time or frame


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


def get_series(
    handler: EregimeHandler,
    y: str,
    xaxis: str = "iter",
    control_file: str = "control",
) -> pd.DataFrame:
    """Return a two-column DataFrame with the requested y-column vs the requested x-axis for the data in eregime.in file.

    xaxis options:
      - 'iter'  : raw iteration index
      - 'frame' : 0..N-1
      - 'time'  : uses control.tstep to convert iteration → time (auto-scales fs/ps/ns)

    Column name resolution (y):
      - Uses the shared alias map (e.g., 'E', 'Magnitude(V/A)' → 'field', etc.).
      - For multi-zone files, you can request 'E1'/'E2' or 'direction1'/'direction2',
        or just 'E'/'direction' and it will fall back to the first pair if needed.

    Returns
    -------
    pd.DataFrame with columns: [<x-label>, <y as requested>], sorted by x.
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
