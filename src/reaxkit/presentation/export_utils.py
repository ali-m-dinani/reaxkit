"""Export tabular datasets and Plotly figures to common file formats.

This module provides utility functions for writing row-oriented table data and
Plotly figures to disk. It focuses on lightweight export flows for utility and
workflow code, including format normalization and parent-directory creation.

**Usage context**

- Reporting pipelines: Write computed tabular outputs to `csv` or `xlsx` files.
- Visualization workflows: Render Plotly figures to raster image artifacts.
- CLI or service tasks: Normalize user-provided format values before export.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import plotly.graph_objects as go
import plotly.io as pio


def write_table(rows: list[dict[str, Any]], target: str | Path, fmt: str) -> Path:
    """Write row dictionaries to a tabular file.

    Supports `csv` and `xlsx` output formats. The format value is normalized by
    trimming whitespace and lowercasing before dispatch. The destination path is
    returned so callers can use the resolved output location directly.

    Parameters
    -----
    rows : list[dict[str, Any]]
        Row data to export. Each dictionary represents one row.
    target : str | Path
        Output file path for the exported table.
    fmt : str
        Requested table format. Supported values are `csv` and `xlsx`.

    Returns
    -----
    Path
        Path to the exported table file.

    Examples
    -----
    >>> write_table([{"a": 1, "b": 2}], "out/data.csv", "csv")
    PosixPath('out/data.csv')
    """
    path = Path(target)
    kind = str(fmt or "").strip().lower()
    if kind == "csv":
        _write_csv(rows, path)
        return path
    if kind == "xlsx":
        _write_xlsx(rows, path)
        return path
    raise ValueError(f"Unsupported table export format: {fmt}")


def write_figure(fig: go.Figure, target: str | Path, fmt: str) -> Path:
    """Write a Plotly figure to a raster image file.

    Accepts `png`, `jpeg`, and `jpg` format requests. The format value is
    normalized to lowercase, `jpg` is mapped to `jpeg`, and the output path
    suffix is adjusted to match the normalized format before rendering.

    Parameters
    -----
    fig : go.Figure
        Plotly figure to render.
    target : str | Path
        Output file path for the figure image.
    fmt : str
        Requested image format. Supported values are `png`, `jpeg`, and `jpg`.

    Returns
    -----
    Path
        Path to the exported figure image file.

    Examples
    -----
    >>> fig = go.Figure()
    >>> write_figure(fig, "out/plot.jpg", "jpg")
    PosixPath('out/plot.jpeg')
    """
    path = Path(target)
    kind = str(fmt or "").strip().lower()
    if kind not in {"png", "jpeg", "jpg"}:
        raise ValueError(f"Unsupported figure export format: {fmt}")
    normalized = "jpeg" if kind == "jpg" else kind
    if path.suffix.lower() != f".{normalized}":
        path = path.with_suffix(f".{normalized}")
    path.parent.mkdir(parents=True, exist_ok=True)
    pio.write_image(fig, str(path), format=normalized, width=1200, height=675, scale=2)
    return path


def _write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    """Write table rows to a CSV file.

    Notes
    -----
    - Preserves first-seen column order across all dictionary rows.
    - Ignores non-dictionary entries in `rows`.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    cols: list[str] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        for key in row.keys():
            skey = str(key)
            if skey not in cols:
                cols.append(skey)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=cols or [])
        if cols:
            writer.writeheader()
        for row in rows:
            if isinstance(row, dict):
                writer.writerow({str(k): row.get(k) for k in row.keys()})


def _write_xlsx(rows: list[dict[str, Any]], path: Path) -> None:
    """Write table rows to an XLSX file.

    Notes
    -----
    - Tries `openpyxl` first, then falls back to `pandas` when needed.
    - Raises `RuntimeError` if neither backend is available.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        from openpyxl import Workbook  # type: ignore

        wb = Workbook()
        ws = wb.active
        ws.title = "data"
        cols: list[str] = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            for key in row.keys():
                skey = str(key)
                if skey not in cols:
                    cols.append(skey)
        if cols:
            ws.append(cols)
        for row in rows:
            if not isinstance(row, dict):
                continue
            ws.append([row.get(col) for col in cols])
        wb.save(str(path))
        return
    except Exception:
        pass
    try:
        import pandas as pd  # type: ignore

        df = pd.DataFrame(rows)
        df.to_excel(str(path), index=False)
        return
    except Exception as exc:
        raise RuntimeError("xlsx export requires openpyxl or pandas.") from exc
