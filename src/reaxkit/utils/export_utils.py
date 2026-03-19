"""Export helpers for tabular data and plot figures."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import plotly.graph_objects as go
import plotly.io as pio


def write_table(rows: list[dict[str, Any]], target: str | Path, fmt: str) -> Path:
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

