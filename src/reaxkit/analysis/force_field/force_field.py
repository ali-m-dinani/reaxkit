"""Force-field analysis tasks."""

from __future__ import annotations

from dataclasses import dataclass, field as dc_field
from typing import Literal, Optional, Sequence

import pandas as pd

from reaxkit.analysis.base import AnalysisTask
from reaxkit.core.analysis_task_registry import register_task
from reaxkit.domain.base_request import BaseRequest
from reaxkit.domain.base_result import BaseResult
from reaxkit.domain.data_models import ForceFieldParametersData

_SECTION_TO_ATTR = {
    "general": "general_parameters",
    "atom": "atom_parameters",
    "bond": "bond_parameters",
    "off_diagonal": "off_diagonal_parameters",
    "angle": "angle_parameters",
    "torsion": "torsion_parameters",
    "hbond": "hydrogen_bond_parameters",
}

_SECTION_ALIASES = {
    "bond": "bond",
    "bonds": "bond",
    "off_diagonal": "off_diagonal",
    "off-diagonal": "off_diagonal",
    "off diagonal": "off_diagonal",
    "offdiag": "off_diagonal",
    "angle": "angle",
    "angles": "angle",
    "torsion": "torsion",
    "torsions": "torsion",
    "hbond": "hbond",
    "hbonds": "hbond",
    "hydrogen_bond": "hbond",
    "hydrogen bond": "hbond",
    "hydrogen_bonds": "hbond",
}


def _normalize_section_name(section: str) -> str:
    norm = section.strip().lower().replace("-", "_").replace(" ", "_")
    if norm not in _SECTION_ALIASES:
        raise KeyError(f"Unknown force-field section {section!r}. Valid options: {sorted(_SECTION_ALIASES)}")
    return _SECTION_ALIASES[norm]


def _atom_index_to_symbol_map(data: ForceFieldParametersData) -> dict[int, str]:
    atom_df = data.atom_parameters
    if atom_df.empty:
        raise ValueError("ForceFieldParametersData.atom_parameters is empty.")
    if "symbol" not in atom_df.columns:
        raise KeyError("ForceFieldParametersData.atom_parameters must include 'symbol' column.")

    out: dict[int, str] = {}
    for idx, row in atom_df.iterrows():
        try:
            atom_index = int(idx)
        except Exception:
            continue
        sym = row.get("symbol")
        if sym is None or (isinstance(sym, float) and pd.isna(sym)) or str(sym).strip() == "":
            sym = f"atom{atom_index}"
        out[atom_index] = str(sym).strip()
    return out


def _add_symbols_for_columns(
    df: pd.DataFrame,
    idx_to_sym: dict[int, str],
    cols: Sequence[str],
    *,
    term_col: str = "term",
    sep: str = "-",
) -> pd.DataFrame:
    out = df.copy()
    symbol_cols: list[str] = []
    for col in cols:
        symbol_col = f"{col}_symbol"
        symbol_cols.append(symbol_col)

        def _map_one(value):
            if value is None or (isinstance(value, float) and pd.isna(value)):
                return None
            try:
                iv = int(value)
            except Exception:
                return None
            return idx_to_sym.get(iv, f"atom{iv}")

        out[symbol_col] = out[col].map(_map_one)

    out[term_col] = out[symbol_cols].apply(
        lambda row: sep.join([x for x in row.tolist() if x is not None]),
        axis=1,
    )
    return out


def _interpret_section(data: ForceFieldParametersData, section: str, sep: str) -> pd.DataFrame:
    idx_to_sym = _atom_index_to_symbol_map(data)
    attr = _SECTION_TO_ATTR[section]
    df = getattr(data, attr).copy()

    if section in {"bond", "off_diagonal"}:
        return _add_symbols_for_columns(df, idx_to_sym, ["i", "j"], sep=sep)
    if section == "angle":
        return _add_symbols_for_columns(df, idx_to_sym, ["i", "j", "k"], sep=sep)
    if section == "torsion":
        return _add_symbols_for_columns(df, idx_to_sym, ["i", "j", "k", "l"], sep=sep)
    return _add_symbols_for_columns(df, idx_to_sym, ["i", "j", "k"], sep=sep)


def _section_frame(data: ForceFieldParametersData, section: str, fmt: str, sep: str) -> pd.DataFrame:
    attr = _SECTION_TO_ATTR[section]
    raw_df = getattr(data, attr).copy()
    if fmt == "raw":
        return raw_df
    if section in {"general", "atom"}:
        return raw_df
    return _interpret_section(data, section, sep)


@dataclass
class ForceFieldDataRequest(BaseRequest):
    """Request for raw or interpreted force-field sections."""

    section: Optional[str] = dc_field(
        default=None,
        metadata={'label': 'Section', 'help': 'Section parameter for ForceFieldDataRequest.'},
    )
    sections: Optional[Sequence[str]] = dc_field(
        default=None,
        metadata={'label': 'Sections', 'help': 'Sections parameter for ForceFieldDataRequest.'},
    )
    format: Literal["raw", "interpreted"] = dc_field(
        default="interpreted",
        metadata={'label': 'Format', 'help': 'Format parameter for ForceFieldDataRequest.', 'choices': ['raw', 'interpreted']},
    )
    sep: str = dc_field(
        default="-",
        metadata={'label': 'Sep', 'help': 'Sep parameter for ForceFieldDataRequest.'},
    )


@dataclass
class ForceFieldDataResult(BaseResult):
    """Raw or interpreted force-field tables."""

    tables: dict[str, pd.DataFrame]
    table: Optional[pd.DataFrame] = None


@register_task("force_field_data")
class ForceFieldDataTask(AnalysisTask):
    """Return raw or interpreted force-field section data."""

    required_data = ForceFieldParametersData

    def run(
        self,
        data: ForceFieldParametersData,
        request: ForceFieldDataRequest,
        reporter=None,
    ) -> ForceFieldDataResult:
        if request.section is not None:
            targets = [_normalize_section_name(request.section)]
        elif request.sections is not None:
            targets = [_normalize_section_name(section) for section in request.sections]
        else:
            targets = list(_SECTION_TO_ATTR)

        tables: dict[str, pd.DataFrame] = {}
        total = len(targets)
        fmt = str(request.format)
        if fmt not in {"raw", "interpreted"}:
            raise ValueError("ForceFieldDataRequest.format must be 'raw' or 'interpreted'.")
        for step_i, section in enumerate(targets, start=1):
            tables[section] = _section_frame(data, section, fmt, str(request.sep))
            if reporter:
                reporter("analyze", step_i, total, f"Loading force-field section: {section} ({fmt})")

        table = tables[targets[0]] if len(targets) == 1 else None
        return ForceFieldDataResult(tables=tables, table=table)


__all__ = [
    "ForceFieldDataRequest",
    "ForceFieldDataResult",
    "ForceFieldDataTask",
]
