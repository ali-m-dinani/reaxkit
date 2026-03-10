"""Force-field analysis tasks."""

from __future__ import annotations

from dataclasses import dataclass, field as dc_field
from typing import Any, Optional

import pandas as pd

from reaxkit.analysis.base import AnalysisTask
from reaxkit.core.analysis_task_registry import register_task
from reaxkit.domain.base_request import BaseRequest
from reaxkit.domain.base_result import BaseResult
from reaxkit.domain.data_models import ForceFieldParametersData
from reaxkit.presentation.specs import PresentationSpec

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


def _interpret_section(data: ForceFieldParametersData, section: str, sep: str = "-") -> pd.DataFrame:
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


def _section_frame(data: ForceFieldParametersData, section: str, interpret: bool) -> pd.DataFrame:
    attr = _SECTION_TO_ATTR[section]
    raw_df = getattr(data, attr).copy()
    if not interpret:
        return raw_df
    if section in {"general", "atom"}:
        return raw_df
    return _interpret_section(data, section, "-")


@dataclass
class ForceFieldDataRequest(BaseRequest):
    """Request for raw or interpreted force-field section output."""

    section: Optional[str] = dc_field(
        default=None,
        metadata={
            "label": "Section",
            "help": (
                "Single force-field section to load. "
                "Example: 'bond' or 'off_diagonal'."
            ),
            "choices": ["general", "atom", "bond", "off_diagonal", "angle", "torsion", "hbond"],
        },
    )
    interpret: bool = dc_field(
        default=True,
        metadata={
            "label": "Interpret",
            "help": (
                "If true, include interpreted symbolic terms for multi-body sections "
                "(for example C-H, C-C-H). If false, keep raw atom-index based rows."
            ),
            "choices": [True, False],
        },
    )


@dataclass
class ForceFieldDataResult(BaseResult):
    """Force-field section extraction result.

    Output structure:
    - request: ForceFieldDataRequest used to generate this result.
    - table: pandas.DataFrame with section rows in tabular form.
      - If one section is requested, rows are returned directly for that section.
      - If multiple sections are requested, rows are concatenated and include a
        leading 'section' column indicating the source section.
      - In interpreted mode for bonded sections, a 'term' column is included
        (for example 'C-H' or 'C-C-H') alongside numeric parameters.
    """

    table: pd.DataFrame
    request: ForceFieldDataRequest


@register_task("force_field_data")
class ForceFieldDataTask(AnalysisTask):
    """Return raw or interpreted force-field section data."""

    required_data = ForceFieldParametersData

    @staticmethod
    def recommended_presentations(_result: ForceFieldDataResult, payload: dict[str, Any]) -> list[PresentationSpec]:
        rows = payload.get("table")
        if not isinstance(rows, list) or not rows:
            return [PresentationSpec(renderer="table", label="Table", view_type="table")]
        sample = rows[0] if isinstance(rows[0], dict) else {}
        numeric_cols = [k for k, v in sample.items() if isinstance(v, (int, float)) and k != "section"]
        if not numeric_cols:
            return [PresentationSpec(renderer="table", label="Table", view_type="table")]

        if "term" in sample:
            x_col = "term"
        elif "symbol" in sample:
            x_col = "symbol"
        elif "i" in sample:
            x_col = "i"
        else:
            x_col = next(iter(sample.keys()), "section")

        y_col = numeric_cols[0]
        group_col = "section" if "section" in sample else ""
        return [
            PresentationSpec(renderer="table", label="Table", view_type="table"),
            PresentationSpec(
                renderer="single_plot",
                label=f"{y_col} vs {x_col}",
                mapping={"x_col": x_col, "y_col": y_col, "group_by_col": group_col},
                options={
                    "title": f"Force-Field Data: {y_col} vs {x_col}",
                    "xlabel": x_col,
                    "ylabel": y_col,
                    "legend": bool(group_col),
                },
                view_type="plot2d",
            ),
        ]

    def run(
        self,
        data: ForceFieldParametersData,
        request: ForceFieldDataRequest,
        reporter=None,
    ) -> ForceFieldDataResult:
        if request.section is not None:
            targets = [_normalize_section_name(request.section)]
        else:
            targets = list(_SECTION_TO_ATTR)

        tables: dict[str, pd.DataFrame] = {}
        total = len(targets)
        interpret = bool(request.interpret)
        for step_i, section in enumerate(targets, start=1):
            tables[section] = _section_frame(data, section, interpret)
            if reporter:
                mode = "interpreted" if interpret else "raw"
                reporter("analyze", step_i, total, f"Loading force-field section: {section} ({mode})")

        if len(targets) == 1:
            table = tables[targets[0]].reset_index(drop=True)
        else:
            frames: list[pd.DataFrame] = []
            for section in targets:
                section_df = tables[section].copy()
                section_df.insert(0, "section", section)
                frames.append(section_df)
            table = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        return ForceFieldDataResult(table=table, request=request)


__all__ = [
    "ForceFieldDataRequest",
    "ForceFieldDataResult",
    "ForceFieldDataTask",
]
