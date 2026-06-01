"""Expose analyzer tasks for structured force-field parameter content.

This module provides section-oriented extraction and filtering over parsed
force-field parameter data. It focuses on parameter-table retrieval and
normalization, and does not run optimization or diagnostic computations.

**Usage context**

- Parameter inspection: Query specific parameter sections for review.
- Data export: Build normalized parameter tables for external tooling.
- Workflow composition: Feed section slices into optimization/report analyzers.
"""

from __future__ import annotations

from dataclasses import dataclass, field as dc_field
from typing import Any, Optional

import pandas as pd

from reaxkit.analysis.base import AnalysisTask
from reaxkit.core.registry.analysis_task_registry import register_task
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
    """Normalize a section alias into a canonical force-field section key."""
    norm = section.strip().lower().replace("-", "_").replace(" ", "_")
    if norm not in _SECTION_ALIASES:
        raise KeyError(f"Unknown force-field section {section!r}. Valid options: {sorted(_SECTION_ALIASES)}")
    return _SECTION_ALIASES[norm]


def _atom_index_to_symbol_map(data: ForceFieldParametersData) -> dict[int, str]:
    """Map atom-parameter row indices to element symbols."""
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
    """Attach per-index symbol columns and a joined ``term`` label."""
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
    """Interpret index-based section rows into symbol-based labeled rows."""
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
    """Return one section table, optionally with interpreted symbolic terms."""
    attr = _SECTION_TO_ATTR[section]
    raw_df = getattr(data, attr).copy()
    if not interpret:
        return raw_df
    if section in {"general", "atom"}:
        return raw_df
    return _interpret_section(data, section, "-")


@dataclass
class FFieldDataRequest(BaseRequest):
    """Request payload for force-field section extraction.

    This request controls section selection and whether bonded/multi-body rows
    are returned as raw atom-index values or augmented with symbolic ``term``
    labels derived from atom parameters.

    Fields
    -----
    section : Optional[str]
        Optional canonical/alias section selector. Allowed values map to
        ``general``, ``atom``, ``bond``, ``off_diagonal``, ``angle``,
        ``torsion``, and ``hbond``. If omitted, all sections are returned.
    interpret : bool
        If ``True``, bonded sections include interpreted symbol columns and a
        joined ``term`` string; if ``False``, rows remain index-based.

    Examples
    -----
    ```python
    request = ForceFieldDataRequest(section="bond", interpret=True)
    ```
    The request returns only the bond section with symbolic term labels.
    """

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
class FFieldDataResult(BaseResult):
    """Result payload for force-field section table analysis.

    The analyzer returns one normalized DataFrame containing either a single
    section or a concatenated multi-section table based on request scope.

    Fields
    -----
    request : ForceFieldDataRequest
        Request object used to generate this result.
    table : pandas.DataFrame
        Output table for the requested section scope. Multi-section outputs
        include a leading ``section`` column; interpreted bonded outputs may
        include symbol columns and ``term`` labels.

    Notes
    -----
    Column schemas vary by section because each force-field block has its own
    parameter layout.

    Examples
    -----
    ```python
    row = {"section": "bond", "i": 1, "j": 2, "term": "C-H"}
    ```
    The sample row shows one interpreted bonded parameter entry.
    """

    table: pd.DataFrame
    request: FFieldDataRequest


@register_task("force_field_data", label="Force Field Data")
class FFieldDataTask(AnalysisTask):
    """Return raw or interpreted force-field section data."""

    required_data = ForceFieldParametersData

    @staticmethod
    def recommended_presentations(_result: FFieldDataResult, payload: dict[str, Any]) -> list[PresentationSpec]:
        """Recommend default table and quick-look plot presentations.

        Selects a table view in all cases and adds a simple plot when at least
        one numeric column can be paired with a suitable categorical/index axis.

        Works on
        Analyzer task output for ``force_field_data``.

        Parameters
        -----
        _result : FFieldDataResult
            Typed analyzer result instance (unused by selection logic).
        payload : dict[str, Any]
            Serialized result payload expected to include ``table`` rows.

        Returns
        -----
        list[PresentationSpec]
            Recommended presentation specifications for rendering.

        Examples
        -----
        ```python
        specs = ForceFieldDataTask.recommended_presentations(
            _result,
            {"table": [{"section": "bond", "term": "C-H", "De": 78.2}]},
        )
        ```
        The output includes a table and a one-series plot when numeric data exists.
        """
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
        request: FFieldDataRequest,
        reporter=None,
    ) -> FFieldDataResult:
        """Run section-oriented extraction over parsed force-field parameters.

        Resolves the requested section scope, materializes section tables in raw
        or interpreted mode, and returns the assembled DataFrame result.

        Works on
        ``ForceFieldParametersData`` parsed from force-field parameter sources.

        Parameters
        -----
        data : ForceFieldParametersData
            Parsed force-field parameter bundle.
        request : FFieldDataRequest
            Section and interpretation options for extraction.
        reporter : Any, optional
            Optional progress callback invoked per processed section.

        Returns
        -----
        FFieldDataResult
            Result containing one extracted/combined section table.

        Examples
        -----
        ```python
        result = ForceFieldDataTask().run(
            data,
            ForceFieldDataRequest(section="angle", interpret=True),
        )
        ```
        The returned table contains only interpreted angle-parameter rows.
        """
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
        return FFieldDataResult(table=table, request=request)


__all__ = [
    "FFieldDataRequest",
    "FFieldDataResult",
    "FFieldDataTask",
]
