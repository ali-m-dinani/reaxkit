"""Optimization-parameter analysis tasks."""

from __future__ import annotations

from dataclasses import dataclass, field as dc_field
from typing import Dict, List, Optional, Tuple

import pandas as pd

from reaxkit.analysis.base import AnalysisTask
from reaxkit.analysis.force_field.force_field import ForceFieldDataRequest, ForceFieldDataTask
from reaxkit.core.analysis_task_registry import register_task
from reaxkit.domain.base_request import BaseRequest
from reaxkit.domain.base_result import BaseResult
from reaxkit.domain.data_models import ForceFieldOptimizationParameterData, ForceFieldParametersData

_SECTION_NUM_MAP: Dict[int, Tuple[str, str]] = {
    1: ("general", "general"),
    2: ("atom", "atom"),
    3: ("bond", "bond"),
    4: ("off_diagonal", "off_diagonal"),
    5: ("angle", "angle"),
    6: ("torsion", "torsion"),
    7: ("hbond", "hbond"),
}

_SECTION_INDEX_COLS: Dict[str, List[str]] = {
    "general": [],
    "atom": ["symbol"],
    "bond": ["i", "j"],
    "off_diagonal": ["i", "j"],
    "angle": ["i", "j", "k"],
    "torsion": ["i", "j", "k", "l"],
    "hbond": ["i", "j", "k"],
}


def _params_frame(data: ForceFieldOptimizationParameterData) -> pd.DataFrame:
    n_rows = len(data.ff_section)
    return pd.DataFrame(
        {
            "ff_section": pd.Series(data.ff_section, dtype=int),
            "ff_section_line": pd.Series(data.ff_section_line, dtype=int),
            "ff_parameter": pd.Series(data.ff_parameter, dtype=int),
            "search_interval": pd.Series(data.search_interval, dtype=float),
            "min_value": pd.Series(data.min_value, dtype=float),
            "max_value": pd.Series(data.max_value, dtype=float),
            "inline_comment": pd.Series(data.inline_comment, dtype=object if n_rows else object),
        }
    )


def _param_columns_for_section(sec_df: pd.DataFrame, section_key: str) -> List[str]:
    idx_cols = set(_SECTION_INDEX_COLS.get(section_key, []))
    return [c for c in sec_df.columns if c not in idx_cols]


def _get_params_data(
    data: ForceFieldOptimizationParameterData,
    *,
    sort_by: str | None = None,
    ascending: bool = True,
    drop_duplicate: bool = True,
) -> pd.DataFrame:
    df = _params_frame(data)
    if drop_duplicate:
        df = df.drop_duplicates(
            subset=["ff_section", "ff_section_line", "ff_parameter"],
            keep="first",
        )
    if sort_by:
        if sort_by not in df.columns:
            raise ValueError(f"'sort_by' must be one of {list(df.columns)}, got {sort_by!r}")
        df = df.sort_values(by=sort_by, ascending=ascending)
    return df.reset_index(drop=True)


def _interpret_params(
    data: ForceFieldOptimizationParameterData,
    force_field: ForceFieldParametersData,
    *,
    add_term: bool = True,
    sep: str = "-",
    sort_by: str | None = None,
    ascending: bool = True,
    drop_duplicate: bool = True,
) -> pd.DataFrame:
    params_df = _get_params_data(
        data,
        sort_by=sort_by,
        ascending=ascending,
        drop_duplicate=drop_duplicate,
    )
    out_rows: List[Dict[str, object]] = []
    sec_cache: Dict[str, pd.DataFrame] = {}

    for row in params_df.itertuples(index=False):
        sec_num = int(row.ff_section)
        line_1b = int(row.ff_section_line)
        par_1b = int(row.ff_parameter)
        if sec_num not in _SECTION_NUM_MAP:
            raise ValueError(f"Unknown ff_section={sec_num}. Expected 1..7.")

        section_key, section_name = _SECTION_NUM_MAP[sec_num]
        if section_key not in sec_cache:
            fmt = "interpreted" if add_term and section_key not in {"general", "atom"} else "raw"
            sec_cache[section_key] = ForceFieldDataTask().run(
                force_field,
                ForceFieldDataRequest(section=section_name, format=fmt, sep=sep),
            ).table

        sec_df = sec_cache[section_key]
        row_idx = line_1b - 1
        if row_idx < 0 or row_idx >= len(sec_df):
            raise IndexError(
                f"params points to {section_name} line {line_1b}, but section has {len(sec_df)} rows."
            )

        param_cols = _param_columns_for_section(sec_df, section_key)
        param_cols = [c for c in param_cols if not (str(c).endswith("_symbol") or c == "term")]
        par_idx = par_1b - 1
        if par_idx < 0 or par_idx >= len(param_cols):
            raise IndexError(
                f"params points to {section_name} parameter {par_1b}, "
                f"but only {len(param_cols)} parameter columns exist: {param_cols}"
            )

        param_name = param_cols[par_idx]
        sec_row = sec_df.iloc[row_idx]
        out_rows.append(
            {
                "ff_section": sec_num,
                "ff_section_line": line_1b,
                "ff_parameter": par_1b,
                "search_interval": row.search_interval,
                "min_value": row.min_value,
                "max_value": row.max_value,
                "inline_comment": row.inline_comment,
                "ffield_section_key": section_key,
                "ffield_section_name": section_name,
                "ffield_row_index": row_idx,
                "ffield_param_name": param_name,
                "ffield_value": sec_row[param_name],
                "term": sec_row.get("term") if add_term else None,
            }
        )

    return pd.DataFrame(out_rows)


@dataclass
class ForceFieldOptimizationParameterRequest(BaseRequest):
    """Request for raw or interpreted params data."""

    sort_by: Optional[str] = dc_field(
        default=None,
        metadata={'label': 'Sort By', 'help': 'Sort By parameter for ForceFieldOptimizationParameterRequest.'},
    )
    ascending: bool = dc_field(
        default=True,
        metadata={'label': 'Ascending', 'help': 'Ascending parameter for ForceFieldOptimizationParameterRequest.', 'choices': [True, False]},
    )
    drop_duplicate: bool = dc_field(
        default=True,
        metadata={'label': 'Drop Duplicate', 'help': 'Drop Duplicate parameter for ForceFieldOptimizationParameterRequest.', 'choices': [True, False]},
    )
    interpret: bool = dc_field(
        default=False,
        metadata={'label': 'Interpret', 'help': 'Interpret parameter for ForceFieldOptimizationParameterRequest.', 'choices': [True, False]},
    )
    force_field: Optional[ForceFieldParametersData] = dc_field(
        default=None,
        metadata={'label': 'Force Field', 'help': 'Force Field parameter for ForceFieldOptimizationParameterRequest.'},
    )
    add_term: bool = dc_field(
        default=True,
        metadata={'label': 'Add Term', 'help': 'Add Term parameter for ForceFieldOptimizationParameterRequest.', 'choices': [True, False]},
    )
    sep: str = dc_field(
        default="-",
        metadata={'label': 'Sep', 'help': 'Sep parameter for ForceFieldOptimizationParameterRequest.'},
    )


@dataclass
class ForceFieldOptimizationParameterResult(BaseResult):
    """Result for raw or interpreted params data."""

    table: pd.DataFrame


@register_task("force_field_optimization_parameters")
class ForceFieldOptimizationParameterTask(AnalysisTask):
    """Return raw or interpreted optimization-parameter definitions from params."""

    required_data = ForceFieldOptimizationParameterData

    def run(
        self,
        data: ForceFieldOptimizationParameterData,
        request: ForceFieldOptimizationParameterRequest,
        reporter=None,
    ) -> ForceFieldOptimizationParameterResult:
        if request.interpret:
            if request.force_field is None:
                raise ValueError("Interpreting params requires request.force_field.")
            table = _interpret_params(
                data,
                request.force_field,
                add_term=bool(request.add_term),
                sep=str(request.sep),
                sort_by=request.sort_by,
                ascending=bool(request.ascending),
                drop_duplicate=bool(request.drop_duplicate),
            )
        else:
            table = _get_params_data(
                data,
                sort_by=request.sort_by,
                ascending=bool(request.ascending),
                drop_duplicate=bool(request.drop_duplicate),
            )
        return ForceFieldOptimizationParameterResult(table=table)


__all__ = [
    "ForceFieldOptimizationParameterRequest",
    "ForceFieldOptimizationParameterResult",
    "ForceFieldOptimizationParameterTask",
]
