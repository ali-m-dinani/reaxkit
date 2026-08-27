"""Cell-parameter target matching and grouped-bar payload helpers."""

from __future__ import annotations

import math
import re

import pandas as pd

from reaxkit.domain.data_models import ForceFieldOptimizationPlotBundleData


CELL_PARAMETER_COLUMNS = [
    "label",
    "iden",
    "parameter_type",
    "ffield_value",
    "qm_value",
    "trainset_lit",
    "weight",
    "group_comment",
    "inline_comment",
    "report_line_number",
    "trainset_line_number",
]

_TYPE_ALIASES = {
    "1": "a",
    "2": "b",
    "3": "c",
    "4": "alpha",
    "5": "beta",
    "6": "gamma",
}


def _number_key(value: object) -> str:
    try:
        return f"{float(value):.12g}"
    except (TypeError, ValueError):
        return ""


def _parameter_type(value: object) -> str:
    raw = str(value).strip().lower()
    return _TYPE_ALIASES.get(raw, raw)


def _report_cell_parameter_table(
    data: ForceFieldOptimizationPlotBundleData,
) -> pd.DataFrame:
    """Recover identifiers inherited by continuation cell-parameter rows."""
    report = pd.DataFrame(
        {
            "report_line_number": data.report.linenos,
            "section": data.report.sections,
            "title": data.report.titles,
            "ffield_value": data.report.ffield_values,
            "qm_value": data.report.qm_values,
            "weight": data.report.weights,
        }
    )
    report = report.loc[
        report["section"]
        .fillna("")
        .astype(str)
        .str.upper()
        .eq("CELL PARAMETERS")
    ]
    rows: list[dict[str, object]] = []
    current_identifier = ""
    pattern = re.compile(
        r"^(?:(?P<identifier>.*?)\s+)?"
        r"(?P<parameter>alpha|beta|gamma|a|b|c)\s*:?\s*$",
        flags=re.IGNORECASE,
    )
    for _, row in report.iterrows():
        match = pattern.match(str(row.get("title", "")).strip())
        if not match:
            continue
        explicit_identifier = str(match.group("identifier") or "").strip()
        if explicit_identifier:
            current_identifier = explicit_identifier
        if not current_identifier:
            continue
        rows.append(
            {
                "iden": current_identifier,
                "parameter_type": _parameter_type(match.group("parameter")),
                "ffield_value": row.get("ffield_value"),
                "qm_value": row.get("qm_value"),
                "weight": row.get("weight"),
                "report_line_number": row.get("report_line_number"),
            }
        )
    return pd.DataFrame(rows)


def build_cell_parameter_table(
    data: ForceFieldOptimizationPlotBundleData,
) -> pd.DataFrame:
    """Match fort.99 cell predictions to parsed training-set targets."""
    report = _report_cell_parameter_table(data)
    if report.empty:
        return pd.DataFrame(columns=CELL_PARAMETER_COLUMNS)

    report["weight_key"] = report["weight"].map(_number_key)
    report["match_occurrence"] = report.groupby(
        ["iden", "parameter_type", "weight_key"], sort=False
    ).cumcount()

    train = data.training_set.cell_parameters.copy()
    if train.empty:
        joined = report.copy()
        joined["trainset_lit"] = pd.NA
        joined["group_comment"] = ""
        joined["inline_comment"] = ""
        joined["trainset_line_number"] = pd.NA
    else:
        train["iden"] = train["iden"].fillna("").astype(str).str.strip()
        train["parameter_type"] = train["type"].map(_parameter_type)
        train["weight_key"] = train["weight"].map(_number_key)
        train["match_occurrence"] = train.groupby(
            ["iden", "parameter_type", "weight_key"], sort=False
        ).cumcount()
        annotations = train.loc[
            :,
            [
                "iden",
                "parameter_type",
                "weight_key",
                "match_occurrence",
                "lit",
                "group_comment",
                "inline_comment",
                "line_number",
            ],
        ].rename(
            columns={"lit": "trainset_lit", "line_number": "trainset_line_number"}
        )
        joined = report.merge(
            annotations,
            on=["iden", "parameter_type", "weight_key", "match_occurrence"],
            how="left",
        )

    joined["label"] = joined.apply(
        lambda row: f"{row['iden']} [{row['parameter_type']}]", axis=1
    )
    for column in ("group_comment", "inline_comment"):
        joined[column] = joined[column].fillna("").astype(str)
    return joined.loc[:, CELL_PARAMETER_COLUMNS].reset_index(drop=True)


def cell_parameter_plot_payloads(
    table: pd.DataFrame,
    *,
    entries_per_figure: int,
) -> list[dict[str, object]]:
    """Split cell targets into consistently styled paired-bar figures."""
    limit = int(entries_per_figure)
    if limit < 1:
        raise ValueError("entries_per_figure must be at least 1.")
    if table.empty:
        return []

    payloads: list[dict[str, object]] = []
    total = len(table)
    figure_count = math.ceil(total / limit)
    for figure_index, start in enumerate(range(0, total, limit), start=1):
        chunk = table.iloc[start : start + limit]
        end = start + len(chunk)
        payloads.append(
            {
                "plot_type": "grouped_bar_plot",
                "labels": chunk["label"].astype(str).tolist(),
                "series": [
                    {
                        "label": "ReaxFF",
                        "values": pd.to_numeric(
                            chunk["ffield_value"], errors="coerce"
                        ).tolist(),
                        "color": "tab:blue",
                    },
                    {
                        "label": "QM/Literature",
                        "values": pd.to_numeric(
                            chunk["qm_value"], errors="coerce"
                        ).tolist(),
                        "color": "tab:orange",
                    },
                ],
                "xlabel": "Training-set cell parameter",
                "ylabel": "Cell parameter value",
                "title": f"Cell Parameters ({start + 1}-{end} of {total})",
                "legend": True,
                "grid": False,
                "group_width": 0.48,
                "label_rotation": 15,
                "label_horizontal_alignment": "right",
                "filename": (
                    f"cell_parameters_{figure_index:03d}_of_"
                    f"{figure_count:03d}.png"
                ),
                "figsize": (max(8.0, 2.2 * len(chunk)), 5.2),
            }
        )
    return payloads


__all__ = ["build_cell_parameter_table", "cell_parameter_plot_payloads"]
