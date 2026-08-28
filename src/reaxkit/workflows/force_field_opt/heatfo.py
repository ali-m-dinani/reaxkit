"""Heat-of-formation extraction and grouped-bar payload helpers."""

from __future__ import annotations

import math
import re

import pandas as pd

from reaxkit.domain.data_models import ForceFieldOptimizationPlotBundleData

HEATFO_COMMENT_PATTERN = r"(?:heat\s+of\s+formation|heatfo)"
HEATFO_COLUMNS = [
    "expression",
    "plot_identifier",
    "first_identifier",
    "ffield_value",
    "qm_value",
    "trainset_lit",
    "weight",
    "group_comment",
    "inline_comment",
    "report_line_number",
    "trainset_line_number",
]


def _number_key(value: object) -> str:
    try:
        return f"{float(value):.12g}"
    except (TypeError, ValueError):
        return ""


def _operand_indices(row: pd.Series) -> list[int]:
    """Return the ordered indices of populated ENERGY operands."""
    return sorted(
        int(match.group(1))
        for column in row.index
        if (match := re.fullmatch(r"id(\d+)", str(column)))
        and pd.notna(row.get(column))
        and str(row.get(column)).strip()
    )


def _format_expression(row: pd.Series) -> str:
    """Reconstruct one signed ENERGY expression from dynamic operand columns."""
    terms: list[str] = []
    for index in _operand_indices(row):
        identifier = str(row.get(f"id{index}")).strip()
        operator = str(row.get(f"op{index}", "+")).strip()
        operator = operator if operator in {"+", "-"} else "+"
        divisor = _number_key(row.get(f"n{index}", 1.0)) or "1"
        terms.append(f"{operator}{identifier}/{divisor}")
    return " ".join(terms)


def _plot_identifier(row: pd.Series) -> str:
    """Select the identifier carrying the unique + or - sign in an expression."""
    operands: list[tuple[str, str]] = []
    for index in _operand_indices(row):
        identifier = str(row.get(f"id{index}")).strip()
        operator = str(row.get(f"op{index}", "+")).strip()
        operands.append((operator if operator in {"+", "-"} else "+", identifier))

    sign_counts = {
        sign: sum(operator == sign for operator, _ in operands) for sign in {"+", "-"}
    }
    unique_signs = [
        sign
        for sign, count in sign_counts.items()
        if count == 1 and sign_counts["-" if sign == "+" else "+"] > 1
    ]
    if len(unique_signs) == 1:
        return next(
            identifier
            for operator, identifier in operands
            if operator == unique_signs[0]
        )
    return operands[0][1] if operands else "expression"


def _energy_heatfo_table(
    data: ForceFieldOptimizationPlotBundleData,
) -> pd.DataFrame:
    """Match comment-labeled training ENERGY expressions to fort.99 results."""
    train = data.training_set.energy.copy()
    if train.empty or "group_comment" not in train.columns:
        return pd.DataFrame(columns=HEATFO_COLUMNS)
    selected = train.loc[
        train["group_comment"].fillna("").astype(str).str.contains(
            HEATFO_COMMENT_PATTERN, case=False, regex=True
        )
    ].copy()
    if selected.empty:
        return pd.DataFrame(columns=HEATFO_COLUMNS)

    selected["first_identifier"] = selected["id1"].fillna("").astype(str).str.strip()
    selected["weight_key"] = selected["weight"].map(_number_key)
    selected["match_occurrence"] = selected.groupby(
        ["first_identifier", "weight_key"], sort=False
    ).cumcount()
    selected["expression"] = selected.apply(_format_expression, axis=1)
    selected["plot_identifier"] = selected.apply(_plot_identifier, axis=1)

    report = pd.DataFrame(
        {
            "report_line_number": data.report.linenos,
            "section": data.report.sections,
            "title": data.report.titles,
            "ffield_value": data.report.ffield_values,
            "qm_value": data.report.qm_values,
            "report_weight": data.report.weights,
        }
    )
    report = report.loc[
        report["section"].fillna("").astype(str).str.upper().eq("ENERGY")
    ].copy()
    report["first_identifier"] = report["title"].fillna("").astype(str).str.extract(
        r"^Energy\s+[+-](?P<identifier>[^\s/]+)", flags=re.IGNORECASE
    )["identifier"]
    wanted = set(selected["first_identifier"])
    report = report.loc[report["first_identifier"].isin(wanted)].copy()
    report["weight_key"] = report["report_weight"].map(_number_key)
    report["match_occurrence"] = report.groupby(
        ["first_identifier", "weight_key"], sort=False
    ).cumcount()

    joined = selected.merge(
        report.loc[
            :,
            [
                "first_identifier",
                "weight_key",
                "match_occurrence",
                "ffield_value",
                "qm_value",
                "report_line_number",
            ],
        ],
        on=["first_identifier", "weight_key", "match_occurrence"],
        how="left",
    )
    joined = joined.rename(
        columns={"lit": "trainset_lit", "line_number": "trainset_line_number"}
    )
    for column in ("group_comment", "inline_comment"):
        if column not in joined.columns:
            joined[column] = ""
        joined[column] = joined[column].fillna("").astype(str)
    return joined.loc[:, HEATFO_COLUMNS].sort_values(
        "trainset_line_number", kind="stable"
    ).reset_index(drop=True)


_REPORT_TARGET_MARKER = re.compile(
    r"\b(?:heat\s+of\s+formation|charge\s+atom|bond(?:\s+distance)?|"
    r"valence\s+angle|torsion\s+angle)\s*:",
    flags=re.IGNORECASE,
)
_REPORT_HEATFO_MARKER = re.compile(
    r"\bheat\s+of\s+formation\s*:?\s*$", flags=re.IGNORECASE
)


def _direct_report_heatfo_table(
    data: ForceFieldOptimizationPlotBundleData,
) -> pd.DataFrame:
    """Recover direct HEATFO identifiers inherited across fort.99 rows."""
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
    rows: list[dict[str, object]] = []
    current_identifier = ""
    for _, row in report.iterrows():
        title = str(row.get("title", "")).strip()
        target_match = _REPORT_TARGET_MARKER.search(title)
        if target_match:
            explicit_identifier = title[: target_match.start()].strip()
            if explicit_identifier:
                current_identifier = explicit_identifier

        if str(row.get("section", "")).upper() != "HEATFO":
            continue
        if not _REPORT_HEATFO_MARKER.search(title) or not current_identifier:
            continue
        rows.append(
            {
                "first_identifier": current_identifier,
                "ffield_value": row.get("ffield_value"),
                "qm_value": row.get("qm_value"),
                "report_weight": row.get("weight"),
                "report_line_number": row.get("report_line_number"),
            }
        )
    return pd.DataFrame(rows)


def _direct_heatfo_table(
    data: ForceFieldOptimizationPlotBundleData,
) -> pd.DataFrame:
    """Match direct HEATFO trainset entries to fort.99 heat rows."""
    train = data.training_set.heatfo.copy()
    if train.empty:
        return pd.DataFrame(columns=HEATFO_COLUMNS)
    train["first_identifier"] = train["iden"].fillna("").astype(str).str.strip()
    train["weight_key"] = train["weight"].map(_number_key)
    train["match_occurrence"] = train.groupby(
        ["first_identifier", "weight_key"], sort=False
    ).cumcount()
    train["expression"] = train["first_identifier"]
    train["plot_identifier"] = train["first_identifier"]

    report = _direct_report_heatfo_table(data)
    if report.empty:
        joined = train.copy()
        joined["ffield_value"] = pd.NA
        joined["qm_value"] = pd.NA
        joined["report_line_number"] = pd.NA
    else:
        report["weight_key"] = report["report_weight"].map(_number_key)
        report["match_occurrence"] = report.groupby(
            ["first_identifier", "weight_key"], sort=False
        ).cumcount()
        joined = train.merge(
            report.loc[
                :,
                [
                    "first_identifier",
                    "weight_key",
                    "match_occurrence",
                    "ffield_value",
                    "qm_value",
                    "report_line_number",
                ],
            ],
            on=["first_identifier", "weight_key", "match_occurrence"],
            how="left",
        )

    joined = joined.rename(
        columns={"lit": "trainset_lit", "line_number": "trainset_line_number"}
    )
    for column in ("group_comment", "inline_comment"):
        if column not in joined.columns:
            joined[column] = ""
        joined[column] = joined[column].fillna("").astype(str)
    return joined.loc[:, HEATFO_COLUMNS].reset_index(drop=True)


def build_heatfo_table(data: ForceFieldOptimizationPlotBundleData) -> pd.DataFrame:
    """Combine direct HEATFO and comment-labeled ENERGY heat targets."""
    tables = [
        table
        for table in (_direct_heatfo_table(data), _energy_heatfo_table(data))
        if not table.empty
    ]
    if not tables:
        return pd.DataFrame(columns=HEATFO_COLUMNS)
    return pd.concat(tables, ignore_index=True).sort_values(
        "trainset_line_number", kind="stable", na_position="last"
    ).reset_index(drop=True)


def heatfo_plot_payloads(
        table: pd.DataFrame,
        *,
        expressions_per_figure: int,
) -> list[dict[str, object]]:
    """Split HeatFO expressions into consistently styled grouped-bar payloads."""
    limit = int(expressions_per_figure)
    if limit < 1:
        raise ValueError("expressions_per_figure must be at least 1.")
    if table.empty:
        return []

    payloads: list[dict[str, object]] = []
    total = len(table)
    figure_count = math.ceil(total / limit)
    for figure_index, start in enumerate(range(0, total, limit), start=1):
        chunk = table.iloc[start: start + limit]
        end = start + len(chunk)
        label_column = (
            "plot_identifier"
            if "plot_identifier" in chunk.columns
            else "first_identifier"
            if "first_identifier" in chunk.columns
            else "expression"
        )
        payloads.append(
            {
                "plot_type": "grouped_bar_plot",
                "labels": chunk[label_column].astype(str).tolist(),
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
                "xlabel": "Structure identifier",
                "ylabel": "Heat of formation (kcal/mol)",
                "title": f"Heat of Formation ({start + 1}-{end} of {total})",
                "legend": True,
                "grid": False,
                "group_width": 0.48,
                "label_rotation": 0,
                "label_horizontal_alignment": "center",
                "filename": f"heatfo_{figure_index:03d}_of_{figure_count:03d}.png",
                "figsize": (max(8.0, 2.2 * len(chunk)), 5.2),
            }
        )
    return payloads


__all__ = ["build_heatfo_table", "heatfo_plot_payloads"]
