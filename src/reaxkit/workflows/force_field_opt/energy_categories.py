"""Classified ENERGY curve and grouped-bar plot helpers."""

from __future__ import annotations

import math
import re

import pandas as pd

from reaxkit.analysis.force_field.trainset import _get_trainset_group_comments
from reaxkit.domain.data_models import ForceFieldOptimizationPlotBundleData
from reaxkit.workflows.force_field_opt.heatfo import (
    _format_expression,
    _number_key,
    _operand_indices,
    _plot_identifier,
)


ENERGY_CATEGORY_COLUMNS = [
    "data_type",
    "block_id",
    "block_label",
    "expression",
    "plot_identifier",
    "curve_identifier",
    "curve_coordinate",
    "curve_coordinate_source",
    "ffield_value",
    "qm_value",
    "trainset_lit",
    "weight",
    "group_comment",
    "inline_comment",
    "report_line_number",
    "trainset_line_number",
]

_REPORT_OPERAND = re.compile(
    r"(?P<operator>[+-])\s*(?P<identifier>[^+\-\s/]+)"
    r"(?:\s*/\s*[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)?"
)


def _train_expression_key(row: pd.Series) -> str:
    """Return an ordered sign/identifier key, ignoring displayed divisors."""
    terms: list[str] = []
    for index in _operand_indices(row):
        operator = str(row.get(f"op{index}", "+")).strip()
        operator = operator if operator in {"+", "-"} else "+"
        terms.append(f"{operator}{str(row.get(f'id{index}')).strip()}")
    return "|".join(terms)


def _report_expression_key(title: object) -> str:
    """Parse the ENERGY operands retained in a fort.99 title."""
    text = re.sub(r"^\s*energy\s+", "", str(title), flags=re.IGNORECASE)
    return "|".join(
        f"{match.group('operator')}{match.group('identifier')}"
        for match in _REPORT_OPERAND.finditer(text)
    )


def _classified_train_energy(
    data: ForceFieldOptimizationPlotBundleData,
) -> pd.DataFrame:
    """Annotate every parsed training ENERGY row with its comment-block type."""
    train = data.training_set.energy.copy()
    if train.empty:
        return train
    train["data_type"] = ""
    train["block_id"] = pd.NA
    train["block_label"] = ""
    train_lines = pd.to_numeric(train["line_number"], errors="coerce")
    summaries = _get_trainset_group_comments(data.training_set)
    summaries = summaries.loc[
        summaries["section"].astype(str).str.lower().eq("energy")
    ]
    for _, summary in summaries.iterrows():
        first = pd.to_numeric(summary.get("first_entry_line_number"), errors="coerce")
        last = pd.to_numeric(summary.get("last_entry_line_number"), errors="coerce")
        if pd.isna(first) or pd.isna(last):
            continue
        mask = train_lines.between(float(first), float(last), inclusive="both")
        block_line = pd.to_numeric(summary.get("line_number"), errors="coerce")
        block_id = int(block_line if pd.notna(block_line) else first)
        comment = str(summary.get("group_comment", "")).strip()
        data_type = str(summary.get("type of data", "")).strip()
        train.loc[mask, "data_type"] = data_type
        train.loc[mask, "block_id"] = block_id
        train.loc[mask, "block_label"] = comment or (
            f"Uncommented block at trainset line {int(first)}"
        )
    return train


def _bar_label(row: pd.Series, data_type: str) -> str:
    """Return a compact label suitable for an ENERGY grouped-bar plot."""
    if data_type == "reaction_energy":
        return _plot_identifier(row)
    identifiers = [
        str(row.get(f"id{index}")).strip() for index in _operand_indices(row)
    ]
    return " - ".join(identifiers) if identifiers else "expression"


def _terminal_coordinate(identifier: object) -> float | None:
    """Extract a decimal or underscore-encoded terminal scan coordinate."""
    text = str(identifier).strip()
    encoded = re.search(r"(?<!\d)([+-]?\d+)_([0-9]+)$", text)
    if encoded:
        integer, fraction = encoded.groups()
        return float(f"{integer}.{fraction}")
    number = re.search(r"(?<![A-Za-z0-9])([+-]?\d+(?:\.\d+)?)$", text)
    return float(number.group(1)) if number else None


def _add_curve_coordinates(table: pd.DataFrame) -> pd.DataFrame:
    """Choose each curve's varying identifier and derive an x coordinate."""
    out = table.copy()
    out["curve_identifier"] = ""
    out["curve_coordinate"] = pd.NA
    out["curve_coordinate_source"] = ""
    for _, group in out.groupby("block_id", sort=False, dropna=False):
        counts: dict[str, int] = {}
        for _, row in group.iterrows():
            for index in _operand_indices(row):
                identifier = str(row.get(f"id{index}")).strip()
                counts[identifier] = counts.get(identifier, 0) + 1

        identifiers: list[str] = []
        coordinates: list[float | None] = []
        for _, row in group.iterrows():
            operands = [
                str(row.get(f"id{index}")).strip()
                for index in _operand_indices(row)
            ]
            if operands:
                minimum_count = min(counts.get(identifier, 0) for identifier in operands)
                candidates = [
                    identifier
                    for identifier in operands
                    if counts.get(identifier, 0) == minimum_count
                ]
                identifier = candidates[-1]
            else:
                identifier = "entry"
            identifiers.append(identifier)
            coordinates.append(_terminal_coordinate(identifier))

        indices = group.index
        out.loc[indices, "curve_identifier"] = identifiers
        if all(coordinate is not None for coordinate in coordinates):
            out.loc[indices, "curve_coordinate"] = coordinates
            out.loc[indices, "curve_coordinate_source"] = "identifier suffix"
        else:
            out.loc[indices, "curve_coordinate"] = list(range(1, len(group) + 1))
            out.loc[indices, "curve_coordinate_source"] = "entry index"
    return out


def build_energy_category_tables(
    data: ForceFieldOptimizationPlotBundleData,
) -> dict[str, pd.DataFrame]:
    """Match selected training ENERGY categories to fort.99 report values."""
    categories = (
        "energy_curve",
        "energy_difference",
        "reaction_energy",
        "single_energy",
    )
    empty = {kind: pd.DataFrame(columns=ENERGY_CATEGORY_COLUMNS) for kind in categories}
    train = _classified_train_energy(data)
    if train.empty:
        return empty
    single_operand = train.apply(lambda row: len(_operand_indices(row)) == 1, axis=1)
    train.loc[single_operand, "data_type"] = "single_energy"
    train = train.loc[train["data_type"].isin(categories)].copy()
    if train.empty:
        return empty

    train["expression_key"] = train.apply(_train_expression_key, axis=1)
    train["weight_key"] = train["weight"].map(_number_key)
    train["match_occurrence"] = train.groupby(
        ["expression_key", "weight_key"], sort=False
    ).cumcount()
    train["expression"] = train.apply(_format_expression, axis=1)
    train["plot_identifier"] = train.apply(
        lambda row: _bar_label(row, str(row["data_type"])), axis=1
    )

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
    report["expression_key"] = report["title"].map(_report_expression_key)
    report["weight_key"] = report["report_weight"].map(_number_key)
    report["match_occurrence"] = report.groupby(
        ["expression_key", "weight_key"], sort=False
    ).cumcount()

    joined = train.merge(
        report.loc[
            :,
            [
                "expression_key",
                "weight_key",
                "match_occurrence",
                "ffield_value",
                "qm_value",
                "report_line_number",
            ],
        ],
        on=["expression_key", "weight_key", "match_occurrence"],
        how="left",
    ).rename(
        columns={"lit": "trainset_lit", "line_number": "trainset_line_number"}
    )
    for column in ("group_comment", "inline_comment"):
        joined[column] = joined[column].fillna("").astype(str)
    joined = joined.loc[joined["report_line_number"].notna()].copy()
    joined = _add_curve_coordinates(joined)

    tables: dict[str, pd.DataFrame] = {}
    for kind in categories:
        table = joined.loc[joined["data_type"].eq(kind)].copy()
        tables[kind] = table.loc[:, ENERGY_CATEGORY_COLUMNS].sort_values(
            "trainset_line_number", kind="stable"
        ).reset_index(drop=True)
    return tables


def energy_curve_plot_groups(table: pd.DataFrame) -> list[dict[str, object]]:
    """Build paired ReaxFF/QM curve groups for ENERGY curve blocks."""
    if table.empty:
        return []
    groups: list[dict[str, object]] = []
    for _, raw_group in table.groupby("block_id", sort=False, dropna=False):
        group = raw_group.copy()
        group["curve_coordinate"] = pd.to_numeric(
            group["curve_coordinate"], errors="coerce"
        )
        group["ffield_value"] = pd.to_numeric(group["ffield_value"], errors="coerce")
        group["qm_value"] = pd.to_numeric(group["qm_value"], errors="coerce")
        group = group.dropna(subset=["curve_coordinate"]).sort_values(
            "curve_coordinate", kind="stable"
        )
        reaxff = group.dropna(subset=["ffield_value"])
        qm = group.dropna(subset=["qm_value"])
        if reaxff.empty and qm.empty:
            continue
        source = str(group.iloc[0]["curve_coordinate_source"])
        groups.append(
            {
                "identifier": str(group.iloc[0]["block_label"]),
                "filename_identifier": (
                    f"{group.iloc[0]['block_label']}_{group.iloc[0]['block_id']}"
                ),
                "xlabel": (
                    "Scan coordinate from identifier"
                    if source == "identifier suffix"
                    else "Entry index"
                ),
                "reaxff_x": reaxff["curve_coordinate"].astype(float).tolist(),
                "reaxff_y": reaxff["ffield_value"].astype(float).tolist(),
                "qm_x": qm["curve_coordinate"].astype(float).tolist(),
                "qm_y": qm["qm_value"].astype(float).tolist(),
            }
        )
    return groups


def energy_bar_plot_payloads(
    table: pd.DataFrame,
    *,
    entries_per_figure: int,
    filename_prefix: str,
    title: str,
    ylabel: str,
) -> list[dict[str, object]]:
    """Split ENERGY entries into consistently styled paired-bar figures."""
    limit = int(entries_per_figure)
    if limit < 1:
        raise ValueError("entries_per_figure must be at least 1.")
    if table.empty:
        return []
    payloads: list[dict[str, object]] = []
    figure_count = math.ceil(len(table) / limit)
    for figure_index, start in enumerate(range(0, len(table), limit), start=1):
        chunk = table.iloc[start : start + limit]
        payloads.append(
            {
                "plot_type": "grouped_bar_plot",
                "labels": chunk["plot_identifier"].astype(str).tolist(),
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
                        "color": "#C0504D",
                    },
                ],
                "title": (
                    title
                    if figure_count == 1
                    else f"{title} ({figure_index}/{figure_count})"
                ),
                "ylabel": ylabel,
                "legend": True,
                "grid": False,
                "group_width": 0.48,
                "minimum_category_slots": limit,
                "filename": f"{filename_prefix}_{figure_index:03d}.png",
            }
        )
    return payloads


__all__ = [
    "ENERGY_CATEGORY_COLUMNS",
    "build_energy_category_tables",
    "energy_bar_plot_payloads",
    "energy_curve_plot_groups",
]
