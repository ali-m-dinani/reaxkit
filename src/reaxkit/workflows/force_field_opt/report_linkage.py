"""Link fort.99 report records to their source training-set entries."""

from __future__ import annotations

import pandas as pd

from reaxkit.domain.data_models import ForceFieldOptimizationPlotBundleData
from reaxkit.workflows.force_field_opt.cell_parameters import (
    build_cell_parameter_table,
)
from reaxkit.workflows.force_field_opt.charge import build_charge_table
from reaxkit.workflows.force_field_opt.energy_categories import (
    _report_expression_key,
    _train_expression_key,
)
from reaxkit.workflows.force_field_opt.geometry_targets import (
    build_geometry_target_table,
)
from reaxkit.workflows.force_field_opt.heatfo import _number_key, build_heatfo_table


TRAINSET_LINK_COLUMNS = [
    "trainset_line_number",
    "group_comment",
    "inline_comment",
]


def _expression_terms(key: object) -> list[tuple[str, str]]:
    """Split a canonical expression key into ordered sign/identifier pairs."""
    terms: list[tuple[str, str]] = []
    for raw_term in str(key).split("|"):
        if len(raw_term) < 2 or raw_term[0] not in {"+", "-"}:
            continue
        terms.append((raw_term[0], raw_term[1:]))
    return terms


def _truncated_expression_score(report_key: object, train_key: object) -> int | None:
    """Score a train expression that uniquely extends a truncated report title."""
    report_terms = _expression_terms(report_key)
    train_terms = _expression_terms(train_key)
    if not report_terms or len(report_terms) > len(train_terms):
        return None
    score = 0
    for (report_sign, report_id), (train_sign, train_id) in zip(
        report_terms, train_terms
    ):
        if report_sign != train_sign or not train_id.startswith(report_id):
            return None
        score += len(report_id)
    return score


def _energy_links(data: ForceFieldOptimizationPlotBundleData) -> pd.DataFrame:
    """Match every ENERGY report expression to its training-set occurrence."""
    train = data.training_set.energy.copy()
    if train.empty:
        return pd.DataFrame(columns=["report_line_number", *TRAINSET_LINK_COLUMNS])
    train["expression_key"] = train.apply(_train_expression_key, axis=1)
    train["weight_key"] = train["weight"].map(_number_key)
    train["match_occurrence"] = train.groupby(
        ["expression_key", "weight_key"], sort=False
    ).cumcount()
    train = train.rename(columns={"line_number": "trainset_line_number"})

    report = pd.DataFrame(
        {
            "report_line_number": data.report.linenos,
            "section": data.report.sections,
            "title": data.report.titles,
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

    links = report.merge(
        train.loc[
            :,
            [
                "expression_key",
                "weight_key",
                "match_occurrence",
                "trainset_line_number",
                "group_comment",
                "inline_comment",
            ],
        ],
        on=["expression_key", "weight_key", "match_occurrence"],
        how="left",
    )
    used_trainset_lines = set(
        pd.to_numeric(links["trainset_line_number"], errors="coerce")
        .dropna()
        .astype(int)
        .tolist()
    )
    for index in links.index[links["trainset_line_number"].isna()]:
        report_row = links.loc[index]
        candidates = train.loc[
            train["weight_key"].eq(report_row["weight_key"])
            & ~pd.to_numeric(train["trainset_line_number"], errors="coerce")
            .isin(used_trainset_lines)
        ].copy()
        if candidates.empty:
            continue
        candidates["prefix_score"] = candidates["expression_key"].map(
            lambda key: _truncated_expression_score(
                report_row["expression_key"], key
            )
        )
        candidates = candidates.dropna(subset=["prefix_score"])
        if candidates.empty:
            continue
        best_score = candidates["prefix_score"].max()
        best = candidates.loc[candidates["prefix_score"].eq(best_score)]
        if len(best) != 1:
            continue
        match = best.iloc[0]
        for column in TRAINSET_LINK_COLUMNS:
            links.at[index, column] = match.get(column, pd.NA)
        used_trainset_lines.add(int(match["trainset_line_number"]))
    return links.loc[:, ["report_line_number", *TRAINSET_LINK_COLUMNS]]


def build_report_trainset_links(
    data: ForceFieldOptimizationPlotBundleData,
) -> pd.DataFrame:
    """Return one training-set annotation row per matched fort.99 line."""
    candidates = [
        _energy_links(data),
        build_charge_table(data),
        build_geometry_target_table(data),
        build_cell_parameter_table(data),
        build_heatfo_table(data),
    ]
    frames: list[pd.DataFrame] = []
    required = {"report_line_number", *TRAINSET_LINK_COLUMNS}
    for table in candidates:
        if table.empty or not required.issubset(table.columns):
            continue
        frame = table.loc[:, ["report_line_number", *TRAINSET_LINK_COLUMNS]].copy()
        frame["report_line_number"] = pd.to_numeric(
            frame["report_line_number"], errors="coerce"
        )
        frame = frame.dropna(subset=["report_line_number"])
        frames.append(frame)
    if not frames:
        return pd.DataFrame(columns=["report_line_number", *TRAINSET_LINK_COLUMNS])

    links = pd.concat(frames, ignore_index=True)
    links["trainset_line_number"] = pd.to_numeric(
        links["trainset_line_number"], errors="coerce"
    ).astype("Int64")
    for column in ("group_comment", "inline_comment"):
        links[column] = links[column].fillna("").astype(str)
    return links.drop_duplicates("report_line_number", keep="first").sort_values(
        "report_line_number", kind="stable"
    ).reset_index(drop=True)


def add_trainset_links(
    table: pd.DataFrame,
    links: pd.DataFrame,
    *,
    report_line_column: str = "report_line_number",
) -> pd.DataFrame:
    """Ensure a result table contains the three training-set linkage columns."""
    out = table.copy()
    if report_line_column not in out.columns:
        for column in TRAINSET_LINK_COLUMNS:
            if column not in out.columns:
                out[column] = pd.NA if column == "trainset_line_number" else ""
        return out

    usable_links = links.copy()
    if usable_links.empty:
        usable_links = pd.DataFrame(
            columns=["report_line_number", *TRAINSET_LINK_COLUMNS]
        )
    rename = {
        column: f"_linked_{column}" for column in TRAINSET_LINK_COLUMNS
    }
    usable_links = usable_links.rename(columns=rename)
    out = out.merge(
        usable_links,
        left_on=report_line_column,
        right_on="report_line_number",
        how="left",
    )
    if report_line_column != "report_line_number":
        out = out.drop(columns=["report_line_number"])

    linked_line = "_linked_trainset_line_number"
    if "trainset_line_number" in out.columns:
        out["trainset_line_number"] = pd.to_numeric(
            out["trainset_line_number"], errors="coerce"
        ).combine_first(pd.to_numeric(out[linked_line], errors="coerce"))
    else:
        out["trainset_line_number"] = pd.to_numeric(
            out[linked_line], errors="coerce"
        )
    out["trainset_line_number"] = out["trainset_line_number"].astype("Int64")

    for column in ("group_comment", "inline_comment"):
        linked = out[f"_linked_{column}"].fillna("").astype(str)
        if column in out.columns:
            existing = out[column].fillna("").astype(str)
            out[column] = existing.where(existing.str.strip().ne(""), linked)
        else:
            out[column] = linked
    out = out.drop(columns=list(rename.values()))
    return out


__all__ = [
    "TRAINSET_LINK_COLUMNS",
    "add_trainset_links",
    "build_report_trainset_links",
]
