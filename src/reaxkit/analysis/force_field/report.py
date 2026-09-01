"""Analyze force-field optimization report artifacts and derived fit summaries.

This module parses report-oriented optimization outputs, including EOS and
bulk-modulus related records, into analyzer-ready tables. It is scoped to
report extraction and derived fit metrics, not parameter optimization itself.

**Usage context**

- Report parsing: Normalize textual optimization-report sections into tables.
- EOS analysis: Build equation-of-state fit inputs and summary outputs.
- Materials metrics: Compute report-level bulk-modulus-related diagnostics.
"""

from __future__ import annotations

from dataclasses import dataclass, field as dc_field
import re
from typing import Any, Optional

import numpy as np
import pandas as pd

from reaxkit.analysis.base import AnalysisTask
from reaxkit.core.platform.constants import const
from reaxkit.core.registry.analysis_task_registry import register_task
from reaxkit.domain.base_request import BaseRequest
from reaxkit.domain.base_result import BaseResult
from reaxkit.domain.data_models import (
    ForceFieldOptimizationPlotBundleData,
    ForceFieldOptimizationReportData,
    ForceFieldOptimizationReportEOSBundleData,
    EnergyMinimizationSummaryData,
)
from reaxkit.presentation.specs import PresentationSpec
from reaxkit.utils.equation_of_states import vinet_energy_ev


def _report_frame(data: ForceFieldOptimizationReportData) -> pd.DataFrame:
    """Build the base optimization-report DataFrame from parsed report fields."""
    df = pd.DataFrame(
        {
            "lineno": pd.Series(data.linenos, dtype=int),
            "section": pd.Series(data.sections, dtype=object),
            "title": pd.Series(data.titles, dtype=object),
            "ffield_value": pd.Series(data.ffield_values, dtype=float),
            "qm_value": pd.Series(data.qm_values, dtype=float),
            "weight": pd.Series(data.weights, dtype=float),
            "error": pd.Series(data.errors, dtype=float),
            "total_ff_error": pd.Series(data.total_ff_error, dtype=float),
        }
    )
    df["section"] = df["section"].replace("", pd.NA)
    return df


def _geometry_summary_frame(data: EnergyMinimizationSummaryData) -> pd.DataFrame:
    """Build an identifier/volume/energy frame from geometry-summary data."""
    n_rows = len(data.identifiers)
    energy_series = None
    if data.minimum_energy is not None:
        energy_series = pd.Series(data.minimum_energy, dtype=float)
    elif data.formation_energy is not None:
        energy_series = pd.Series(data.formation_energy, dtype=float)
    else:
        energy_series = pd.Series([pd.NA] * n_rows)
    return pd.DataFrame(
        {
            "identifier": pd.Series(data.identifiers, dtype=object),
            "V": (
                pd.Series(data.volume, dtype=float)
                if data.volume is not None
                else pd.Series([pd.NA] * n_rows)
            ),
            "E": energy_series,
        }
    )


def _get_report_data(
    data: ForceFieldOptimizationReportData,
) -> pd.DataFrame:
    """Return sorted report rows augmented with QM-minus-FF differences."""
    df = _report_frame(data)
    df["qm_ff_difference"] = df["qm_value"] - df["ffield_value"]
    return df.sort_values("lineno", ascending=True).reset_index(drop=True)


def _parse_two_body_energy_terms(data: ForceFieldOptimizationReportData) -> pd.DataFrame:
    """Parse two-body ENERGY titles with optional displayed divisors."""
    df = _report_frame(data)
    if "section" not in df.columns or "title" not in df.columns:
        raise KeyError("Expected 'section' and 'title' columns in report DataFrame.")

    energy_df = df[df["section"].astype(str).str.upper() == "ENERGY"].copy()

    pattern = re.compile(
        r"^\s*Energy\s+"
        r"(?P<sign1>[+-])\s*(?P<iden1>[^\s/]+)"
        r"(?:\s*/\s*(?P<n1>\d+(?:\.\d+)?))?\s+"
        r"(?P<sign2>[+-])\s*(?P<iden2>[^\s/]+)"
        r"(?:\s*/\s*(?P<n2>\d+(?:\.\d+)?))?\s*$",
        flags=re.IGNORECASE,
    )

    def _parse_title(title: str) -> dict[str, object]:
        m = pattern.search(title)
        if not m:
            return {
                "opt1": np.nan,
                "iden1": np.nan,
                "n1": np.nan,
                "opt2": np.nan,
                "iden2": np.nan,
                "n2": np.nan,
            }

        g = m.groupdict()
        return {
            "opt1": 1 if g["sign1"] == "+" else -1,
            "iden1": g["iden1"].strip(),
            "n1": float(g["n1"]) if g["n1"] is not None else np.nan,
            "opt2": 1 if g["sign2"] == "+" else -1,
            "iden2": g["iden2"].strip(),
            "n2": float(g["n2"]) if g["n2"] is not None else np.nan,
        }

    parsed = energy_df["title"].astype(str).apply(_parse_title)
    parsed_df = pd.DataFrame(list(parsed))
    energy_df = pd.concat([energy_df.reset_index(drop=True), parsed_df], axis=1)
    return energy_df.dropna(subset=["iden1", "iden2"]).reset_index(drop=True)


def _energy_vs_volume(
    report: ForceFieldOptimizationReportData,
    geometry_summary: EnergyMinimizationSummaryData,
) -> pd.DataFrame:
    """Join repeated two-body report terms with geometry volumes."""
    energy_df = _parse_two_body_energy_terms(report)
    if energy_df.empty:
        return pd.DataFrame(columns=["iden1", "iden2", "ffield_value", "qm_value", "V_iden2"])

    repeated = energy_df.groupby("iden1").filter(lambda g: len(g) > 1).copy()
    if repeated.empty:
        return pd.DataFrame(columns=["iden1", "iden2", "ffield_value", "qm_value", "V_iden2"])

    repeated = repeated[repeated["iden1"] != repeated["iden2"]]
    if repeated.empty:
        return pd.DataFrame(columns=["iden1", "iden2", "ffield_value", "qm_value", "V_iden2"])

    geometry_df = _geometry_summary_frame(geometry_summary)
    if geometry_df.empty or "identifier" not in geometry_df.columns or "V" not in geometry_df.columns:
        return pd.DataFrame(columns=["iden1", "iden2", "ffield_value", "qm_value", "V_iden2"])

    vol_df = geometry_df[["identifier", "V"]].drop_duplicates()
    merged = repeated.merge(vol_df, left_on="iden2", right_on="identifier", how="left")
    out = merged[["iden1", "iden2", "ffield_value", "qm_value", "V"]].rename(columns={"V": "V_iden2"})
    return out.sort_values(["iden1", "iden2"]).reset_index(drop=True)


def _base_other_energy_volume_table(
    report: ForceFieldOptimizationReportData,
    geometry_summary: EnergyMinimizationSummaryData,
) -> pd.DataFrame:
    """Build base/other EOS data with both ReaxFF and QM energies."""
    columns = [
        "base_iden",
        "other_iden",
        "V_other_iden",
        "E_other_iden",
        "ffield_value",
        "qm_value",
    ]
    energy_df = _parse_two_body_energy_terms(report)
    if energy_df.empty:
        return pd.DataFrame(columns=columns)

    # Explicit joined identifier used for downstream duplicate inspection/debugging.
    energy_df = energy_df.copy()
    energy_df["joined_iden"] = (
        energy_df["iden1"].astype(str).str.strip() + "|" + energy_df["iden2"].astype(str).str.strip()
    )

    repeated_bases = (
        energy_df["iden1"]
        .astype(str)
        .str.strip()
        .value_counts()
        .loc[lambda s: s > 1]
        .index
        .tolist()
    )
    if not repeated_bases:
        return pd.DataFrame(columns=columns)

    geometry_df = _geometry_summary_frame(geometry_summary)
    if geometry_df.empty or "identifier" not in geometry_df.columns:
        return pd.DataFrame(columns=columns)

    geometry_df = geometry_df[["identifier", "V", "E"]].drop_duplicates(subset=["identifier"], keep="first")
    geo_map: dict[str, tuple[Any, Any]] = {
        str(row["identifier"]).strip(): (row.get("V", pd.NA), row.get("E", pd.NA))
        for _, row in geometry_df.iterrows()
    }

    rows: list[dict[str, Any]] = []
    for base in repeated_bases:
        sub = energy_df[energy_df["iden1"].astype(str).str.strip() == base].copy()
        sub["_other_iden"] = sub["iden2"].astype(str).str.strip()
        report_rows = sub.drop_duplicates(subset=["_other_iden"], keep="first").set_index("_other_iden")
        other_order: list[str] = []
        for raw in sub["_other_iden"].tolist():
            other = str(raw).strip()
            if other not in other_order:
                other_order.append(other)
        if base not in other_order:
            other_order.append(base)

        for other in other_order:
            vol, eng = geo_map.get(other, (pd.NA, pd.NA))
            if other in report_rows.index:
                report_row = report_rows.loc[other]
                ffield_value = report_row.get("ffield_value", pd.NA)
                qm_value = report_row.get("qm_value", pd.NA)
            else:
                # The base is the zero-energy reference for a ``+base -other``
                # EOS series; fort.99 normally omits base-minus-base.
                ffield_value = 0.0 if other == base else pd.NA
                qm_value = 0.0 if other == base else pd.NA
            rows.append(
                {
                    "base_iden": base,
                    "other_iden": other,
                    "V_other_iden": vol,
                    "E_other_iden": eng,
                    "ffield_value": ffield_value,
                    "qm_value": qm_value,
                }
            )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.loc[:, columns].sort_values(
        ["base_iden", "V_other_iden"],
        ascending=[True, True],
        na_position="last",
    ).reset_index(drop=True)


_CURVE_COLUMNS = [
    "curve_type",
    "base_iden",
    "other_iden",
    "iden1",
    "iden2",
    "V_other_iden",
    "E_other_iden",
    "strain_percent",
    "strain_type",
    "strain_definition",
    "scan_type",
    "scan_coordinate",
    "scan_unit",
    "coordinate_source",
    "geo_restraint_line_number",
    "ffield_value",
    "qm_value",
    "group_comment",
    "inline_comment",
    "classification_reason",
    "report_line_number",
    "trainset_line_number",
]


_ORTHORHOMBIC_STRAIN_AXES = {
    "c12": ("a", "b"),
    "c13": ("a", "c"),
    "c23": ("b", "c"),
}
_SHEAR_STRAIN_ANGLES = {
    "c44": "alpha",
    "c55": "beta",
    "c66": "gamma",
}


def elastic_family(identifier: object) -> str | None:
    """Return the supported elastic family encoded at the identifier start."""
    match = re.match(
        r"^(c(?:12|13|23|44|55|66))(?:[_\-.]|$)",
        str(identifier).strip(),
        re.IGNORECASE,
    )
    return match.group(1).lower() if match else None


def _normalized_elastic_identifier(identifier: object) -> str:
    """Collapse generator zero-padding in elastic point identifiers."""
    text = str(identifier).strip()
    return re.sub(
        r"^(c(?:12|13|23|44|55|66)[_\-.][ce])0+(\d+)(?=[_\-.]|$)",
        lambda match: f"{match.group(1)}{int(match.group(2))}",
        text,
        flags=re.IGNORECASE,
    ).lower()


def _elastic_strain_metadata(
    base_iden: object,
    other_iden: object,
    geometry_cells: pd.DataFrame | None,
) -> dict[str, Any]:
    """Calculate orthorhombic or shear strain from paired ``CRYSTX`` cells."""
    family = elastic_family(base_iden)
    empty = {
        "strain_percent": pd.NA,
        "strain_type": "",
        "strain_definition": "",
    }
    if family is None:
        return empty

    if family in _ORTHORHOMBIC_STRAIN_AXES:
        first_axis, second_axis = _ORTHORHOMBIC_STRAIN_AXES[family]
        strain_type = "orthorhombic"
        definition = (
            f"0.5 * (({first_axis}-{first_axis}0)/{first_axis}0 - "
            f"({second_axis}-{second_axis}0)/{second_axis}0) * 100"
        )
    else:
        angle = _SHEAR_STRAIN_ANGLES[family]
        strain_type = "shear_angle"
        definition = f"radians({angle}-{angle}0) * 100"

    unavailable = {
        "strain_percent": pd.NA,
        "strain_type": strain_type,
        "strain_definition": definition,
    }
    required = {"descriptor", "a", "b", "c", "alpha", "beta", "gamma"}
    if (
        geometry_cells is None
        or geometry_cells.empty
        or not required.issubset(geometry_cells.columns)
    ):
        return unavailable

    cells = geometry_cells.drop_duplicates("descriptor", keep="first").copy()
    cells["_elastic_key"] = cells["descriptor"].map(_normalized_elastic_identifier)
    cells = cells.drop_duplicates("_elastic_key", keep="first").set_index("_elastic_key")
    base_key = _normalized_elastic_identifier(base_iden)
    other_key = _normalized_elastic_identifier(other_iden)
    if base_key not in cells.index or other_key not in cells.index:
        return unavailable

    base_cell, other_cell = cells.loc[base_key], cells.loc[other_key]
    if family in _ORTHORHOMBIC_STRAIN_AXES:
        first_axis, second_axis = _ORTHORHOMBIC_STRAIN_AXES[family]
        first_0 = pd.to_numeric(base_cell[first_axis], errors="coerce")
        second_0 = pd.to_numeric(base_cell[second_axis], errors="coerce")
        first = pd.to_numeric(other_cell[first_axis], errors="coerce")
        second = pd.to_numeric(other_cell[second_axis], errors="coerce")
        values = np.asarray([first_0, second_0, first, second], dtype=float)
        if not np.isfinite(values).all() or first_0 == 0.0 or second_0 == 0.0:
            return unavailable
        strain_percent = 50.0 * (
            (float(first) - float(first_0)) / float(first_0)
            - (float(second) - float(second_0)) / float(second_0)
        )
    else:
        angle = _SHEAR_STRAIN_ANGLES[family]
        angle_0 = pd.to_numeric(base_cell[angle], errors="coerce")
        distorted_angle = pd.to_numeric(other_cell[angle], errors="coerce")
        values = np.asarray([angle_0, distorted_angle], dtype=float)
        if not np.isfinite(values).all():
            return unavailable
        strain_percent = float(np.deg2rad(float(distorted_angle) - float(angle_0)) * 100.0)

    return {
        "strain_percent": strain_percent,
        "strain_type": strain_type,
        "strain_definition": definition,
    }


def _two_body_expression_key(
    op1: object,
    iden1: object,
    n1: object,
    op2: object,
    iden2: object,
    n2: object,
) -> str:
    """Build an ordered sign/identifier key independent of displayed divisors.

    fort.99 may omit common divisors that are explicit in the training set, so
    the divisor values cannot be required for comment linkage.
    """
    sign1 = "+" if str(op1).strip() in {"+", "1", "1.0"} or op1 == 1 else "-"
    sign2 = "+" if str(op2).strip() in {"+", "1", "1.0"} or op2 == 1 else "-"
    return "|".join(
        (sign1, str(iden1).strip(), sign2, str(iden2).strip())
    )


def _annotated_two_body_energy_terms(
    data: ForceFieldOptimizationPlotBundleData,
) -> pd.DataFrame:
    """Join fort.99 two-body rows to trainset comments by expression occurrence."""
    report_rows = _parse_two_body_energy_terms(data.report).copy()
    train_rows = data.training_set.energy.copy()
    if report_rows.empty or train_rows.empty:
        return pd.DataFrame()

    extra_id_cols = [
        col
        for col in train_rows.columns
        if re.fullmatch(r"id\d+", str(col)) and str(col) not in {"id1", "id2"}
    ]
    mask = train_rows["id1"].notna() & train_rows["id2"].notna()
    for col in extra_id_cols:
        mask &= train_rows[col].isna() | train_rows[col].astype(str).str.strip().eq("")
    train_rows = train_rows.loc[mask].copy()
    if train_rows.empty:
        return pd.DataFrame()

    report_rows["expression_key"] = report_rows.apply(
        lambda row: _two_body_expression_key(
            row.get("opt1"), row.get("iden1"), row.get("n1"),
            row.get("opt2"), row.get("iden2"), row.get("n2"),
        ),
        axis=1,
    )
    train_rows["expression_key"] = train_rows.apply(
        lambda row: _two_body_expression_key(
            row.get("op1"), row.get("id1"), row.get("n1"),
            row.get("op2"), row.get("id2"), row.get("n2"),
        ),
        axis=1,
    )
    report_rows["expression_occurrence"] = report_rows.groupby(
        "expression_key", sort=False
    ).cumcount()
    train_rows["expression_occurrence"] = train_rows.groupby(
        "expression_key", sort=False
    ).cumcount()
    annotations = train_rows.loc[
        :,
        [
            "expression_key",
            "expression_occurrence",
            "group_comment",
            "inline_comment",
            "line_number",
        ],
    ].rename(columns={"line_number": "trainset_line_number"})
    joined = report_rows.merge(
        annotations,
        on=["expression_key", "expression_occurrence"],
        how="left",
    )
    joined["group_comment"] = joined["group_comment"].fillna("").astype(str)
    joined["inline_comment"] = joined["inline_comment"].fillna("").astype(str)
    joined["report_line_number"] = joined["lineno"]
    return joined


def _identifier_suggests_eos(identifier: object) -> bool:
    """Recognize common EOS and elastic-family identifier conventions."""
    text = str(identifier).strip().lower()
    return bool(
        re.search(r"(?:^|[_\-.])(bulk|volume|eos)(?:[_\-.]|$)", text)
        or re.search(r"(?:^|[_\-.])c\d+(?:[_\-.]|$)", text)
    )


def _geo_restraint_lookup(restraints: pd.DataFrame) -> dict[str, dict[str, Any]]:
    """Return unambiguous per-descriptor BOND/ANGLE restraint metadata."""
    required = {"descriptor", "restraint_type", "coordinate"}
    if restraints.empty or not required.issubset(restraints.columns):
        return {}
    lookup: dict[str, dict[str, Any]] = {}
    for descriptor, group in restraints.groupby("descriptor", sort=False, dropna=False):
        name = str(descriptor).strip()
        kinds = {
            str(value).strip().lower()
            for value in group["restraint_type"].dropna()
            if str(value).strip().lower() in {"bond", "angle"}
        }
        if not name or len(kinds) != 1:
            continue
        first = group.loc[
            group["restraint_type"].astype(str).str.lower().isin(kinds)
        ].iloc[0]
        kind = next(iter(kinds))
        lookup[name] = {
            "restraint_type": kind,
            "coordinate": pd.to_numeric(first.get("coordinate"), errors="coerce"),
            "restraint_line_number": first.get("restraint_line_number", pd.NA),
        }
    return lookup


def _identifier_scan_coordinate(identifier: object) -> float | None:
    """Extract a terminal decimal scan coordinate when geo has no restraint."""
    text = str(identifier).strip()
    encoded_decimal = re.search(r"(?<!\d)([+-]?\d+)_([0-9]+)$", text)
    if encoded_decimal:
        integer, fraction = encoded_decimal.groups()
        return float(f"{integer}.{fraction}")
    number = re.search(r"(?<![A-Za-z0-9])([+-]?\d+(?:\.\d+)?)$", text)
    return float(number.group(1)) if number else None


def _classify_curve_rows(
    rows: pd.DataFrame,
    geometry_restraints: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Classify two-structure candidates as EOS, bond, angle, or other curves."""
    if rows.empty:
        return rows
    out = rows.copy()
    comments = (
        out["group_comment"].fillna("").astype(str)
        + " "
        + out["inline_comment"].fillna("").astype(str)
    ).str.lower()
    out["legacy_restraint"] = comments.str.contains(r"\brestraint\b", regex=True)
    eos_comment = comments.str.contains(r"\b(?:eos|volume|bulk)\b", regex=True)
    eos_identifier = out.apply(
        lambda row: _identifier_suggests_eos(row.get("iden1"))
        or _identifier_suggests_eos(row.get("iden2")),
        axis=1,
    )
    restraint_lookup = _geo_restraint_lookup(
        geometry_restraints if geometry_restraints is not None else pd.DataFrame()
    )

    out["curve_type"] = "other_curve"
    out["classification_reason"] = "no unique BOND/ANGLE RESTRAINT in geo"
    out["_comment_bucket"] = out["group_comment"].str.strip()
    out.loc[out["_comment_bucket"] == "", "_comment_bucket"] = "<uncommented>"
    for _, comment_rows in out.groupby("_comment_bucket", sort=False, dropna=False):
        for indices in _component_row_indices(comment_rows):
            component = out.loc[indices]
            if eos_comment.loc[indices].any():
                curve_type = "eos"
                reason = "EOS/volume/bulk comment"
            elif eos_identifier.loc[indices].any():
                curve_type = "eos"
                reason = "EOS identifier family"
            else:
                identifiers = {
                    str(value).strip()
                    for column in ("iden1", "iden2")
                    for value in component[column]
                }
                kinds = {
                    restraint_lookup[identifier]["restraint_type"]
                    for identifier in identifiers
                    if identifier in restraint_lookup
                }
                curve_type = next(iter(kinds)) if len(kinds) == 1 else "other_curve"
                reason = {
                    "bond": "BOND RESTRAINT in geo",
                    "angle": "ANGLE RESTRAINT in geo",
                }.get(curve_type, "no unique BOND/ANGLE RESTRAINT in geo")
            out.loc[indices, "curve_type"] = curve_type
            out.loc[indices, "classification_reason"] = reason
    out = out.drop(columns=["_comment_bucket"])
    return out


def _component_row_indices(rows: pd.DataFrame) -> list[list[int]]:
    """Return connected row components for identifier-pair expressions."""
    parent: dict[str, str] = {}

    def find(value: str) -> str:
        parent.setdefault(value, value)
        if parent[value] != value:
            parent[value] = find(parent[value])
        return parent[value]

    def union(left: str, right: str) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    for _, row in rows.iterrows():
        union(str(row["iden1"]).strip(), str(row["iden2"]).strip())

    grouped: dict[str, list[int]] = {}
    for index, row in rows.iterrows():
        root = find(str(row["iden1"]).strip())
        grouped.setdefault(root, []).append(int(index))
    return list(grouped.values())


def _curve_table_from_classified_rows(
    rows: pd.DataFrame,
    geometry_summary: EnergyMinimizationSummaryData,
    geometry_restraints: pd.DataFrame | None = None,
    geometry_cells: pd.DataFrame | None = None,
    *,
    curve_type: str,
) -> pd.DataFrame:
    """Build one row per curve point while allowing either operand to repeat."""
    if rows.empty or "curve_type" not in rows.columns:
        return pd.DataFrame(columns=_CURVE_COLUMNS)
    selected = rows.loc[rows["curve_type"] == curve_type].copy()
    if selected.empty:
        return pd.DataFrame(columns=_CURVE_COLUMNS)

    geometry_df = _geometry_summary_frame(geometry_summary)
    geo_map = {
        str(row["identifier"]).strip(): (row.get("V", pd.NA), row.get("E", pd.NA))
        for _, row in geometry_df.drop_duplicates("identifier", keep="first").iterrows()
    }
    restraint_lookup = _geo_restraint_lookup(
        geometry_restraints if geometry_restraints is not None else pd.DataFrame()
    )
    def scan_metadata(identifier: str) -> dict[str, Any]:
        record = restraint_lookup.get(identifier)
        if record is not None:
            kind = str(record["restraint_type"])
            return {
                "scan_type": kind,
                "scan_coordinate": record["coordinate"],
                "scan_unit": "angstrom" if kind == "bond" else "degree",
                "coordinate_source": "geo restraint",
                "geo_restraint_line_number": record["restraint_line_number"],
            }
        return {
            "scan_type": "other" if curve_type == "other_curve" else curve_type,
            "scan_coordinate": _identifier_scan_coordinate(identifier),
            "scan_unit": {
                "bond": "angstrom",
                "angle": "degree",
            }.get(curve_type, ""),
            "coordinate_source": "identifier suffix",
            "geo_restraint_line_number": pd.NA,
        }
    output_rows: list[dict[str, Any]] = []
    selected["_comment_bucket"] = selected["group_comment"].str.strip()
    selected.loc[selected["_comment_bucket"] == "", "_comment_bucket"] = "<uncommented>"

    for _, comment_rows in selected.groupby("_comment_bucket", sort=False, dropna=False):
        for indices in _component_row_indices(comment_rows):
            component = comment_rows.loc[indices].copy()
            occurrence_counts: dict[str, int] = {}
            for _, row in component.iterrows():
                for identifier in {str(row["iden1"]).strip(), str(row["iden2"]).strip()}:
                    occurrence_counts[identifier] = occurrence_counts.get(identifier, 0) + 1
            if not occurrence_counts:
                continue
            base_iden, repeat_count = sorted(
                occurrence_counts.items(), key=lambda item: (-item[1], item[0])
            )[0]
            if repeat_count < 2:
                continue

            seen_other: set[str] = set()
            for _, row in component.iterrows():
                iden1, iden2 = str(row["iden1"]).strip(), str(row["iden2"]).strip()
                if iden1 == base_iden:
                    other_iden = iden2
                elif iden2 == base_iden:
                    other_iden = iden1
                else:
                    continue
                if other_iden in seen_other:
                    continue
                seen_other.add(other_iden)
                volume, energy = geo_map.get(other_iden, (pd.NA, pd.NA))
                scan = scan_metadata(other_iden)
                strain = _elastic_strain_metadata(base_iden, other_iden, geometry_cells)
                output_rows.append(
                    {
                        "curve_type": curve_type,
                        "base_iden": base_iden,
                        "other_iden": other_iden,
                        "iden1": iden1,
                        "iden2": iden2,
                        "V_other_iden": volume,
                        "E_other_iden": energy,
                        **strain,
                        **scan,
                        "ffield_value": row.get("ffield_value", pd.NA),
                        "qm_value": row.get("qm_value", pd.NA),
                        "group_comment": row.get("group_comment", ""),
                        "inline_comment": row.get("inline_comment", ""),
                        "classification_reason": row.get("classification_reason", ""),
                        "report_line_number": row.get("report_line_number", pd.NA),
                        "trainset_line_number": row.get("trainset_line_number", pd.NA),
                    }
                )

            if base_iden not in seen_other:
                volume, energy = geo_map.get(base_iden, (pd.NA, pd.NA))
                scan = scan_metadata(base_iden)
                strain = _elastic_strain_metadata(base_iden, base_iden, geometry_cells)
                first = component.iloc[0]
                output_rows.append(
                    {
                        "curve_type": curve_type,
                        "base_iden": base_iden,
                        "other_iden": base_iden,
                        "iden1": base_iden,
                        "iden2": base_iden,
                        "V_other_iden": volume,
                        "E_other_iden": energy,
                        **strain,
                        **scan,
                        "ffield_value": 0.0,
                        "qm_value": 0.0,
                        "group_comment": first.get("group_comment", ""),
                        "inline_comment": "",
                        "classification_reason": first.get("classification_reason", ""),
                        "report_line_number": pd.NA,
                        "trainset_line_number": pd.NA,
                    }
                )

    if not output_rows:
        return pd.DataFrame(columns=_CURVE_COLUMNS)
    sort_coordinate = "V_other_iden" if curve_type == "eos" else "scan_coordinate"
    return pd.DataFrame(output_rows, columns=_CURVE_COLUMNS).sort_values(
        ["base_iden", sort_coordinate, "other_iden"],
        kind="stable",
        na_position="last",
    ).reset_index(drop=True)


def _force_field_optimization_curve_tables(
    data: ForceFieldOptimizationPlotBundleData,
) -> dict[str, pd.DataFrame]:
    """Return EOS plus geo-classified bond, angle, and fallback curve tables."""
    classified = _classify_curve_rows(
        _annotated_two_body_energy_terms(data), data.geometry_restraints
    )
    tables = {
        curve_type: _curve_table_from_classified_rows(
            classified,
            data.geometry_summary,
            data.geometry_restraints,
            getattr(data, "geometry_cells", pd.DataFrame()),
            curve_type=curve_type,
        )
        for curve_type in ("eos", "bond", "angle", "other_curve")
    }
    if classified.empty:
        legacy = classified.copy()
    else:
        legacy = classified.loc[
            classified["legacy_restraint"].astype(bool)
            & classified["curve_type"].ne("eos")
        ].copy()
    legacy["curve_type"] = "restraint"
    tables["restraint"] = _curve_table_from_classified_rows(
        legacy,
        data.geometry_summary,
        data.geometry_restraints,
        getattr(data, "geometry_cells", pd.DataFrame()),
        curve_type="restraint",
    )
    return tables


def _fit_vinet_bulk_modulus(
    *,
    volumes: np.ndarray,
    energies: np.ndarray,
    shift_min_to_zero: bool,
    flip_sign: bool,
) -> dict[str, float]:
    """Fit a Vinet EOS curve and return bulk-modulus-related fit parameters."""
    from scipy.optimize import curve_fit

    V = np.asarray(volumes, dtype=float)
    E_kcal = np.asarray(energies, dtype=float)

    order = np.argsort(V)
    V = V[order]
    E_kcal = E_kcal[order]

    if flip_sign:
        E_kcal = -E_kcal

    E = E_kcal * const("energy_kcalmol_to_eV")
    if shift_min_to_zero:
        E = E - np.nanmin(E)

    V0_guess = float(V[np.nanargmin(E)])
    E0_guess = float(np.nanmin(E))
    K0_guess = 0.5 / const("eV_per_A3_to_GPa")
    C_guess = 4.0
    p0 = [E0_guess, K0_guess, V0_guess, C_guess]
    bounds = (
        [-np.inf, 1e-12, 1e-9, 1e-6],
        [np.inf, 1e3, np.inf, 1e3],
    )

    popt, _ = curve_fit(
        vinet_energy_ev,
        V,
        E,
        p0=p0,
        bounds=bounds,
        maxfev=20000,
    )

    E0_fit, K0_fit, V0_fit, C_fit = popt
    K0_GPa = float(K0_fit * const("eV_per_A3_to_GPa"))
    return {
        "V0_A3": float(V0_fit),
        "K0_eV_A3": float(K0_fit),
        "K0_GPa": K0_GPa,
        "E0_eV": float(E0_fit),
        "C": float(C_fit),
    }


def _bulk_modulus_table_from_eos(
    eos_table: pd.DataFrame,
    *,
    base_iden: Optional[str] = None,
    shift_min_to_zero: bool = True,
    flip_sign: bool = False,
    min_points: int = 6,
) -> pd.DataFrame:
    """Compute per-base Vinet bulk-modulus fits from an EOS input table."""
    out_cols = ["base_iden", "n_points", "V0_A3", "K0_eV_A3", "K0_GPa", "E0_eV", "C", "success"]
    if eos_table.empty:
        return pd.DataFrame(columns=out_cols)

    work = eos_table.copy()
    if base_iden and str(base_iden).lower() != "all":
        work = work[work["base_iden"] == str(base_iden)].copy()
    if work.empty:
        return pd.DataFrame(columns=out_cols)

    rows: list[dict[str, object]] = []
    for b, grp in work.groupby("base_iden", dropna=False):
        g = grp.copy()
        g["V_other_iden"] = pd.to_numeric(g["V_other_iden"], errors="coerce")
        g["E_other_iden"] = pd.to_numeric(g["E_other_iden"], errors="coerce")
        g = g[np.isfinite(g["V_other_iden"]) & np.isfinite(g["E_other_iden"])].copy()
        if len(g) < int(min_points):
            continue
        try:
            fit = _fit_vinet_bulk_modulus(
                volumes=g["V_other_iden"].to_numpy(dtype=float),
                energies=g["E_other_iden"].to_numpy(dtype=float),
                shift_min_to_zero=bool(shift_min_to_zero),
                flip_sign=bool(flip_sign),
            )
        except Exception:
            continue
        rows.append(
            {
                "base_iden": str(b),
                "n_points": int(len(g)),
                "V0_A3": fit["V0_A3"],
                "K0_eV_A3": fit["K0_eV_A3"],
                "K0_GPa": fit["K0_GPa"],
                "E0_eV": fit["E0_eV"],
                "C": fit["C"],
                "success": True,
            }
        )
    if not rows:
        return pd.DataFrame(columns=out_cols)
    return pd.DataFrame(rows, columns=out_cols).sort_values("base_iden").reset_index(drop=True)


@dataclass
class FFieldOptimizationReportRequest(BaseRequest):
    """Request payload for optimization-report table extraction.

    This request configures the base optimization-report analyzer. The current
    task returns all parsed report rows and does not expose request-time
    filtering parameters.

    Fields
    -----
    None.

    Examples
    -----
    ```python
    request = ForceFieldOptimizationReportRequest()
    ```
    The request returns the full parsed report table.
    """


@dataclass
class FFieldOptimizationReportResult(BaseResult):
    """Result payload containing parsed optimization-report rows.

    The analyzer returns normalized report records and a derived
    ``qm_ff_difference`` metric to simplify review of model-vs-reference gaps.

    Fields
    -----
    request : ForceFieldOptimizationReportRequest
        Request object used to generate this result.
    table : pandas.DataFrame
        Table with columns ``lineno``, ``section``, ``title``,
        ``ffield_value``, ``qm_value``, ``weight``, ``error``,
        ``total_ff_error``, and ``qm_ff_difference``.

    Examples
    -----
    ```python
    row = {
        "lineno": 12,
        "section": "ENERGY",
        "ffield_value": 11.9,
        "qm_value": 12.4,
        "qm_ff_difference": 0.5,
    }
    ```
    ``qm_ff_difference`` is computed as ``qm_value - ffield_value``.
    """

    table: pd.DataFrame
    request: FFieldOptimizationReportRequest


@dataclass
class FFieldOptimizationReportEOSRequest(BaseRequest):
    """Request payload for EOS table extraction from report artifacts.

    This request optionally filters the derived base/other energy-volume table
    by one base identifier.

    Fields
    -----
    iden : Optional[str]
        Optional ``base_iden`` selector. Use ``None`` or ``"all"`` to keep all
        identifiers, or provide one identifier (for example ``"MgO"``).

    Examples
    -----
    ```python
    request = ForceFieldOptimizationReportEOSRequest(iden="bulk_0")
    ```
    The request returns EOS rows only for ``base_iden == "bulk_0"``.
    """
    iden: Optional[str] = dc_field(
        default=None,
        metadata={
            "label": "Identifier Filter",
            "help": (
                "Optional identifier filter for base_iden. "
                "Example: 'MgO'. Use 'all' or None to keep all identifiers."
            ),
        },
    )


@dataclass
class FFieldOptimizationReportEOSResult(BaseResult):
    """Result payload containing EOS-compatible energy-volume rows.

    The analyzer joins repeated report identifiers with geometry summary values
    to produce per-base equation-of-state inputs.

    Fields
    -----
    request : ForceFieldOptimizationReportEOSRequest
        Request object used to generate this result.
    table : pandas.DataFrame
        Table with columns ``base_iden``, ``other_iden``, ``V_other_iden``,
        ``E_other_iden``, ``ffield_value`` (ReaxFF prediction), and
        ``qm_value`` (QM/literature target).

    Examples
    -----
    ```python
    rows = [
        {"base_iden": "bulk_0", "other_iden": "bulk_1", "V_other_iden": 10.0, "E_other_iden": -120.0},
        {"base_iden": "bulk_0", "other_iden": "bulk_0", "V_other_iden": 9.0, "E_other_iden": -90.0},
    ]
    ```
    Each row is one point in a base-identifier EOS series.
    """

    table: pd.DataFrame
    request: FFieldOptimizationReportEOSRequest


@dataclass
class FFieldOptimizationReportRestraintRequest(BaseRequest):
    """Select all restraint curves or one repeated reference identifier."""

    iden: Optional[str] = None


@dataclass
class FFieldOptimizationReportRestraintResult(BaseResult):
    """Comment-annotated ReaxFF/QM restraint curve rows."""

    table: pd.DataFrame
    request: FFieldOptimizationReportRestraintRequest


@dataclass
class FFieldOptimizationReportBulkModulusRequest(BaseRequest):
    """Request payload for Vinet bulk-modulus fitting from EOS rows.

    This request controls optional base filtering and fitting options used to
    derive equilibrium and bulk-modulus parameters from energy-volume data.

    Fields
    -----
    iden : Optional[str]
        Optional ``base_iden`` selector. Use ``None`` or ``"all"`` to fit all
        eligible base identifier groups.
    shift_min_to_zero : bool
        Whether to shift each energy series by its minimum before fitting.
    flip_sign : bool
        Whether to multiply input energies by ``-1`` before fitting.
    min_points : int
        Minimum number of finite ``(V, E)`` points required per base series.

    Examples
    -----
    ```python
    request = ForceFieldOptimizationReportBulkModulusRequest(
        iden="bulk_0",
        shift_min_to_zero=True,
        flip_sign=False,
        min_points=6,
    )
    ```
    The request fits one selected base EOS series using standard defaults.
    """
    iden: Optional[str] = dc_field(
        default=None,
        metadata={
            "label": "Base Identifier Filter",
            "help": (
                "Optional base_iden filter. Example: 'bulk_0'. "
                "Use 'all' or None to evaluate all eligible base identifiers."
            ),
        },
    )
    shift_min_to_zero: bool = dc_field(
        default=True,
        metadata={
            "label": "Shift Min To Zero",
            "help": "If true, shift each fitted energy series so its minimum is zero before EOS fitting.",
            "choices": [True, False],
        },
    )
    flip_sign: bool = dc_field(
        default=False,
        metadata={
            "label": "Flip Sign",
            "help": "If true, multiply energies by -1 before fitting.",
            "choices": [True, False],
        },
    )
    min_points: int = dc_field(
        default=6,
        metadata={
            "label": "Minimum Points",
            "help": "Minimum number of finite (V_other_iden, E_other_iden) rows needed to fit one base_iden series.",
            "min": 3,
        },
    )


@dataclass
class FFieldOptimizationReportBulkModulusResult(BaseResult):
    """Result payload for EOS-derived bulk-modulus fitting.

    The analyzer returns one row per successfully fitted base identifier,
    including equilibrium volume and bulk modulus in multiple units.

    Fields
    -----
    request : ForceFieldOptimizationReportBulkModulusRequest
        Request object used to generate this result.
    table : pandas.DataFrame
        Table with columns ``base_iden``, ``n_points``, ``V0_A3``,
        ``K0_eV_A3``, ``K0_GPa``, ``E0_eV``, ``C``, and ``success``.

    Examples
    -----
    ```python
    row = {
        "base_iden": "bulk_0",
        "n_points": 8,
        "V0_A3": 11.2,
        "K0_GPa": 76.9,
        "success": True,
    }
    ```
    The row indicates a successful Vinet fit for one base EOS series.
    """

    table: pd.DataFrame
    request: FFieldOptimizationReportBulkModulusRequest


@register_task("force_field_optimization_report", label="Force Field Optimization Report")
class FFieldOptimizationReportTask(AnalysisTask):
    """Return the parsed optimization-report table with QM-FF differences."""

    required_data = ForceFieldOptimizationReportData

    @staticmethod
    def recommended_presentations(
        _result: FFieldOptimizationReportResult, payload: dict[str, Any]
    ) -> list[PresentationSpec]:
        """Recommend table and QM-FF difference plot views for report output.

        Returns a table view by default and adds a line/scatter-style plot when
        serialized rows expose ``lineno`` and ``qm_ff_difference`` columns.

        Works on
        Analyzer task output for ``force_field_optimization_report``.

        Parameters
        -----
        _result : FFieldOptimizationReportResult
            Typed analyzer result instance (unused by current selection logic).
        payload : dict[str, Any]
            Serialized analyzer payload expected to include a ``table`` list.

        Returns
        -----
        list[PresentationSpec]
            Recommended presentation specifications for UI rendering.

        Examples
        -----
        ```python
        specs = ForceFieldOptimizationReportTask.recommended_presentations(
            _result,
            {"table": [{"lineno": 12, "qm_ff_difference": 0.5}]},
        )
        ```
        The returned list includes a table and a ``qm_ff_difference`` plot.
        """
        rows = payload.get("table")
        if not isinstance(rows, list) or not rows:
            return [PresentationSpec(renderer="table", label="Table", view_type="table")]
        sample = rows[0] if isinstance(rows[0], dict) else {}
        if "lineno" not in sample or "qm_ff_difference" not in sample:
            return [PresentationSpec(renderer="table", label="Table", view_type="table")]
        return [
            PresentationSpec(renderer="table", label="Table", view_type="table"),
            PresentationSpec(
                renderer="single_plot",
                label="QM-FF Difference vs Line",
                mapping={"x_col": "lineno", "y_col": "qm_ff_difference", "group_by_col": ""},
                options={
                    "title": "QM-FF Difference vs Line",
                    "xlabel": "lineno",
                    "ylabel": "qm_ff_difference",
                    "legend": False,
                },
                view_type="plot2d",
            ),
        ]

    def run(
        self,
        data: ForceFieldOptimizationReportData,
        request: FFieldOptimizationReportRequest,
        reporter=None,
    ) -> FFieldOptimizationReportResult:
        """Run base optimization-report extraction and difference computation.

        Converts parsed report rows into a normalized table and augments each
        row with a QM-minus-force-field difference metric.

        Works on
        ``ForceFieldOptimizationReportData``.

        Parameters
        -----
        data : ForceFieldOptimizationReportData
            Parsed optimization-report data source.
        request : FFieldOptimizationReportRequest
            Request object for the analysis task.
        reporter : Any, optional
            Progress callback accepted by analyzer tasks; unused here.

        Returns
        -----
        FFieldOptimizationReportResult
            Result containing the normalized report table.

        Examples
        -----
        ```python
        result = ForceFieldOptimizationReportTask().run(
            data,
            ForceFieldOptimizationReportRequest(),
        )
        ```
        The result table includes ``qm_ff_difference`` for each report row.
        """
        table = _get_report_data(data)
        return FFieldOptimizationReportResult(table=table, request=request)


@register_task("force_field_optimization_report_eos", label="Force Field Optimization Report EOS")
class FFieldOptimizationReportEOSTask(AnalysisTask):
    """Return comment-aware ENERGY-vs-volume EOS data."""

    required_data = ForceFieldOptimizationPlotBundleData

    @staticmethod
    def recommended_presentations(
        _result: FFieldOptimizationReportEOSResult, payload: dict[str, Any]
    ) -> list[PresentationSpec]:
        """Recommend table and energy-vs-volume presentations for EOS output.

        Provides a table for all outputs and adds an EOS plot when
        ``V_other_iden`` and ``E_other_iden`` columns are available.

        Works on
        Analyzer task output for ``force_field_optimization_report_eos``.

        Parameters
        -----
        _result : FFieldOptimizationReportEOSResult
            Typed analyzer result instance (unused by current logic).
        payload : dict[str, Any]
            Serialized payload expected to include ``table`` rows.

        Returns
        -----
        list[PresentationSpec]
            Recommended specs for EOS table and plot rendering.

        Examples
        -----
        ```python
        specs = ForceFieldOptimizationReportEOSTask.recommended_presentations(
            _result,
            {"table": [{"base_iden": "bulk_0", "V_other_iden": 10.0, "E_other_iden": -120.0}]},
        )
        ```
        The output includes a table and an energy-vs-volume plot.
        """
        rows = payload.get("table")
        if not isinstance(rows, list) or not rows:
            return [PresentationSpec(renderer="table", label="Table", view_type="table")]
        sample = rows[0] if isinstance(rows[0], dict) else {}
        if "V_other_iden" not in sample or "E_other_iden" not in sample:
            return [PresentationSpec(renderer="table", label="Table", view_type="table")]
        group_col = "base_iden" if "base_iden" in sample else ""
        return [
            PresentationSpec(renderer="table", label="Table", view_type="table"),
            PresentationSpec(
                renderer="single_plot",
                label="Energy vs Volume",
                mapping={
                    "x_col": "V_other_iden",
                    "y_col": "E_other_iden",
                    "group_by_col": group_col,
                },
                options={
                    "title": "Energy vs Volume",
                    "xlabel": "V_other_iden",
                    "ylabel": "E_other_iden",
                    "legend": bool(group_col),
                },
                view_type="plot2d",
            ),
        ]

    def run(
        self,
        data: ForceFieldOptimizationPlotBundleData,
        request: FFieldOptimizationReportEOSRequest,
        reporter=None,
    ) -> FFieldOptimizationReportEOSResult:
        """Run EOS table extraction from report and geometry summary bundles.

        Builds the base/other energy-volume table and optionally filters it by
        one base identifier from the request.

        Works on
        ``ForceFieldOptimizationPlotBundleData``.

        Parameters
        -----
        data : ForceFieldOptimizationPlotBundleData
            Bundle with parsed report rows, geometry values, and training-set comments.
        request : FFieldOptimizationReportEOSRequest
            Optional identifier filter for EOS table scope.
        reporter : Any, optional
            Progress callback accepted by analyzer tasks; unused here.

        Returns
        -----
        FFieldOptimizationReportEOSResult
            Result containing EOS-compatible energy-volume rows.

        Examples
        -----
        ```python
        result = ForceFieldOptimizationReportEOSTask().run(
            data,
            ForceFieldOptimizationReportEOSRequest(iden="all"),
        )
        ```
        The output table includes all eligible base-identifier EOS rows.
        """
        table = _force_field_optimization_curve_tables(data)["eos"]
        if request.iden and str(request.iden).lower() != "all":
            table = table[table["base_iden"] == request.iden].reset_index(drop=True)
        return FFieldOptimizationReportEOSResult(table=table, request=request)


@register_task(
    "force_field_optimization_report_restraints",
    label="Force Field Optimization Report Restraints",
)
class FFieldOptimizationReportRestraintTask(AnalysisTask):
    """Return repeated two-structure ENERGY terms labeled as restraints."""

    required_data = ForceFieldOptimizationPlotBundleData

    def run(
        self,
        data: ForceFieldOptimizationPlotBundleData,
        request: FFieldOptimizationReportRestraintRequest,
        reporter=None,
    ) -> FFieldOptimizationReportRestraintResult:
        table = _force_field_optimization_curve_tables(data)["restraint"]
        if request.iden and str(request.iden).lower() != "all":
            table = table[table["base_iden"] == request.iden].reset_index(drop=True)
        return FFieldOptimizationReportRestraintResult(table=table, request=request)


@register_task("force_field_optimization_report_bulk_modulus", label="Force Field Optimization Report Bulk Modulus")
class FFieldOptimizationReportBulkModulusTask(AnalysisTask):
    """Return a Vinet bulk-modulus fit derived from report + geometry summary."""

    required_data = ForceFieldOptimizationReportEOSBundleData

    @staticmethod
    def recommended_presentations(
        _result: FFieldOptimizationReportBulkModulusResult, payload: dict[str, Any]
    ) -> list[PresentationSpec]:
        """Recommend table and bulk-modulus summary plot presentations.

        Returns a table view by default and adds a ``K0_GPa`` plot keyed by
        ``base_iden`` when those fields are present in serialized rows.

        Works on
        Analyzer task output for ``force_field_optimization_report_bulk_modulus``.

        Parameters
        -----
        _result : FFieldOptimizationReportBulkModulusResult
            Typed analyzer result instance (unused by current selection logic).
        payload : dict[str, Any]
            Serialized payload expected to include ``table`` rows.

        Returns
        -----
        list[PresentationSpec]
            Recommended presentation specifications for fit summaries.

        Examples
        -----
        ```python
        specs = ForceFieldOptimizationReportBulkModulusTask.recommended_presentations(
            _result,
            {"table": [{"base_iden": "bulk_0", "K0_GPa": 76.9}]},
        )
        ```
        The returned specs include a table and one bulk-modulus comparison plot.
        """
        rows = payload.get("table")
        if not isinstance(rows, list) or not rows:
            return [PresentationSpec(renderer="table", label="Table", view_type="table")]
        sample = rows[0] if isinstance(rows[0], dict) else {}
        if "base_iden" not in sample or "K0_GPa" not in sample:
            return [PresentationSpec(renderer="table", label="Table", view_type="table")]
        return [
            PresentationSpec(renderer="table", label="Table", view_type="table"),
            PresentationSpec(
                renderer="single_plot",
                label="Bulk Modulus vs Base Identifier",
                mapping={"x_col": "base_iden", "y_col": "K0_GPa", "group_by_col": ""},
                options={
                    "title": "Bulk Modulus vs Base Identifier",
                    "xlabel": "base_iden",
                    "ylabel": "K0_GPa",
                    "legend": False,
                },
                view_type="plot2d",
            ),
        ]

    def run(
        self,
        data: ForceFieldOptimizationReportEOSBundleData,
        request: FFieldOptimizationReportBulkModulusRequest,
        reporter=None,
    ) -> FFieldOptimizationReportBulkModulusResult:
        """Run bulk-modulus fitting on EOS rows derived from report artifacts.

        Derives EOS rows from report and geometry summary data, applies request
        fit options, and returns one fit summary row per eligible base series.

        Works on
        ``ForceFieldOptimizationReportEOSBundleData``.

        Parameters
        -----
        data : ForceFieldOptimizationReportEOSBundleData
            Bundle with report and geometry summary inputs for EOS derivation.
        request : FFieldOptimizationReportBulkModulusRequest
            Fit configuration and optional identifier filter.
        reporter : Any, optional
            Progress callback accepted by analyzer tasks; unused here.

        Returns
        -----
        FFieldOptimizationReportBulkModulusResult
            Result containing Vinet fit parameters per base identifier.

        Examples
        -----
        ```python
        result = ForceFieldOptimizationReportBulkModulusTask().run(
            data,
            ForceFieldOptimizationReportBulkModulusRequest(min_points=6),
        )
        ```
        The output table includes one row per successful bulk-modulus fit.
        """
        eos_table = _base_other_energy_volume_table(data.report, data.geometry_summary)
        table = _bulk_modulus_table_from_eos(
            eos_table,
            base_iden=request.iden,
            shift_min_to_zero=bool(request.shift_min_to_zero),
            flip_sign=bool(request.flip_sign),
            min_points=max(3, int(request.min_points)),
        )
        return FFieldOptimizationReportBulkModulusResult(table=table, request=request)


__all__ = [
    "FFieldOptimizationReportRequest",
    "FFieldOptimizationReportResult",
    "FFieldOptimizationReportTask",
    "FFieldOptimizationReportEOSRequest",
    "FFieldOptimizationReportEOSResult",
    "FFieldOptimizationReportEOSTask",
    "FFieldOptimizationReportRestraintRequest",
    "FFieldOptimizationReportRestraintResult",
    "FFieldOptimizationReportRestraintTask",
    "FFieldOptimizationReportBulkModulusRequest",
    "FFieldOptimizationReportBulkModulusResult",
    "FFieldOptimizationReportBulkModulusTask",
]
