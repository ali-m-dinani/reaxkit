"""Geometry-target matching and grouped-bar payload helpers."""

from __future__ import annotations

import math
import re

import pandas as pd

from reaxkit.domain.data_models import ForceFieldOptimizationPlotBundleData


GEOMETRY_TARGET_COLUMNS = [
    "label",
    "iden",
    "geometry_type",
    "atoms",
    "ffield_value",
    "qm_value",
    "trainset_lit",
    "weight",
    "group_comment",
    "inline_comment",
    "report_line_number",
    "trainset_line_number",
]

_GEOMETRY_TITLE = re.compile(
    r"\b(?P<kind>bond(?:\s+distance)?|valence\s+angle|torsion\s+angle|angle)"
    r"\s*:\s*(?P<atoms>\d+(?:\s+\d+){1,3})\s*$",
    flags=re.IGNORECASE,
)
_CHARGE_TITLE = re.compile(
    r"^(?:(?P<identifier>.*?)\s+)?charge\s+atom\s*:",
    flags=re.IGNORECASE,
)


def _number_key(value: object) -> str:
    try:
        return f"{float(value):.12g}"
    except (TypeError, ValueError):
        return ""


def _canonical_atoms(atoms: tuple[int, ...]) -> tuple[int, ...]:
    """Treat an atom path and its reverse as the same geometry target."""
    reversed_atoms = tuple(reversed(atoms))
    return min(atoms, reversed_atoms)


def _geometry_type(atom_count: int, title_kind: object = "") -> str:
    kind = re.sub(r"\s+", "_", str(title_kind).strip().lower())
    if atom_count == 2 or kind.startswith("bond"):
        return "bond"
    if atom_count == 3 or kind in {"angle", "valence_angle"}:
        return "valence_angle"
    if atom_count == 4 or kind == "torsion_angle":
        return "torsion_angle"
    return "geometry"


def _report_geometry_table(
    data: ForceFieldOptimizationPlotBundleData,
) -> pd.DataFrame:
    """Recover identifiers inherited by fort.99 geometry continuation rows."""
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
        charge_match = _CHARGE_TITLE.match(title)
        if charge_match:
            explicit_identifier = str(
                charge_match.group("identifier") or ""
            ).strip()
            if explicit_identifier:
                current_identifier = explicit_identifier

        if str(row.get("section", "")).upper() != "GEOMETRY":
            continue
        match = _GEOMETRY_TITLE.search(title)
        if not match:
            continue
        explicit_identifier = title[: match.start()].strip()
        if explicit_identifier:
            current_identifier = explicit_identifier
        if not current_identifier:
            continue
        atoms = tuple(int(value) for value in match.group("atoms").split())
        if len(atoms) not in {2, 3, 4}:
            continue
        rows.append(
            {
                "iden": current_identifier,
                "geometry_type": _geometry_type(len(atoms), match.group("kind")),
                "atoms": " ".join(str(atom) for atom in atoms),
                "atom_key": _canonical_atoms(atoms),
                "ffield_value": row.get("ffield_value"),
                "qm_value": row.get("qm_value"),
                "weight": row.get("weight"),
                "report_line_number": row.get("report_line_number"),
            }
        )
    return pd.DataFrame(rows)


def _training_atoms(row: pd.Series) -> tuple[int, ...]:
    atoms: list[int] = []
    for column in ("at1", "at2", "at3", "at4"):
        value = row.get(column, pd.NA)
        if pd.isna(value):
            continue
        try:
            atoms.append(int(value))
        except (TypeError, ValueError):
            continue
    return tuple(atoms)


def build_geometry_target_table(
    data: ForceFieldOptimizationPlotBundleData,
) -> pd.DataFrame:
    """Match fort.99 geometry predictions to parsed training-set atom tuples."""
    report = _report_geometry_table(data)
    if report.empty:
        return pd.DataFrame(columns=GEOMETRY_TARGET_COLUMNS)

    report["weight_key"] = report["weight"].map(_number_key)
    report["match_occurrence"] = report.groupby(
        ["iden", "atom_key", "weight_key"], sort=False
    ).cumcount()

    train = data.training_set.geometry.copy()
    if train.empty:
        joined = report.copy()
        joined["trainset_lit"] = pd.NA
        joined["group_comment"] = ""
        joined["inline_comment"] = ""
        joined["trainset_line_number"] = pd.NA
        joined["training_atoms"] = ""
    else:
        train["iden"] = train["iden"].fillna("").astype(str).str.strip()
        train["atom_tuple"] = train.apply(_training_atoms, axis=1)
        train = train.loc[train["atom_tuple"].map(len).isin({2, 3, 4})].copy()
        train["atom_key"] = train["atom_tuple"].map(_canonical_atoms)
        train["training_atoms"] = train["atom_tuple"].map(
            lambda atoms: " ".join(str(atom) for atom in atoms)
        )
        train["weight_key"] = train["weight"].map(_number_key)
        train["match_occurrence"] = train.groupby(
            ["iden", "atom_key", "weight_key"], sort=False
        ).cumcount()
        annotations = train.loc[
            :,
            [
                "iden",
                "atom_key",
                "weight_key",
                "match_occurrence",
                "training_atoms",
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
            on=["iden", "atom_key", "weight_key", "match_occurrence"],
            how="left",
        )

    joined["atoms"] = joined["training_atoms"].fillna("").where(
        joined["training_atoms"].fillna("").astype(str).str.strip().ne(""),
        joined["atoms"],
    )
    joined["label"] = joined.apply(
        lambda row: f"{row['iden']} [atoms {row['atoms']}]", axis=1
    )
    for column in ("group_comment", "inline_comment"):
        joined[column] = joined[column].fillna("").astype(str)
    return joined.loc[:, GEOMETRY_TARGET_COLUMNS].reset_index(drop=True)


def geometry_target_plot_payloads(
    table: pd.DataFrame,
    *,
    entries_per_figure: int,
) -> list[dict[str, object]]:
    """Split geometry targets into consistently styled paired-bar figures."""
    limit = int(entries_per_figure)
    if limit < 1:
        raise ValueError("entries_per_figure must be at least 1.")
    if table.empty:
        return []

    work = table.copy()
    if "geometry_type" not in work.columns:
        work["geometry_type"] = "geometry"

    payloads: list[dict[str, object]] = []
    names = {
        "bond": ("Bond", "Bond length (angstrom)"),
        "valence_angle": ("Valence Angle", "Angle (degrees)"),
        "torsion_angle": ("Torsion Angle", "Angle (degrees)"),
        "geometry": ("Geometry", "Geometry target value"),
    }
    for geometry_type, typed_table in work.groupby(
        "geometry_type", sort=False, dropna=False
    ):
        type_key = str(geometry_type or "geometry")
        title_name, ylabel = names.get(
            type_key, (type_key.replace("_", " ").title(), "Geometry target value")
        )
        filename_type = re.sub(r"[^a-z0-9]+", "_", type_key.lower()).strip("_")
        filename_type = filename_type or "geometry"
        total = len(typed_table)
        figure_count = math.ceil(total / limit)
        for figure_index, start in enumerate(range(0, total, limit), start=1):
            chunk = typed_table.iloc[start : start + limit]
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
                    "xlabel": "Training-set geometry entry",
                    "ylabel": ylabel,
                    "title": (
                        f"{title_name} Targets ({start + 1}-{end} of {total})"
                    ),
                    "legend": True,
                    "grid": False,
                    "group_width": 0.48,
                    "minimum_category_slots": limit,
                    "label_rotation": 15,
                    "label_horizontal_alignment": "right",
                    "filename": (
                        f"geometry_{filename_type}_{figure_index:03d}_of_"
                        f"{figure_count:03d}.png"
                    ),
                    "figsize": (max(8.0, 2.2 * len(chunk)), 5.2),
                }
            )
    return payloads


__all__ = ["build_geometry_target_table", "geometry_target_plot_payloads"]
