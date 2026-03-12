"""Force-field optimization report analysis tasks."""

from __future__ import annotations

from dataclasses import dataclass, field as dc_field
import re
from typing import Any, Optional

import numpy as np
import pandas as pd

from reaxkit.analysis.base import AnalysisTask
from reaxkit.core.constants import const
from reaxkit.core.analysis_task_registry import register_task
from reaxkit.domain.base_request import BaseRequest
from reaxkit.domain.base_result import BaseResult
from reaxkit.domain.data_models import (
    ForceFieldOptimizationReportData,
    ForceFieldOptimizationReportEOSBundleData,
    GeometrySummaryData,
)
from reaxkit.presentation.specs import PresentationSpec
from reaxkit.utils.equation_of_states import vinet_energy_ev


def _report_frame(data: ForceFieldOptimizationReportData) -> pd.DataFrame:
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


def _geometry_summary_frame(data: GeometrySummaryData) -> pd.DataFrame:
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
    df = _report_frame(data)
    df["qm_ff_difference"] = df["qm_value"] - df["ffield_value"]
    return df.sort_values("lineno", ascending=True).reset_index(drop=True)


def _parse_two_body_energy_terms(data: ForceFieldOptimizationReportData) -> pd.DataFrame:
    df = _report_frame(data)
    if "section" not in df.columns or "title" not in df.columns:
        raise KeyError("Expected 'section' and 'title' columns in report DataFrame.")

    energy_df = df[df["section"].astype(str).str.upper() == "ENERGY"].copy()
    energy_df = energy_df[energy_df["title"].astype(str).str.count("/") == 2].copy()

    pattern = re.compile(
        r"^Energy\s+"
        r"(?P<sign1>[+-])(?P<iden1>[^/]+?)\s*/\s*(?P<n1>\d+(?:\.\d+)?)\s+"
        r"(?P<sign2>[+-])(?P<iden2>[^/]+?)\s*/\s*(?P<n2>\d+(?:\.\d+)?)",
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
            "n1": float(g["n1"]),
            "opt2": 1 if g["sign2"] == "+" else -1,
            "iden2": g["iden2"].strip(),
            "n2": float(g["n2"]),
        }

    parsed = energy_df["title"].astype(str).apply(_parse_title)
    parsed_df = pd.DataFrame(list(parsed))
    energy_df = pd.concat([energy_df.reset_index(drop=True), parsed_df], axis=1)
    return energy_df.dropna(subset=["iden1", "iden2"]).reset_index(drop=True)


def _energy_vs_volume(
    report: ForceFieldOptimizationReportData,
    geometry_summary: GeometrySummaryData,
) -> pd.DataFrame:
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
    geometry_summary: GeometrySummaryData,
) -> pd.DataFrame:
    """Build base/other EOS table from repeated base identifiers."""
    energy_df = _parse_two_body_energy_terms(report)
    if energy_df.empty:
        return pd.DataFrame(columns=["base_iden", "other_iden", "V_other_iden", "E_other_iden"])

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
        return pd.DataFrame(columns=["base_iden", "other_iden", "V_other_iden", "E_other_iden"])

    geometry_df = _geometry_summary_frame(geometry_summary)
    if geometry_df.empty or "identifier" not in geometry_df.columns:
        return pd.DataFrame(columns=["base_iden", "other_iden", "V_other_iden", "E_other_iden"])

    geometry_df = geometry_df[["identifier", "V", "E"]].drop_duplicates(subset=["identifier"], keep="first")
    geo_map: dict[str, tuple[Any, Any]] = {
        str(row["identifier"]).strip(): (row.get("V", pd.NA), row.get("E", pd.NA))
        for _, row in geometry_df.iterrows()
    }

    rows: list[dict[str, Any]] = []
    for base in repeated_bases:
        sub = energy_df[energy_df["iden1"].astype(str).str.strip() == base]
        other_order: list[str] = []
        for raw in sub["iden2"].astype(str).tolist():
            other = str(raw).strip()
            if other not in other_order:
                other_order.append(other)
        if base not in other_order:
            other_order.append(base)

        for other in other_order:
            vol, eng = geo_map.get(other, (pd.NA, pd.NA))
            rows.append(
                {
                    "base_iden": base,
                    "other_iden": other,
                    "V_other_iden": vol,
                    "E_other_iden": eng,
                }
            )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(["base_iden", "V_other_iden"], ascending=[True, True], na_position="last").reset_index(drop=True)


def _fit_vinet_bulk_modulus(
    *,
    volumes: np.ndarray,
    energies: np.ndarray,
    shift_min_to_zero: bool,
    flip_sign: bool,
) -> dict[str, float]:
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
class ForceFieldOptimizationReportRequest(BaseRequest):
    """Request for optimization-report rows."""


@dataclass
class ForceFieldOptimizationReportResult(BaseResult):
    """Optimization-report table result.

    Output structure:
    - request: ForceFieldOptimizationReportRequest used to generate this result.
    - table: pandas.DataFrame with columns:
      ['lineno', 'section', 'title', 'ffield_value', 'qm_value',
       'weight', 'error', 'total_ff_error', 'qm_ff_difference']
      - lineno: source row index from the optimization report
      - section: report section label (for example ENERGY, CHARGE)
      - title: descriptive row text
      - ffield_value/qm_value: compared model and reference values
      - weight: weighting factor in the objective
      - error: row-level weighted error contribution
      - total_ff_error: aggregate error value at parse time
      - qm_ff_difference: qm_value - ffield_value

    Example:
    If qm_value=12.4 and ffield_value=11.9, qm_ff_difference is 0.5.
    """

    table: pd.DataFrame
    request: ForceFieldOptimizationReportRequest


@dataclass
class ForceFieldOptimizationReportEOSRequest(BaseRequest):
    """Request for ENERGY-vs-volume data derived from report + geometry summary."""
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
class ForceFieldOptimizationReportEOSResult(BaseResult):
    """EOS (energy-vs-volume) analysis result.

    Output structure:
    - request: ForceFieldOptimizationReportEOSRequest used to generate this result.
    - table: pandas.DataFrame with columns:
      ['base_iden', 'other_iden', 'V_other_iden', 'E_other_iden']
      - base_iden: repeated base identifier used to form an EOS group
      - other_iden: related identifier connected to the base (including base itself)
      - V_other_iden: volume of other_iden from geometry summary
      - E_other_iden: energy of other_iden from geometry summary

    Example:
    If base_iden='bulk_0', rows can include:
    - (bulk_0, bulk_1, 10.0, -120.0)
    - (bulk_0, bulk_2, 13.0, -65.0)
    - (bulk_0, bulk_0, 9.0, -90.0)
    """

    table: pd.DataFrame
    request: ForceFieldOptimizationReportEOSRequest


@dataclass
class ForceFieldOptimizationReportBulkModulusRequest(BaseRequest):
    """Request for a Vinet bulk-modulus fit derived from report + geometry summary."""
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
class ForceFieldOptimizationReportBulkModulusResult(BaseResult):
    """Bulk-modulus fit result from EOS table rows.

    Output structure:
    - request: ForceFieldOptimizationReportBulkModulusRequest used to generate this result.
    - table: pandas.DataFrame with one row per fitted base_iden and columns:
      ['base_iden', 'n_points', 'V0_A3', 'K0_eV_A3', 'K0_GPa', 'E0_eV', 'C', 'success']
      - base_iden: EOS base identifier
      - n_points: number of finite points used in fit
      - V0_A3: fitted equilibrium volume
      - K0_eV_A3/K0_GPa: fitted bulk modulus in two units
      - E0_eV: fitted minimum energy
      - C: fitted Vinet shape parameter
      - success: True when fit succeeded

    Example:
    A row like (bulk_0, 8, 11.2, 0.48, 76.9, -2.14, 3.8, True)
    means bulk_0 was fitted with 8 points and produced K0=76.9 GPa.
    """

    table: pd.DataFrame
    request: ForceFieldOptimizationReportBulkModulusRequest


@register_task("force_field_optimization_report", label="Force Field Optimization Report")
class ForceFieldOptimizationReportTask(AnalysisTask):
    """Return the parsed optimization-report table with QM-FF differences."""

    required_data = ForceFieldOptimizationReportData

    @staticmethod
    def recommended_presentations(
        _result: ForceFieldOptimizationReportResult, payload: dict[str, Any]
    ) -> list[PresentationSpec]:
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
        request: ForceFieldOptimizationReportRequest,
        reporter=None,
    ) -> ForceFieldOptimizationReportResult:
        table = _get_report_data(data)
        return ForceFieldOptimizationReportResult(table=table, request=request)


@register_task("force_field_optimization_report_eos", label="Force Field Optimization Report EOS")
class ForceFieldOptimizationReportEOSTask(AnalysisTask):
    """Return ENERGY-vs-volume data derived from report + geometry summary."""

    required_data = ForceFieldOptimizationReportEOSBundleData

    @staticmethod
    def recommended_presentations(
        _result: ForceFieldOptimizationReportEOSResult, payload: dict[str, Any]
    ) -> list[PresentationSpec]:
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
        data: ForceFieldOptimizationReportEOSBundleData,
        request: ForceFieldOptimizationReportEOSRequest,
        reporter=None,
    ) -> ForceFieldOptimizationReportEOSResult:
        table = _base_other_energy_volume_table(data.report, data.geometry_summary)
        if request.iden and str(request.iden).lower() != "all":
            table = table[table["base_iden"] == request.iden].reset_index(drop=True)
        return ForceFieldOptimizationReportEOSResult(table=table, request=request)


@register_task("force_field_optimization_report_bulk_modulus", label="Force Field Optimization Report Bulk Modulus")
class ForceFieldOptimizationReportBulkModulusTask(AnalysisTask):
    """Return a Vinet bulk-modulus fit derived from report + geometry summary."""

    required_data = ForceFieldOptimizationReportEOSBundleData

    @staticmethod
    def recommended_presentations(
        _result: ForceFieldOptimizationReportBulkModulusResult, payload: dict[str, Any]
    ) -> list[PresentationSpec]:
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
        request: ForceFieldOptimizationReportBulkModulusRequest,
        reporter=None,
    ) -> ForceFieldOptimizationReportBulkModulusResult:
        eos_table = _base_other_energy_volume_table(data.report, data.geometry_summary)
        table = _bulk_modulus_table_from_eos(
            eos_table,
            base_iden=request.iden,
            shift_min_to_zero=bool(request.shift_min_to_zero),
            flip_sign=bool(request.flip_sign),
            min_points=max(3, int(request.min_points)),
        )
        return ForceFieldOptimizationReportBulkModulusResult(table=table, request=request)


__all__ = [
    "ForceFieldOptimizationReportRequest",
    "ForceFieldOptimizationReportResult",
    "ForceFieldOptimizationReportTask",
    "ForceFieldOptimizationReportEOSRequest",
    "ForceFieldOptimizationReportEOSResult",
    "ForceFieldOptimizationReportEOSTask",
    "ForceFieldOptimizationReportBulkModulusRequest",
    "ForceFieldOptimizationReportBulkModulusResult",
    "ForceFieldOptimizationReportBulkModulusTask",
]
