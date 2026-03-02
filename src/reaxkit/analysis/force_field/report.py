"""Force-field optimization report analysis tasks."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Literal, Optional

import numpy as np
import pandas as pd

from reaxkit.analysis.base import AnalysisTask
from reaxkit.core.constants import const
from reaxkit.core.task_registry import register_task
from reaxkit.domain.base_request import BaseRequest
from reaxkit.domain.base_result import BaseResult
from reaxkit.domain.data_models import ForceFieldOptimizationReportData, GeometrySummaryData
from reaxkit.utils.equation_of_states import vinet_energy_ev


def _fort99_frame(data: ForceFieldOptimizationReportData) -> pd.DataFrame:
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
    return pd.DataFrame(
        {
            "identifier": pd.Series(data.identifiers, dtype=object),
            "V": (
                pd.Series(data.volume, dtype=float)
                if data.volume is not None
                else pd.Series([pd.NA] * n_rows)
            ),
        }
    )


def _get_fort99_data(
    data: ForceFieldOptimizationReportData,
    *,
    sortby: str = "lineno",
    ascending: bool = True,
) -> pd.DataFrame:
    df = _fort99_frame(data)
    df["qm_ff_difference"] = df["qm_value"] - df["ffield_value"]
    if sortby not in df.columns:
        raise ValueError(f"Invalid sort key: '{sortby}'. Available columns: {list(df.columns)}")
    return df.sort_values(sortby, ascending=ascending).reset_index(drop=True)


def _parse_fort99_two_body_energy_terms(data: ForceFieldOptimizationReportData) -> pd.DataFrame:
    df = _fort99_frame(data)
    if "section" not in df.columns or "title" not in df.columns:
        raise KeyError("Expected 'section' and 'title' columns in fort.99 DataFrame.")

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


def _fort99_energy_vs_volume(
    report: ForceFieldOptimizationReportData,
    geometry_summary: GeometrySummaryData,
) -> pd.DataFrame:
    energy_df = _parse_fort99_two_body_energy_terms(report)
    if energy_df.empty:
        return pd.DataFrame(columns=["iden1", "iden2", "ffield_value", "qm_value", "V_iden2"])

    repeated = energy_df.groupby("iden1").filter(lambda g: len(g) > 1).copy()
    if repeated.empty:
        return pd.DataFrame(columns=["iden1", "iden2", "ffield_value", "qm_value", "V_iden2"])

    repeated = repeated[repeated["iden1"] != repeated["iden2"]]
    if repeated.empty:
        return pd.DataFrame(columns=["iden1", "iden2", "ffield_value", "qm_value", "V_iden2"])

    fort74_df = _geometry_summary_frame(geometry_summary)
    if fort74_df.empty or "identifier" not in fort74_df.columns or "V" not in fort74_df.columns:
        return pd.DataFrame(columns=["iden1", "iden2", "ffield_value", "qm_value", "V_iden2"])

    vol_df = fort74_df[["identifier", "V"]].drop_duplicates()
    merged = repeated.merge(vol_df, left_on="iden2", right_on="identifier", how="left")
    out = merged[["iden1", "iden2", "ffield_value", "qm_value", "V"]].rename(columns={"V": "V_iden2"})
    return out.sort_values(["iden1", "iden2"]).reset_index(drop=True)


def _get_fort99_bulk_modulus(
    report: ForceFieldOptimizationReportData,
    geometry_summary: GeometrySummaryData,
    *,
    iden: str,
    source: str = "ffield",
    shift_min_to_zero: bool = True,
    flip_sign: bool = False,
    dropna: bool = True,
) -> dict[str, object]:
    from scipy.optimize import curve_fit

    df = _fort99_energy_vs_volume(report, geometry_summary)
    if df.empty:
        raise ValueError("No ENERGY vs volume data found (fort99_energy_vs_volume returned empty).")

    group = df[df["iden1"] == iden].copy()
    if group.empty:
        raise ValueError(f"No rows found for iden1 == {iden!r}.")

    src = (source or "").strip().lower()
    if src in {"ffield", "ff", "forcefield", "force-field"}:
        e_col = "ffield_value"
        src_name = "ffield"
    elif src in {"qm", "dft", "reference"}:
        e_col = "qm_value"
        src_name = "qm"
    else:
        raise ValueError("source must be one of {'ffield','qm'}.")

    V = group["V_iden2"].to_numpy(dtype=float)
    E_kcal = group[e_col].to_numpy(dtype=float)

    if dropna:
        mask = np.isfinite(V) & np.isfinite(E_kcal)
        V = V[mask]
        E_kcal = E_kcal[mask]

    if len(V) < 6:
        raise ValueError(f"Need at least ~6 E(V) points for a stable EOS fit; got {len(V)} for iden={iden!r}.")

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
        "iden": iden,
        "source": src_name,
        "n_points": int(len(V)),
        "V0_A3": float(V0_fit),
        "K0_eV_A3": float(K0_fit),
        "K0_GPa": K0_GPa,
        "E0_eV": float(E0_fit),
        "C": float(C_fit),
        "success": True,
    }


@dataclass
class ForceFieldOptimizationReportRequest(BaseRequest):
    """Request for fort.99 report rows."""

    sortby: str = "lineno"
    ascending: bool = True


@dataclass
class ForceFieldOptimizationReportResult(BaseResult):
    """Result for fort.99 report rows."""

    table: pd.DataFrame


@dataclass
class ForceFieldOptimizationReportEOSRequest(BaseRequest):
    """Request for ENERGY-vs-volume data derived from fort.99 and fort.74."""

    geometry_summary: GeometrySummaryData
    iden: Optional[str] = None


@dataclass
class ForceFieldOptimizationReportEOSResult(BaseResult):
    """Result for ENERGY-vs-volume data derived from fort.99 and fort.74."""

    table: pd.DataFrame


@dataclass
class ForceFieldOptimizationReportBulkModulusRequest(BaseRequest):
    """Request for a Vinet bulk-modulus fit derived from fort.99 and fort.74."""

    geometry_summary: GeometrySummaryData
    iden: str
    source: Literal["ffield", "qm"] = "ffield"
    shift_min_to_zero: bool = True
    flip_sign: bool = False
    dropna: bool = True


@dataclass
class ForceFieldOptimizationReportBulkModulusResult(BaseResult):
    """Result for a Vinet bulk-modulus fit derived from fort.99 and fort.74."""

    values: dict[str, object]


@register_task("force_field_optimization_report")
class ForceFieldOptimizationReportTask(AnalysisTask):
    """Return the parsed fort.99 report table with QM-FF differences."""

    required_data = ForceFieldOptimizationReportData

    def run(
        self,
        data: ForceFieldOptimizationReportData,
        request: ForceFieldOptimizationReportRequest,
        reporter=None,
    ) -> ForceFieldOptimizationReportResult:
        table = _get_fort99_data(data, sortby=str(request.sortby), ascending=bool(request.ascending))
        return ForceFieldOptimizationReportResult(table=table)


@register_task("force_field_optimization_report_eos")
class ForceFieldOptimizationReportEOSTask(AnalysisTask):
    """Return ENERGY-vs-volume data derived from fort.99 and fort.74."""

    required_data = ForceFieldOptimizationReportData

    def run(
        self,
        data: ForceFieldOptimizationReportData,
        request: ForceFieldOptimizationReportEOSRequest,
        reporter=None,
    ) -> ForceFieldOptimizationReportEOSResult:
        table = _fort99_energy_vs_volume(data, request.geometry_summary)
        if request.iden and str(request.iden).lower() != "all":
            table = table[table["iden1"] == request.iden].reset_index(drop=True)
        return ForceFieldOptimizationReportEOSResult(table=table)


@register_task("force_field_optimization_report_bulk_modulus")
class ForceFieldOptimizationReportBulkModulusTask(AnalysisTask):
    """Return a Vinet bulk-modulus fit derived from fort.99 and fort.74."""

    required_data = ForceFieldOptimizationReportData

    def run(
        self,
        data: ForceFieldOptimizationReportData,
        request: ForceFieldOptimizationReportBulkModulusRequest,
        reporter=None,
    ) -> ForceFieldOptimizationReportBulkModulusResult:
        values = _get_fort99_bulk_modulus(
            data,
            request.geometry_summary,
            iden=str(request.iden),
            source=str(request.source),
            shift_min_to_zero=bool(request.shift_min_to_zero),
            flip_sign=bool(request.flip_sign),
            dropna=bool(request.dropna),
        )
        return ForceFieldOptimizationReportBulkModulusResult(values=values)


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
