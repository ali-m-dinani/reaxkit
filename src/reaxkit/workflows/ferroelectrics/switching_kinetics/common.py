"""Shared tabular workflow for KAI, NLS, and SNNG model fitting.

Model time zero is the supplied field onset, or each group's earliest input
sample when onset is omitted. Fits, exported tables, and plots use this origin.
For timestamps already relative to field onset, field_start_time=0 preserves
elapsed times even if the first recorded sample occurs after zero.
"""

from __future__ import annotations

from reaxkit.presentation.workflow_artifacts import write_workflow_csv

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from reaxkit.analysis.ferroelectrics.switching_kinetics import (
    estimate_kai_t0,
    fit_switching_model,
)
from reaxkit.analysis.ferroelectrics.switching_kinetics.fitting import MODEL_PARAMETERS
from reaxkit.core.storage.storage_layout import default_project_root, generate_run_id

from .plotting import generate_switching_plots


@dataclass(frozen=True)
class SwitchingWorkflowResult:
    """Tables written by a switching-kinetics workflow."""

    normalized_data: pd.DataFrame
    fitted_curves: pd.DataFrame
    parameters: pd.DataFrame
    metrics: pd.DataFrame
    sweep_curves: pd.DataFrame
    sweep_parameters: pd.DataFrame
    sweep_metrics: pd.DataFrame


def _sheet_name(value: str) -> str | int:
    """Interpret a numeric sheet selector as a zero-based index."""
    stripped = str(value).strip()
    return int(stripped) if stripped.isdigit() else stripped


def read_table(path: Path, sheet: str | int = 0) -> pd.DataFrame:
    """Read a CSV or Excel switching table."""
    source = path.expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Switching input table not found: {source}")
    suffix = source.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(source)
    if suffix in {".xlsx", ".xlsm", ".xls"}:
        try:
            return pd.read_excel(source, sheet_name=sheet)
        except ImportError as exc:
            raise RuntimeError("Excel input requires openpyxl. Install reaxkit with Excel support.") from exc
    raise ValueError("Input must be a .csv, .xlsx, .xlsm, or .xls table.")


def _numeric_column(data: pd.DataFrame, column: str) -> np.ndarray:
    """Extract a finite numeric column or report missing or invalid values."""
    if column not in data.columns:
        available = ", ".join(map(str, data.columns))
        raise ValueError(f"Column {column!r} was not found. Available columns: {available}.")
    values = pd.to_numeric(data[column], errors="coerce").to_numpy(dtype=float)
    if not np.all(np.isfinite(values)):
        raise ValueError(f"Column {column!r} must contain only finite numeric values.")
    return values


def prepare_switching_fraction(
    data: pd.DataFrame,
    *,
    time_column: str,
    data_kind: str,
    value_column: str | None = None,
    polarity_columns: Sequence[str] | None = None,
    initial_polarization: float | None = None,
    final_polarization: float | None = None,
    tail_fraction: float = 0.1,
    clip_fraction: bool = False,
    field_start_time: float | None = None,
) -> pd.DataFrame:
    """Convert fraction, polarization, or per-domain polarity to ``time,fraction``.

    Polarization is normalized as ``(P-P_initial)/(P_final-P_initial)``.
    Rows are sorted by time and, by default, shifted to start at zero.
    This default identifies the earliest sample in each group with field onset;
    it does not detect when the field was applied. With field_start_time, model
    time zero instead means that explicit onset in input time units. For times
    already measured from onset, field_start_time=0 avoids shifting a late
    first sample back to zero. Output times retain the input time units.
    With field_start_time, retain only rows at or after onset and subtract
    onset without resetting the first retained sample to zero. The baseline
    is the last sample at or before onset, or the earliest sample if none
    precedes onset. Without an onset, use the earliest sample as the baseline.
    Explicit polarization endpoints override the baseline and the median of
    the final tail_fraction of retained rows. Per-domain polarity is converted
    to the fraction whose sign opposes its baseline sign; baseline zeros are rejected
    and later zeros do not count as flipped. Fractions outside [0, 1] beyond
    numerical tolerance are rejected unless clip_fraction is enabled.
    """
    time = _numeric_column(data, time_column)
    order = np.argsort(time, kind="stable")
    time = time[order]
    if time.size == 0:
        raise ValueError("Each curve must contain at least one time sample.")
    reference_index = 0
    retained = np.ones(time.size, dtype=bool)
    origin = time[0]
    if field_start_time is not None:
        if not np.isfinite(field_start_time):
            raise ValueError("--field-start-time must be finite.")
        origin = float(field_start_time)
        retained = time >= origin
        if not np.any(retained):
            raise ValueError("No samples remain at or after --field-start-time.")
        baseline_indices = np.flatnonzero(time <= origin)
        if baseline_indices.size:
            reference_index = int(baseline_indices[-1])
    time = time[retained] - origin
    if data_kind in {"fraction", "polarization"}:
        if not value_column:
            raise ValueError("--value-column is required for fraction or polarization data.")
        values = _numeric_column(data, value_column)[order]
        baseline = float(values[reference_index])
        values = values[retained]
        if data_kind == "fraction":
            fraction = values
        else:
            if not 0.0 < tail_fraction <= 1.0:
                raise ValueError("tail_fraction must satisfy 0 < tail_fraction <= 1.")
            initial = baseline if initial_polarization is None else float(initial_polarization)
            tail_count = max(1, int(np.ceil(values.size * tail_fraction)))
            final = (
                float(np.median(values[-tail_count:]))
                if final_polarization is None
                else float(final_polarization)
            )
            scale = final - initial
            if not np.isfinite(scale) or np.isclose(scale, 0.0):
                raise ValueError("Initial and final polarization must be finite and different.")
            fraction = (values - initial) / scale
    elif data_kind == "polarity-columns":
        columns = list(polarity_columns or [])
        if not columns:
            raise ValueError("--polarity-columns is required for polarity-columns data.")
        missing = [column for column in columns if column not in data.columns]
        if missing:
            raise ValueError(f"Polarity column(s) not found: {', '.join(missing)}.")
        matrix = data[columns].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)[order]
        if not np.all(np.isfinite(matrix)):
            raise ValueError("Polarity columns must contain only finite numeric values.")
        reference = np.sign(matrix[reference_index])
        matrix = matrix[retained]
        if np.any(reference == 0.0):
            zero_columns = [columns[index] for index in np.flatnonzero(reference == 0.0)]
            raise ValueError(
                "The reference polarity must be nonzero for every domain; zero in: "
                + ", ".join(zero_columns)
            )
        fraction = np.mean(np.sign(matrix) * reference[None, :] < 0.0, axis=1)
    else:
        raise ValueError("data_kind must be fraction, polarization, or polarity-columns.")

    tolerance = 1.0e-8
    if clip_fraction:
        fraction = np.clip(fraction, 0.0, 1.0)
    elif np.any((fraction < -tolerance) | (fraction > 1.0 + tolerance)):
        raise ValueError(
            "Normalized switched fraction falls outside [0, 1]. Check endpoints or use --clip-fraction."
        )
    fraction = np.clip(fraction, 0.0, 1.0)
    fraction = np.asarray(fraction, dtype=float)
    if np.any(np.diff(time) <= 0.0):
        raise ValueError("Each curve must have unique, strictly increasing time values.")
    return pd.DataFrame({"time": time, "fraction": fraction})


def _parse_assignment(
    token: str,
    *,
    bound: bool,
    allow_auto: bool = False,
) -> tuple[str | None, str, object]:
    """Parse a scoped parameter value, bound pair, or allowed auto estimate."""
    if "=" in token:
        key, raw_value = token.split("=", 1)
    elif allow_auto:
        key, raw_value = token, "auto"
    else:
        raise ValueError(f"Expected NAME=VALUE, received {token!r}.")
    model = None
    parameter = key.strip().lower()
    if "." in parameter:
        model, parameter = parameter.split(".", 1)
    if raw_value == "auto":
        value = "auto"
    elif bound:
        try:
            low_text, high_text = raw_value.split(":", 1)
            value: object = (float(low_text), float(high_text))
        except ValueError as exc:
            raise ValueError(f"Expected NAME=LOW:HIGH, received {token!r}.") from exc
    else:
        value = float(raw_value)
    return model, parameter, value


def _parse_sweep_values(specification: str) -> tuple[float, ...]:
    """Parse comma-separated values or an inclusive START:STOP[:STEP] range."""
    text = specification.strip()
    if "," in text:
        values = tuple(float(item.strip()) for item in text.split(",") if item.strip())
    elif ":" in text:
        parts = [float(item.strip()) for item in text.split(":")]
        if len(parts) not in (2, 3):
            raise ValueError("Sweep ranges must use START:STOP or START:STOP:STEP.")
        start, stop = parts[:2]
        step = parts[2] if len(parts) == 3 else 1.0
        if step == 0.0 or (stop - start) * step < 0.0:
            raise ValueError("Sweep step must move from START toward STOP.")
        count = int(np.floor((stop - start) / step + 1.0e-12)) + 1
        values_array = start + step * np.arange(count, dtype=float)
        if values_array.size == 0 or not np.isclose(values_array[-1], stop):
            values_array = np.append(values_array, stop)
        values = tuple(float(value) for value in values_array)
    else:
        values = (float(text),)
    if not values or any(not np.isfinite(value) or value <= 0.0 for value in values):
        raise ValueError("Sweep values must be finite and greater than zero.")
    if len(set(values)) != len(values):
        raise ValueError("Sweep values must be unique.")
    return values


def _parse_sweep(token: str) -> tuple[str | None, str, tuple[float, ...]]:
    """Split a sweep into optional model scope, parameter, and positive values."""
    try:
        key, specification = token.split("=", 1)
    except ValueError as exc:
        raise ValueError(f"Expected [MODEL.]NAME=VALUES, received {token!r}.") from exc
    scope = None
    parameter = key.strip().lower()
    if "." in parameter:
        scope, parameter = parameter.split(".", 1)
    return scope, parameter, _parse_sweep_values(specification)


def _model_options(
    args: argparse.Namespace,
    model: str,
    *,
    time: np.ndarray,
    fraction: np.ndarray,
) -> tuple[dict, dict, dict]:
    """Resolve fixed values, bounds, and guesses for one model and curve."""
    fixed: dict[str, float] = {}
    bounds: dict[str, tuple[float, float]] = {}
    initial: dict[str, float] = {}
    for token in args.fixed:
        scope, name, value = _parse_assignment(token, bound=False, allow_auto=True)
        if scope in (None, model) and name in MODEL_PARAMETERS[model]:
            if value == "auto":
                if model != "kai" or name != "t0":
                    raise ValueError(
                        f"Automatic --fixed is supported only for KAI t0, not {model}.{name}."
                    )
                fixed[name] = estimate_kai_t0(time, fraction)
            else:
                fixed[name] = float(value)
    for token in args.bounds:
        scope, name, value = _parse_assignment(token, bound=True)
        if scope in (None, model) and name in MODEL_PARAMETERS[model]:
            bounds[name] = value
    for token in args.initial:
        scope, name, value = _parse_assignment(token, bound=False)
        if scope in (None, model) and name in MODEL_PARAMETERS[model]:
            initial[name] = float(value)
    if model == "nls" and not args.fit_nls_amplitude and "amplitude" not in fixed:
        fixed["amplitude"] = 1.0
    return fixed, bounds, initial


def _validate_parameter_options(args: argparse.Namespace, models: Sequence[str]) -> None:
    """Reject parameter options and sweeps that do not apply to selected models."""
    selected = set(models)
    for tokens, is_bound, allow_auto in (
        (args.fixed, False, True),
        (args.bounds, True, False),
        (args.initial, False, False),
    ):
        for token in tokens:
            scope, name, value = _parse_assignment(
                token,
                bound=is_bound,
                allow_auto=allow_auto,
            )
            if scope is not None and scope not in MODEL_PARAMETERS:
                raise ValueError(f"Unknown model scope {scope!r} in {token!r}.")
            if scope is not None and scope not in selected:
                raise ValueError(f"Parameter option {token!r} does not apply to this workflow.")
            applicable = selected if scope is None else {scope}
            if not any(name in MODEL_PARAMETERS[model] for model in applicable):
                raise ValueError(f"Unknown parameter {name!r} in {token!r}.")
            if value == "auto" and not any(
                model == "kai" and name == "t0" for model in applicable
            ):
                raise ValueError(
                    "A value-free --fixed option is supported only for KAI t0."
                )
    for token in args.sweep:
        scope, name, _ = _parse_sweep(token)
        if scope is not None and scope not in MODEL_PARAMETERS:
            raise ValueError(f"Unknown model scope {scope!r} in {token!r}.")
        if scope is not None and scope not in selected:
            raise ValueError(f"Sweep {token!r} does not apply to this workflow.")
        applicable = selected if scope is None else {scope}
        if not any(name in MODEL_PARAMETERS[model] for model in applicable):
            raise ValueError(f"Unknown sweep parameter {name!r} in {token!r}.")


def _model_sweeps(args: argparse.Namespace, model: str) -> list[tuple[str, tuple[float, ...]]]:
    """Select a model's independent sweeps, rejecting duplicate parameters."""
    sweeps: list[tuple[str, tuple[float, ...]]] = []
    seen: set[str] = set()
    for token in args.sweep:
        scope, name, values = _parse_sweep(token)
        if scope in (None, model) and name in MODEL_PARAMETERS[model]:
            if name in seen:
                raise ValueError(f"Parameter {model}.{name} has more than one sweep specification.")
            seen.add(name)
            sweeps.append((name, values))
    return sweeps


def fit_table(
    data: pd.DataFrame,
    args: argparse.Namespace,
    models: Iterable[str],
) -> SwitchingWorkflowResult:
    """Normalize and fit requested models independently for each input group.

    Baseline fits and sweeps use elapsed time from args.field_start_time,
    or from each group's earliest sample when onset is omitted. The latter
    assumes that sample corresponds to field onset; onset is not inferred.
    Rank baseline fits by AICc, breaking ties by RMSE. Each parameter sweep
    fixes one parameter at a time and refits the others, preserving other
    fixed values. Return seven tables, including empty tables without sweeps.
    Supplied physical factors derive SNNG density or velocity after fitting.
    """
    models = tuple(models)
    _validate_parameter_options(args, models)
    for name in ("thickness", "wall_velocity", "nucleation_density"):
        value = getattr(args, name)
        if value is not None and (not np.isfinite(value) or value <= 0.0):
            raise ValueError(f"--{name.replace('_', '-')} must be finite and greater than zero.")
    if args.group_column:
        if args.group_column not in data.columns:
            raise ValueError(f"Group column {args.group_column!r} was not found.")
        grouped = data.groupby(args.group_column, sort=False, dropna=False)
    else:
        grouped = [("all", data)]

    normalized_tables: list[pd.DataFrame] = []
    curve_tables: list[pd.DataFrame] = []
    parameter_tables: list[pd.DataFrame] = []
    metric_rows: list[dict] = []
    sweep_curve_tables: list[pd.DataFrame] = []
    sweep_parameter_tables: list[pd.DataFrame] = []
    sweep_metric_rows: list[dict] = []
    for group, subset in grouped:
        normalized = prepare_switching_fraction(
            subset,
            time_column=args.time_column,
            data_kind=args.data_kind,
            value_column=args.value_column,
            polarity_columns=args.polarity_columns,
            initial_polarization=args.initial_polarization,
            final_polarization=args.final_polarization,
            tail_fraction=args.tail_fraction,
            clip_fraction=args.clip_fraction,
            field_start_time=getattr(args, "field_start_time", None),
        )
        normalized.insert(0, "group", group)
        normalized_tables.append(normalized)
        group_results = []
        for model in models:
            fit_time = normalized["time"].to_numpy()
            fit_fraction = normalized["fraction"].to_numpy()
            fixed, bounds, initial = _model_options(
                args,
                model,
                time=fit_time,
                fraction=fit_fraction,
            )
            result = fit_switching_model(
                fit_time,
                fit_fraction,
                model,
                fixed=fixed,
                bounds=bounds,
                initial=initial,
                n_starts=args.starts,
                random_state=args.seed,
                loss=args.loss,
            )
            group_results.append(result)
            for sweep_parameter, sweep_values in _model_sweeps(args, model):
                for sweep_value in sweep_values:
                    sweep_fixed = dict(fixed)
                    sweep_fixed[sweep_parameter] = sweep_value
                    sweep_result = fit_switching_model(
                        fit_time,
                        fit_fraction,
                        model,
                        fixed=sweep_fixed,
                        bounds=bounds,
                        initial=initial,
                        n_starts=args.starts,
                        random_state=args.seed,
                        loss=args.loss,
                    )
                    sweep_curve_tables.append(
                        pd.DataFrame(
                            {
                                "group": group,
                                "model": model,
                                "sweep_parameter": sweep_parameter,
                                "sweep_value": sweep_value,
                                "time": normalized["time"],
                                "observed_fraction": normalized["fraction"],
                                "predicted_fraction": sweep_result.predicted,
                                "residual": sweep_result.residuals,
                            }
                        )
                    )
                    sweep_parameter_table = sweep_result.parameter_table()
                    sweep_parameter_table.insert(0, "sweep_value", sweep_value)
                    sweep_parameter_table.insert(0, "sweep_parameter", sweep_parameter)
                    sweep_parameter_table.insert(0, "group", group)
                    sweep_parameter_tables.append(sweep_parameter_table)
                    sweep_metric_rows.append(
                        {
                            "group": group,
                            "model": model,
                            "sweep_parameter": sweep_parameter,
                            "sweep_value": sweep_value,
                            **sweep_result.metrics,
                            "success": sweep_result.success,
                            "message": sweep_result.message,
                        }
                    )
        group_results.sort(key=lambda item: (item.metrics["aicc"], item.metrics["rmse"]))
        for rank, result in enumerate(group_results, start=1):
            curve_tables.append(
                pd.DataFrame(
                    {
                        "group": group,
                        "model": result.model,
                        "time": normalized["time"],
                        "observed_fraction": normalized["fraction"],
                        "predicted_fraction": result.predicted,
                        "residual": result.residuals,
                    }
                )
            )
            parameter_table = result.parameter_table()
            parameter_table.insert(0, "group", group)
            if result.model == "snng":
                prefactor = result.parameters["prefactor"]
                if args.thickness and args.wall_velocity:
                    derived = prefactor / (
                        2.0 * np.pi * args.thickness * args.wall_velocity**2
                    )
                    parameter_table.loc[len(parameter_table)] = {
                        "group": group,
                        "model": "snng",
                        "parameter": "derived_nucleation_density",
                        "value": derived,
                        "standard_error": np.nan,
                        "fixed": False,
                    }
                if args.thickness and args.nucleation_density:
                    derived = np.sqrt(
                        prefactor / (2.0 * np.pi * args.thickness * args.nucleation_density)
                    )
                    parameter_table.loc[len(parameter_table)] = {
                        "group": group,
                        "model": "snng",
                        "parameter": "derived_wall_velocity",
                        "value": derived,
                        "standard_error": np.nan,
                        "fixed": False,
                    }
            parameter_tables.append(parameter_table)
            metric_rows.append(
                {
                    "group": group,
                    "rank_by_aicc": rank,
                    "model": result.model,
                    **result.metrics,
                    "success": result.success,
                    "message": result.message,
                    "n_starts": result.n_starts,
                }
            )
    return SwitchingWorkflowResult(
        normalized_data=pd.concat(normalized_tables, ignore_index=True),
        fitted_curves=pd.concat(curve_tables, ignore_index=True),
        parameters=pd.concat(parameter_tables, ignore_index=True),
        metrics=pd.DataFrame(metric_rows),
        sweep_curves=(
            pd.concat(sweep_curve_tables, ignore_index=True)
            if sweep_curve_tables
            else pd.DataFrame(
                columns=(
                    "group", "model", "sweep_parameter", "sweep_value", "time",
                    "observed_fraction", "predicted_fraction", "residual",
                )
            )
        ),
        sweep_parameters=(
            pd.concat(sweep_parameter_tables, ignore_index=True)
            if sweep_parameter_tables
            else pd.DataFrame(
                columns=(
                    "group", "sweep_parameter", "sweep_value", "model",
                    "parameter", "value", "standard_error", "fixed",
                )
            )
        ),
        sweep_metrics=pd.DataFrame(
            sweep_metric_rows,
            columns=(
                "group", "model", "sweep_parameter", "sweep_value", "sse", "rmse",
                "mae", "r_squared", "aic", "aicc", "bic", "n_observations",
                "n_free_parameters", "success", "message",
            ),
        ),
    )


def write_results(result: SwitchingWorkflowResult, output: Path) -> Path:
    """Write seven result tables to an Excel workbook or CSV directory.

    Export normalized data, fitted curves, parameters, metrics, and sweep
    curves, parameters, and metrics, including empty sweep tables.
    Exported time zero is the supplied field onset, or the earliest input
    sample per group if onset was omitted; original timestamps are not restored.
    """
    destination = output.expanduser().resolve()
    if destination.suffix.lower() in {".xlsx", ".xlsm"}:
        destination.parent.mkdir(parents=True, exist_ok=True)
        with pd.ExcelWriter(destination, engine="openpyxl") as writer:
            result.normalized_data.to_excel(writer, sheet_name="normalized_data", index=False)
            result.fitted_curves.to_excel(writer, sheet_name="fitted_curves", index=False)
            result.parameters.to_excel(writer, sheet_name="parameters", index=False)
            result.metrics.to_excel(writer, sheet_name="metrics", index=False)
            result.sweep_curves.to_excel(writer, sheet_name="sweep_curves", index=False)
            result.sweep_parameters.to_excel(writer, sheet_name="sweep_parameters", index=False)
            result.sweep_metrics.to_excel(writer, sheet_name="sweep_metrics", index=False)
        return destination
    if destination.suffix:
        raise ValueError("Output must be an .xlsx/.xlsm workbook or a directory for CSV files.")
    destination.mkdir(parents=True, exist_ok=True)
    write_workflow_csv(result.normalized_data, destination / "normalized_data.csv", index=False)
    write_workflow_csv(result.fitted_curves, destination / "fitted_curves.csv", index=False)
    write_workflow_csv(result.parameters, destination / "parameters.csv", index=False)
    write_workflow_csv(result.metrics, destination / "metrics.csv", index=False)
    write_workflow_csv(result.sweep_curves, destination / "sweep_curves.csv", index=False)
    write_workflow_csv(result.sweep_parameters, destination / "sweep_parameters.csv", index=False)
    write_workflow_csv(result.sweep_metrics, destination / "sweep_metrics.csv", index=False)
    return destination


def add_common_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Add documented input, normalization, fitting, sweep, and output flags.

    Model-specific options apply only to selected models. Physical SNNG inputs
    derive quantities after fitting. Return the supplied parser.
    The time origin defaults to each group's earliest sample; an explicit
    --field-start-time makes time zero correspond to the supplied field onset.
    """
    parser.add_argument("--input", type=Path, required=True, help="Read switching data from a CSV or Excel table. Example: --input switching.xlsx, read curves from that workbook.")
    parser.add_argument("--sheet", default="0", help="Select an Excel sheet by zero-based index or name; ignored for CSV. Example: --sheet pulses, read only the pulses sheet.")
    parser.add_argument("--time-column", default="time", help="Select numeric times; curves are sorted and shifted relative to field onset, or the first sample when onset is omitted, without unit conversion. Example: --time-column time_ns, fit using time_ns.")
    parser.add_argument("--field-start-time", type=float, default=None, help="Set field onset in input time units for every group; exclude earlier rows and subtract onset. Use the last sample at or before onset as baseline, or the earliest if none exists. Default: shift each curve by its first time. Example: --field-start-time 125, fit only times at least 125 using time minus 125.")
    parser.add_argument("--data-kind", choices=("fraction", "polarization", "polarity-columns"), required=True, help="Choose fraction, polarization, or per-domain polarity input. Example: --data-kind polarization, normalize the selected value column to switched fractions.")
    parser.add_argument("--value-column", default=None, help="Select the required value column for fraction or polarization input. Example: --value-column Pz, normalize Pz in polarization mode.")
    parser.add_argument("--polarity-columns", nargs="+", default=None, help="Select domain columns for polarity-columns input; unlisted domains are excluded and initial signs must be nonzero. Example: --polarity-columns d1 d2 d3, count sign reversals across these three domains.")
    parser.add_argument("--group-column", default=None, help="Fit every distinct column value as a separate curve; otherwise fit all rows together. Example: --group-column field, fit each applied field independently.")
    parser.add_argument("--initial-polarization", type=float, default=None, help="Override the polarization baseline selected at field onset, or the earliest value when onset is omitted. Example: --initial-polarization -100, map -100 to zero switched fraction.")
    parser.add_argument("--final-polarization", type=float, default=None, help="Set the polarization normalization endpoint instead of estimating its plateau. Example: --final-polarization 100, map 100 to unit switched fraction.")
    parser.add_argument("--tail-fraction", type=float, default=0.1, help="Set the fraction of final time-sorted rows used for the median polarization plateau, in (0, 1]; ignored with --final-polarization. Example: --tail-fraction 0.2, use the last 20 percent of each curve.")
    parser.add_argument("--clip-fraction", action="store_true", help="Clip normalized fractions to [0, 1] instead of rejecting values outside the numerical tolerance. Example: --clip-fraction, clamp overshoots before fitting.")
    parser.add_argument("--fixed", action="append", default=[], metavar="[MODEL.]NAME[=VALUE]", help="Fix [MODEL.]NAME=VALUE; repeat as needed. Only KAI t0 accepts an omitted value or auto for the 63.2 percent crossing estimate. Example: --fixed kai.t0, infer and fix KAI t0 when KAI is selected.")
    parser.add_argument("--bounds", action="append", default=[], metavar="[MODEL.]NAME=LOW:HIGH", help="Override positive bounds with [MODEL.]NAME=LOW:HIGH; repeat as needed. Example: --bounds n=0.5:8, constrain n in selected KAI or NLS fits.")
    parser.add_argument("--initial", action="append", default=[], metavar="[MODEL.]NAME=VALUE", help="Override a starting guess with [MODEL.]NAME=VALUE; repeat as needed. Example: --initial n=2, start n at 2 in selected KAI or NLS fits.")
    parser.add_argument("--sweep", action="append", default=[], metavar="[MODEL.]NAME=VALUES", help="Fix one parameter at each value and refit the others; accept lists or inclusive START:STOP[:STEP] ranges and repeat for independent sweeps. Example: --sweep snng.m=1:3:0.5, refit selected SNNG at five m values.")
    parser.add_argument("--fit-nls-amplitude", action="store_true", help="Fit NLS amplitude instead of fixing it at 1; explicit --fixed amplitude takes precedence. Ignored without NLS. Example: --fit-nls-amplitude, estimate amplitude from the curve.")
    parser.add_argument("--starts", type=int, default=12, help="Set optimization starts per fit; more starts increase search effort and runtime. Example: --starts 24, try 24 starting parameter sets.")
    parser.add_argument("--seed", type=int, default=0, help="Set the seed for reproducible multi-start initialization. Example: --seed 7, reuse randomized guesses for otherwise identical fits.")
    parser.add_argument("--loss", choices=("linear", "soft_l1", "huber", "cauchy", "arctan"), default="linear", help="Choose the least-squares optimization loss. Example: --loss soft_l1, reduce the influence of large residuals compared with linear loss.")
    parser.add_argument("--thickness", type=float, default=None, help="Supply positive SNNG thickness to derive density or velocity using consistent length and time units; does not constrain fitting. Example: --thickness 100e-9, use a 100 nm film when lengths are in metres.")
    parser.add_argument("--wall-velocity", type=float, default=None, help="Supply positive SNNG wall velocity and --thickness to derive nucleation density from the fitted prefactor. Example: --wall-velocity 2.5, use 2.5 length units per input time unit.")
    parser.add_argument("--nucleation-density", type=float, default=None, help="Supply positive SNNG nucleation density and --thickness to derive wall velocity from the fitted prefactor. Example: --nucleation-density 1e12, use 1e12 per cubic length unit.")
    parser.add_argument("--time-unit", default="input units", help="Label plot time axes without rescaling times or fitted parameters. Example: --time-unit ps, label plotted times as picoseconds.")
    parser.add_argument("--figure-dpi", type=int, default=180, help="Set positive PNG resolution in dots per inch; larger values produce more pixels. Example: --figure-dpi 300, export plots at 300 dpi.")
    parser.add_argument("--output", type=Path, default=None, help="Name an .xlsx/.xlsm workbook or suffix-free CSV directory inside the unique run folder; parent paths are ignored. Defaults to <input-stem>_switching_fits.xlsx. Example: --output switching_fits.xlsx, write that workbook and sibling plots directory.")
    return parser


def run_workflow(
    args: argparse.Namespace,
    models: Sequence[str],
    *,
    command_name: str,
) -> int:
    """Read, normalize, fit, rank, and export switching-model results.

    Write under the project storage root in other/<command>/<run-id>, using
    only the basename of args.output. Export PNG fit and residual plots plus
    sweep or model-comparison plots when applicable, then return zero.
    All fitted, exported, and plotted times share the explicit field onset
    as zero, or each group's earliest sample when onset is omitted. For input
    already relative to field onset, --field-start-time 0 preserves that origin.
    """
    if int(args.figure_dpi) <= 0:
        raise ValueError("--figure-dpi must be greater than zero.")
    source = Path(args.input)
    table = read_table(source, _sheet_name(args.sheet))
    result = fit_table(table, args, models)
    run_directory = (
        default_project_root() / "other" / command_name / generate_run_id()
    )
    output_name = (
        Path(args.output).name
        if args.output is not None
        else f"{source.stem}_switching_fits.xlsx"
    )
    destination = run_directory / output_name
    saved = write_results(result, Path(destination))
    plot_directory = (
        Path(saved) / "plots" if Path(saved).is_dir() else Path(saved).parent / "plots"
    )
    plots = generate_switching_plots(
        result,
        plot_directory,
        time_unit=str(args.time_unit),
        dpi=int(args.figure_dpi),
    )
    for group, group_metrics in result.metrics.groupby("group", sort=False):
        best = group_metrics.sort_values("rank_by_aicc").iloc[0]
        print(
            f"Group {group!r}: best AICc model is {str(best['model']).upper()} "
            f"(RMSE={float(best['rmse']):.6g}, R^2={float(best['r_squared']):.6g})."
        )
    print(f"Wrote normalized data, fitted curves, parameters, and metrics to {saved}")
    print(f"Wrote {len(plots)} switching-fit plot(s) to {plot_directory.resolve()}")
    return 0


__all__ = [
    "SwitchingWorkflowResult",
    "add_common_arguments",
    "fit_table",
    "_parse_sweep_values",
    "prepare_switching_fraction",
    "read_table",
    "run_workflow",
    "write_results",
]
