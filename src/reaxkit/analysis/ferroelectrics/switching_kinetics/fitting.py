"""Constrained fitting and comparison of ferroelectric switching models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from .kai import kai_fraction
from .nls import nls_fraction
from .snng import snng_fraction

MODEL_FUNCTIONS: dict[str, Callable[..., np.ndarray]] = {
    "kai": kai_fraction,
    "nls": nls_fraction,
    "snng": snng_fraction,
}
MODEL_PARAMETERS: dict[str, tuple[str, ...]] = {
    "kai": ("t0", "n"),
    "nls": ("t1", "width", "n", "amplitude"),
    "snng": ("prefactor", "alpha", "m"),
}


@dataclass(frozen=True)
class SwitchingFitResult:
    """Best-fit parameters, predictions, uncertainty estimates, and metrics."""

    model: str
    parameters: dict[str, float]
    standard_errors: dict[str, float]
    predicted: np.ndarray
    residuals: np.ndarray
    metrics: dict[str, float]
    success: bool
    message: str
    n_starts: int

    def parameter_table(self) -> pd.DataFrame:
        """Return one row per parameter, including fixed parameters."""
        return pd.DataFrame(
            [
                {
                    "model": self.model,
                    "parameter": name,
                    "value": value,
                    "standard_error": self.standard_errors.get(name, np.nan),
                    "fixed": name not in self.standard_errors,
                }
                for name, value in self.parameters.items()
            ]
        )


def _validated_data(time, fraction) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(time, dtype=float)
    y = np.asarray(fraction, dtype=float)
    if x.ndim != 1 or y.ndim != 1 or x.size != y.size:
        raise ValueError("Time and switched fraction must be one-dimensional arrays of equal length.")
    if x.size < 5:
        raise ValueError("At least five samples are required for switching-model fitting.")
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise ValueError("Time and switched fraction must contain only finite values.")
    if np.any(x < 0.0) or np.any(np.diff(x) <= 0.0):
        raise ValueError("Time values must be non-negative and strictly increasing.")
    if np.any((y < 0.0) | (y > 1.0)):
        raise ValueError("Switched fractions must lie between zero and one.")
    return x, y


def _default_bounds(model: str, time: np.ndarray) -> dict[str, tuple[float, float]]:
    positive = time[time > 0.0]
    scale_min = float(np.min(positive)) if positive.size else 1.0e-12
    scale_max = max(float(np.max(time)), scale_min * 10.0)
    tiny = max(scale_min * 1.0e-4, np.finfo(float).tiny)
    if model == "kai":
        return {"t0": (tiny, scale_max * 100.0), "n": (0.05, 30.0)}
    if model == "nls":
        return {
            "t1": (tiny, scale_max * 100.0),
            "width": (1.0e-4, 10.0),
            "n": (0.05, 30.0),
            "amplitude": (0.05, 1.5),
        }
    if model == "snng":
        # Broad positive ranges are necessary because units of time are chosen
        # by the caller.  Optimization is performed in log-parameter space.
        return {
            "prefactor": (1.0e-18 / scale_max**2, 1.0e18 / scale_min**2),
            "alpha": (1.0e-18 / scale_max, 1.0e18 / scale_min),
            "m": (0.05, 30.0),
        }
    raise ValueError(f"Unknown switching model: {model!r}.")


def _initial_parameters(model: str, time: np.ndarray, fraction: np.ndarray) -> dict[str, float]:
    target_time = float(np.interp(0.5, np.maximum.accumulate(fraction), time))
    positive = time[time > 0.0]
    fallback = float(np.median(positive)) if positive.size else 1.0
    target_time = target_time if target_time > 0.0 else fallback
    if model == "kai":
        return {"t0": target_time / (-np.log(0.5)) ** 0.5, "n": 2.0}
    if model == "nls":
        return {"t1": target_time, "width": 0.5, "n": 2.0, "amplitude": 1.0}
    if model == "snng":
        alpha = target_time ** -1.0
        integral = max(float(target_time**2 * (1.0 - np.exp(-1.0))), 1.0e-30)
        return {"prefactor": -np.log(0.5) / integral, "alpha": alpha, "m": 1.0}
    raise ValueError(f"Unknown switching model: {model!r}.")


def fit_switching_model(
    time,
    fraction,
    model: str,
    *,
    fixed: Mapping[str, float] | None = None,
    bounds: Mapping[str, tuple[float, float]] | None = None,
    initial: Mapping[str, float] | None = None,
    n_starts: int = 12,
    random_state: int | None = 0,
    loss: str = "linear",
) -> SwitchingFitResult:
    """Fit one model with bounded, multi-start nonlinear least squares.

    All model parameters are positive, so optimization occurs in logarithmic
    parameter space.  ``fixed`` and ``bounds`` make it possible to reproduce
    literature assumptions such as NLS ``n=2`` or SNNG ``m=n_KAI-2``.
    """
    model_key = str(model).lower()
    if model_key not in MODEL_FUNCTIONS:
        raise ValueError(f"Unknown model {model!r}; choose kai, nls, or snng.")
    x, y = _validated_data(time, fraction)
    if n_starts < 1:
        raise ValueError("n_starts must be at least one.")

    names = MODEL_PARAMETERS[model_key]
    fixed_values = {str(k): float(v) for k, v in (fixed or {}).items()}
    unknown_fixed = set(fixed_values) - set(names)
    if unknown_fixed:
        raise ValueError(f"Unknown {model_key} fixed parameter(s): {sorted(unknown_fixed)}.")
    if any(value <= 0.0 or not np.isfinite(value) for value in fixed_values.values()):
        raise ValueError("Fixed model parameters must be finite and greater than zero.")

    resolved_bounds = _default_bounds(model_key, x)
    for name, pair in (bounds or {}).items():
        if name not in names:
            raise ValueError(f"Unknown {model_key} bounded parameter: {name!r}.")
        low, high = map(float, pair)
        if not (0.0 < low < high < np.inf):
            raise ValueError(f"Bounds for {name} must satisfy 0 < lower < upper.")
        resolved_bounds[name] = (low, high)

    guesses = _initial_parameters(model_key, x, y)
    guesses.update({str(k): float(v) for k, v in (initial or {}).items()})
    free_names = tuple(name for name in names if name not in fixed_values)
    function = MODEL_FUNCTIONS[model_key]

    def assemble(log_values: np.ndarray) -> dict[str, float]:
        params = dict(fixed_values)
        params.update({name: float(np.exp(value)) for name, value in zip(free_names, log_values)})
        return {name: params[name] for name in names}

    def residual(log_values: np.ndarray) -> np.ndarray:
        params = assemble(log_values)
        return np.asarray(function(x, **params), dtype=float) - y

    lower = np.log([resolved_bounds[name][0] for name in free_names])
    upper = np.log([resolved_bounds[name][1] for name in free_names])
    initial_log = np.log(
        [
            np.clip(guesses[name], resolved_bounds[name][0], resolved_bounds[name][1])
            for name in free_names
        ]
    )
    starts = [initial_log]
    rng = np.random.default_rng(random_state)
    for _ in range(n_starts - 1):
        starts.append(rng.uniform(lower, upper))

    best = None
    best_sse = np.inf
    if not free_names:
        predicted = np.asarray(function(x, **assemble(np.empty(0))), dtype=float)
        residuals = predicted - y
        best_sse = float(np.dot(residuals, residuals))
    else:
        for start in starts:
            try:
                candidate = least_squares(
                    residual,
                    start,
                    bounds=(lower, upper),
                    loss=loss,
                    max_nfev=5000,
                )
            except (FloatingPointError, ValueError, OverflowError):
                continue
            sse = float(np.dot(candidate.fun, candidate.fun))
            if np.isfinite(sse) and sse < best_sse:
                best, best_sse = candidate, sse
        if best is None:
            raise RuntimeError(f"All {model_key.upper()} optimization starts failed.")
        predicted = np.asarray(function(x, **assemble(best.x)), dtype=float)
        residuals = predicted - y

    parameters = assemble(best.x if best is not None else np.empty(0))
    sample_count = int(x.size)
    parameter_count = len(free_names)
    mse = best_sse / sample_count
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(residuals)))
    centered = y - float(np.mean(y))
    total_sum = float(np.dot(centered, centered))
    r_squared = 1.0 - best_sse / total_sum if total_sum > 0.0 else float("nan")
    safe_sse = max(best_sse, np.finfo(float).tiny)
    aic = sample_count * np.log(safe_sse / sample_count) + 2.0 * parameter_count
    denominator = sample_count - parameter_count - 1
    aicc = (
        aic + 2.0 * parameter_count * (parameter_count + 1) / denominator
        if denominator > 0
        else float("inf")
    )
    bic = sample_count * np.log(safe_sse / sample_count) + parameter_count * np.log(sample_count)

    standard_errors: dict[str, float] = {}
    if best is not None and sample_count > parameter_count:
        covariance_log = np.linalg.pinv(best.jac.T @ best.jac) * (
            best_sse / (sample_count - parameter_count)
        )
        errors_log = np.sqrt(np.maximum(np.diag(covariance_log), 0.0))
        for name, error_log in zip(free_names, errors_log):
            standard_errors[name] = parameters[name] * float(error_log)

    return SwitchingFitResult(
        model=model_key,
        parameters=parameters,
        standard_errors=standard_errors,
        predicted=predicted,
        residuals=residuals,
        metrics={
            "sse": best_sse,
            "rmse": rmse,
            "mae": mae,
            "r_squared": r_squared,
            "aic": float(aic),
            "aicc": float(aicc),
            "bic": float(bic),
            "n_observations": float(sample_count),
            "n_free_parameters": float(parameter_count),
        },
        success=bool(best.success) if best is not None else True,
        message=str(best.message) if best is not None else "All parameters were fixed.",
        n_starts=int(n_starts),
    )


def compare_switching_models(
    time,
    fraction,
    models: Sequence[str] = ("kai", "nls", "snng"),
    **fit_options,
) -> list[SwitchingFitResult]:
    """Fit multiple models and return them from lowest to highest AICc."""
    results = [fit_switching_model(time, fraction, model, **fit_options) for model in models]
    return sorted(results, key=lambda result: (result.metrics["aicc"], result.metrics["rmse"]))


__all__ = [
    "MODEL_FUNCTIONS",
    "MODEL_PARAMETERS",
    "SwitchingFitResult",
    "compare_switching_models",
    "fit_switching_model",
]
