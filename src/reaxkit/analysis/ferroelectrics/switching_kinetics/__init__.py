"""KAI, NLS, and SNNG ferroelectric switching models and fitting."""

from .fitting import SwitchingFitResult, compare_switching_models, fit_switching_model
from .kai import KAI_CHARACTERISTIC_FRACTION, estimate_kai_t0, kai_fraction
from .nls import nls_fraction
from .snng import snng_fraction, snng_fraction_physical, snng_growth_integral

__all__ = [
    "SwitchingFitResult",
    "compare_switching_models",
    "fit_switching_model",
    "KAI_CHARACTERISTIC_FRACTION",
    "estimate_kai_t0",
    "kai_fraction",
    "nls_fraction",
    "snng_fraction",
    "snng_fraction_physical",
    "snng_growth_integral",
]
