"""CSV and Excel workflows for ferroelectric switching kinetics.

Time zero denotes the explicit field onset when supplied, otherwise the
earliest input sample per group. Use field_start_time=0 for timestamps already
relative to field onset, especially when recording starts after onset.
"""

from .common import SwitchingWorkflowResult, fit_table, prepare_switching_fraction

__all__ = ["SwitchingWorkflowResult", "fit_table", "prepare_switching_fraction"]
