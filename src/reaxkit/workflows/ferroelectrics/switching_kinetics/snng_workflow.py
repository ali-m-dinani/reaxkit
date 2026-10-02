"""Fit tabulated ferroelectric switching data with the SNNG model.

Accept switched fractions, polarization, or per-domain polarities and export
normalized data, fit parameters, metrics, sweep results, and PNG plots.
"""

from __future__ import annotations

import argparse

from .common import add_common_arguments, run_workflow

COMMAND = "fit-snng-switching"
ALL_COMMANDS = (COMMAND,)
ALL_LEGACY_COMMANDS = ("fit_snng_switching",)


def build_parser(parser: argparse.ArgumentParser, *, command: str) -> argparse.ArgumentParser:
    """Document SNNG fitting modes and add the shared switching arguments.

    Time zero is the explicit field onset, or each group's earliest sample
    when onset is omitted. Document --field-start-time 0 for input timestamps
    already relative to onset, so late first samples keep their elapsed time.
    """
    if command not in (*ALL_COMMANDS, *ALL_LEGACY_COMMANDS):
        raise KeyError(command)
    parser.set_defaults(command=COMMAND)
    parser.formatter_class = argparse.RawTextHelpFormatter
    parser.description = """Fit the site-saturated nucleation with nucleation and growth (SNNG) model to a CSV or Excel table.

Estimate nucleation and growth kinetics from switched fractions or normalized polarization.
Optionally derive density or wall velocity from the fitted prefactor and supplied physical inputs.
Fit existing data only; write result tables and PNG plots in a unique run folder.

[Note]
Model time zero represents electric-field onset only when the time origin is set correctly.
With --field-start-time T, fitted/exported/plotted time is input time minus T;
rows before T are excluded, and the first retained sample may have positive time.
Without this flag, time zero is the earliest input sample in each group, which
must correspond to field onset for a physical switching-time interpretation.
If timestamps are already relative to field onset, use --field-start-time 0
to preserve that origin, including when the first recorded sample is later than zero.
Specify T in the input time column's units; --time-unit only labels plots.

Examples:
  1. Fit an existing switched-fraction curve:
     reaxkit fit-snng-switching --input switching.csv --data-kind fraction --value-column fraction

  2. Normalize polarization and fit each applied field separately:
     reaxkit fit-snng-switching --input switching.xlsx --sheet pulses --data-kind polarization --value-column Pz --group-column field

  3. Fit the fraction of domains whose polarity has reversed:
     reaxkit fit-snng-switching --input domains.csv --data-kind polarity-columns --polarity-columns d1 d2 d3

  4. Refit at fixed parameter values and export tables to a directory:
     reaxkit fit-snng-switching --input switching.csv --data-kind fraction --value-column fraction --sweep m=1,2,3 --output snng_fits

  5. Fit switching after the field is applied at time 125:
     reaxkit fit-snng-switching --input switching.csv --data-kind fraction --value-column fraction --field-start-time 125
"""
    return add_common_arguments(parser)


def run_main(command: str, args: argparse.Namespace) -> int:
    """Run SNNG fitting for a supported command name or legacy alias."""
    if command not in (*ALL_COMMANDS, *ALL_LEGACY_COMMANDS):
        raise KeyError(command)
    return run_workflow(args, ("snng",), command_name=COMMAND)


__all__ = ["ALL_COMMANDS", "ALL_LEGACY_COMMANDS", "COMMAND", "build_parser", "run_main"]
