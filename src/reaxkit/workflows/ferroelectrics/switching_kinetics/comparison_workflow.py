"""Fit and compare KAI, NLS, and SNNG models for tabulated switching data.

Normalize each curve, rank fits by corrected Akaike information criterion
(AICc), and export parameters, metrics, sweep results, and comparison plots.
"""

from __future__ import annotations

import argparse

from .common import add_common_arguments, run_workflow

COMMAND = "fit-switching-kinetics"
ALL_COMMANDS = (COMMAND,)
ALL_LEGACY_COMMANDS = ("fit_switching_kinetics",)


def build_parser(parser: argparse.ArgumentParser, *, command: str) -> argparse.ArgumentParser:
    """Document model comparison and add the shared switching arguments.

    Time zero is the explicit field onset, or each group's earliest sample
    when onset is omitted. Document --field-start-time 0 for input timestamps
    already relative to onset, so late first samples keep their elapsed time.
    """
    if command not in (*ALL_COMMANDS, *ALL_LEGACY_COMMANDS):
        raise KeyError(command)
    parser.set_defaults(command=COMMAND)
    parser.formatter_class = argparse.RawTextHelpFormatter
    parser.description = """Fit and rank KAI, NLS, and SNNG models for tabulated switching data.

Compare all three models on each curve using AICc, with RMSE as the tie-breaker.
Accept fractions, polarization, or domain polarities; optionally group curves or sweep parameters.
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
  1. Compare models for an existing switched-fraction curve:
     reaxkit fit-switching-kinetics --input switching.csv --data-kind fraction --value-column fraction

  2. Normalize polarization and compare models for each applied field:
     reaxkit fit-switching-kinetics --input switching.xlsx --sheet pulses --data-kind polarization --value-column Pz --group-column field

  3. Compare models for the fraction of reversed domains:
     reaxkit fit-switching-kinetics --input domains.csv --data-kind polarity-columns --polarity-columns d1 d2 d3

  4. Sweep KAI exponents while also fitting NLS and SNNG:
     reaxkit fit-switching-kinetics --input switching.csv --data-kind fraction --value-column fraction --sweep kai.n=1:6 --fit-nls-amplitude --output model_comparison

  5. Fit switching after the field is applied at time 125:
     reaxkit fit-switching-kinetics --input switching.csv --data-kind fraction --value-column fraction --field-start-time 125
"""
    return add_common_arguments(parser)


def run_main(command: str, args: argparse.Namespace) -> int:
    """Run all three model fits and rank them for each input group."""
    if command not in (*ALL_COMMANDS, *ALL_LEGACY_COMMANDS):
        raise KeyError(command)
    return run_workflow(
        args,
        ("kai", "nls", "snng"),
        command_name=COMMAND,
    )


__all__ = ["ALL_COMMANDS", "ALL_LEGACY_COMMANDS", "COMMAND", "build_parser", "run_main"]
