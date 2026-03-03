"""Shared presentation dispatch for tabular workflow results."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

from reaxkit.presentation.plot import plot as render_plot


PlotPayloadBuilder = Callable[[str, object, object], dict[str, object] | None]


def export_result_csv(result, path: str) -> None:
    """Export a result table to CSV."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    result.table.to_csv(out, index=False)


def print_result_table(result) -> None:
    """Print a result table to stdout."""
    print(result.table.to_string(index=False))


def present_result(
    command: str,
    result,
    args,
    *,
    plot_payload_builder: PlotPayloadBuilder | None = None,
) -> None:
    """Dispatch result presentation from CLI-style arguments."""
    export_csv = getattr(args, "export", None)
    save = getattr(args, "save", None)
    plot_mode = getattr(args, "plot", None)
    show = bool(getattr(args, "show", False))
    wants_plot = bool(plot_mode or save or show)

    if export_csv:
        export_result_csv(result, export_csv)

    if wants_plot:
        if plot_payload_builder is None:
            print("Plotting is not available for this command.")
        else:
            payload = plot_payload_builder(command, result, args)
            if payload is None:
                print("No data available for plotting.")
            else:
                if save:
                    render_plot({**payload, "save": save})
                if show or (plot_mode and not save):
                    render_plot(payload)

    if not (wants_plot or export_csv):
        print_result_table(result)
