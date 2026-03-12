"""Shared presentation dispatch for tabular workflow results."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

from reaxkit.cli.path import resolve_output_path
from reaxkit.core.storage_layout import normalize_storage_args
from reaxkit.presentation.persist import persist_analysis_result
from reaxkit.presentation.plot import plot as render_plot
from reaxkit.presentation.specs import ensure_presentation_spec, spec_to_plot_payload


PlotPayloadBuilder = Callable[[str, object, object], dict[str, object] | None]

_RAW_PLOT_DATA_KEYS = frozenset(
    {
        "x",
        "y",
        "z",
        "series",
        "subplots",
        "values",
        "bins",
        "vectors",
        "u",
        "v",
        "labels",
        "min_vals",
        "max_vals",
        "median_vals",
    }
)


def _looks_like_raw_plot_payload(payload: object) -> bool:
    """Return True when payload already targets a concrete plot renderer."""
    if not isinstance(payload, dict):
        return False
    return any(key in payload for key in _RAW_PLOT_DATA_KEYS)


def export_result_csv(result, path: str) -> None:
    """Export a result table to CSV."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    result.table.to_csv(out, index=False)


def print_result_table(result) -> None:
    """Print a result table to stdout."""
    print(result.table.to_string(index=False))


def _print_output_dirs(paths: list[Path]) -> None:
    seen: set[str] = set()
    ordered_dirs: list[str] = []
    for path in paths:
        directory = str(path.resolve())
        if directory in seen:
            continue
        seen.add(directory)
        ordered_dirs.append(directory)
    if not ordered_dirs:
        return
    print("Results saved in:")
    for directory in ordered_dirs:
        print(f"  {directory}")


def present_result(
    command: str,
    result,
    args,
    *,
    plot_payload_builder: PlotPayloadBuilder | None = None,
) -> None:
    """Dispatch result presentation from CLI-style arguments."""
    normalized = normalize_storage_args(vars(args), snapshot=False)
    for key, value in normalized.items():
        setattr(args, key, value)
    result_dirs: list[Path] = []
    export_csv = getattr(args, "export", None)
    analysis_dir = persist_analysis_result(command, result, args, write_csv=not bool(export_csv))
    result_dirs.append(analysis_dir)

    save = getattr(args, "save", None)
    plot_mode = getattr(args, "plot", None)
    show = bool(getattr(args, "show", False))
    wants_plot = bool(plot_mode or save or show)

    if export_csv:
        export_path = resolve_output_path(
            export_csv,
            command,
            run_id=getattr(args, "run_id", None),
            project_root=getattr(args, "project_root", "."),
            analysis_id=getattr(args, "analysis_id", None),
        )
        export_result_csv(result, str(export_path))
        result_dirs.append(export_path.parent)

    if wants_plot:
        if plot_payload_builder is None:
            print("Plotting is not available for this command.")
        else:
            payload = plot_payload_builder(command, result, args)
            if payload is None:
                print("No data available for plotting.")
            else:
                if not _looks_like_raw_plot_payload(payload):
                    # Typed presentation specs are adapted to renderer payloads here.
                    spec = ensure_presentation_spec(payload)
                    if spec is None and isinstance(payload, list):
                        for item in payload:
                            cand = ensure_presentation_spec(item)
                            if cand is not None and cand.renderer != "table":
                                spec = cand
                                break
                    if spec is not None:
                        payload = spec_to_plot_payload(spec, result)
                        if payload is None:
                            print("No plot-compatible presentation available for this result.")
                            return
                if save:
                    save_path = resolve_output_path(
                        save,
                        command,
                        run_id=getattr(args, "run_id", None),
                        project_root=getattr(args, "project_root", "."),
                        analysis_id=getattr(args, "analysis_id", None),
                    )
                    render_plot({**payload, "save": str(save_path)})
                    result_dirs.append(save_path.parent)
                if show or (plot_mode and not save):
                    render_plot(payload)

    if not (wants_plot or export_csv):
        print_result_table(result)
    _print_output_dirs(result_dirs)
