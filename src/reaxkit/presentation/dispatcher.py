"""
Shared presentation dispatch for tabular workflow results.

**Usage context**

- Import these helpers from presentation workflows that produce tables, files, or plots.
- Reuse the public APIs here to keep output formatting and artifact behavior consistent.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

from reaxkit.cli.path import resolve_output_path
from reaxkit.core.storage.storage_layout import ReaxkitStorageLayout, normalize_storage_args
from reaxkit.presentation.persist import append_artifacts_to_settings, persist_analysis_result
from reaxkit.presentation.plot import plot as render_plot
from reaxkit.presentation.report_registry import get_report_payload_builder
from reaxkit.presentation.reporting import normalize_report_formats, write_report_artifacts
from reaxkit.presentation.specs import ensure_presentation_spec, spec_to_plot_payload


PlotPayloadBuilder = Callable[[str, object, object], dict[str, object] | None]
ReportPayloadBuilder = Callable[[str, object, object, Path], dict[str, object] | None]

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
        "segments",
        "points",
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
    """
    Export a result table to CSV.
    
    This function is part of the ReaxKit presentation API and performs the operation
    described by its name and arguments.
    
    Parameters
    -----
    result : Any
        Input parameter used by this function.
    path : str
        Input parameter used by this function.
    
    Returns
    -----
    None
        Value produced by this function call.
    
    Examples
    -----
    ```python
    from reaxkit.presentation.dispatcher import export_result_csv
    result = export_result_csv(...)
    print(type(result).__name__)
    ```
    Sample output:
    ```text
    str
    ```
    The output type reflects the return contract for this API call.
    """
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    result.table.to_csv(out, index=False)


def print_result_table(result) -> None:
    """
    Print a result table to stdout.
    
    This function is part of the ReaxKit presentation API and performs the operation
    described by its name and arguments.
    
    Parameters
    -----
    result : Any
        Input parameter used by this function.
    
    Returns
    -----
    None
        Value produced by this function call.
    
    Examples
    -----
    ```python
    from reaxkit.presentation.dispatcher import print_result_table
    result = print_result_table(...)
    print(type(result).__name__)
    ```
    Sample output:
    ```text
    str
    ```
    The output type reflects the return contract for this API call.
    """
    print(result.table.to_string(index=False))


def _print_output_dirs(paths: list[Path]) -> None:
    """
    Print output dirs.
    """
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
    report_payload_builder: ReportPayloadBuilder | None = None,
) -> None:
    """
    Dispatch result presentation from CLI-style arguments.
    
    This function is part of the ReaxKit presentation API and performs the operation
    described by its name and arguments.
    
    Parameters
    -----
    command : str
        Input parameter used by this function.
    result : Any
        Input parameter used by this function.
    args : Any
        Input parameter used by this function.
    plot_payload_builder : PlotPayloadBuilder | None, optional
        Input parameter used by this function.
    report_payload_builder : ReportPayloadBuilder | None, optional
        Input parameter used by this function.
    
    Returns
    -----
    None
        Value produced by this function call.
    
    Examples
    -----
    ```python
    from reaxkit.presentation.dispatcher import present_result
    result = present_result(...)
    print(type(result).__name__)
    ```
    Sample output:
    ```text
    str
    ```
    The output type reflects the return contract for this API call.
    """
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
    wants_report = bool(getattr(args, "report", False))
    report_mode = str(getattr(args, "report_format", "both") or "both")

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

    if wants_report:
        if report_payload_builder is None:
            report_payload_builder = get_report_payload_builder(str(command))
        if report_payload_builder is None:
            print("Report generation is not available for this command.")
        else:
            payload = report_payload_builder(command, result, args, analysis_dir)
            if payload is None:
                print("No data available for report generation.")
            else:
                analysis_id = (
                    getattr(args, "analysis_id", None)
                    or getattr(args, "run_id", None)
                    or getattr(args, "_analysis_id", None)
                    or "analysis"
                )
                layout = ReaxkitStorageLayout(project_root=Path(getattr(args, "project_root", ".")))
                report_dir = layout.reports_root / str(command) / str(analysis_id)
                report_files, report_notes = write_report_artifacts(
                    payload,
                    out_dir=report_dir,
                    stem=str(analysis_id),
                    formats=normalize_report_formats(report_mode),
                )
                if report_files:
                    append_artifacts_to_settings(analysis_dir, reports=report_files)
                    result_dirs.append(report_dir)
                for note in report_notes:
                    print(f"[report] {note}")

    if not (wants_plot or export_csv or wants_report):
        print_result_table(result)
    _print_output_dirs(result_dirs)
