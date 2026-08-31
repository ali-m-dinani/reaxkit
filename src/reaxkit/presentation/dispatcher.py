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
from reaxkit.core.runtime.generator_runtime import maybe_copy_output_to_dot
from reaxkit.core.storage.storage_layout import ReaxkitStorageLayout, normalize_storage_args
from reaxkit.presentation.persist import append_artifacts_to_settings, persist_analysis_result
from reaxkit.presentation.plot import plot as render_plot
from reaxkit.presentation.report_registry import get_report_payload_builder
from reaxkit.presentation.reporting import normalize_report_formats, write_report_artifacts
from reaxkit.presentation.specs import ensure_presentation_spec, spec_to_plot_payload


PlotPayloadBuilder = Callable[
    [str, object, object],
    dict[str, object] | list[dict[str, object]] | None,
]
ReportPayloadBuilder = Callable[[str, object, object, Path], dict[str, object] | None]
ExportHandler = Callable[[str, object, object], list[Path]]

_PLOT_FILE_SUFFIXES = frozenset(
    {".png", ".jpg", ".jpeg", ".svg", ".pdf", ".tif", ".tiff", ".bmp"}
)

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
    csv_tables = getattr(result, "csv_tables", None)
    if isinstance(csv_tables, dict) and csv_tables:
        all_frames = csv_tables.get("all_frames")
        if not hasattr(all_frames, "to_csv"):
            all_frames = result.table
        all_frames.to_csv(out, index=False)
        for name, table in csv_tables.items():
            if name == "all_frames" or not hasattr(table, "to_csv"):
                continue
            safe = "".join(
                ch if (ch.isalnum() or ch in {"_", "-"}) else "_"
                for ch in str(name)
            ).strip("_") or "table"
            sibling = out.with_name(f"{out.stem}_{safe}{out.suffix or '.csv'}")
            table.to_csv(sibling, index=False)
        return
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
    export_handler: ExportHandler | None = None,
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
    export_handler : ExportHandler | None, optional
        Command-specific CSV export callback. The callback returns output
        directories to include in the saved-results summary.
    
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
    output_artifacts: list[Path] = []
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
        if export_handler is not None:
            result_dirs.extend(export_handler(command, result, args))
        else:
            export_path = resolve_output_path(
                export_csv,
                command,
                run_id=getattr(args, "run_id", None),
                project_root=getattr(args, "project_root", "."),
                analysis_id=getattr(args, "analysis_id", None),
            )
            export_result_csv(result, str(export_path))
            result_dirs.append(export_path.parent)
            output_artifacts.append(export_path)

    if wants_plot:
        if plot_payload_builder is None:
            print("Plotting is not available for this command.")
        else:
            payload = plot_payload_builder(command, result, args)
            if payload is None:
                print("No data available for plotting.")
            else:
                payload_batch = (
                    isinstance(payload, list)
                    and bool(payload)
                    and all(_looks_like_raw_plot_payload(item) for item in payload)
                )
                if not payload_batch and not _looks_like_raw_plot_payload(payload):
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
                plot_payloads = list(payload) if payload_batch else [payload]
                if save:
                    save_path = resolve_output_path(
                        save,
                        command,
                        run_id=getattr(args, "run_id", None),
                        project_root=getattr(args, "project_root", "."),
                        analysis_id=getattr(args, "analysis_id", None),
                    )
                    if payload_batch:
                        if save_path.suffix.lower() in _PLOT_FILE_SUFFIXES:
                            raise ValueError(
                                "Saving multiple plots requires a directory path, "
                                f"not a figure filename: {save_path}"
                            )
                        save_path.mkdir(parents=True, exist_ok=True)
                        for index, item in enumerate(plot_payloads, start=1):
                            filename = Path(
                                str(item.get("filename") or f"plot_{index}.png")
                            ).name
                            if Path(filename).suffix.lower() not in _PLOT_FILE_SUFFIXES:
                                filename = f"{filename}.png"
                            subdirectory = Path(str(item.get("subdirectory") or ""))
                            if subdirectory.is_absolute() or ".." in subdirectory.parts:
                                raise ValueError(
                                    "Plot payload subdirectory must be a safe relative path: "
                                    f"{subdirectory}"
                                )
                            item_path = save_path / subdirectory / filename
                            item_path.parent.mkdir(parents=True, exist_ok=True)
                            render_plot(
                                {
                                    **item,
                                    "save": str(item_path),
                                }
                            )
                        result_dirs.append(save_path)
                        output_artifacts.append(save_path)
                    else:
                        render_plot({**plot_payloads[0], "save": str(save_path)})
                        result_dirs.append(save_path.parent)
                        output_artifacts.append(save_path)
                if show or (plot_mode and not save):
                    for item in plot_payloads:
                        render_plot(item)

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

    if bool(getattr(args, "copy_to_dot", False)):
        for artifact in output_artifacts:
            copied = maybe_copy_output_to_dot(artifact, enabled=True)
            if copied is not None:
                result_dirs.append(copied if copied.is_dir() else copied.parent)

    if not (wants_plot or export_csv or wants_report):
        print_result_table(result)
    _print_output_dirs(result_dirs)
