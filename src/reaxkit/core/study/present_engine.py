"""
Presentation orchestration for study aggregate plots.

**Usage context**

- Import these helpers from ReaxKit core modules when implementing CLI and workflow logic.
- Reuse the public APIs here to keep behavior consistent across commands and engines.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable


def sort_plot_x(values: list[Any], *, sort_values_fn: Callable[[list[Any]], list[Any]]) -> list[Any]:
    """
    Sort plot x.
    
    This function is part of the ReaxKit core API and performs the operation described by its name and arguments.
    
    Parameters
    -----
    values : list[Any]
        Input parameter used by this function.
    sort_values_fn : Callable[[list[Any]], list[Any]]
        Input parameter used by this function.
    
    Returns
    -----
    list[Any]
        Value produced by this function call.
    
    Examples
    -----
    ```python
    from reaxkit.core.study.present_engine import sort_plot_x
    # Configure required arguments for your case.
    result = sort_plot_x(...)
    print(type(result).__name__)
    ```
    Sample output:
    ```text
    str
    ```
    The output type reflects the return contract for this API call.
    """
    return sort_values_fn(list(values))


def row_key_from_params(row: dict[str, str], param_cols: list[str]) -> str:
    """
    Row key from params.
    
    This function is part of the ReaxKit core API and performs the operation described by its name and arguments.
    
    Parameters
    -----
    row : dict[str, str]
        Input parameter used by this function.
    param_cols : list[str]
        Input parameter used by this function.
    
    Returns
    -----
    str
        Value produced by this function call.
    
    Examples
    -----
    ```python
    from reaxkit.core.study.present_engine import row_key_from_params
    # Configure required arguments for your case.
    result = row_key_from_params(...)
    print(type(result).__name__)
    ```
    Sample output:
    ```text
    str
    ```
    The output type reflects the return contract for this API call.
    """
    parts = [f"{p}={row.get(p)}" for p in param_cols]
    return ", ".join(parts)


def find_param_columns(rows: list[dict[str, str]]) -> list[str]:
    """
    Find param columns.
    
    This function is part of the ReaxKit core API and performs the operation described by its name and arguments.
    
    Parameters
    -----
    rows : list[dict[str, str]]
        Input parameter used by this function.
    
    Returns
    -----
    list[str]
        Value produced by this function call.
    
    Examples
    -----
    ```python
    from reaxkit.core.study.present_engine import find_param_columns
    # Configure required arguments for your case.
    result = find_param_columns(...)
    print(type(result).__name__)
    ```
    Sample output:
    ```text
    str
    ```
    The output type reflects the return contract for this API call.
    """
    if not rows:
        return []
    known = {"case_id", "replicate_id", "x_name", "x_value", "y_name", "y_value", "run_stage"}
    out: list[str] = []
    for key in rows[0].keys():
        if key in known:
            continue
        if key.startswith("y_"):
            continue
        out.append(key)
    return out


def make_errorbar_plots_for_case_aggregate(
    case_agg_dir: Path,
    *,
    aggregate_title: str,
    load_csv_rows_fn: Callable[[Path], list[dict[str, str]]],
    to_plot_value_fn: Callable[[Any], Any],
    safe_float_fn: Callable[[Any], float | None],
    slug_underscore_fn: Callable[[Any], str],
    render_plot_fn: Callable[[dict[str, Any]], Any],
) -> list[Path]:
    """
    Make errorbar plots for case aggregate.
    
    This function is part of the ReaxKit core API and performs the operation described by its name and arguments.
    
    Parameters
    -----
    case_agg_dir : Path
        Input parameter used by this function.
    aggregate_title : str
        Input parameter used by this function.
    load_csv_rows_fn : Callable[[Path], list[dict[str, str]]]
        Input parameter used by this function.
    to_plot_value_fn : Callable[[Any], Any]
        Input parameter used by this function.
    safe_float_fn : Callable[[Any], float | None]
        Input parameter used by this function.
    slug_underscore_fn : Callable[[Any], str]
        Input parameter used by this function.
    render_plot_fn : Callable[[dict[str, Any]], Any]
        Input parameter used by this function.
    
    Returns
    -----
    list[Path]
        Value produced by this function call.
    
    Examples
    -----
    ```python
    from reaxkit.core.study.present_engine import make_errorbar_plots_for_case_aggregate
    # Configure required arguments for your case.
    result = make_errorbar_plots_for_case_aggregate(...)
    print(type(result).__name__)
    ```
    Sample output:
    ```text
    str
    ```
    The output type reflects the return contract for this API call.
    """
    per_x_csv = case_agg_dir / "per_x_stats.csv"
    if not per_x_csv.exists():
        return []
    rows = load_csv_rows_fn(per_x_csv)
    if not rows:
        return []
    grouped: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        y_name = str(row.get("y_name") or "").strip()
        if not y_name:
            continue
        grouped.setdefault(y_name, []).append(row)

    outputs: list[Path] = []
    for y_name, items in grouped.items():
        items_sorted = sorted(items, key=lambda r: to_plot_value_fn(r.get("x_value")))
        x_vals: list[Any] = []
        y_vals: list[float] = []
        err_vals: list[float] = []
        for r in items_sorted:
            x_raw = r.get("x_value")
            x_num = safe_float_fn(x_raw)
            x_vals.append(x_num if x_num is not None else str(x_raw))
            y_mean = safe_float_fn(r.get("y_mean"))
            y_std = safe_float_fn(r.get("y_std"))
            y_sem = safe_float_fn(r.get("y_sem"))
            if y_mean is None:
                continue
            y_vals.append(y_mean)
            err_vals.append(y_sem if y_sem is not None else (y_std if y_std is not None else 0.0))
        if not y_vals:
            continue
        out_path = case_agg_dir / f"errorbar_{slug_underscore_fn(y_name)}.png"
        render_plot_fn(
            {
                "plot_type": "errorbar_plot",
                "x": x_vals[: len(y_vals)],
                "y": y_vals,
                "yerr": err_vals,
                "xlabel": "x",
                "ylabel": y_name,
                "title": f"{aggregate_title} - {y_name} (errorbar)",
                "legend": False,
                "save": str(out_path),
            }
        )
        outputs.append(out_path)
    return outputs


def make_boxplots_for_case_aggregate(
    case_agg_dir: Path,
    *,
    aggregate_title: str,
    load_csv_rows_fn: Callable[[Path], list[dict[str, str]]],
    safe_float_fn: Callable[[Any], float | None],
    slug_underscore_fn: Callable[[Any], str],
    sort_plot_x_fn: Callable[[list[Any]], list[Any]],
    render_plot_fn: Callable[[dict[str, Any]], Any],
) -> list[Path]:
    """
    Make boxplots for case aggregate.
    
    This function is part of the ReaxKit core API and performs the operation described by its name and arguments.
    
    Parameters
    -----
    case_agg_dir : Path
        Input parameter used by this function.
    aggregate_title : str
        Input parameter used by this function.
    load_csv_rows_fn : Callable[[Path], list[dict[str, str]]]
        Input parameter used by this function.
    safe_float_fn : Callable[[Any], float | None]
        Input parameter used by this function.
    slug_underscore_fn : Callable[[Any], str]
        Input parameter used by this function.
    sort_plot_x_fn : Callable[[list[Any]], list[Any]]
        Input parameter used by this function.
    render_plot_fn : Callable[[dict[str, Any]], Any]
        Input parameter used by this function.
    
    Returns
    -----
    list[Path]
        Value produced by this function call.
    
    Examples
    -----
    ```python
    from reaxkit.core.study.present_engine import make_boxplots_for_case_aggregate
    # Configure required arguments for your case.
    result = make_boxplots_for_case_aggregate(...)
    print(type(result).__name__)
    ```
    Sample output:
    ```text
    str
    ```
    The output type reflects the return contract for this API call.
    """
    raw_csv = case_agg_dir / "raw_replicates.csv"
    if not raw_csv.exists():
        return []
    rows = load_csv_rows_fn(raw_csv)
    if not rows:
        return []

    grouped_y: dict[str, dict[Any, list[float]]] = {}
    for row in rows:
        y_name = str(row.get("y_name") or "").strip()
        if not y_name:
            continue
        x_val = row.get("x_value")
        y_val = safe_float_fn(row.get("y_value"))
        if y_val is None:
            continue
        grouped_y.setdefault(y_name, {}).setdefault(x_val, []).append(y_val)

    outputs: list[Path] = []
    for y_name, xmap in grouped_y.items():
        x_keys = sort_plot_x_fn(list(xmap.keys()))
        data: list[list[float]] = []
        labels: list[str] = []
        for xv in x_keys:
            vals = xmap.get(xv)
            if not vals:
                continue
            data.append(vals)
            labels.append(str(xv))
        if not data:
            continue
        out_path = case_agg_dir / f"boxplot_{slug_underscore_fn(y_name)}.png"
        render_plot_fn(
            {
                "plot_type": "box_whisker_plot",
                "data": data,
                "labels": labels,
                "xlabel": "x",
                "ylabel": y_name,
                "title": f"{aggregate_title} - {y_name} (boxplot)",
                "save": str(out_path),
            }
        )
        outputs.append(out_path)
    return outputs


def make_all_cases_errorbar_plots(
    across_dir: Path,
    *,
    aggregate_title: str,
    load_csv_rows_fn: Callable[[Path], list[dict[str, str]]],
    find_param_columns_fn: Callable[[list[dict[str, str]]], list[str]],
    row_key_from_params_fn: Callable[[dict[str, str], list[str]], str],
    to_plot_value_fn: Callable[[Any], Any],
    safe_float_fn: Callable[[Any], float | None],
    stat_value_fn: Callable[[dict[str, Any], str], Any],
    slug_underscore_fn: Callable[[Any], str],
    render_plot_fn: Callable[[dict[str, Any]], Any],
) -> list[Path]:
    """
    Make all cases errorbar plots.
    
    This function is part of the ReaxKit core API and performs the operation described by its name and arguments.
    
    Parameters
    -----
    across_dir : Path
        Input parameter used by this function.
    aggregate_title : str
        Input parameter used by this function.
    load_csv_rows_fn : Callable[[Path], list[dict[str, str]]]
        Input parameter used by this function.
    find_param_columns_fn : Callable[[list[dict[str, str]]], list[str]]
        Input parameter used by this function.
    row_key_from_params_fn : Callable[[dict[str, str], list[str]], str]
        Input parameter used by this function.
    to_plot_value_fn : Callable[[Any], Any]
        Input parameter used by this function.
    safe_float_fn : Callable[[Any], float | None]
        Input parameter used by this function.
    stat_value_fn : Callable[[dict[str, Any], str], Any]
        Input parameter used by this function.
    slug_underscore_fn : Callable[[Any], str]
        Input parameter used by this function.
    render_plot_fn : Callable[[dict[str, Any]], Any]
        Input parameter used by this function.
    
    Returns
    -----
    list[Path]
        Value produced by this function call.
    
    Examples
    -----
    ```python
    from reaxkit.core.study.present_engine import make_all_cases_errorbar_plots
    # Configure required arguments for your case.
    result = make_all_cases_errorbar_plots(...)
    print(type(result).__name__)
    ```
    Sample output:
    ```text
    str
    ```
    The output type reflects the return contract for this API call.
    """
    src = across_dir / "across_cases_per_x_stats.csv"
    if not src.exists():
        return []
    rows = load_csv_rows_fn(src)
    if not rows:
        return []
    param_cols = find_param_columns_fn(rows)
    grouped_by_y: dict[str, dict[str, list[dict[str, str]]]] = {}
    for row in rows:
        y_name = str(row.get("y_name") or "").strip()
        if not y_name:
            continue
        curve_key = row_key_from_params_fn(row, param_cols)
        grouped_by_y.setdefault(y_name, {}).setdefault(curve_key, []).append(row)

    outputs: list[Path] = []
    for y_name, curve_map in grouped_by_y.items():
        series = []
        for curve_key, items in curve_map.items():
            items_sorted = sorted(items, key=lambda r: to_plot_value_fn(r.get("x_value")))
            xs: list[Any] = []
            ys: list[float] = []
            yerrs: list[float] = []
            for r in items_sorted:
                xv_raw = r.get("x_value")
                xv_num = safe_float_fn(xv_raw)
                x_val: Any = xv_num if xv_num is not None else str(xv_raw)
                ym = safe_float_fn(stat_value_fn(r, "mean"))
                ys_err = safe_float_fn(stat_value_fn(r, "sem"))
                ystd = safe_float_fn(stat_value_fn(r, "std"))
                if ym is None:
                    continue
                xs.append(x_val)
                ys.append(ym)
                yerrs.append(ys_err if ys_err is not None else (ystd if ystd is not None else 0.0))
            if ys:
                series.append({"x": xs, "y": ys, "yerr": yerrs, "label": curve_key})
        if not series:
            continue
        out_path = across_dir / f"errorbar_all_cases_{slug_underscore_fn(y_name)}.png"
        render_plot_fn(
            {
                "plot_type": "errorbar_plot",
                "series": series,
                "xlabel": "x",
                "ylabel": y_name,
                "title": f"{aggregate_title} - {y_name} (all cases errorbar)",
                "legend": True,
                "save": str(out_path),
            }
        )
        outputs.append(out_path)
    return outputs


def make_heatmap_from_rows(
    *,
    rows: list[dict[str, str]],
    x_param: str,
    y_param: str,
    value_col: str,
    out_path: Path,
    title: str,
    safe_float_fn: Callable[[Any], float | None],
    stat_value_fn: Callable[[dict[str, Any], str], Any],
    render_plot_fn: Callable[[dict[str, Any]], Any],
) -> Path | None:
    """
    Make heatmap from rows.
    
    This function is part of the ReaxKit core API and performs the operation described by its name and arguments.
    
    Parameters
    -----
    rows : list[dict[str, str]]
        Input parameter used by this function.
    x_param : str
        Input parameter used by this function.
    y_param : str
        Input parameter used by this function.
    value_col : str
        Input parameter used by this function.
    out_path : Path
        Input parameter used by this function.
    title : str
        Input parameter used by this function.
    safe_float_fn : Callable[[Any], float | None]
        Input parameter used by this function.
    stat_value_fn : Callable[[dict[str, Any], str], Any]
        Input parameter used by this function.
    render_plot_fn : Callable[[dict[str, Any]], Any]
        Input parameter used by this function.
    
    Returns
    -----
    Path | None
        Value produced by this function call.
    
    Examples
    -----
    ```python
    from reaxkit.core.study.present_engine import make_heatmap_from_rows
    # Configure required arguments for your case.
    result = make_heatmap_from_rows(...)
    print(type(result).__name__)
    ```
    Sample output:
    ```text
    str
    ```
    The output type reflects the return contract for this API call.
    """
    stat_key = value_col[2:] if value_col.startswith("y_") else value_col
    coords: list[list[float]] = []
    values: list[float] = []
    for row in rows:
        xv = safe_float_fn(row.get(x_param))
        yv = safe_float_fn(row.get(y_param))
        zv_raw = row.get(value_col)
        if zv_raw is None:
            zv_raw = stat_value_fn(row, stat_key)
        zv = safe_float_fn(zv_raw)
        if xv is None or yv is None or zv is None:
            continue
        coords.append([xv, yv, 0.0])
        values.append(zv)
    if not values:
        return None
    render_plot_fn(
        {
            "plot_type": "heatmap2d_from_3d",
            "coords": coords,
            "values": values,
            "xlabel": x_param,
            "ylabel": y_param,
            "title": title,
            "save": str(out_path),
        }
    )
    return out_path


def make_all_cases_heatmaps(
    across_dir: Path,
    *,
    aggregate_title: str,
    load_csv_rows_fn: Callable[[Path], list[dict[str, str]]],
    find_param_columns_fn: Callable[[list[dict[str, str]]], list[str]],
    safe_float_fn: Callable[[Any], float | None],
    slug_underscore_fn: Callable[[Any], str],
    make_heatmap_from_rows_fn: Callable[..., Path | None],
) -> list[Path]:
    """
    Make all cases heatmaps.
    
    This function is part of the ReaxKit core API and performs the operation described by its name and arguments.
    
    Parameters
    -----
    across_dir : Path
        Input parameter used by this function.
    aggregate_title : str
        Input parameter used by this function.
    load_csv_rows_fn : Callable[[Path], list[dict[str, str]]]
        Input parameter used by this function.
    find_param_columns_fn : Callable[[list[dict[str, str]]], list[str]]
        Input parameter used by this function.
    safe_float_fn : Callable[[Any], float | None]
        Input parameter used by this function.
    slug_underscore_fn : Callable[[Any], str]
        Input parameter used by this function.
    make_heatmap_from_rows_fn : Callable[..., Path | None]
        Input parameter used by this function.
    
    Returns
    -----
    list[Path]
        Value produced by this function call.
    
    Examples
    -----
    ```python
    from reaxkit.core.study.present_engine import make_all_cases_heatmaps
    # Configure required arguments for your case.
    result = make_all_cases_heatmaps(...)
    print(type(result).__name__)
    ```
    Sample output:
    ```text
    str
    ```
    The output type reflects the return contract for this API call.
    """
    outputs: list[Path] = []
    per_x_csv = across_dir / "across_cases_per_x_stats.csv"
    global_csv = across_dir / "across_cases_global_stats.csv"

    if per_x_csv.exists():
        rows = load_csv_rows_fn(per_x_csv)
        if rows:
            param_cols = find_param_columns_fn(rows)
            if len(param_cols) >= 2:
                x_param, y_param = param_cols[0], param_cols[1]
                by_y: dict[str, list[dict[str, str]]] = {}
                for row in rows:
                    y_name = str(row.get("y_name") or "").strip()
                    if y_name:
                        by_y.setdefault(y_name, []).append(row)
                for y_name, y_rows in by_y.items():
                    x_vals = [safe_float_fn(r.get("x_value")) for r in y_rows]
                    x_vals = [v for v in x_vals if v is not None]
                    if not x_vals:
                        continue
                    final_x = max(x_vals)
                    final_rows = [r for r in y_rows if safe_float_fn(r.get("x_value")) == final_x]
                    for metric in ("y_mean", "y_std", "y_sem"):
                        out_path = across_dir / f"heatmap_final_x_{slug_underscore_fn(y_name)}_{metric}.png"
                        made = make_heatmap_from_rows_fn(
                            rows=final_rows,
                            x_param=x_param,
                            y_param=y_param,
                            value_col=metric,
                            out_path=out_path,
                            title=f"{aggregate_title} - {y_name} final x {metric}",
                        )
                        if made is not None:
                            outputs.append(made)

    if global_csv.exists():
        rows = load_csv_rows_fn(global_csv)
        if rows:
            param_cols = find_param_columns_fn(rows)
            if len(param_cols) >= 2:
                x_param, y_param = param_cols[0], param_cols[1]
                by_y: dict[str, list[dict[str, str]]] = {}
                for row in rows:
                    y_name = str(row.get("y_name") or "").strip()
                    if y_name:
                        by_y.setdefault(y_name, []).append(row)
                for y_name, y_rows in by_y.items():
                    for metric in ("y_mean", "y_std", "y_sem"):
                        out_path = across_dir / f"heatmap_global_{slug_underscore_fn(y_name)}_{metric}.png"
                        made = make_heatmap_from_rows_fn(
                            rows=y_rows,
                            x_param=x_param,
                            y_param=y_param,
                            value_col=metric,
                            out_path=out_path,
                            title=f"{aggregate_title} - {y_name} global {metric}",
                        )
                        if made is not None:
                            outputs.append(made)
    return outputs


def make_all_cases_per_iter_boxplots(
    across_dir: Path,
    *,
    aggregate_title: str,
    load_csv_rows_fn: Callable[[Path], list[dict[str, str]]],
    find_param_columns_fn: Callable[[list[dict[str, str]]], list[str]],
    sort_plot_x_fn: Callable[[list[Any]], list[Any]],
    row_key_from_params_fn: Callable[[dict[str, str], list[str]], str],
    safe_float_fn: Callable[[Any], float | None],
    slug_underscore_fn: Callable[[Any], str],
    render_plot_fn: Callable[[dict[str, Any]], Any],
) -> list[Path]:
    """
    Make all cases per iter boxplots.
    
    This function is part of the ReaxKit core API and performs the operation described by its name and arguments.
    
    Parameters
    -----
    across_dir : Path
        Input parameter used by this function.
    aggregate_title : str
        Input parameter used by this function.
    load_csv_rows_fn : Callable[[Path], list[dict[str, str]]]
        Input parameter used by this function.
    find_param_columns_fn : Callable[[list[dict[str, str]]], list[str]]
        Input parameter used by this function.
    sort_plot_x_fn : Callable[[list[Any]], list[Any]]
        Input parameter used by this function.
    row_key_from_params_fn : Callable[[dict[str, str], list[str]], str]
        Input parameter used by this function.
    safe_float_fn : Callable[[Any], float | None]
        Input parameter used by this function.
    slug_underscore_fn : Callable[[Any], str]
        Input parameter used by this function.
    render_plot_fn : Callable[[dict[str, Any]], Any]
        Input parameter used by this function.
    
    Returns
    -----
    list[Path]
        Value produced by this function call.
    
    Examples
    -----
    ```python
    from reaxkit.core.study.present_engine import make_all_cases_per_iter_boxplots
    # Configure required arguments for your case.
    result = make_all_cases_per_iter_boxplots(...)
    print(type(result).__name__)
    ```
    Sample output:
    ```text
    str
    ```
    The output type reflects the return contract for this API call.
    """
    src = across_dir / "raw_all_cases.csv"
    if not src.exists():
        return []
    rows = load_csv_rows_fn(src)
    if not rows:
        return []
    param_cols = find_param_columns_fn(rows)
    if not param_cols:
        return []
    by_y: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        y_name = str(row.get("y_name") or "").strip()
        if y_name:
            by_y.setdefault(y_name, []).append(row)

    outputs: list[Path] = []
    for y_name, y_rows in by_y.items():
        x_values = sort_plot_x_fn(list({row.get("x_value") for row in y_rows}))
        for xv in x_values:
            subset = [r for r in y_rows if str(r.get("x_value")) == str(xv)]
            dist: dict[str, list[float]] = {}
            for r in subset:
                case_key = row_key_from_params_fn(r, param_cols)
                yv = safe_float_fn(r.get("y_value"))
                if yv is None:
                    continue
                dist.setdefault(case_key, []).append(yv)
            labels = [k for k, v in dist.items() if v]
            data = [v for v in dist.values() if v]
            if not data:
                continue
            out_path = across_dir / f"boxplot_all_cases_{slug_underscore_fn(y_name)}_x_{slug_underscore_fn(xv)}.png"
            render_plot_fn(
                {
                    "plot_type": "box_whisker_plot",
                    "data": data,
                    "labels": labels,
                    "xlabel": "case",
                    "ylabel": y_name,
                    "title": f"{aggregate_title} - {y_name} boxplot at x={xv}",
                    "save": str(out_path),
                }
            )
            outputs.append(out_path)
    return outputs


def plot_study_aggregates(
    *,
    study_root: Path,
    aggregate_title_filter: str | None,
    case_filter: str | None,
    read_json_fn: Callable[[Path], dict[str, Any]],
    load_source_study_doc_fn: Callable[[dict[str, Any]], dict[str, Any]],
    aggregate_defs_from_doc_fn: Callable[[dict[str, Any]], dict[str, Any]],
    case_matches_selector_fn: Callable[[dict[str, Any], str | None], bool],
    make_errorbar_plots_for_case_aggregate_fn: Callable[[Path], list[Path]] | Callable[..., list[Path]],
    make_boxplots_for_case_aggregate_fn: Callable[[Path], list[Path]] | Callable[..., list[Path]],
    make_all_cases_errorbar_plots_fn: Callable[[Path], list[Path]] | Callable[..., list[Path]],
    make_all_cases_heatmaps_fn: Callable[[Path], list[Path]] | Callable[..., list[Path]],
    make_all_cases_per_iter_boxplots_fn: Callable[[Path], list[Path]] | Callable[..., list[Path]],
    local_now_fn: Callable[[], str],
    duration_minutes_fn: Callable[[str | None, str | None], float | None],
    log_task_event_fn: Callable[..., None],
    write_named_status_fn: Callable[..., tuple[Path, Path]],
    plot_status_csv_file: str,
    plot_status_json_file: str,
) -> dict[str, Any]:
    """
    Plot study aggregates.
    
    This function is part of the ReaxKit core API and performs the operation described by its name and arguments.
    
    Parameters
    -----
    study_root : Path
        Input parameter used by this function.
    aggregate_title_filter : str | None
        Input parameter used by this function.
    case_filter : str | None
        Input parameter used by this function.
    read_json_fn : Callable[[Path], dict[str, Any]]
        Input parameter used by this function.
    load_source_study_doc_fn : Callable[[dict[str, Any]], dict[str, Any]]
        Input parameter used by this function.
    aggregate_defs_from_doc_fn : Callable[[dict[str, Any]], dict[str, Any]]
        Input parameter used by this function.
    case_matches_selector_fn : Callable[[dict[str, Any], str | None], bool]
        Input parameter used by this function.
    make_errorbar_plots_for_case_aggregate_fn : Callable[[Path], list[Path]] | Callable[..., list[Path]]
        Input parameter used by this function.
    make_boxplots_for_case_aggregate_fn : Callable[[Path], list[Path]] | Callable[..., list[Path]]
        Input parameter used by this function.
    make_all_cases_errorbar_plots_fn : Callable[[Path], list[Path]] | Callable[..., list[Path]]
        Input parameter used by this function.
    make_all_cases_heatmaps_fn : Callable[[Path], list[Path]] | Callable[..., list[Path]]
        Input parameter used by this function.
    make_all_cases_per_iter_boxplots_fn : Callable[[Path], list[Path]] | Callable[..., list[Path]]
        Input parameter used by this function.
    local_now_fn : Callable[[], str]
        Input parameter used by this function.
    duration_minutes_fn : Callable[[str | None, str | None], float | None]
        Input parameter used by this function.
    log_task_event_fn : Callable[..., None]
        Input parameter used by this function.
    write_named_status_fn : Callable[..., tuple[Path, Path]]
        Input parameter used by this function.
    plot_status_csv_file : str
        Input parameter used by this function.
    plot_status_json_file : str
        Input parameter used by this function.
    
    Returns
    -----
    dict[str, Any]
        Value produced by this function call.
    
    Examples
    -----
    ```python
    from reaxkit.core.study.present_engine import plot_study_aggregates
    # Configure required arguments for your case.
    result = plot_study_aggregates(...)
    print(type(result).__name__)
    ```
    Sample output:
    ```text
    str
    ```
    The output type reflects the return contract for this API call.
    """
    study_manifest = read_json_fn(study_root / "study_manifest.json")
    study_doc = load_source_study_doc_fn(study_manifest)
    aggregate_defs = aggregate_defs_from_doc_fn(study_doc)
    if not aggregate_defs:
        raise ValueError("No top-level aggregate definitions found in study.yaml.")

    if aggregate_title_filter:
        if aggregate_title_filter not in aggregate_defs:
            raise KeyError(f"Unknown aggregate title: {aggregate_title_filter}")
        target_titles = [aggregate_title_filter]
    else:
        target_titles = list(aggregate_defs.keys())

    generated: list[Path] = []
    missing: list[str] = []
    target_set = set(target_titles)
    status_rows: list[dict[str, Any]] = []
    for case in study_manifest.get("cases") or []:
        if not isinstance(case, dict):
            continue
        if not case_matches_selector_fn(case, case_filter):
            continue
        case_path = Path(str(case.get("path") or ""))
        for agg_title in target_titles:
            case_agg_dir = case_path / "aggregate" / agg_title
            if not case_agg_dir.exists():
                missing.append(str(case_agg_dir))
                continue
            generated.extend(make_errorbar_plots_for_case_aggregate_fn(case_agg_dir, aggregate_title=agg_title))
            generated.extend(make_boxplots_for_case_aggregate_fn(case_agg_dir, aggregate_title=agg_title))

    for agg_title in target_titles:
        started_at = local_now_fn()
        before_count = len(generated)
        across_dir = study_root / "cases" / "aggregate" / agg_title
        if not across_dir.exists():
            if agg_title in target_set:
                missing.append(str(across_dir))
            finished_at = local_now_fn()
            status_rows.append(
                {
                    "title": agg_title,
                    "status": "skip",
                    "reason": "missing_aggregate_dir",
                    "generated_count": 0,
                    "missing_count": 1,
                    "started_at": started_at,
                    "finished_at": finished_at,
                    "duration_min": duration_minutes_fn(started_at, finished_at),
                }
            )
            log_task_event_fn("SKIP", f"plot {agg_title}", "missing aggregate directory")
            continue
        generated.extend(make_all_cases_errorbar_plots_fn(across_dir, aggregate_title=agg_title))
        generated.extend(make_all_cases_heatmaps_fn(across_dir, aggregate_title=agg_title))
        generated.extend(make_all_cases_per_iter_boxplots_fn(across_dir, aggregate_title=agg_title))
        created_n = len(generated) - before_count
        finished_at = local_now_fn()
        status_rows.append(
            {
                "title": agg_title,
                "status": "done",
                "reason": "",
                "generated_count": created_n,
                "missing_count": 0,
                "started_at": started_at,
                "finished_at": finished_at,
                "duration_min": duration_minutes_fn(started_at, finished_at),
            }
        )
        log_task_event_fn("DONE", f"plot {agg_title}", f"generated={created_n}")

    generated_files = [str(p) for p in generated]
    summary = {
        "total": len(status_rows),
        "done": sum(1 for r in status_rows if r.get("status") == "done"),
        "skip": sum(1 for r in status_rows if r.get("status") == "skip"),
        "generated_count": len(generated),
        "generated_files": generated_files,
        "missing_dirs": missing,
    }
    csv_path, json_path = write_named_status_fn(
        study_root=study_root,
        rows=status_rows,
        summary=summary,
        csv_name=plot_status_csv_file,
        json_name=plot_status_json_file,
    )
    return {"generated": generated, "missing": missing, "manifest": json_path, "csv": csv_path}
