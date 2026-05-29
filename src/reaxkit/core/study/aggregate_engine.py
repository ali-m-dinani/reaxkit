"""
Aggregation orchestration for studies.

**Usage context**

- Import these helpers from ReaxKit core modules when implementing CLI and workflow logic.
- Reuse the public APIs here to keep behavior consistent across commands and engines.
"""

from __future__ import annotations

import csv
import math
import statistics
from pathlib import Path
from typing import Any, Callable


def safe_float(value: Any) -> float | None:
    """
    Safe float.
    
    This function is part of the ReaxKit core API and performs the operation described by its name and arguments.
    
    Parameters
    -----
    value : Any
        Input parameter used by this function.
    
    Returns
    -----
    float | None
        Value produced by this function call.
    
    Examples
    -----
    ```python
    from reaxkit.core.study.aggregate_engine import safe_float
    # Configure required arguments for your case.
    result = safe_float(...)
    print(type(result).__name__)
    ```
    Sample output:
    ```text
    str
    ```
    The output type reflects the return contract for this API call.
    """
    try:
        fv = float(value)
    except Exception:
        return None
    if not math.isfinite(fv):
        return None
    return fv


def stat_value(row: dict[str, Any], key: str) -> Any:
    """
    Stat value.
    
    This function is part of the ReaxKit core API and performs the operation described by its name and arguments.
    
    Parameters
    -----
    row : dict[str, Any]
        Input parameter used by this function.
    key : str
        Input parameter used by this function.
    
    Returns
    -----
    Any
        Value produced by this function call.
    
    Examples
    -----
    ```python
    from reaxkit.core.study.aggregate_engine import stat_value
    # Configure required arguments for your case.
    result = stat_value(...)
    print(type(result).__name__)
    ```
    Sample output:
    ```text
    str
    ```
    The output type reflects the return contract for this API call.
    """
    prefixed = f"y_{key}"
    if prefixed in row:
        return row.get(prefixed)
    return row.get(key)


def load_csv_rows(path: Path) -> list[dict[str, str]]:
    """
    Load csv rows.
    
    This function is part of the ReaxKit core API and performs the operation described by its name and arguments.
    
    Parameters
    -----
    path : Path
        Input parameter used by this function.
    
    Returns
    -----
    list[dict[str, str]]
        Value produced by this function call.
    
    Examples
    -----
    ```python
    from reaxkit.core.study.aggregate_engine import load_csv_rows
    # Configure required arguments for your case.
    result = load_csv_rows(...)
    print(type(result).__name__)
    ```
    Sample output:
    ```text
    str
    ```
    The output type reflects the return contract for this API call.
    """
    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        return [dict(row) for row in reader]


def extract_scalar_from_csv(
    csv_path: Path,
    *,
    value_column: str | None,
) -> tuple[float | None, str | None]:
    """
    Extract scalar from csv.
    
    This function is part of the ReaxKit core API and performs the operation described by its name and arguments.
    
    Parameters
    -----
    csv_path : Path
        Input parameter used by this function.
    value_column : str | None
        Input parameter used by this function.
    
    Returns
    -----
    tuple[float | None, str | None]
        Value produced by this function call.
    
    Examples
    -----
    ```python
    from reaxkit.core.study.aggregate_engine import extract_scalar_from_csv
    # Configure required arguments for your case.
    result = extract_scalar_from_csv(...)
    print(type(result).__name__)
    ```
    Sample output:
    ```text
    str
    ```
    The output type reflects the return contract for this API call.
    """
    rows = load_csv_rows(csv_path)
    if not rows:
        return None, None

    columns = list(rows[0].keys())
    target_col: str | None = None
    if value_column:
        if value_column not in columns:
            raise KeyError(f"Requested value column '{value_column}' not found in {csv_path}.")
        target_col = value_column
    else:
        preferred = [
            "polarization",
            "P_z (uC/cm^2)",
            "p_z",
            "value",
            "dipole_magnitude",
            "mu_z (debye)",
            "mu_z",
        ]
        for cand in preferred:
            if cand in columns:
                target_col = cand
                break

        if target_col is None:
            ignore = {"iter", "frame", "frame_index", "time", "atom_id", "x", "y", "z"}
            for col in columns:
                if col.lower() in ignore:
                    continue
                vals = [safe_float(row.get(col)) for row in rows]
                vals = [v for v in vals if v is not None]
                if vals:
                    target_col = col
                    break

    if target_col is None:
        return None, None
    vals = [safe_float(row.get(target_col)) for row in rows]
    vals = [v for v in vals if v is not None]
    if not vals:
        return None, target_col
    return float(sum(vals) / len(vals)), target_col


def load_column_values(csv_path: Path, column: str) -> list[Any]:
    """
    Load column values.
    
    This function is part of the ReaxKit core API and performs the operation described by its name and arguments.
    
    Parameters
    -----
    csv_path : Path
        Input parameter used by this function.
    column : str
        Input parameter used by this function.
    
    Returns
    -----
    list[Any]
        Value produced by this function call.
    
    Examples
    -----
    ```python
    from reaxkit.core.study.aggregate_engine import load_column_values
    # Configure required arguments for your case.
    result = load_column_values(...)
    print(type(result).__name__)
    ```
    Sample output:
    ```text
    str
    ```
    The output type reflects the return contract for this API call.
    """
    rows = load_csv_rows(csv_path)
    if not rows:
        return []
    if column not in rows[0]:
        raise KeyError(f"Column '{column}' not found in {csv_path}.")
    return [row.get(column) for row in rows]


def resolve_variable_csv_path(*, stage_dir: Path, spec: dict[str, str]) -> Path | None:
    """
    Resolve variable csv path.
    
    This function is part of the ReaxKit core API and performs the operation described by its name and arguments.
    
    Parameters
    -----
    stage_dir : Path
        Input parameter used by this function.
    spec : dict[str, str]
        Input parameter used by this function.
    
    Returns
    -----
    Path | None
        Value produced by this function call.
    
    Examples
    -----
    ```python
    from reaxkit.core.study.aggregate_engine import resolve_variable_csv_path
    # Configure required arguments for your case.
    result = resolve_variable_csv_path(...)
    print(type(result).__name__)
    ```
    Sample output:
    ```text
    str
    ```
    The output type reflects the return contract for this API call.
    """
    directory_value = str(spec.get("directory") or "").strip()
    folder_id_value = str(spec.get("folder_id") or "").strip()
    file_value = str(spec.get("file") or "").strip()
    if not file_value:
        return None

    file_path = Path(file_value)
    if file_path.is_absolute():
        return file_path

    if not directory_value:
        return stage_dir / file_value

    directory_path = stage_dir / directory_value
    if not directory_path.exists() or not directory_path.is_dir():
        return None

    target_dir = directory_path
    if folder_id_value:
        if folder_id_value.lower() == "latest":
            children = [p for p in directory_path.iterdir() if p.is_dir()]
            if not children:
                return None
            target_dir = max(children, key=lambda p: p.stat().st_mtime)
        else:
            explicit = directory_path / folder_id_value
            if not explicit.exists() or not explicit.is_dir():
                return None
            target_dir = explicit

    return target_dir / file_value


def resolve_variable_file_for_run(
    *,
    stage_dir: Path,
    spec: dict[str, str],
    result_dirs: list[str] | None,
) -> Path | None:
    """
    Resolve variable file for run.
    
    This function is part of the ReaxKit core API and performs the operation described by its name and arguments.
    
    Parameters
    -----
    stage_dir : Path
        Input parameter used by this function.
    spec : dict[str, str]
        Input parameter used by this function.
    result_dirs : list[str] | None
        Input parameter used by this function.
    
    Returns
    -----
    Path | None
        Value produced by this function call.
    
    Examples
    -----
    ```python
    from reaxkit.core.study.aggregate_engine import resolve_variable_file_for_run
    # Configure required arguments for your case.
    result = resolve_variable_file_for_run(...)
    print(type(result).__name__)
    ```
    Sample output:
    ```text
    str
    ```
    The output type reflects the return contract for this API call.
    """
    file_value = str(spec.get("file") or "").strip()
    if not file_value:
        return None
    if result_dirs:
        for result_dir in result_dirs:
            candidate = Path(str(result_dir)) / file_value
            if candidate.exists():
                return candidate
    return resolve_variable_csv_path(stage_dir=stage_dir, spec=spec)


def apply_reducer_to_series(
    *,
    pairs: list[tuple[Any, Any]],
    reducer: str,
) -> list[tuple[Any, float]]:
    """
    Apply reducer to series.
    
    This function is part of the ReaxKit core API and performs the operation described by its name and arguments.
    
    Parameters
    -----
    pairs : list[tuple[Any, Any]]
        Input parameter used by this function.
    reducer : str
        Input parameter used by this function.
    
    Returns
    -----
    list[tuple[Any, float]]
        Value produced by this function call.
    
    Examples
    -----
    ```python
    from reaxkit.core.study.aggregate_engine import apply_reducer_to_series
    # Configure required arguments for your case.
    result = apply_reducer_to_series(...)
    print(type(result).__name__)
    ```
    Sample output:
    ```text
    str
    ```
    The output type reflects the return contract for this API call.
    """
    vals: list[tuple[Any, float]] = []
    for x, y in pairs:
        fy = safe_float(y)
        if fy is None:
            continue
        vals.append((x, fy))
    if not vals:
        return []
    r = str(reducer or "identity").strip().lower()
    if r == "identity":
        return vals
    ys = [v for _, v in vals]
    xs_num = [safe_float(x) for x, _ in vals]
    if r == "mean":
        return [("__reduced__", float(statistics.fmean(ys)))]
    if r == "median":
        return [("__reduced__", float(statistics.median(ys)))]
    if r == "first":
        return [("__reduced__", float(ys[0]))]
    if r == "last":
        return [("__reduced__", float(ys[-1]))]
    if r == "max":
        return [("__reduced__", float(max(ys)))]
    if r == "min":
        return [("__reduced__", float(min(ys)))]
    if r == "trapz":
        import numpy as np

        if any(v is None for v in xs_num):
            return []
        area = float(np.trapz(ys, [float(v) for v in xs_num]))
        return [("__reduced__", area)]
    if r == "slope":
        import numpy as np

        if any(v is None for v in xs_num):
            return []
        xarr = np.array([float(v) for v in xs_num], dtype=float)
        yarr = np.array(ys, dtype=float)
        if len(xarr) < 2:
            return []
        a, _b = np.polyfit(xarr, yarr, 1)
        return [("__reduced__", float(a))]
    raise ValueError(f"Unsupported reducer: {reducer}")


def compute_stats(values: list[float], wanted: list[str]) -> dict[str, float]:
    """
    Compute stats.
    
    This function is part of the ReaxKit core API and performs the operation described by its name and arguments.
    
    Parameters
    -----
    values : list[float]
        Input parameter used by this function.
    wanted : list[str]
        Input parameter used by this function.
    
    Returns
    -----
    dict[str, float]
        Value produced by this function call.
    
    Examples
    -----
    ```python
    from reaxkit.core.study.aggregate_engine import compute_stats
    # Configure required arguments for your case.
    result = compute_stats(...)
    print(type(result).__name__)
    ```
    Sample output:
    ```text
    str
    ```
    The output type reflects the return contract for this API call.
    """
    out: dict[str, float] = {}
    n = len(values)
    if "n" in wanted:
        out["n"] = float(n)
    if n == 0:
        return out
    mean_v = float(statistics.fmean(values))
    if "mean" in wanted:
        out["mean"] = mean_v
    if "min" in wanted:
        out["min"] = float(min(values))
    if "max" in wanted:
        out["max"] = float(max(values))
    std_v = float(statistics.stdev(values)) if n > 1 else 0.0
    if "std" in wanted:
        out["std"] = std_v
    if "sem" in wanted:
        out["sem"] = float(std_v / math.sqrt(n)) if n > 0 else 0.0
    return out


def normalize_stage_variables(rendered_stage: dict[str, Any]) -> dict[str, dict[str, str]]:
    """
    Normalize stage variables.
    
    This function is part of the ReaxKit core API and performs the operation described by its name and arguments.
    
    Parameters
    -----
    rendered_stage : dict[str, Any]
        Input parameter used by this function.
    
    Returns
    -----
    dict[str, dict[str, str]]
        Value produced by this function call.
    
    Examples
    -----
    ```python
    from reaxkit.core.study.aggregate_engine import normalize_stage_variables
    # Configure required arguments for your case.
    result = normalize_stage_variables(...)
    print(type(result).__name__)
    ```
    Sample output:
    ```text
    str
    ```
    The output type reflects the return contract for this API call.
    """
    variables: dict[str, dict[str, str]] = {}
    raw = rendered_stage.get("variables")
    if not isinstance(raw, dict):
        return variables
    for variable_name, spec in raw.items():
        name = str(variable_name).strip()
        if not name or not isinstance(spec, dict):
            continue
        payload: dict[str, str] = {}
        directory_value = str(spec.get("directory") or "").strip()
        folder_id_value = str(spec.get("folder_id") or "").strip()
        file_value = str(spec.get("file") or "").strip()
        column_value = str(spec.get("column") or "").strip()
        if directory_value:
            payload["directory"] = directory_value
        if folder_id_value:
            payload["folder_id"] = folder_id_value
        if file_value:
            payload["file"] = file_value
        if column_value:
            payload["column"] = column_value
        if "file" not in payload:
            continue
        variables[name] = payload
    return variables


def to_plot_value(v: Any):
    """
    To plot value.
    
    This function is part of the ReaxKit core API and performs the operation described by its name and arguments.
    
    Parameters
    -----
    v : Any
        Input parameter used by this function.
    
    Returns
    -----
    Any
        Value produced by this function call.
    
    Examples
    -----
    ```python
    from reaxkit.core.study.aggregate_engine import to_plot_value
    # Configure required arguments for your case.
    result = to_plot_value(...)
    print(type(result).__name__)
    ```
    Sample output:
    ```text
    str
    ```
    The output type reflects the return contract for this API call.
    """
    fv = safe_float(v)
    return fv if fv is not None else str(v)


def sort_values(values: list[Any]) -> list[Any]:
    """
    Sort values.
    
    This function is part of the ReaxKit core API and performs the operation described by its name and arguments.
    
    Parameters
    -----
    values : list[Any]
        Input parameter used by this function.
    
    Returns
    -----
    list[Any]
        Value produced by this function call.
    
    Examples
    -----
    ```python
    from reaxkit.core.study.aggregate_engine import sort_values
    # Configure required arguments for your case.
    result = sort_values(...)
    print(type(result).__name__)
    ```
    Sample output:
    ```text
    str
    ```
    The output type reflects the return contract for this API call.
    """
    def _key(v: Any):
        fv = safe_float(v)
        if fv is not None:
            return (0, fv, str(v))
        return (1, str(v), "")

    return sorted(values, key=_key)


def aggregate_study_analysis(
    *,
    study_root: Path,
    analysis_name: str,
    value_column: str | None,
    stage_filter: str | None,
    read_json_fn: Callable[[Path], dict[str, Any]],
    load_source_study_doc_fn: Callable[[dict[str, Any]], dict[str, Any]],
    aggregate_defs_from_doc_fn: Callable[[dict[str, Any]], dict[str, Any]],
    analysis_defs_from_doc_fn: Callable[[dict[str, Any]], dict[str, Any]],
    aggregate_from_definition_fn: Callable[..., dict[str, Any]],
    iter_replicate_variable_records_fn: Callable[..., tuple[list[dict[str, Any]], str | None]],
    to_plot_value_fn: Callable[[Any], Any],
    make_aggregate_plot_fn: Callable[..., tuple[bool, str]],
    utc_now_fn: Callable[[], str],
    write_json_fn: Callable[[Path, dict[str, Any]], None],
) -> dict[str, Any]:
    """
    Aggregate study analysis.
    
    This function is part of the ReaxKit core API and performs the operation described by its name and arguments.
    
    Parameters
    -----
    study_root : Path
        Input parameter used by this function.
    analysis_name : str
        Input parameter used by this function.
    value_column : str | None
        Input parameter used by this function.
    stage_filter : str | None
        Input parameter used by this function.
    read_json_fn : Callable[[Path], dict[str, Any]]
        Input parameter used by this function.
    load_source_study_doc_fn : Callable[[dict[str, Any]], dict[str, Any]]
        Input parameter used by this function.
    aggregate_defs_from_doc_fn : Callable[[dict[str, Any]], dict[str, Any]]
        Input parameter used by this function.
    analysis_defs_from_doc_fn : Callable[[dict[str, Any]], dict[str, Any]]
        Input parameter used by this function.
    aggregate_from_definition_fn : Callable[..., dict[str, Any]]
        Input parameter used by this function.
    iter_replicate_variable_records_fn : Callable[..., tuple[list[dict[str, Any]], str | None]]
        Input parameter used by this function.
    to_plot_value_fn : Callable[[Any], Any]
        Input parameter used by this function.
    make_aggregate_plot_fn : Callable[..., tuple[bool, str]]
        Input parameter used by this function.
    utc_now_fn : Callable[[], str]
        Input parameter used by this function.
    write_json_fn : Callable[[Path, dict[str, Any]], None]
        Input parameter used by this function.
    
    Returns
    -----
    dict[str, Any]
        Value produced by this function call.
    
    Examples
    -----
    ```python
    from reaxkit.core.study.aggregate_engine import aggregate_study_analysis
    # Configure required arguments for your case.
    result = aggregate_study_analysis(...)
    print(type(result).__name__)
    ```
    Sample output:
    ```text
    str
    ```
    The output type reflects the return contract for this API call.
    """
    study_manifest = read_json_fn(study_root / "study_manifest.json")
    try:
        study_doc = load_source_study_doc_fn(study_manifest)
        aggregate_defs = aggregate_defs_from_doc_fn(study_doc)
        analysis_defs_by_title = analysis_defs_from_doc_fn(study_doc)
    except Exception:
        aggregate_defs = {}
        analysis_defs_by_title = {}

    if analysis_name in aggregate_defs:
        agg_def = aggregate_defs[analysis_name]
        if agg_def.analysis_title not in analysis_defs_by_title:
            raise KeyError(
                f"Aggregate '{agg_def.title}' references unknown analysis_title '{agg_def.analysis_title}'."
            )
        return aggregate_from_definition_fn(
            study_root=study_root,
            study_manifest=study_manifest,
            aggregate_def=agg_def,
            analysis_def=analysis_defs_by_title[agg_def.analysis_title],
            stage_filter=stage_filter,
        )

    param_names = list((study_manifest.get("parameters") or {}).keys())
    raw_rows, chosen_column = iter_replicate_variable_records_fn(
        study_manifest=study_manifest,
        variable_name=analysis_name,
        stage_filter=stage_filter,
        value_column_override=value_column,
    )

    out_dir = study_root / "analysis" / analysis_name / "aggregate"
    out_dir.mkdir(parents=True, exist_ok=True)

    raw_csv = out_dir / f"{analysis_name}_raw.csv"
    raw_fields = [
        *param_names,
        "replicate",
        analysis_name,
        "case_id",
        "stage",
        "analysis_title",
        "analysis_variable",
        "source_file",
        "source_column",
    ]
    with raw_csv.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=raw_fields)
        writer.writeheader()
        for row in raw_rows:
            writer.writerow({k: row.get(k) for k in raw_fields})

    grouped: dict[tuple[Any, ...], list[float]] = {}
    for row in raw_rows:
        key = tuple(row.get(p) for p in param_names)
        grouped.setdefault(key, []).append(float(row[analysis_name]))

    mean_col = f"mean_{analysis_name}"
    std_col = f"std_{analysis_name}"
    sem_col = f"sem_{analysis_name}"
    grouped_rows: list[dict[str, Any]] = []
    for key, vals in grouped.items():
        n = len(vals)
        mean_v = statistics.fmean(vals)
        std_v = statistics.stdev(vals) if n > 1 else 0.0
        sem_v = (std_v / math.sqrt(n)) if n > 0 else 0.0
        out = {p: key[i] for i, p in enumerate(param_names)}
        out.update({mean_col: mean_v, std_col: std_v, sem_col: sem_v, "n": n})
        grouped_rows.append(out)

    grouped_rows.sort(key=lambda row: tuple(to_plot_value_fn(row[p]) for p in param_names))
    grouped_csv = out_dir / f"{analysis_name}_grouped.csv"
    grouped_fields = [*param_names, mean_col, std_col, sem_col, "n"]
    with grouped_csv.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=grouped_fields)
        writer.writeheader()
        for row in grouped_rows:
            writer.writerow({k: row.get(k) for k in grouped_fields})

    plot_path = out_dir / f"{analysis_name}_plot.png"
    plot_ok, plot_msg = make_aggregate_plot_fn(
        grouped_rows,
        params=param_names,
        metric_mean_col=mean_col,
        metric_std_col=std_col,
        out_path=plot_path,
    )
    if not plot_ok and plot_path.exists():
        plot_path.unlink()

    manifest = {
        "study_root": str(study_root),
        "variable": analysis_name,
        "stage_filter": stage_filter,
        "value_column": value_column,
        "resolved_value_column": chosen_column,
        "raw_count": len(raw_rows),
        "group_count": len(grouped_rows),
        "raw_csv": str(raw_csv),
        "grouped_csv": str(grouped_csv),
        "plot": str(plot_path) if plot_ok else None,
        "plot_status": plot_msg,
        "generated_at_utc": utc_now_fn(),
    }
    manifest_path = out_dir / "aggregate_manifest.json"
    write_json_fn(manifest_path, manifest)
    return {
        "manifest": manifest_path,
        "raw_csv": raw_csv,
        "grouped_csv": grouped_csv,
        "plot": plot_path if plot_ok else None,
        "raw_count": len(raw_rows),
        "group_count": len(grouped_rows),
    }


def aggregate_study_all(
    *,
    study_root: Path,
    aggregate_title_filter: str | None,
    stage_filter: str | None,
    value_column: str | None,
    legacy_analysis_name: str | None,
    read_json_fn: Callable[[Path], dict[str, Any]],
    load_source_study_doc_fn: Callable[[dict[str, Any]], dict[str, Any]],
    aggregate_defs_from_doc_fn: Callable[[dict[str, Any]], dict[str, Any]],
    aggregate_study_analysis_fn: Callable[..., dict[str, Any]],
    local_now_fn: Callable[[], str],
    utc_now_fn: Callable[[], str],
    duration_minutes_fn: Callable[[str | None, str | None], float | None],
    log_task_event_fn: Callable[..., None],
    write_named_status_fn: Callable[..., tuple[Path, Path]],
    aggregate_status_csv_file: str,
    aggregate_status_json_file: str,
) -> list[dict[str, Any]]:
    """
    Aggregate study all.
    
    This function is part of the ReaxKit core API and performs the operation described by its name and arguments.
    
    Parameters
    -----
    study_root : Path
        Input parameter used by this function.
    aggregate_title_filter : str | None
        Input parameter used by this function.
    stage_filter : str | None
        Input parameter used by this function.
    value_column : str | None
        Input parameter used by this function.
    legacy_analysis_name : str | None
        Input parameter used by this function.
    read_json_fn : Callable[[Path], dict[str, Any]]
        Input parameter used by this function.
    load_source_study_doc_fn : Callable[[dict[str, Any]], dict[str, Any]]
        Input parameter used by this function.
    aggregate_defs_from_doc_fn : Callable[[dict[str, Any]], dict[str, Any]]
        Input parameter used by this function.
    aggregate_study_analysis_fn : Callable[..., dict[str, Any]]
        Input parameter used by this function.
    local_now_fn : Callable[[], str]
        Input parameter used by this function.
    utc_now_fn : Callable[[], str]
        Input parameter used by this function.
    duration_minutes_fn : Callable[[str | None, str | None], float | None]
        Input parameter used by this function.
    log_task_event_fn : Callable[..., None]
        Input parameter used by this function.
    write_named_status_fn : Callable[..., tuple[Path, Path]]
        Input parameter used by this function.
    aggregate_status_csv_file : str
        Input parameter used by this function.
    aggregate_status_json_file : str
        Input parameter used by this function.
    
    Returns
    -----
    list[dict[str, Any]]
        Value produced by this function call.
    
    Examples
    -----
    ```python
    from reaxkit.core.study.aggregate_engine import aggregate_study_all
    # Configure required arguments for your case.
    result = aggregate_study_all(...)
    print(type(result).__name__)
    ```
    Sample output:
    ```text
    str
    ```
    The output type reflects the return contract for this API call.
    """
    study_manifest = read_json_fn(study_root / "study_manifest.json")
    aggregate_defs: dict[str, Any] = {}
    try:
        study_doc = load_source_study_doc_fn(study_manifest)
        aggregate_defs = aggregate_defs_from_doc_fn(study_doc)
    except Exception:
        aggregate_defs = {}

    if aggregate_defs:
        if aggregate_title_filter:
            if aggregate_title_filter not in aggregate_defs:
                raise KeyError(f"Unknown aggregate title: {aggregate_title_filter}")
            targets = [aggregate_title_filter]
        else:
            targets = list(aggregate_defs.keys())
        results: list[dict[str, Any]] = []
        status_rows: list[dict[str, Any]] = []
        for title in targets:
            started_at = local_now_fn()
            started_utc = utc_now_fn()
            res = aggregate_study_analysis_fn(
                study_root=study_root,
                analysis_name=title,
                value_column=value_column,
                stage_filter=stage_filter,
            )
            finished_at = local_now_fn()
            duration_min = duration_minutes_fn(started_at, finished_at)
            out = dict(res)
            out["title"] = title
            results.append(out)
            log_task_event_fn("DONE", f"aggregate {title}")
            status_rows.append(
                {
                    "title": title,
                    "status": "done",
                    "reason": "",
                    "raw_csv": str(out.get("raw_csv") or ""),
                    "grouped_csv": str(out.get("grouped_csv") or ""),
                    "manifest": str(out.get("manifest") or ""),
                    "raw_count": int(out.get("raw_count", 0) or 0),
                    "group_count": int(out.get("group_count", 0) or 0),
                    "started_at": started_at,
                    "finished_at": finished_at,
                    "duration_min": duration_min,
                    "started_at_utc": started_utc,
                    "finished_at_utc": utc_now_fn(),
                }
            )
        summary = {
            "total": len(status_rows),
            "done": sum(1 for r in status_rows if r.get("status") == "done"),
            "failed": sum(1 for r in status_rows if r.get("status") == "failed"),
        }
        write_named_status_fn(
            study_root=study_root,
            rows=status_rows,
            summary=summary,
            csv_name=aggregate_status_csv_file,
            json_name=aggregate_status_json_file,
        )
        return results

    if legacy_analysis_name is None:
        raise ValueError(
            "No top-level aggregate definitions found in study.yaml. "
            "Provide --analysis <name> to use legacy variable aggregation."
        )
    res = aggregate_study_analysis_fn(
        study_root=study_root,
        analysis_name=legacy_analysis_name,
        value_column=value_column,
        stage_filter=stage_filter,
    )
    out = dict(res)
    out["title"] = legacy_analysis_name
    log_task_event_fn("DONE", f"aggregate {legacy_analysis_name}")
    write_named_status_fn(
        study_root=study_root,
        rows=[
            {
                "title": legacy_analysis_name,
                "status": "done",
                "reason": "",
                "raw_csv": str(out.get("raw_csv") or ""),
                "grouped_csv": str(out.get("grouped_csv") or ""),
                "manifest": str(out.get("manifest") or ""),
                "raw_count": int(out.get("raw_count", 0) or 0),
                "group_count": int(out.get("group_count", 0) or 0),
                "started_at": None,
                "finished_at": local_now_fn(),
                "duration_min": None,
                "started_at_utc": None,
                "finished_at_utc": utc_now_fn(),
            }
        ],
        summary={"total": 1, "done": 1, "failed": 0},
        csv_name=aggregate_status_csv_file,
        json_name=aggregate_status_json_file,
    )
    return [out]
