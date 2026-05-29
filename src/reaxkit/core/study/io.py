"""
Study IO/status helpers.

**Usage context**

- Import these helpers from ReaxKit core modules when implementing CLI and workflow logic.
- Reuse the public APIs here to keep behavior consistent across commands and engines.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from reaxkit.core.study.logging import utc_now


def write_json(path: Path, payload: dict[str, Any]) -> None:
    """
    Write json.
    
    This function is part of the ReaxKit core API and performs the operation described by its name and arguments.
    
    Parameters
    -----
    path : Path
        Input parameter used by this function.
    payload : dict[str, Any]
        Input parameter used by this function.
    
    Returns
    -----
    None
        Value produced by this function call.
    
    Examples
    -----
    ```python
    from reaxkit.core.study.io import write_json
    # Configure required arguments for your case.
    result = write_json(...)
    print(type(result).__name__)
    ```
    Sample output:
    ```text
    str
    ```
    The output type reflects the return contract for this API call.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def read_json(path: Path) -> dict[str, Any]:
    """
    Read json.
    
    This function is part of the ReaxKit core API and performs the operation described by its name and arguments.
    
    Parameters
    -----
    path : Path
        Input parameter used by this function.
    
    Returns
    -----
    dict[str, Any]
        Value produced by this function call.
    
    Examples
    -----
    ```python
    from reaxkit.core.study.io import read_json
    # Configure required arguments for your case.
    result = read_json(...)
    print(type(result).__name__)
    ```
    Sample output:
    ```text
    str
    ```
    The output type reflects the return contract for this API call.
    """
    if not path.exists():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}.")
    return payload


def resolve_existing_file(primary: Path, *fallbacks: str) -> Path:
    """
    Resolve existing file.
    
    This function is part of the ReaxKit core API and performs the operation described by its name and arguments.
    
    Parameters
    -----
    primary : Path
        Input parameter used by this function.
    *fallbacks : str
        Input parameter used by this function.
    
    Returns
    -----
    Path
        Value produced by this function call.
    
    Examples
    -----
    ```python
    from reaxkit.core.study.io import resolve_existing_file
    # Configure required arguments for your case.
    result = resolve_existing_file(...)
    print(type(result).__name__)
    ```
    Sample output:
    ```text
    str
    ```
    The output type reflects the return contract for this API call.
    """
    if primary.exists():
        return primary
    for name in fallbacks:
        candidate = primary.parent / name
        if candidate.exists():
            return candidate
    return primary


def write_named_status(
    *,
    study_root: Path,
    rows: list[dict[str, Any]],
    summary: dict[str, Any],
    csv_name: str,
    json_name: str,
) -> tuple[Path, Path]:
    """
    Write named status.
    
    This function is part of the ReaxKit core API and performs the operation described by its name and arguments.
    
    Parameters
    -----
    study_root : Path
        Input parameter used by this function.
    rows : list[dict[str, Any]]
        Input parameter used by this function.
    summary : dict[str, Any]
        Input parameter used by this function.
    csv_name : str
        Input parameter used by this function.
    json_name : str
        Input parameter used by this function.
    
    Returns
    -----
    tuple[Path, Path]
        Value produced by this function call.
    
    Examples
    -----
    ```python
    from reaxkit.core.study.io import write_named_status
    # Configure required arguments for your case.
    result = write_named_status(...)
    print(type(result).__name__)
    ```
    Sample output:
    ```text
    str
    ```
    The output type reflects the return contract for this API call.
    """
    json_path = study_root / json_name
    csv_path = study_root / csv_name
    payload = {
        "generated_at_utc": utc_now(),
        "summary": summary,
        "rows": rows,
    }
    write_json(json_path, payload)

    if rows:
        fieldnames = list(rows[0].keys())
    else:
        fieldnames = []
    with csv_path.open("w", encoding="utf-8", newline="") as fh:
        if fieldnames:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow({k: row.get(k) for k in fieldnames})
        else:
            fh.write("")
    return csv_path, json_path


def write_study_run_status(
    *,
    study_root: Path,
    rows: list[dict[str, Any]],
    summary: dict[str, Any],
    run_status_json_file: str,
    run_status_csv_file: str,
) -> tuple[Path, Path]:
    """
    Write study run status.
    
    This function is part of the ReaxKit core API and performs the operation described by its name and arguments.
    
    Parameters
    -----
    study_root : Path
        Input parameter used by this function.
    rows : list[dict[str, Any]]
        Input parameter used by this function.
    summary : dict[str, Any]
        Input parameter used by this function.
    run_status_json_file : str
        Input parameter used by this function.
    run_status_csv_file : str
        Input parameter used by this function.
    
    Returns
    -----
    tuple[Path, Path]
        Value produced by this function call.
    
    Examples
    -----
    ```python
    from reaxkit.core.study.io import write_study_run_status
    # Configure required arguments for your case.
    result = write_study_run_status(...)
    print(type(result).__name__)
    ```
    Sample output:
    ```text
    str
    ```
    The output type reflects the return contract for this API call.
    """
    status_dir = study_root
    status_dir.mkdir(parents=True, exist_ok=True)

    json_path = status_dir / run_status_json_file
    csv_path = status_dir / run_status_csv_file

    payload = {
        "generated_at_utc": utc_now(),
        "summary": summary,
        "rows": rows,
    }
    write_json(json_path, payload)

    fieldnames = [
        "case_id",
        "replicate_id",
        "status",
        "run",
        "done",
        "skip",
        "wait",
        "fail",
        "started_at",
        "finished_at",
        "duration_min",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in fieldnames})

    return json_path, csv_path


def run_stage_manifest_path(stage_dir: Path, *, run_stage_manifest_file: str, legacy_stage_manifest_file: str) -> Path:
    """
    Run stage manifest path.
    
    This function is part of the ReaxKit core API and performs the operation described by its name and arguments.
    
    Parameters
    -----
    stage_dir : Path
        Input parameter used by this function.
    run_stage_manifest_file : str
        Input parameter used by this function.
    legacy_stage_manifest_file : str
        Input parameter used by this function.
    
    Returns
    -----
    Path
        Value produced by this function call.
    
    Examples
    -----
    ```python
    from reaxkit.core.study.io import run_stage_manifest_path
    # Configure required arguments for your case.
    result = run_stage_manifest_path(...)
    print(type(result).__name__)
    ```
    Sample output:
    ```text
    str
    ```
    The output type reflects the return contract for this API call.
    """
    return resolve_existing_file(stage_dir / run_stage_manifest_file, legacy_stage_manifest_file)


def run_replicate_manifest_path(
    rep_dir: Path, *, run_replicate_manifest_file: str, legacy_replicate_manifest_file: str
) -> Path:
    """
    Run replicate manifest path.
    
    This function is part of the ReaxKit core API and performs the operation described by its name and arguments.
    
    Parameters
    -----
    rep_dir : Path
        Input parameter used by this function.
    run_replicate_manifest_file : str
        Input parameter used by this function.
    legacy_replicate_manifest_file : str
        Input parameter used by this function.
    
    Returns
    -----
    Path
        Value produced by this function call.
    
    Examples
    -----
    ```python
    from reaxkit.core.study.io import run_replicate_manifest_path
    # Configure required arguments for your case.
    result = run_replicate_manifest_path(...)
    print(type(result).__name__)
    ```
    Sample output:
    ```text
    str
    ```
    The output type reflects the return contract for this API call.
    """
    return resolve_existing_file(rep_dir / run_replicate_manifest_file, legacy_replicate_manifest_file)


def run_case_manifest_path(case_dir: Path, *, run_case_manifest_file: str, legacy_case_manifest_file: str) -> Path:
    """
    Run case manifest path.
    
    This function is part of the ReaxKit core API and performs the operation described by its name and arguments.
    
    Parameters
    -----
    case_dir : Path
        Input parameter used by this function.
    run_case_manifest_file : str
        Input parameter used by this function.
    legacy_case_manifest_file : str
        Input parameter used by this function.
    
    Returns
    -----
    Path
        Value produced by this function call.
    
    Examples
    -----
    ```python
    from reaxkit.core.study.io import run_case_manifest_path
    # Configure required arguments for your case.
    result = run_case_manifest_path(...)
    print(type(result).__name__)
    ```
    Sample output:
    ```text
    str
    ```
    The output type reflects the return contract for this API call.
    """
    return resolve_existing_file(case_dir / run_case_manifest_file, legacy_case_manifest_file)


def run_status_csv_path(study_root: Path, *, run_status_csv_file: str, legacy_run_status_csv_file: str) -> Path:
    """
    Run status csv path.
    
    This function is part of the ReaxKit core API and performs the operation described by its name and arguments.
    
    Parameters
    -----
    study_root : Path
        Input parameter used by this function.
    run_status_csv_file : str
        Input parameter used by this function.
    legacy_run_status_csv_file : str
        Input parameter used by this function.
    
    Returns
    -----
    Path
        Value produced by this function call.
    
    Examples
    -----
    ```python
    from reaxkit.core.study.io import run_status_csv_path
    # Configure required arguments for your case.
    result = run_status_csv_path(...)
    print(type(result).__name__)
    ```
    Sample output:
    ```text
    str
    ```
    The output type reflects the return contract for this API call.
    """
    return resolve_existing_file(study_root / run_status_csv_file, legacy_run_status_csv_file)


def analysis_status_csv_path(study_root: Path, *, analysis_status_csv_file: str, legacy_analysis_status_csv_file: str) -> Path:
    """
    Analysis status csv path.
    
    This function is part of the ReaxKit core API and performs the operation described by its name and arguments.
    
    Parameters
    -----
    study_root : Path
        Input parameter used by this function.
    analysis_status_csv_file : str
        Input parameter used by this function.
    legacy_analysis_status_csv_file : str
        Input parameter used by this function.
    
    Returns
    -----
    Path
        Value produced by this function call.
    
    Examples
    -----
    ```python
    from reaxkit.core.study.io import analysis_status_csv_path
    # Configure required arguments for your case.
    result = analysis_status_csv_path(...)
    print(type(result).__name__)
    ```
    Sample output:
    ```text
    str
    ```
    The output type reflects the return contract for this API call.
    """
    return resolve_existing_file(study_root / analysis_status_csv_file, legacy_analysis_status_csv_file)


def analysis_status_json_path(
    study_root: Path, *, analysis_status_json_file: str, legacy_analysis_status_json_file: str
) -> Path:
    """
    Analysis status json path.
    
    This function is part of the ReaxKit core API and performs the operation described by its name and arguments.
    
    Parameters
    -----
    study_root : Path
        Input parameter used by this function.
    analysis_status_json_file : str
        Input parameter used by this function.
    legacy_analysis_status_json_file : str
        Input parameter used by this function.
    
    Returns
    -----
    Path
        Value produced by this function call.
    
    Examples
    -----
    ```python
    from reaxkit.core.study.io import analysis_status_json_path
    # Configure required arguments for your case.
    result = analysis_status_json_path(...)
    print(type(result).__name__)
    ```
    Sample output:
    ```text
    str
    ```
    The output type reflects the return contract for this API call.
    """
    return resolve_existing_file(study_root / analysis_status_json_file, legacy_analysis_status_json_file)


def load_status_rows(csv_path: Path, *, not_found_message: str) -> list[dict[str, str]]:
    """
    Load status rows.
    
    This function is part of the ReaxKit core API and performs the operation described by its name and arguments.
    
    Parameters
    -----
    csv_path : Path
        Input parameter used by this function.
    not_found_message : str
        Input parameter used by this function.
    
    Returns
    -----
    list[dict[str, str]]
        Value produced by this function call.
    
    Examples
    -----
    ```python
    from reaxkit.core.study.io import load_status_rows
    # Configure required arguments for your case.
    result = load_status_rows(...)
    print(type(result).__name__)
    ```
    Sample output:
    ```text
    str
    ```
    The output type reflects the return contract for this API call.
    """
    if not csv_path.exists():
        raise FileNotFoundError(not_found_message)
    with csv_path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        return [dict(row) for row in reader]


def stage_status_path(stage_dir: Path, *, stage_status_file: str) -> Path:
    """
    Stage status path.
    
    This function is part of the ReaxKit core API and performs the operation described by its name and arguments.
    
    Parameters
    -----
    stage_dir : Path
        Input parameter used by this function.
    stage_status_file : str
        Input parameter used by this function.
    
    Returns
    -----
    Path
        Value produced by this function call.
    
    Examples
    -----
    ```python
    from reaxkit.core.study.io import stage_status_path
    # Configure required arguments for your case.
    result = stage_status_path(...)
    print(type(result).__name__)
    ```
    Sample output:
    ```text
    str
    ```
    The output type reflects the return contract for this API call.
    """
    return stage_dir / stage_status_file


def load_stage_status(stage_dir: Path, *, stage_status_file: str) -> dict[str, Any]:
    """
    Load stage status.
    
    This function is part of the ReaxKit core API and performs the operation described by its name and arguments.
    
    Parameters
    -----
    stage_dir : Path
        Input parameter used by this function.
    stage_status_file : str
        Input parameter used by this function.
    
    Returns
    -----
    dict[str, Any]
        Value produced by this function call.
    
    Examples
    -----
    ```python
    from reaxkit.core.study.io import load_stage_status
    # Configure required arguments for your case.
    result = load_stage_status(...)
    print(type(result).__name__)
    ```
    Sample output:
    ```text
    str
    ```
    The output type reflects the return contract for this API call.
    """
    path = stage_status_path(stage_dir, stage_status_file=stage_status_file)
    if not path.exists():
        return {"status": "pending", "jobs": [], "updated_at_utc": None}
    payload = read_json(path)
    payload.setdefault("jobs", [])
    payload.setdefault("status", "pending")
    return payload


def write_stage_status(stage_dir: Path, status: dict[str, Any], *, stage_status_file: str) -> None:
    """
    Write stage status.
    
    This function is part of the ReaxKit core API and performs the operation described by its name and arguments.
    
    Parameters
    -----
    stage_dir : Path
        Input parameter used by this function.
    status : dict[str, Any]
        Input parameter used by this function.
    stage_status_file : str
        Input parameter used by this function.
    
    Returns
    -----
    None
        Value produced by this function call.
    
    Examples
    -----
    ```python
    from reaxkit.core.study.io import write_stage_status
    # Configure required arguments for your case.
    result = write_stage_status(...)
    print(type(result).__name__)
    ```
    Sample output:
    ```text
    str
    ```
    The output type reflects the return contract for this API call.
    """
    payload = dict(status)
    payload["updated_at_utc"] = utc_now()
    write_json(stage_status_path(stage_dir, stage_status_file=stage_status_file), payload)


def analysis_status_path(stage_dir: Path, *, analysis_status_file: str) -> Path:
    """
    Analysis status path.
    
    This function is part of the ReaxKit core API and performs the operation described by its name and arguments.
    
    Parameters
    -----
    stage_dir : Path
        Input parameter used by this function.
    analysis_status_file : str
        Input parameter used by this function.
    
    Returns
    -----
    Path
        Value produced by this function call.
    
    Examples
    -----
    ```python
    from reaxkit.core.study.io import analysis_status_path
    # Configure required arguments for your case.
    result = analysis_status_path(...)
    print(type(result).__name__)
    ```
    Sample output:
    ```text
    str
    ```
    The output type reflects the return contract for this API call.
    """
    return stage_dir / analysis_status_file


def load_analysis_status(stage_dir: Path, *, analysis_status_file: str) -> dict[str, Any]:
    """
    Load analysis status.
    
    This function is part of the ReaxKit core API and performs the operation described by its name and arguments.
    
    Parameters
    -----
    stage_dir : Path
        Input parameter used by this function.
    analysis_status_file : str
        Input parameter used by this function.
    
    Returns
    -----
    dict[str, Any]
        Value produced by this function call.
    
    Examples
    -----
    ```python
    from reaxkit.core.study.io import load_analysis_status
    # Configure required arguments for your case.
    result = load_analysis_status(...)
    print(type(result).__name__)
    ```
    Sample output:
    ```text
    str
    ```
    The output type reflects the return contract for this API call.
    """
    path = analysis_status_path(stage_dir, analysis_status_file=analysis_status_file)
    if not path.exists():
        return {"analyses": {}, "updated_at_utc": None}
    payload = read_json(path)
    if not isinstance(payload.get("analyses"), dict):
        payload["analyses"] = {}
    return payload


def write_analysis_status(stage_dir: Path, status: dict[str, Any], *, analysis_status_file: str) -> None:
    """
    Write analysis status.
    
    This function is part of the ReaxKit core API and performs the operation described by its name and arguments.
    
    Parameters
    -----
    stage_dir : Path
        Input parameter used by this function.
    status : dict[str, Any]
        Input parameter used by this function.
    analysis_status_file : str
        Input parameter used by this function.
    
    Returns
    -----
    None
        Value produced by this function call.
    
    Examples
    -----
    ```python
    from reaxkit.core.study.io import write_analysis_status
    # Configure required arguments for your case.
    result = write_analysis_status(...)
    print(type(result).__name__)
    ```
    Sample output:
    ```text
    str
    ```
    The output type reflects the return contract for this API call.
    """
    payload = dict(status)
    payload["updated_at_utc"] = utc_now()
    write_json(analysis_status_path(stage_dir, analysis_status_file=analysis_status_file), payload)


def analysis_manifest_path(stage_dir: Path, *, analysis_manifest_file: str) -> Path:
    """
    Analysis manifest path.
    
    This function is part of the ReaxKit core API and performs the operation described by its name and arguments.
    
    Parameters
    -----
    stage_dir : Path
        Input parameter used by this function.
    analysis_manifest_file : str
        Input parameter used by this function.
    
    Returns
    -----
    Path
        Value produced by this function call.
    
    Examples
    -----
    ```python
    from reaxkit.core.study.io import analysis_manifest_path
    # Configure required arguments for your case.
    result = analysis_manifest_path(...)
    print(type(result).__name__)
    ```
    Sample output:
    ```text
    str
    ```
    The output type reflects the return contract for this API call.
    """
    return stage_dir / analysis_manifest_file


def load_analysis_manifest(stage_dir: Path, *, analysis_manifest_file: str) -> dict[str, Any]:
    """
    Load analysis manifest.
    
    This function is part of the ReaxKit core API and performs the operation described by its name and arguments.
    
    Parameters
    -----
    stage_dir : Path
        Input parameter used by this function.
    analysis_manifest_file : str
        Input parameter used by this function.
    
    Returns
    -----
    dict[str, Any]
        Value produced by this function call.
    
    Examples
    -----
    ```python
    from reaxkit.core.study.io import load_analysis_manifest
    # Configure required arguments for your case.
    result = load_analysis_manifest(...)
    print(type(result).__name__)
    ```
    Sample output:
    ```text
    str
    ```
    The output type reflects the return contract for this API call.
    """
    path = analysis_manifest_path(stage_dir, analysis_manifest_file=analysis_manifest_file)
    if not path.exists():
        return {"runs": [], "updated_at_utc": None}
    payload = read_json(path)
    runs = payload.get("runs")
    if not isinstance(runs, list):
        payload["runs"] = []
    return payload


def write_analysis_manifest(stage_dir: Path, payload: dict[str, Any], *, analysis_manifest_file: str) -> None:
    """
    Write analysis manifest.
    
    This function is part of the ReaxKit core API and performs the operation described by its name and arguments.
    
    Parameters
    -----
    stage_dir : Path
        Input parameter used by this function.
    payload : dict[str, Any]
        Input parameter used by this function.
    analysis_manifest_file : str
        Input parameter used by this function.
    
    Returns
    -----
    None
        Value produced by this function call.
    
    Examples
    -----
    ```python
    from reaxkit.core.study.io import write_analysis_manifest
    # Configure required arguments for your case.
    result = write_analysis_manifest(...)
    print(type(result).__name__)
    ```
    Sample output:
    ```text
    str
    ```
    The output type reflects the return contract for this API call.
    """
    out = dict(payload)
    out["updated_at_utc"] = utc_now()
    write_json(analysis_manifest_path(stage_dir, analysis_manifest_file=analysis_manifest_file), out)

