"""
Run orchestration for studies.

**Usage context**

- Import these helpers from ReaxKit core modules when implementing CLI and workflow logic.
- Reuse the public APIs here to keep behavior consistent across commands and engines.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable


def run_study(
    *,
    study_root: Path,
    stage_filter: str | None,
    case_filter: str | None,
    replicate_filter: str | None,
    artifact_transfer: str,
    run_geometry_generator: bool,
    strict_actions: bool,
    parallel_workers: int,
    rerun_failed: bool,
    read_json_fn: Callable[[Path], dict[str, Any]],
    load_study_run_status_rows_fn: Callable[[Path], list[dict[str, str]]],
    to_int_fn: Callable[[Any], int],
    case_matches_selector_fn: Callable[[dict[str, Any], str | None], bool],
    replicate_matches_selector_fn: Callable[[str, str | None], bool],
    run_case_manifest_path_fn: Callable[[Path], Path],
    run_single_replicate_pipeline_fn: Callable[..., dict[str, Any]],
    write_study_run_status_fn: Callable[..., tuple[Path, Path]],
) -> dict[str, int]:
    """
    Run study.
    
    This function is part of the ReaxKit core API and performs the operation described by its name and arguments.
    
    Parameters
    -----
    study_root : Path
        Input parameter used by this function.
    stage_filter : str | None
        Input parameter used by this function.
    case_filter : str | None
        Input parameter used by this function.
    replicate_filter : str | None
        Input parameter used by this function.
    artifact_transfer : str
        Input parameter used by this function.
    run_geometry_generator : bool
        Input parameter used by this function.
    strict_actions : bool
        Input parameter used by this function.
    parallel_workers : int
        Input parameter used by this function.
    rerun_failed : bool
        Input parameter used by this function.
    read_json_fn : Callable[[Path], dict[str, Any]]
        Input parameter used by this function.
    load_study_run_status_rows_fn : Callable[[Path], list[dict[str, str]]]
        Input parameter used by this function.
    to_int_fn : Callable[[Any], int]
        Input parameter used by this function.
    case_matches_selector_fn : Callable[[dict[str, Any], str | None], bool]
        Input parameter used by this function.
    replicate_matches_selector_fn : Callable[[str, str | None], bool]
        Input parameter used by this function.
    run_case_manifest_path_fn : Callable[[Path], Path]
        Input parameter used by this function.
    run_single_replicate_pipeline_fn : Callable[..., dict[str, Any]]
        Input parameter used by this function.
    write_study_run_status_fn : Callable[..., tuple[Path, Path]]
        Input parameter used by this function.
    
    Returns
    -----
    dict[str, int]
        Value produced by this function call.
    
    Examples
    -----
    ```python
    from reaxkit.core.study.run_engine import run_study
    # Configure required arguments for your case.
    result = run_study(...)
    print(type(result).__name__)
    ```
    Sample output:
    ```text
    str
    ```
    The output type reflects the return contract for this API call.
    """
    manifest = read_json_fn(study_root / "study_manifest.json")
    case_entries = manifest.get("cases") or []
    if not isinstance(case_entries, list):
        raise ValueError("Invalid study_manifest.json: cases must be a list.")

    counts = {"completed": 0, "skipped": 0, "failed": 0, "not_ready": 0}
    tasks: list[tuple[dict[str, Any], dict[str, Any]]] = []
    row_map: dict[tuple[str, str], dict[str, Any]] = {}

    rerun_targets: set[tuple[str, str]] | None = None
    if rerun_failed:
        rerun_targets = set()
        for row in load_study_run_status_rows_fn(study_root):
            case_id = str(row.get("case_id") or "").strip()
            rep_id = str(row.get("replicate_id") or "").strip()
            fail_n = to_int_fn(row.get("fail"))
            wait_n = to_int_fn(row.get("wait"))
            status = str(row.get("status") or "").strip().lower()
            if not case_id or not rep_id:
                continue
            if fail_n > 0 or wait_n > 0 or status in {"failed", "waiting"}:
                rerun_targets.add((case_id, rep_id))

    for case in case_entries:
        if not isinstance(case, dict):
            continue
        if not case_matches_selector_fn(case, case_filter):
            continue
        case_id = str(case.get("case_id") or "")
        case_path = Path(str(case.get("path") or ""))
        case_manifest = read_json_fn(run_case_manifest_path_fn(case_path))
        reps = case_manifest.get("replicates") or []
        for rep in reps:
            if not isinstance(rep, dict):
                continue
            rep_id = str(rep.get("replicate_id") or "")
            if not replicate_matches_selector_fn(rep_id, replicate_filter):
                continue
            if rerun_targets is not None and (case_id, rep_id) not in rerun_targets:
                continue
            tasks.append((case, rep))
            row_map[(case_id, rep_id)] = {
                "case_id": case_id,
                "replicate_id": rep_id,
                "status": "pending",
                "run": 0,
                "done": 0,
                "skip": 0,
                "wait": 0,
                "fail": 0,
                "started_at": None,
                "finished_at": None,
                "duration_min": None,
            }

    def persist_status() -> None:
        rows = sorted(row_map.values(), key=lambda r: (str(r.get("case_id") or ""), str(r.get("replicate_id") or "")))
        summary = {
            "completed": counts["completed"],
            "skipped": counts["skipped"],
            "not_ready": counts["not_ready"],
            "failed": counts["failed"],
            "total_replicates": len(rows),
        }
        write_study_run_status_fn(study_root=study_root, rows=rows, summary=summary)

    if rerun_failed and not tasks:
        print("[info] No failed/waiting replicates found in run_status.csv.")
        persist_status()
        return counts

    persist_status()

    workers = max(1, int(parallel_workers))
    if strict_actions and workers > 1:
        print("[warn] --strict-actions with --parallel-workers>1 is downgraded to sequential execution.")
        workers = 1

    if workers == 1:
        for case, rep in tasks:
            rec = run_single_replicate_pipeline_fn(
                case=case,
                rep=rep,
                manifest=manifest,
                stage_filter=stage_filter,
                artifact_transfer=artifact_transfer,
                run_geometry_generator=run_geometry_generator,
                strict_actions=strict_actions,
                cleanup_before_stage=rerun_failed,
            )
            for key in counts:
                counts[key] += int(rec.get(key, 0))
            row_map[(str(rec.get("case_id") or ""), str(rec.get("replicate_id") or ""))] = {
                "case_id": str(rec.get("case_id") or ""),
                "replicate_id": str(rec.get("replicate_id") or ""),
                "status": str(rec.get("status") or "unknown"),
                "run": int(rec.get("run", 0)),
                "done": int(rec.get("done", 0)),
                "skip": int(rec.get("skip", 0)),
                "wait": int(rec.get("wait", 0)),
                "fail": int(rec.get("fail", 0)),
                "started_at": rec.get("started_at"),
                "finished_at": rec.get("finished_at"),
                "duration_min": rec.get("duration_min"),
            }
            persist_status()
            if strict_actions and rec.get("failed", 0) > 0:
                return counts
        return counts

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(
                run_single_replicate_pipeline_fn,
                case=case,
                rep=rep,
                manifest=manifest,
                stage_filter=stage_filter,
                artifact_transfer=artifact_transfer,
                run_geometry_generator=run_geometry_generator,
                strict_actions=False,
                cleanup_before_stage=rerun_failed,
            )
            for case, rep in tasks
        ]
        for future in as_completed(futures):
            rec = future.result()
            for key in counts:
                counts[key] += int(rec.get(key, 0))
            row_map[(str(rec.get("case_id") or ""), str(rec.get("replicate_id") or ""))] = {
                "case_id": str(rec.get("case_id") or ""),
                "replicate_id": str(rec.get("replicate_id") or ""),
                "status": str(rec.get("status") or "unknown"),
                "run": int(rec.get("run", 0)),
                "done": int(rec.get("done", 0)),
                "skip": int(rec.get("skip", 0)),
                "wait": int(rec.get("wait", 0)),
                "fail": int(rec.get("fail", 0)),
                "started_at": rec.get("started_at"),
                "finished_at": rec.get("finished_at"),
                "duration_min": rec.get("duration_min"),
            }
            persist_status()
    return counts

