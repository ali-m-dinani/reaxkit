"""
Analyze orchestration for studies.

**Usage context**

- Import these helpers from ReaxKit core modules when implementing CLI and workflow logic.
- Reuse the public APIs here to keep behavior consistent across commands and engines.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Callable


def analyze_study(
    *,
    study_root: Path,
    analysis_filter: str | None,
    case_filter: str | None,
    replicate_filter: str | None,
    strict_actions: bool,
    rerun_failed: bool,
    read_json_fn: Callable[[Path], dict[str, Any]],
    load_study_yaml_fn: Callable[[Path], dict[str, Any]],
    validate_study_fn: Callable[[dict[str, Any]], tuple[Any, Any, Any, Any, list[Any], Any]],
    analysis_label_width_fn: Callable[[int], int],
    load_analysis_status_rows_fn: Callable[[Path], list[dict[str, str]]],
    to_int_fn: Callable[[Any], int],
    case_matches_selector_fn: Callable[[dict[str, Any], str | None], bool],
    replicate_matches_selector_fn: Callable[[str, str | None], bool],
    run_case_manifest_path_fn: Callable[[Path], Path],
    run_replicate_manifest_path_fn: Callable[[Path], Path],
    load_stage_status_fn: Callable[[Path], dict[str, Any]],
    render_value_fn: Callable[[Any, dict[str, Any]], Any],
    utc_now_fn: Callable[[], str],
    local_now_fn: Callable[[], str],
    run_analysis_steps_fn: Callable[..., list[dict[str, Any]]],
    duration_minutes_fn: Callable[[str | None, str | None], float | None],
    collect_result_dirs_from_step_records_fn: Callable[[list[dict[str, Any]]], list[str]],
    load_analysis_status_fn: Callable[[Path], dict[str, Any]],
    write_analysis_status_fn: Callable[[Path, dict[str, Any]], None],
    load_analysis_manifest_fn: Callable[[Path], dict[str, Any]],
    write_analysis_manifest_fn: Callable[[Path, dict[str, Any]], None],
    log_stage_event_fn: Callable[..., None],
    analysis_status_json_file: str,
    analysis_status_csv_file: str,
    write_json_fn: Callable[[Path, dict[str, Any]], None],
) -> dict[str, Any]:
    """
    Analyze study.
    
    This function is part of the ReaxKit core API and performs the operation described by its name and arguments.
    
    Parameters
    -----
    study_root : Path
        Input parameter used by this function.
    analysis_filter : str | None
        Input parameter used by this function.
    case_filter : str | None
        Input parameter used by this function.
    replicate_filter : str | None
        Input parameter used by this function.
    strict_actions : bool
        Input parameter used by this function.
    rerun_failed : bool
        Input parameter used by this function.
    read_json_fn : Callable[[Path], dict[str, Any]]
        Input parameter used by this function.
    load_study_yaml_fn : Callable[[Path], dict[str, Any]]
        Input parameter used by this function.
    validate_study_fn : Callable[[dict[str, Any]], tuple[Any, Any, Any, Any, list[Any], Any]]
        Input parameter used by this function.
    analysis_label_width_fn : Callable[[int], int]
        Input parameter used by this function.
    load_analysis_status_rows_fn : Callable[[Path], list[dict[str, str]]]
        Input parameter used by this function.
    to_int_fn : Callable[[Any], int]
        Input parameter used by this function.
    case_matches_selector_fn : Callable[[dict[str, Any], str | None], bool]
        Input parameter used by this function.
    replicate_matches_selector_fn : Callable[[str, str | None], bool]
        Input parameter used by this function.
    run_case_manifest_path_fn : Callable[[Path], Path]
        Input parameter used by this function.
    run_replicate_manifest_path_fn : Callable[[Path], Path]
        Input parameter used by this function.
    load_stage_status_fn : Callable[[Path], dict[str, Any]]
        Input parameter used by this function.
    render_value_fn : Callable[[Any, dict[str, Any]], Any]
        Input parameter used by this function.
    utc_now_fn : Callable[[], str]
        Input parameter used by this function.
    local_now_fn : Callable[[], str]
        Input parameter used by this function.
    run_analysis_steps_fn : Callable[..., list[dict[str, Any]]]
        Input parameter used by this function.
    duration_minutes_fn : Callable[[str | None, str | None], float | None]
        Input parameter used by this function.
    collect_result_dirs_from_step_records_fn : Callable[[list[dict[str, Any]]], list[str]]
        Input parameter used by this function.
    load_analysis_status_fn : Callable[[Path], dict[str, Any]]
        Input parameter used by this function.
    write_analysis_status_fn : Callable[[Path, dict[str, Any]], None]
        Input parameter used by this function.
    load_analysis_manifest_fn : Callable[[Path], dict[str, Any]]
        Input parameter used by this function.
    write_analysis_manifest_fn : Callable[[Path, dict[str, Any]], None]
        Input parameter used by this function.
    log_stage_event_fn : Callable[..., None]
        Input parameter used by this function.
    analysis_status_json_file : str
        Input parameter used by this function.
    analysis_status_csv_file : str
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
    from reaxkit.core.study.analyze_engine import analyze_study
    # Configure required arguments for your case.
    result = analyze_study(...)
    print(type(result).__name__)
    ```
    Sample output:
    ```text
    str
    ```
    The output type reflects the return contract for this API call.
    """
    manifest = read_json_fn(study_root / "study_manifest.json")
    source_yaml = Path(str(manifest.get("source_yaml") or "")).resolve()
    if not source_yaml.exists():
        raise FileNotFoundError(f"Study source YAML not found: {source_yaml}")
    study_doc = load_study_yaml_fn(source_yaml)
    _, _, _, _, analyses, _ = validate_study_fn(study_doc)
    analysis_defs: list[dict[str, Any]] = []
    for entry in analyses:
        if analysis_filter and entry.title != analysis_filter:
            continue
        analysis_defs.append(
            {
                "analysis_id": entry.analysis_id,
                "title": entry.title,
                "run_stage": entry.run_stage,
                "payload": entry.payload,
            }
        )

    if not analysis_defs:
        raise ValueError("No analysis entries found to run. Check study 'analysis' and --analysis filter.")
    max_analysis_chars = 0
    for entry in analysis_defs:
        t = str(entry.get("title") or "").strip()
        if len(t) > max_analysis_chars:
            max_analysis_chars = len(t)
    analysis_block_width = analysis_label_width_fn(max_analysis_chars=max_analysis_chars)

    records: list[dict[str, Any]] = []
    counts = {"completed": 0, "failed": 0, "skipped": 0}
    study_dir = source_yaml.parent
    rerun_targets: set[tuple[str, str, str]] | None = None
    if rerun_failed:
        rerun_targets = set()
        for row in load_analysis_status_rows_fn(study_root):
            case_id = str(row.get("case_id") or "").strip()
            rep_id = str(row.get("replicate_id") or "").strip()
            analysis_id = str(row.get("analysis_id") or "").strip()
            fail_n = to_int_fn(row.get("fail"))
            wait_n = to_int_fn(row.get("wait"))
            status_v = str(row.get("status") or "").strip().lower()
            if not case_id or not rep_id or not analysis_id:
                continue
            if fail_n > 0 or wait_n > 0 or status_v in {"failed", "waiting"}:
                rerun_targets.add((case_id, rep_id, analysis_id))
        if not rerun_targets:
            print("[info] No failed/waiting analysis entries found in analysis_status.csv.")

    def _analysis_counters(*, status: str, reason: str | None = None) -> dict[str, int]:
        s = str(status or "").strip().lower()
        r = str(reason or "").strip().lower()
        run = 1 if s in {"completed", "failed"} else 0
        done = 1 if s == "completed" else 0
        fail = 1 if s == "failed" else 0
        wait = 1 if (s == "skipped" and r.startswith("run_stage_not_completed")) else 0
        skip = 1 if (s == "skipped" and wait == 0) else 0
        return {"run": run, "done": done, "skip": skip, "wait": wait, "fail": fail}

    for case in manifest.get("cases") or []:
        if not isinstance(case, dict):
            continue
        if not case_matches_selector_fn(case, case_filter):
            continue
        case_id = str(case.get("case_id") or "")
        case_path = Path(str(case.get("path") or ""))
        case_manifest = read_json_fn(run_case_manifest_path_fn(case_path))
        for rep in case_manifest.get("replicates") or []:
            if not isinstance(rep, dict):
                continue
            rep_id = str(rep.get("replicate_id") or "")
            if not replicate_matches_selector_fn(rep_id, replicate_filter):
                continue
            rep_path = Path(str(rep.get("path") or ""))
            rep_manifest = read_json_fn(run_replicate_manifest_path_fn(rep_path))
            stage_entries = rep_manifest.get("stages") or []
            stage_entry_by_name = {
                str(entry.get("stage") or "").strip(): entry for entry in stage_entries if isinstance(entry, dict)
            }
            context_base = {
                **(rep_manifest.get("parameters") if isinstance(rep_manifest.get("parameters"), dict) else {}),
                "study_name": str(manifest.get("study_name") or ""),
                "case_id": case_id,
                "replicate_id": rep_id,
            }
            for analysis_def in analysis_defs:
                analysis_id = str(analysis_def.get("analysis_id") or "").strip()
                title = str(analysis_def.get("title") or "").strip()
                run_stage = str(analysis_def.get("run_stage") or "").strip()
                if rerun_targets is not None and (case_id, rep_id, analysis_id) not in rerun_targets:
                    continue
                stage_entry = stage_entry_by_name.get(run_stage)
                if stage_entry is None:
                    counts["skipped"] += 1
                    status_value = "skipped"
                    reason = "run_stage_not_found"
                    ctr = _analysis_counters(status=status_value, reason=reason)
                    log_stage_event_fn(case_id, rep_id, title, "SKIP", reason, stage_block_width=analysis_block_width)
                    records.append(
                        {
                            "case_id": case_id,
                            "replicate_id": rep_id,
                            "analysis_id": analysis_id,
                            "title": title,
                            "run_stage": run_stage,
                            "status": status_value,
                            "reason": reason,
                            **ctr,
                            "started_at": None,
                            "finished_at": None,
                            "duration_min": None,
                        }
                    )
                    continue

                stage_dir = Path(str(stage_entry.get("path") or ""))
                stage_status = load_stage_status_fn(stage_dir)
                if str(stage_status.get("status") or "") != "completed":
                    counts["skipped"] += 1
                    status_value = "skipped"
                    reason = f"run_stage_not_completed:{stage_status.get('status')}"
                    ctr = _analysis_counters(status=status_value, reason=reason)
                    log_stage_event_fn(case_id, rep_id, title, "WAIT", reason, stage_block_width=analysis_block_width)
                    records.append(
                        {
                            "case_id": case_id,
                            "replicate_id": rep_id,
                            "analysis_id": analysis_id,
                            "title": title,
                            "run_stage": run_stage,
                            "stage_path": str(stage_dir),
                            "status": status_value,
                            "reason": reason,
                            **ctr,
                            "started_at": None,
                            "finished_at": None,
                            "duration_min": None,
                        }
                    )
                    continue

                context = dict(context_base)
                context["stage"] = run_stage
                rendered_analysis = render_value_fn((analysis_def.get("payload") or analysis_def), context)
                started = utc_now_fn()
                started_local = local_now_fn()
                log_stage_event_fn(case_id, rep_id, title, "RUN", stage_block_width=analysis_block_width)
                try:
                    step_records = run_analysis_steps_fn(
                        stage_dir=stage_dir,
                        rendered_analysis=rendered_analysis if isinstance(rendered_analysis, dict) else {},
                        study_dir=study_dir,
                    )
                    failed = any(str(r.get("status") or "").lower() == "failed" for r in step_records)
                    status_value = "failed" if failed else "completed"
                    reason = None if not failed else "step_failed"
                except Exception as exc:
                    step_records = []
                    status_value = "failed"
                    reason = str(exc)

                finished = utc_now_fn()
                finished_local = local_now_fn()
                duration_min = duration_minutes_fn(started_local, finished_local)
                result_dirs = collect_result_dirs_from_step_records_fn(step_records)
                analysis_status = load_analysis_status_fn(stage_dir)
                analysis_status.setdefault("analyses", {})
                analysis_status["analyses"][analysis_id] = {
                    "status": status_value,
                    "title": title,
                    "run_stage": run_stage,
                    "started_at_utc": started,
                    "finished_at_utc": finished,
                    "result_dirs": result_dirs,
                }
                write_analysis_status_fn(stage_dir, analysis_status)
                analysis_manifest = load_analysis_manifest_fn(stage_dir)
                analysis_manifest.setdefault("runs", [])
                analysis_manifest["runs"].append(
                    {
                        "analysis_id": analysis_id,
                        "title": title,
                        "run_stage": run_stage,
                        "status": status_value,
                        "reason": reason,
                        "started_at_utc": started,
                        "finished_at_utc": finished,
                        "started_at": started_local,
                        "finished_at": finished_local,
                        "duration_min": duration_min,
                        "result_dirs": result_dirs,
                        "step_records": step_records,
                    }
                )
                write_analysis_manifest_fn(stage_dir, analysis_manifest)
                ctr = _analysis_counters(status=status_value, reason=reason)

                rec = {
                    "case_id": case_id,
                    "replicate_id": rep_id,
                    "analysis_id": analysis_id,
                    "title": title,
                    "run_stage": run_stage,
                    "stage_path": str(stage_dir),
                    "status": status_value,
                    "reason": reason,
                    "n_steps": len(step_records),
                    "result_dirs": result_dirs,
                    **ctr,
                    "started_at": started_local,
                    "finished_at": finished_local,
                    "duration_min": duration_min,
                }
                records.append(rec)
                if status_value == "completed":
                    counts["completed"] += 1
                    log_stage_event_fn(case_id, rep_id, title, "DONE", stage_block_width=analysis_block_width)
                else:
                    counts["failed"] += 1
                    log_stage_event_fn(case_id, rep_id, title, "FAIL", stage_block_width=analysis_block_width)
                    if strict_actions:
                        break
            if strict_actions and counts["failed"] > 0:
                break
        if strict_actions and counts["failed"] > 0:
            break

    out_dir = study_root
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "study_root": str(study_root),
        "generated_at_utc": utc_now_fn(),
        "counts": counts,
        "records": records,
        "analysis_filter": analysis_filter,
    }
    manifest_path = out_dir / analysis_status_json_file
    write_json_fn(manifest_path, payload)

    csv_path = out_dir / analysis_status_csv_file
    fieldnames = [
        "case_id",
        "replicate_id",
        "analysis_id",
        "title",
        "run_stage",
        "stage_path",
        "status",
        "run",
        "done",
        "skip",
        "wait",
        "fail",
        "started_at",
        "finished_at",
        "duration_min",
        "reason",
        "n_steps",
        "result_dirs",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for rec in records:
            writer.writerow({k: rec.get(k) for k in fieldnames})

    return {"counts": counts, "manifest": manifest_path, "csv": csv_path}

