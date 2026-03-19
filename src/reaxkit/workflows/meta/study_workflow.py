"""Study planning workflow: generate sweep/replicate folder structures from YAML."""

from __future__ import annotations

import argparse
import csv
import math
import shutil
import subprocess
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from reaxkit.core.study.logging import (
    analysis_label_width as _analysis_label_width,
    duration_minutes as _duration_minutes,
    local_now as _local_now,
    log_stage_event as _log_stage_event,
    log_task_event as _log_task_event,
    stage_label_width as _stage_label_width,
    utc_now as _utc_now,
)
from reaxkit.core.study.io import (
    analysis_manifest_path as _io_analysis_manifest_path,
    analysis_status_csv_path as _io_analysis_status_csv_path,
    analysis_status_path as _io_analysis_status_path,
    load_analysis_manifest as _io_load_analysis_manifest,
    load_analysis_status as _io_load_analysis_status,
    load_status_rows as _io_load_status_rows,
    load_stage_status as _io_load_stage_status,
    read_json as _io_read_json,
    resolve_existing_file as _io_resolve_existing_file,
    run_case_manifest_path as _io_run_case_manifest_path,
    run_replicate_manifest_path as _io_run_replicate_manifest_path,
    run_stage_manifest_path as _io_run_stage_manifest_path,
    run_status_csv_path as _io_run_status_csv_path,
    stage_status_path as _io_stage_status_path,
    write_analysis_manifest as _io_write_analysis_manifest,
    write_analysis_status as _io_write_analysis_status,
    write_json as _io_write_json,
    write_named_status as _io_write_named_status,
    write_stage_status as _io_write_stage_status,
    write_study_run_status as _io_write_study_run_status,
)
from reaxkit.core.study.init import (
    apply_stage_slurm_job_name as _init_apply_stage_slurm_job_name,
    collect_template_stage_relpaths as _init_collect_template_stage_relpaths,
    copy_template_into_replicate as _init_copy_template_into_replicate,
    enumerate_cases as _init_enumerate_cases,
    render_value as _init_render_value,
    resolve_template_path as _init_resolve_template_path,
)
from reaxkit.core.study.aggregate_engine import (
    apply_reducer_to_series as _agg_apply_reducer_to_series,
    compute_stats as _agg_compute_stats,
    aggregate_study_all as _engine_aggregate_study_all,
    aggregate_study_analysis as _engine_aggregate_study_analysis,
    extract_scalar_from_csv as _agg_extract_scalar_from_csv,
    load_column_values as _agg_load_column_values,
    load_csv_rows as _agg_load_csv_rows,
    normalize_stage_variables as _agg_normalize_stage_variables,
    resolve_variable_csv_path as _agg_resolve_variable_csv_path,
    resolve_variable_file_for_run as _agg_resolve_variable_file_for_run,
    safe_float as _agg_safe_float,
    sort_values as _agg_sort_values,
    stat_value as _agg_stat_value,
    to_plot_value as _agg_to_plot_value,
)
from reaxkit.core.study.analyze_engine import analyze_study as _engine_analyze_study
from reaxkit.core.study.present_engine import plot_study_aggregates as _engine_plot_study_aggregates
from reaxkit.core.study.present_engine import (
    make_all_cases_errorbar_plots as _present_make_all_cases_errorbar_plots,
    make_all_cases_heatmaps as _present_make_all_cases_heatmaps,
    make_all_cases_per_iter_boxplots as _present_make_all_cases_per_iter_boxplots,
    make_boxplots_for_case_aggregate as _present_make_boxplots_for_case_aggregate,
    make_errorbar_plots_for_case_aggregate as _present_make_errorbar_plots_for_case_aggregate,
    make_heatmap_from_rows as _present_make_heatmap_from_rows,
    row_key_from_params as _present_row_key_from_params,
    sort_plot_x as _present_sort_plot_x,
    find_param_columns as _present_find_param_columns,
)
from reaxkit.core.study.run_engine import run_study as _engine_run_study
from reaxkit.core.study.manage_engine import (
    rename_case_directories as _manage_rename_case_directories,
    update_study_directory_paths as _manage_update_study_directory_paths,
)
from reaxkit.core.study.naming import (
    canonical_token as _canonical_token,
    case_label_from_params as _case_label_from_params,
    slug as _slug,
    slug_underscore as _slug_underscore,
)
from reaxkit.core.study.schema import (
    AggregateDef,
    AnalysisDef,
    ArtifactRef,
    StageDef,
    aggregate_defs_from_doc as _aggregate_defs_from_doc,
    analysis_defs_from_doc as _analysis_defs_from_doc,
    load_source_study_doc as _load_source_study_doc,
    load_study_yaml as _load_study_yaml,
    validate_study as _validate_study,
)
from reaxkit.engine.reaxff.generators.control_generator import (
    ControlGeneratorSpec,
    write_control,
    write_control_template_with_overrides,
)
from reaxkit.presentation.plot.registry import plot as render_plot

STUDY_COMMAND = "study"
CONTROL_PARAMETER_ALIASES = {
    "temperature": "mdtemp",
}
DEFAULT_ARTIFACT_TRANSFER = "copy"
SUPPORTED_ARTIFACT_TRANSFER = {"copy", "symlink", "hardlink"}
STAGE_STATUS_FILE = ".stage_status.json"
ANALYSIS_STATUS_FILE = ".analysis_status.json"
ANALYSIS_MANIFEST_FILE = "analysis_manifest.json"
RUN_STAGE_MANIFEST_FILE = "run_stage_manifest.json"
RUN_REPLICATE_MANIFEST_FILE = "run_replicate_manifest.json"
RUN_CASE_MANIFEST_FILE = "run_case_manifest.json"
RUN_STATUS_CSV_FILE = "run_status.csv"
RUN_STATUS_JSON_FILE = "run_status.json"
ANALYSIS_STATUS_CSV_FILE = "analysis_status.csv"
ANALYSIS_STATUS_JSON_FILE = "analysis_status.json"
AGGREGATE_STATUS_CSV_FILE = "aggregate_status.csv"
AGGREGATE_STATUS_JSON_FILE = "aggregate_status.json"
PLOT_STATUS_CSV_FILE = "plot_status.csv"
PLOT_STATUS_JSON_FILE = "plot_status.json"
LEGACY_STAGE_MANIFEST_FILE = "stage_manifest.json"
LEGACY_REPLICATE_MANIFEST_FILE = "replicate_manifest.json"
LEGACY_CASE_MANIFEST_FILE = "case_manifest.json"
LEGACY_RUN_STATUS_CSV_FILE = "study_run_status.csv"
LEGACY_RUN_STATUS_JSON_FILE = "study_run_status.json"
LEGACY_ANALYSIS_STATUS_CSV_FILE = "study_analyze_results.csv"
LEGACY_ANALYSIS_STATUS_JSON_FILE = "study_analyze_manifest.json"
RR_CLEANUP_PATTERNS = [
    "job.*",
    "fort.*",
    "59s",
    "57s",
    "58s",
    "4s",
    "13s",
    "13s2",
    "vels23",
    "6*",
    "7*",
    "8*",
    "9*",
    "Kid*",
    "Unknown*",
    "*log",
    "*out",
    "*.recover",
    "*thermolog",
    "energy*",
    "reax*",
    "output*",
    "mol*",
    "xmolout",
    "summary.txt",
    "ffieldss",
]

STUDY_TEMPLATE_YAML = """study_name: mg_temp_sweep
template: "./template"
parameters:
  mg: [40, 60]     # Mg percent
  te: [300, 500]   # Temperature (K)
replicates: 2

run:
  - stage: MM
    steps:
      - "python hybrid_generator.py Z {mg_percent}"
      - "reaxkit xtob --file xmol_hybrid_sortby_Z.xyz --dims 11.37,13.12,450 --angles 90,90,90 --sort z --output geo --copy-to-dot"
      - "sbatch submit_and_wait.sh"
    produces:
      final_geometry: "./fort.90"

  - stage: NPT
    consumes:
      initial_geometry:
        from: MM.final_geometry
        to: ./geo
    steps:
      - "reaxkit write-control --parameter mdtemp --value {temperature} --output control --copy-to-dot"
      - "sbatch submit_and_wait.sh"
      - "python fix_charges.py 32"
    produces:

  - stage: NVT
    consumes:
    steps:
      - "reaxkit write-control --parameter mdtemp --value {temperature} --output control --copy-to-dot"
      - "sbatch submit_and_wait.sh"
    produces:

analysis:
  - title: msd_atom1
    run_stage: NVT
    steps:
      - "reaxkit msd --atom-ids 1 --export results.csv"
    variables:
      iter:
        directory: "reaxkit_workspace/analysis/msd"
        file: "results.csv"
        column: "iter"
      msd:
        directory: "reaxkit_workspace/analysis/msd"
        file: "results.csv"
        column: "msd"

aggregate:
  - title: msd_atom1_aggregation
    analysis_title: msd_atom1
    x: iter
    y: [msd]
    reducer: identity
    stats: [mean, std, min, max, sem, n]
    on_missing: skip
"""

def _write_study_run_status(
    *,
    study_root: Path,
    rows: list[dict[str, Any]],
    summary: dict[str, Any],
) -> tuple[Path, Path]:
    return _io_write_study_run_status(
        study_root=study_root,
        rows=rows,
        summary=summary,
        run_status_json_file=RUN_STATUS_JSON_FILE,
        run_status_csv_file=RUN_STATUS_CSV_FILE,
    )


def _write_named_status(
    *,
    study_root: Path,
    rows: list[dict[str, Any]],
    summary: dict[str, Any],
    csv_name: str,
    json_name: str,
) -> tuple[Path, Path]:
    return _io_write_named_status(
        study_root=study_root,
        rows=rows,
        summary=summary,
        csv_name=csv_name,
        json_name=json_name,
    )


def _resolve_existing_file(primary: Path, *fallbacks: str) -> Path:
    return _io_resolve_existing_file(primary, *fallbacks)


def _run_stage_manifest_path(stage_dir: Path) -> Path:
    return _io_run_stage_manifest_path(
        stage_dir, run_stage_manifest_file=RUN_STAGE_MANIFEST_FILE, legacy_stage_manifest_file=LEGACY_STAGE_MANIFEST_FILE
    )


def _run_replicate_manifest_path(rep_dir: Path) -> Path:
    return _io_run_replicate_manifest_path(
        rep_dir,
        run_replicate_manifest_file=RUN_REPLICATE_MANIFEST_FILE,
        legacy_replicate_manifest_file=LEGACY_REPLICATE_MANIFEST_FILE,
    )


def _run_case_manifest_path(case_dir: Path) -> Path:
    return _io_run_case_manifest_path(
        case_dir, run_case_manifest_file=RUN_CASE_MANIFEST_FILE, legacy_case_manifest_file=LEGACY_CASE_MANIFEST_FILE
    )


def _run_status_csv_path(study_root: Path) -> Path:
    return _io_run_status_csv_path(
        study_root, run_status_csv_file=RUN_STATUS_CSV_FILE, legacy_run_status_csv_file=LEGACY_RUN_STATUS_CSV_FILE
    )


def _analysis_status_csv_path(study_root: Path) -> Path:
    return _io_analysis_status_csv_path(
        study_root,
        analysis_status_csv_file=ANALYSIS_STATUS_CSV_FILE,
        legacy_analysis_status_csv_file=LEGACY_ANALYSIS_STATUS_CSV_FILE,
    )


def _load_study_run_status_rows(study_root: Path) -> list[dict[str, str]]:
    csv_path = _run_status_csv_path(study_root)
    return _io_load_status_rows(
        csv_path,
        not_found_message=(
            f"Cannot use rerun mode because status file does not exist: {csv_path}. "
            "Run 'reaxkit study --run <study_root>' once first."
        ),
    )


def _load_analysis_status_rows(study_root: Path) -> list[dict[str, str]]:
    csv_path = _analysis_status_csv_path(study_root)
    return _io_load_status_rows(
        csv_path,
        not_found_message=(
            f"Cannot use analyze rerun mode because status file does not exist: {csv_path}. "
            "Run 'reaxkit study --analyze <study_root>' once first."
        ),
    )


def _to_int(value: Any) -> int:
    try:
        return int(value)
    except Exception:
        return 0


def _cleanup_stage_artifacts(stage_dir: Path) -> int:
    removed = 0
    for pattern in RR_CLEANUP_PATTERNS:
        for target in stage_dir.glob(pattern):
            if target.name in {STAGE_STATUS_FILE, RUN_STAGE_MANIFEST_FILE, LEGACY_STAGE_MANIFEST_FILE}:
                continue
            try:
                if target.is_dir() and not target.is_symlink():
                    shutil.rmtree(target)
                else:
                    target.unlink()
                removed += 1
            except FileNotFoundError:
                continue
    return removed


def _render_value(value: Any, context: dict[str, Any]) -> Any:
    return _init_render_value(value, context)


def _enumerate_cases(parameters: dict[str, list[Any]]) -> list[dict[str, Any]]:
    return _init_enumerate_cases(parameters)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    _io_write_json(path, payload)


def _resolve_template_path(template_value: str, *, study_dir: Path) -> Path:
    return _init_resolve_template_path(template_value, study_dir=study_dir)


def _copy_template_into_replicate(*, template_root: Path, replicate_dir: Path) -> None:
    _init_copy_template_into_replicate(template_root=template_root, replicate_dir=replicate_dir)


def _apply_stage_slurm_job_name(*, stage_dir: Path, case_number: int, replicate_number: int, stage_name: str) -> None:
    _init_apply_stage_slurm_job_name(
        stage_dir=stage_dir,
        case_number=case_number,
        replicate_number=replicate_number,
        stage_name=stage_name,
    )


def _collect_template_stage_relpaths(*, template_root: Path, stage_names: list[str]) -> dict[str, Path]:
    return _init_collect_template_stage_relpaths(template_root=template_root, stage_names=stage_names)


def _normalize_control_overrides(overrides: dict[str, Any] | None) -> dict[str, Any] | None:
    if overrides is None:
        return None
    out: dict[str, Any] = {}
    for key, value in overrides.items():
        key_norm = str(key).strip().lower()
        canonical = CONTROL_PARAMETER_ALIASES.get(key_norm, key_norm)
        out[canonical] = value
    return out


def _as_stage_artifact_ref(value: str) -> ArtifactRef:
    text = str(value).strip()
    if "." not in text:
        raise ValueError(f"Artifact reference must be '<stage>.<artifact>', got: {text!r}")
    stage_name, artifact_name = text.split(".", 1)
    stage_name = stage_name.strip()
    artifact_name = artifact_name.strip()
    if not stage_name or not artifact_name:
        raise ValueError(f"Invalid artifact reference: {text!r}")
    return ArtifactRef(stage=stage_name, artifact=artifact_name)


def _collect_stage_produces(rendered_stage: dict[str, Any]) -> dict[str, str]:
    produced: dict[str, str] = {}
    produces = rendered_stage.get("produces")
    if isinstance(produces, dict):
        for artifact_name, rel_path in produces.items():
            name = str(artifact_name).strip()
            path_text = str(rel_path).strip()
            if name and path_text:
                produced[name] = path_text

    # Backward-compatible support for legacy 'outputs' mapping.
    outputs = rendered_stage.get("outputs")
    if isinstance(outputs, dict):
        for artifact_name, rel_path in outputs.items():
            name = str(artifact_name).strip()
            path_text = str(rel_path).strip()
            if name and path_text and name not in produced:
                produced[name] = path_text
    return produced


def _normalize_stage_consumes(rendered_stage: dict[str, Any]) -> dict[str, dict[str, Any]]:
    consumes: dict[str, dict[str, Any]] = {}
    raw = rendered_stage.get("consumes")
    if isinstance(raw, dict):
        for local_artifact, value in raw.items():
            local_name = str(local_artifact).strip()
            if not local_name:
                continue
            if isinstance(value, str):
                consumes[local_name] = {"from": value}
            elif isinstance(value, dict):
                if "from" not in value:
                    raise ValueError(f"consumes.{local_name} must include 'from'.")
                consumes[local_name] = dict(value)
            else:
                raise ValueError(f"consumes.{local_name} must be string or mapping.")

    # Backward-compatible legacy input_geometry_from.
    if "input_geometry_from" in rendered_stage and "initial_geometry" not in consumes:
        consumes["initial_geometry"] = {"from": rendered_stage["input_geometry_from"], "to": "inputs/initial_geometry.xyz"}
    return consumes


def _resolve_cli_script_path(script_value: str, *, study_dir: Path) -> str:
    script_path = Path(str(script_value))
    if script_path.is_absolute():
        return str(script_path)
    return str((study_dir / script_path).resolve())


def _run_geometry_generation_if_needed(
    stage_dir: Path,
    rendered_stage: dict[str, Any],
    *,
    study_dir: Path,
    enabled: bool,
) -> dict[str, Any] | None:
    cfg = rendered_stage.get("geometry_generator")
    if not isinstance(cfg, dict):
        return None

    script = cfg.get("script")
    cli_template = cfg.get("cli_template")
    if cli_template is None and script is None:
        return None
    if not enabled:
        return {
            "status": "skipped",
            "reason": "geometry_generator execution disabled",
            "command": str(cli_template) if cli_template is not None else f"python {script}",
        }

    command = str(cli_template) if cli_template is not None else f"python {script}"
    if script:
        script_resolved = _resolve_cli_script_path(str(script), study_dir=study_dir)
        command = command.replace(str(script), script_resolved)

    proc = subprocess.run(
        command,
        shell=True,
        cwd=str(stage_dir),
        capture_output=True,
        text=True,
    )
    result = {
        "status": "success" if proc.returncode == 0 else "failed",
        "return_code": int(proc.returncode),
        "command": command,
        "cwd": str(stage_dir),
        "stdout": proc.stdout,
        "stderr": proc.stderr,
    }
    if proc.returncode != 0:
        raise RuntimeError(
            f"Geometry generator failed in stage '{rendered_stage.get('stage')}' with return code {proc.returncode}.\n"
            f"Command: {command}\n"
            f"stderr:\n{proc.stderr}"
        )
    return result


def _default_consume_destination(stage_dir: Path, *, local_artifact: str, source: Path) -> Path:
    suffix = source.suffix if source.suffix else ".dat"
    return stage_dir / "inputs" / f"{local_artifact}{suffix}"


def _transfer_artifact(
    source: Path,
    destination: Path,
    *,
    transfer_mode: str,
) -> dict[str, Any]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        if destination.is_dir() and not destination.is_symlink():
            raise IsADirectoryError(f"Artifact destination exists as directory: {destination}")
        destination.unlink()

    if transfer_mode == "copy":
        shutil.copy2(source, destination)
    elif transfer_mode == "symlink":
        destination.symlink_to(source)
    elif transfer_mode == "hardlink":
        destination.hardlink_to(source)
    else:
        raise ValueError(f"Unsupported artifact transfer mode: {transfer_mode}")

    return {
        "status": "linked" if transfer_mode in {"symlink", "hardlink"} else "copied",
        "mode": transfer_mode,
        "source": str(source),
        "destination": str(destination),
    }


def _propagate_stage_consumes(
    stage_dir: Path,
    rendered_stage: dict[str, Any],
    *,
    produced_artifacts: dict[tuple[str, str], Path],
    transfer_mode: str,
) -> list[dict[str, Any]]:
    consumes = _normalize_stage_consumes(rendered_stage)
    if not consumes:
        return []

    events: list[dict[str, Any]] = []
    for local_artifact, spec in consumes.items():
        source_ref = _as_stage_artifact_ref(spec.get("from"))
        source_key = (source_ref.stage, source_ref.artifact)
        if source_key not in produced_artifacts:
            raise KeyError(
                f"Stage '{rendered_stage.get('stage')}' consumes '{source_ref.stage}.{source_ref.artifact}', "
                "but that artifact is not declared by upstream stages."
            )
        source_path = produced_artifacts[source_key]
        to_value = spec.get("to")
        destination = stage_dir / str(to_value) if to_value else _default_consume_destination(
            stage_dir,
            local_artifact=local_artifact,
            source=source_path,
        )

        if source_path.exists():
            try:
                event = _transfer_artifact(source_path, destination, transfer_mode=transfer_mode)
            except OSError as exc:
                # Common on Windows when symlink perms are not available.
                if transfer_mode in {"symlink", "hardlink"}:
                    event = _transfer_artifact(source_path, destination, transfer_mode="copy")
                    event["fallback_from"] = transfer_mode
                    event["fallback_reason"] = str(exc)
                else:
                    raise
        else:
            event = {
                "status": "pending_missing_source",
                "mode": transfer_mode,
                "source": str(source_path),
                "destination": str(destination),
            }

        event["local_artifact"] = local_artifact
        event["from"] = f"{source_ref.stage}.{source_ref.artifact}"
        events.append(event)
    return events


def _write_stage_control_if_needed(
    stage_dir: Path,
    stage_payload: dict[str, Any],
    *,
    rendered_stage: dict[str, Any],
    study_dir: Path,
) -> tuple[Path | None, str | None]:
    control_cfg = rendered_stage.get("control")
    if not isinstance(control_cfg, dict):
        return None, None

    generator_name = str(control_cfg.get("generator") or "").strip().lower()
    if generator_name and generator_name != "control_generator":
        raise ValueError(f"Unsupported control generator: {generator_name}")

    control_out = stage_dir / "control"
    overrides = control_cfg.get("parameters")
    if overrides is not None and not isinstance(overrides, dict):
        raise ValueError("control.parameters must be a mapping when provided.")
    overrides = _normalize_control_overrides(overrides)

    template_value = control_cfg.get("template")
    template_used: str | None = None
    if template_value:
        template_path = _resolve_template_path(str(template_value), study_dir=study_dir)
        if template_path.exists():
            template_text = template_path.read_text(encoding="utf-8")
            write_control_template_with_overrides(
                out_path=control_out,
                spec=ControlGeneratorSpec(template_text=template_text),
                overrides=overrides,
            )
            template_used = str(template_path)
        else:
            # Phase-1 planning should still initialize even if external templates
            # are not present yet; fall back to the bundled control template.
            write_control(
                out_path=control_out,
                overrides=overrides,
            )
            template_used = "default_control_template"
    else:
        write_control(
            out_path=control_out,
            overrides=overrides,
        )
        template_used = "default_control_template"
    _ = stage_payload
    return control_out, template_used


def _init_study(
    study_file: Path,
    *,
    root: Path,
    force: bool,
    run_geometry_generator: bool,
    artifact_transfer: str,
    strict_actions: bool,
) -> Path:
    if artifact_transfer not in SUPPORTED_ARTIFACT_TRANSFER:
        raise ValueError(
            f"Unsupported artifact transfer mode: {artifact_transfer}. "
            f"Supported: {', '.join(sorted(SUPPORTED_ARTIFACT_TRANSFER))}"
        )

    study_doc = _load_study_yaml(study_file)
    study_name, parameters, replicates, stages, analyses, template_dir_raw = _validate_study(study_doc)
    combos = _enumerate_cases(parameters)

    study_root = (root / study_name).resolve()
    if study_root.exists() and any(study_root.iterdir()) and not force:
        raise FileExistsError(f"Study folder already exists and is not empty: {study_root}. Use --force to continue.")
    study_root.mkdir(parents=True, exist_ok=True)

    stage_names = [s.name for s in stages]
    stage_set = set(stage_names)
    for stage in stages:
        dep = str(stage.payload.get("depends_on") or "").strip()
        if dep and dep not in stage_set:
            raise ValueError(f"Stage '{stage.name}' depends_on unknown stage '{dep}'.")
    for analysis in analyses:
        if analysis.run_stage not in stage_set:
            raise ValueError(
                f"Analysis '{analysis.analysis_id}' references unknown run_stage '{analysis.run_stage}'."
            )

    cases_dir = study_root / "cases"
    cases_dir.mkdir(parents=True, exist_ok=True)

    case_entries: list[dict[str, Any]] = []
    study_yaml_dir = study_file.resolve().parent
    template_dir = _resolve_template_path(template_dir_raw, study_dir=study_yaml_dir) if template_dir_raw else None
    if template_dir is not None and not template_dir.exists():
        raise FileNotFoundError(f"Study template directory not found: {template_dir}")
    stage_template_relpaths: dict[str, Path] | None = None
    if template_dir is not None:
        stage_template_relpaths = _collect_template_stage_relpaths(
            template_root=template_dir,
            stage_names=stage_names,
        )

    for case_idx, param_values in enumerate(combos, start=1):
        combo_slug = "_".join(f"{_slug_underscore(k)}_{_slug_underscore(v)}" for k, v in param_values.items())
        case_id = f"case_{case_idx:04d}"
        case_dir = cases_dir / f"{case_id}_{combo_slug}"
        case_dir.mkdir(parents=True, exist_ok=True)

        replicate_entries: list[dict[str, Any]] = []
        for rep_idx in range(1, replicates + 1):
            rep_id = f"rep_{rep_idx:02d}"
            rep_dir = case_dir / rep_id
            rep_dir.mkdir(parents=True, exist_ok=True)
            if template_dir is not None:
                _copy_template_into_replicate(template_root=template_dir, replicate_dir=rep_dir)

            stage_paths: dict[str, Path] = {}
            produced_artifacts: dict[tuple[str, str], Path] = {}
            stage_entries: list[dict[str, Any]] = []
            for stage in stages:
                dep = str(stage.payload.get("depends_on") or "").strip()
                if stage_template_relpaths is not None:
                    stage_rel = stage_template_relpaths[stage.name]
                    stage_dir = rep_dir / stage_rel
                else:
                    base = stage_paths.get(dep, rep_dir) if dep else rep_dir
                    stage_dir = base / stage.name
                stage_dir.mkdir(parents=True, exist_ok=True)
                _apply_stage_slurm_job_name(
                    stage_dir=stage_dir,
                    case_number=case_idx,
                    replicate_number=rep_idx,
                    stage_name=stage.name,
                )
                stage_paths[stage.name] = stage_dir

                context = {
                    **param_values,
                    "study_name": study_name,
                    "case_id": case_id,
                    "replicate_id": rep_id,
                    "stage": stage.name,
                }
                rendered_stage = _render_value(stage.payload, context)
                rendered_stage["stage"] = stage.name
                control_path, control_template_used = _write_stage_control_if_needed(
                    stage_dir=stage_dir,
                    stage_payload=stage.payload,
                    rendered_stage=rendered_stage,
                    study_dir=study_yaml_dir,
                )
                stage_actions: list[dict[str, Any]] = []
                geometry_run = None
                try:
                    geometry_run = _run_geometry_generation_if_needed(
                        stage_dir=stage_dir,
                        rendered_stage=rendered_stage,
                        study_dir=study_yaml_dir,
                        enabled=run_geometry_generator,
                    )
                except Exception as exc:
                    if strict_actions:
                        raise
                    geometry_run = {
                        "status": "failed",
                        "reason": str(exc),
                    }
                if geometry_run is not None:
                    stage_actions.append({"action": "geometry_generator", **geometry_run})

                consumed_artifacts: list[dict[str, Any]] = []
                try:
                    consumed_artifacts = _propagate_stage_consumes(
                        stage_dir=stage_dir,
                        rendered_stage=rendered_stage,
                        produced_artifacts=produced_artifacts,
                        transfer_mode=artifact_transfer,
                    )
                except Exception as exc:
                    if strict_actions:
                        raise
                    consumed_artifacts.append(
                        {
                            "status": "failed",
                            "reason": str(exc),
                        }
                    )
                for event in consumed_artifacts:
                    stage_actions.append({"action": "artifact_consume", **event})

                produced_declared = _collect_stage_produces(rendered_stage)
                if control_path is not None and "control_file" not in produced_declared:
                    produced_declared["control_file"] = "control"

                produced_manifest: list[dict[str, Any]] = []
                for artifact_name, rel_path in produced_declared.items():
                    artifact_path = stage_dir / str(rel_path)
                    produced_artifacts[(stage.name, artifact_name)] = artifact_path
                    produced_manifest.append(
                        {
                            "name": artifact_name,
                            "path": str(artifact_path),
                            "exists": bool(artifact_path.exists()),
                        }
                    )

                stage_manifest = {
                    "study_name": study_name,
                    "case_id": case_id,
                    "replicate_id": rep_id,
                    "stage": stage.name,
                    "parameters": param_values,
                    "depends_on": dep or None,
                    "stage_path": str(stage_dir),
                    "rendered_stage": rendered_stage,
                    "control_file": str(control_path) if control_path is not None else None,
                    "control_template_used": control_template_used,
                    "artifacts": {
                        "produced": produced_manifest,
                        "consumed": consumed_artifacts,
                    },
                    "actions": stage_actions,
                    "generated_at_utc": _utc_now(),
                }
                _write_json(stage_dir / RUN_STAGE_MANIFEST_FILE, stage_manifest)
                stage_entries.append(
                    {
                        "stage": stage.name,
                        "path": str(stage_dir),
                        "depends_on": dep or None,
                        "control_file": str(control_path) if control_path is not None else None,
                        "control_template_used": control_template_used,
                        "artifacts": {
                            "produced": produced_manifest,
                            "consumed": consumed_artifacts,
                        },
                    }
                )

            rep_manifest = {
                "study_name": study_name,
                "case_id": case_id,
                "replicate_id": rep_id,
                "parameters": param_values,
                "replicate_path": str(rep_dir),
                "stages": stage_entries,
                "generated_at_utc": _utc_now(),
            }
            _write_json(rep_dir / RUN_REPLICATE_MANIFEST_FILE, rep_manifest)
            replicate_entries.append(
                {
                    "replicate_id": rep_id,
                    "path": str(rep_dir),
                }
            )

        case_manifest = {
            "study_name": study_name,
            "case_id": case_id,
            "parameters": param_values,
            "case_path": str(case_dir),
            "replicates": replicate_entries,
            "generated_at_utc": _utc_now(),
        }
        _write_json(case_dir / RUN_CASE_MANIFEST_FILE, case_manifest)
        case_entries.append(
            {
                "case_id": case_id,
                "combo_slug": combo_slug,
                "parameters": param_values,
                "path": str(case_dir),
                "replicates": replicate_entries,
            }
        )

    manifest = {
        "study_name": study_name,
        "study_root": str(study_root),
        "source_yaml": str(study_file.resolve()),
        "template_dir": str(template_dir) if template_dir is not None else None,
        "parameters": parameters,
        "replicates": replicates,
        "run_stages": stage_names,
        "workflow_stages": stage_names,  # backward-compatible alias
        "analysis": [
            {
                "analysis_id": a.analysis_id,
                "title": a.title,
                "run_stage": a.run_stage,
                "payload": a.payload,
            }
            for a in analyses
        ],
        "n_cases": len(case_entries),
        "n_total_runs": len(case_entries) * replicates,
        "cases": case_entries,
        "generated_at_utc": _utc_now(),
    }
    _write_json(study_root / "study_manifest.json", manifest)
    return study_root


def _write_study_template(path: Path, *, force: bool) -> Path:
    if path.exists() and not force:
        raise FileExistsError(f"Refusing to overwrite existing file: {path}. Use --force.")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(STUDY_TEMPLATE_YAML, encoding="utf-8")
    return path


def _read_json(path: Path) -> dict[str, Any]:
    return _io_read_json(path)


def _rename_case_directories(
    *,
    study_root: Path,
    case_filter: str | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    return _manage_rename_case_directories(
        study_root=study_root,
        case_filter=case_filter,
        dry_run=dry_run,
        case_matches_selector_fn=_case_matches_selector,
    )


def _prompt_with_default(label: str, default: str | None) -> str | None:
    base = (default or "").strip()
    msg = f"{label} [{base}]: " if base else f"{label}: "
    entered = input(msg).strip()
    if entered:
        return entered
    return base or None


def _update_study_directory_paths(
    *,
    study_root: Path,
    case_filter: str | None = None,
    replicate_filter: str | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    return _manage_update_study_directory_paths(
        study_root=study_root,
        case_filter=case_filter,
        replicate_filter=replicate_filter,
        dry_run=dry_run,
        prompt_with_default_fn=_prompt_with_default,
        replicate_variants_fn=_replicate_variants,
    )


def _stage_status_path(stage_dir: Path) -> Path:
    return _io_stage_status_path(stage_dir, stage_status_file=STAGE_STATUS_FILE)


def _load_stage_status(stage_dir: Path) -> dict[str, Any]:
    return _io_load_stage_status(stage_dir, stage_status_file=STAGE_STATUS_FILE)


def _write_stage_status(stage_dir: Path, status: dict[str, Any]) -> None:
    _io_write_stage_status(stage_dir, status, stage_status_file=STAGE_STATUS_FILE)


def _analysis_status_path(stage_dir: Path) -> Path:
    return _io_analysis_status_path(stage_dir, analysis_status_file=ANALYSIS_STATUS_FILE)


def _load_analysis_status(stage_dir: Path) -> dict[str, Any]:
    return _io_load_analysis_status(stage_dir, analysis_status_file=ANALYSIS_STATUS_FILE)


def _write_analysis_status(stage_dir: Path, status: dict[str, Any]) -> None:
    _io_write_analysis_status(stage_dir, status, analysis_status_file=ANALYSIS_STATUS_FILE)


def _analysis_manifest_path(stage_dir: Path) -> Path:
    return _io_analysis_manifest_path(stage_dir, analysis_manifest_file=ANALYSIS_MANIFEST_FILE)


def _load_analysis_manifest(stage_dir: Path) -> dict[str, Any]:
    return _io_load_analysis_manifest(stage_dir, analysis_manifest_file=ANALYSIS_MANIFEST_FILE)


def _write_analysis_manifest(stage_dir: Path, payload: dict[str, Any]) -> None:
    _io_write_analysis_manifest(stage_dir, payload, analysis_manifest_file=ANALYSIS_MANIFEST_FILE)


def _extract_result_dirs_from_text(text: str) -> list[str]:
    lines = str(text or "").splitlines()
    out: list[str] = []
    in_block = False
    for raw in lines:
        line = raw.rstrip()
        if line.strip() == "Results saved in:":
            in_block = True
            continue
        if not in_block:
            continue
        if not line.strip():
            if out:
                break
            continue
        if line.startswith("  "):
            out.append(line.strip())
            continue
        # End of block when indentation pattern stops.
        if out:
            break
    return out


def _collect_result_dirs_from_step_records(step_records: list[dict[str, Any]]) -> list[str]:
    dirs: list[str] = []
    for rec in step_records:
        if not isinstance(rec, dict):
            continue
        for key in ("stdout", "stderr"):
            text = str(rec.get(key) or "")
            if not text:
                continue
            dirs.extend(_extract_result_dirs_from_text(text))
    # dedupe preserving order
    seen: set[str] = set()
    uniq: list[str] = []
    for d in dirs:
        if d in seen:
            continue
        seen.add(d)
        uniq.append(d)
    return uniq


def _case_matches_selector(case_entry: dict[str, Any], selector: str | None) -> bool:
    if not selector:
        return True
    needle = _canonical_token(selector)
    if not needle:
        return True
    params = case_entry.get("parameters") or {}
    candidates = [
        str(case_entry.get("case_id") or ""),
        str(case_entry.get("combo_slug") or ""),
        str(case_entry.get("path") or ""),
        _case_label_from_params(params if isinstance(params, dict) else {}),
    ]
    for cand in candidates:
        if needle in _canonical_token(cand):
            return True
    return False


def _replicate_variants(value: str | None) -> set[str]:
    base = str(value or "").strip()
    if not base:
        return set()
    out = {base}
    if base.startswith("replicate_"):
        out.add(f"rep_{base[len('replicate_'):]}")
    elif base.startswith("rep_"):
        out.add(f"replicate_{base[len('rep_'):]}")
    return out


def _replicate_matches_selector(replicate_id: str, selector: str | None) -> bool:
    if not selector:
        return True
    rep_tokens = {_canonical_token(v) for v in _replicate_variants(replicate_id)}
    sel_tokens = {_canonical_token(v) for v in _replicate_variants(str(selector))}
    rep_tokens.discard("")
    sel_tokens.discard("")
    if not sel_tokens:
        return True
    return bool(rep_tokens.intersection(sel_tokens))


def _build_declared_produced_map(replicate_manifest: dict[str, Any]) -> dict[tuple[str, str], Path]:
    produced: dict[tuple[str, str], Path] = {}
    for stage_entry in replicate_manifest.get("stages") or []:
        if not isinstance(stage_entry, dict):
            continue
        stage_name = str(stage_entry.get("stage") or "").strip()
        if not stage_name:
            continue
        artifacts = (stage_entry.get("artifacts") or {}).get("produced") or []
        if isinstance(artifacts, list):
            for item in artifacts:
                if not isinstance(item, dict):
                    continue
                art_name = str(item.get("name") or "").strip()
                art_path = str(item.get("path") or "").strip()
                if art_name and art_path:
                    produced[(stage_name, art_name)] = Path(art_path)
    return produced


def _is_stage_ready(
    stage_manifest: dict[str, Any],
    *,
    status_by_stage: dict[str, str],
    produced_artifacts: dict[tuple[str, str], Path],
) -> tuple[bool, str]:
    stage_name = str(stage_manifest.get("stage") or "")
    depends_on = str(stage_manifest.get("depends_on") or "").strip()
    if depends_on and status_by_stage.get(depends_on) != "completed":
        return False, f"waiting_on_dependency:{depends_on}"

    rendered_stage = stage_manifest.get("rendered_stage") or {}
    consumes = _normalize_stage_consumes(rendered_stage if isinstance(rendered_stage, dict) else {})
    for local_name, spec in consumes.items():
        source_ref = _as_stage_artifact_ref(spec.get("from"))
        src_key = (source_ref.stage, source_ref.artifact)
        src = produced_artifacts.get(src_key)
        if src is None:
            return False, f"missing_declared_artifact:{source_ref.stage}.{source_ref.artifact}"
        if not src.exists():
            return False, f"missing_source_file:{source_ref.stage}.{source_ref.artifact}"
        _ = local_name
    _ = stage_name
    return True, "ready"


def _submit_stage_job(
    stage_dir: Path,
    rendered_stage: dict[str, Any],
    *,
    study_dir: Path,
) -> dict[str, Any] | None:
    submit_cfg = rendered_stage.get("submit")
    if not isinstance(submit_cfg, dict):
        return None
    command = str(submit_cfg.get("command") or "").strip()
    if not command:
        return None

    script = submit_cfg.get("script")
    if script:
        script_resolved = _resolve_cli_script_path(str(script), study_dir=study_dir)
        command = command.replace(str(script), script_resolved)
    command = _rewrite_python_script_tokens(command, study_dir=study_dir)
    workdir_value = str(submit_cfg.get("workdir") or ".").strip()
    workdir = stage_dir / workdir_value
    workdir.mkdir(parents=True, exist_ok=True)

    job_id = f"local-{uuid.uuid4().hex[:12]}"
    proc = subprocess.run(
        command,
        shell=True,
        cwd=str(workdir),
        capture_output=True,
        text=True,
    )
    return {
        "job_id": job_id,
        "command": command,
        "cwd": str(workdir),
        "return_code": int(proc.returncode),
        "stdout": proc.stdout,
        "stderr": proc.stderr,
        "status": "completed" if proc.returncode == 0 else "failed",
        "submitted_at_utc": _utc_now(),
        "finished_at_utc": _utc_now(),
    }


def _rewrite_python_script_tokens(command: str, *, study_dir: Path) -> str:
    rewritten = str(command)
    for token in str(command).split():
        raw = token.strip().strip("'\"")
        if not raw.lower().endswith(".py"):
            continue
        candidate = (study_dir / raw).resolve()
        if not candidate.exists():
            continue
        rewritten = rewritten.replace(raw, str(candidate))
    return rewritten


def _run_single_step_command(
    *,
    stage_dir: Path,
    command: str,
    workdir: str | None = None,
) -> dict[str, Any]:
    command_tokens = str(command).strip().split()
    if command_tokens and command_tokens[0].lower() == "sbatch" and "--wait" not in command_tokens:
        command = "sbatch --wait " + " ".join(command_tokens[1:])

    wd = stage_dir / str(workdir or ".")
    wd.mkdir(parents=True, exist_ok=True)
    job_id = f"local-{uuid.uuid4().hex[:12]}"
    proc = subprocess.run(
        command,
        shell=True,
        cwd=str(wd),
        capture_output=True,
        text=True,
    )
    return {
        "job_id": job_id,
        "command": command,
        "cwd": str(wd),
        "return_code": int(proc.returncode),
        "stdout": proc.stdout,
        "stderr": proc.stderr,
        "status": "completed" if proc.returncode == 0 else "failed",
        "submitted_at_utc": _utc_now(),
        "finished_at_utc": _utc_now(),
    }


def _execute_stage_steps_if_any(
    *,
    stage_dir: Path,
    rendered_stage: dict[str, Any],
    study_dir: Path,
) -> list[dict[str, Any]] | None:
    raw_steps = rendered_stage.get("steps")
    if raw_steps is None:
        return None
    if not isinstance(raw_steps, list):
        raise ValueError("stage.steps must be a list.")

    step_records: list[dict[str, Any]] = []
    for idx, step in enumerate(raw_steps, start=1):
        if isinstance(step, str):
            cmd = str(step).strip()
            if not cmd:
                continue
            cmd = _rewrite_python_script_tokens(cmd, study_dir=study_dir)
            record = _run_single_step_command(stage_dir=stage_dir, command=cmd)
            record["step_index"] = idx
            record["step_type"] = "run"
            step_records.append(record)
            continue

        if not isinstance(step, dict):
            raise ValueError(f"stage.steps[{idx}] must be string or mapping.")
        cmd = str(step.get("run") or "").strip()
        if not cmd:
            raise ValueError(f"stage.steps[{idx}] missing required 'run' command.")

        script = step.get("script")
        if script:
            script_resolved = _resolve_cli_script_path(str(script), study_dir=study_dir)
            cmd = cmd.replace(str(script), script_resolved)
        cmd = _rewrite_python_script_tokens(cmd, study_dir=study_dir)

        record = _run_single_step_command(
            stage_dir=stage_dir,
            command=cmd,
            workdir=(str(step.get("workdir")) if step.get("workdir") is not None else None),
        )
        record["step_index"] = idx
        record["step_type"] = str(step.get("type") or "run")
        if "name" in step:
            record["step_name"] = str(step.get("name"))
        step_records.append(record)
    return step_records


def _execute_stage_run(
    *,
    stage_dir: Path,
    stage_manifest: dict[str, Any],
    produced_artifacts: dict[tuple[str, str], Path],
    artifact_transfer: str,
    run_geometry_generator: bool,
    strict_actions: bool,
) -> dict[str, Any]:
    rendered_stage = stage_manifest.get("rendered_stage") or {}
    if not isinstance(rendered_stage, dict):
        raise ValueError(f"Invalid rendered_stage in {stage_dir}")

    actions: list[dict[str, Any]] = []

    consumed_artifacts: list[dict[str, Any]] = []
    try:
        consumed_artifacts = _propagate_stage_consumes(
            stage_dir=stage_dir,
            rendered_stage=rendered_stage,
            produced_artifacts=produced_artifacts,
            transfer_mode=artifact_transfer,
        )
    except Exception as exc:
        if strict_actions:
            raise
        consumed_artifacts.append({"status": "failed", "reason": str(exc)})
    for event in consumed_artifacts:
        actions.append({"action": "artifact_consume", **event})

    study_dir = Path(stage_manifest.get("study_source_dir") or stage_dir)
    step_jobs: list[dict[str, Any]] | None = None
    try:
        step_jobs = _execute_stage_steps_if_any(
            stage_dir=stage_dir,
            rendered_stage=rendered_stage,
            study_dir=study_dir,
        )
    except Exception as exc:
        if strict_actions:
            raise
        actions.append({"action": "steps", "status": "failed", "reason": str(exc)})
        step_jobs = []

    if step_jobs is not None:
        for item in step_jobs:
            actions.append({"action": "step", **item})
    else:
        # Backward-compatibility path for legacy stage schema.
        geometry_action = None
        try:
            geometry_action = _run_geometry_generation_if_needed(
                stage_dir=stage_dir,
                rendered_stage=rendered_stage,
                study_dir=study_dir,
                enabled=run_geometry_generator,
            )
        except Exception as exc:
            if strict_actions:
                raise
            geometry_action = {"status": "failed", "reason": str(exc)}
        if geometry_action is not None:
            actions.append({"action": "geometry_generator", **geometry_action})

        job_record = _submit_stage_job(
            stage_dir,
            rendered_stage,
            study_dir=study_dir,
        )
        if job_record is not None:
            actions.append({"action": "submit_job", **job_record})

    failed = any(str(a.get("status", "")).lower() == "failed" for a in actions)

    stage_status = _load_stage_status(stage_dir)
    stage_status.setdefault("jobs", [])
    job_records = [a for a in actions if "job_id" in a and "command" in a]
    stage_status["jobs"].extend(job_records)
    stage_status["last_actions"] = actions
    stage_status["status"] = "failed" if failed else "completed"
    _write_stage_status(stage_dir, stage_status)

    stage_manifest["last_run"] = {
        "status": stage_status["status"],
        "actions": actions,
        "updated_at_utc": _utc_now(),
    }
    _write_json(stage_dir / RUN_STAGE_MANIFEST_FILE, stage_manifest)

    return {
        "status": stage_status["status"],
        "actions": actions,
        "job_id": job_records[-1].get("job_id") if job_records else None,
    }


def _run_single_replicate_pipeline(
    *,
    case: dict[str, Any],
    rep: dict[str, Any],
    manifest: dict[str, Any],
    stage_filter: str | None,
    artifact_transfer: str,
    run_geometry_generator: bool,
    strict_actions: bool,
    cleanup_before_stage: bool,
) -> dict[str, Any]:
    counts = {"completed": 0, "skipped": 0, "failed": 0, "not_ready": 0}
    started_at = _local_now()
    rep_id = str(rep.get("replicate_id") or "")
    case_id = str(case.get("case_id") or "")
    rep_path = Path(str(rep.get("path") or ""))
    rep_manifest = _read_json(_run_replicate_manifest_path(rep_path))
    produced_map = _build_declared_produced_map(rep_manifest)

    stage_entries = rep_manifest.get("stages") or []
    max_stage_chars = 0
    for _entry in stage_entries:
        if not isinstance(_entry, dict):
            continue
        sname = str(_entry.get("stage") or "").strip()
        if len(sname) > max_stage_chars:
            max_stage_chars = len(sname)
    stage_block_width = _stage_label_width(max_stage_chars=max_stage_chars)

    status_by_stage: dict[str, str] = {}
    for entry in stage_entries:
        if not isinstance(entry, dict):
            continue
        stage_name = str(entry.get("stage") or "").strip()
        if not stage_name:
            continue
        stage_dir = Path(str(entry.get("path") or ""))
        status_by_stage[stage_name] = str(_load_stage_status(stage_dir).get("status") or "pending")

    for entry in stage_entries:
        if not isinstance(entry, dict):
            continue
        stage_name = str(entry.get("stage") or "").strip()
        if not stage_name:
            continue
        if stage_filter and stage_name != stage_filter:
            continue

        stage_dir = Path(str(entry.get("path") or ""))
        stage_manifest = _read_json(_run_stage_manifest_path(stage_dir))
        stage_manifest["study_source_dir"] = str(Path(str(manifest.get("source_yaml") or "")).parent)

        status = _load_stage_status(stage_dir)
        current_status = str(status.get("status") or "pending")
        if current_status == "completed":
            counts["skipped"] += 1
            _log_stage_event(
                case_id,
                rep_id,
                stage_name,
                "SKIP",
                "already completed",
                stage_block_width=stage_block_width,
            )
            continue

        ready, reason = _is_stage_ready(
            stage_manifest,
            status_by_stage=status_by_stage,
            produced_artifacts=produced_map,
        )
        if not ready:
            counts["not_ready"] += 1
            _log_stage_event(
                case_id,
                rep_id,
                stage_name,
                "WAIT",
                reason,
                stage_block_width=stage_block_width,
            )
            continue

        _log_stage_event(case_id, rep_id, stage_name, "RUN", stage_block_width=stage_block_width)
        if cleanup_before_stage:
            removed_n = _cleanup_stage_artifacts(stage_dir)
            _log_stage_event(
                case_id,
                rep_id,
                stage_name,
                "CLEAN",
                f"removed={removed_n}",
                stage_block_width=stage_block_width,
            )
        result = _execute_stage_run(
            stage_dir=stage_dir,
            stage_manifest=stage_manifest,
            produced_artifacts=produced_map,
            artifact_transfer=artifact_transfer,
            run_geometry_generator=run_geometry_generator,
            strict_actions=strict_actions,
        )
        final_status = str(result.get("status") or "failed")
        status_by_stage[stage_name] = final_status
        if final_status == "completed":
            counts["completed"] += 1
            _log_stage_event(case_id, rep_id, stage_name, "DONE", stage_block_width=stage_block_width)
        else:
            counts["failed"] += 1
            _log_stage_event(case_id, rep_id, stage_name, "FAIL", stage_block_width=stage_block_width)
            if strict_actions:
                finished_at = _local_now()
                return {
                    **counts,
                    "case_id": case_id,
                    "replicate_id": rep_id,
                    "run": counts["completed"] + counts["failed"],
                    "done": counts["completed"],
                    "skip": counts["skipped"],
                    "wait": counts["not_ready"],
                    "fail": counts["failed"],
                    "status": "failed",
                    "started_at": started_at,
                    "finished_at": finished_at,
                    "duration_min": _duration_minutes(started_at, finished_at),
                }

    finished_at = _local_now()
    if counts["failed"] > 0:
        final_status = "failed"
    elif counts["not_ready"] > 0:
        final_status = "waiting"
    elif counts["completed"] > 0:
        final_status = "done"
    elif counts["skipped"] > 0:
        final_status = "skipped"
    else:
        final_status = "idle"

    return {
        **counts,
        "case_id": case_id,
        "replicate_id": rep_id,
        "run": counts["completed"] + counts["failed"],
        "done": counts["completed"],
        "skip": counts["skipped"],
        "wait": counts["not_ready"],
        "fail": counts["failed"],
        "status": final_status,
        "started_at": started_at,
        "finished_at": finished_at,
        "duration_min": _duration_minutes(started_at, finished_at),
    }


def _run_study(
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
) -> dict[str, int]:
    return _engine_run_study(
        study_root=study_root,
        stage_filter=stage_filter,
        case_filter=case_filter,
        replicate_filter=replicate_filter,
        artifact_transfer=artifact_transfer,
        run_geometry_generator=run_geometry_generator,
        strict_actions=strict_actions,
        parallel_workers=parallel_workers,
        rerun_failed=rerun_failed,
        read_json_fn=_read_json,
        load_study_run_status_rows_fn=_load_study_run_status_rows,
        to_int_fn=_to_int,
        case_matches_selector_fn=_case_matches_selector,
        replicate_matches_selector_fn=_replicate_matches_selector,
        run_case_manifest_path_fn=_run_case_manifest_path,
        run_single_replicate_pipeline_fn=_run_single_replicate_pipeline,
        write_study_run_status_fn=_write_study_run_status,
    )


def _normalize_analysis_variables(payload: dict[str, Any]) -> dict[str, dict[str, str]]:
    variables: dict[str, dict[str, str]] = {}
    raw = payload.get("variables")
    if not isinstance(raw, dict):
        return variables
    for variable_name, spec in raw.items():
        name = str(variable_name).strip()
        if not name or not isinstance(spec, dict):
            continue
        item: dict[str, str] = {}
        for k in ("directory", "folder_id", "file", "column"):
            val = str(spec.get(k) or "").strip()
            if val:
                item[k] = val
        if "file" in item:
            variables[name] = item
    return variables


def _run_analysis_steps(
    *,
    stage_dir: Path,
    rendered_analysis: dict[str, Any],
    study_dir: Path,
) -> list[dict[str, Any]]:
    raw_steps = rendered_analysis.get("steps")
    command = str(rendered_analysis.get("command") or "").strip()
    if raw_steps is None:
        if not command:
            raise ValueError("Analysis is missing both steps and command.")
        raw_steps = [f"reaxkit {command}"]
    if not isinstance(raw_steps, list):
        raise ValueError("analysis.steps must be a list when provided.")

    records: list[dict[str, Any]] = []
    for idx, step in enumerate(raw_steps, start=1):
        if isinstance(step, str):
            cmd = _rewrite_python_script_tokens(str(step).strip(), study_dir=study_dir)
            if not cmd:
                continue
            rec = _run_single_step_command(stage_dir=stage_dir, command=cmd)
            rec["step_index"] = idx
            rec["step_type"] = "run"
            records.append(rec)
            continue
        if not isinstance(step, dict):
            raise ValueError(f"analysis.steps[{idx}] must be string or mapping.")
        cmd = str(step.get("run") or "").strip()
        if not cmd:
            raise ValueError(f"analysis.steps[{idx}] missing required 'run'.")
        script = step.get("script")
        if script:
            script_resolved = _resolve_cli_script_path(str(script), study_dir=study_dir)
            cmd = cmd.replace(str(script), script_resolved)
        cmd = _rewrite_python_script_tokens(cmd, study_dir=study_dir)
        rec = _run_single_step_command(
            stage_dir=stage_dir,
            command=cmd,
            workdir=(str(step.get("workdir")) if step.get("workdir") is not None else None),
        )
        rec["step_index"] = idx
        rec["step_type"] = str(step.get("type") or "run")
        records.append(rec)
    return records


def _analyze_study(
    *,
    study_root: Path,
    analysis_filter: str | None,
    case_filter: str | None,
    replicate_filter: str | None,
    strict_actions: bool,
    rerun_failed: bool,
) -> dict[str, Any]:
    return _engine_analyze_study(
        study_root=study_root,
        analysis_filter=analysis_filter,
        case_filter=case_filter,
        replicate_filter=replicate_filter,
        strict_actions=strict_actions,
        rerun_failed=rerun_failed,
        read_json_fn=_read_json,
        load_study_yaml_fn=_load_study_yaml,
        validate_study_fn=_validate_study,
        analysis_label_width_fn=_analysis_label_width,
        load_analysis_status_rows_fn=_load_analysis_status_rows,
        to_int_fn=_to_int,
        case_matches_selector_fn=_case_matches_selector,
        replicate_matches_selector_fn=_replicate_matches_selector,
        run_case_manifest_path_fn=_run_case_manifest_path,
        run_replicate_manifest_path_fn=_run_replicate_manifest_path,
        load_stage_status_fn=_load_stage_status,
        render_value_fn=_render_value,
        utc_now_fn=_utc_now,
        local_now_fn=_local_now,
        run_analysis_steps_fn=_run_analysis_steps,
        duration_minutes_fn=_duration_minutes,
        collect_result_dirs_from_step_records_fn=_collect_result_dirs_from_step_records,
        load_analysis_status_fn=_load_analysis_status,
        write_analysis_status_fn=_write_analysis_status,
        load_analysis_manifest_fn=_load_analysis_manifest,
        write_analysis_manifest_fn=_write_analysis_manifest,
        log_stage_event_fn=_log_stage_event,
        analysis_status_json_file=ANALYSIS_STATUS_JSON_FILE,
        analysis_status_csv_file=ANALYSIS_STATUS_CSV_FILE,
        write_json_fn=_write_json,
    )


def _safe_float(value: Any) -> float | None:
    return _agg_safe_float(value)


def _stat_value(row: dict[str, Any], key: str) -> Any:
    return _agg_stat_value(row, key)


def _load_csv_rows(path: Path) -> list[dict[str, str]]:
    return _agg_load_csv_rows(path)


def _extract_scalar_from_csv(
    csv_path: Path,
    *,
    value_column: str | None,
) -> tuple[float | None, str | None]:
    return _agg_extract_scalar_from_csv(csv_path, value_column=value_column)


def _load_column_values(csv_path: Path, column: str) -> list[Any]:
    return _agg_load_column_values(csv_path, column)


def _resolve_variable_file_for_run(
    *,
    stage_dir: Path,
    spec: dict[str, str],
    result_dirs: list[str] | None,
) -> Path | None:
    return _agg_resolve_variable_file_for_run(stage_dir=stage_dir, spec=spec, result_dirs=result_dirs)


def _apply_reducer_to_series(
    *,
    pairs: list[tuple[Any, Any]],
    reducer: str,
) -> list[tuple[Any, float]]:
    return _agg_apply_reducer_to_series(pairs=pairs, reducer=reducer)


def _compute_stats(values: list[float], wanted: list[str]) -> dict[str, float]:
    return _agg_compute_stats(values, wanted)


def _normalize_stage_variables(rendered_stage: dict[str, Any]) -> dict[str, dict[str, str]]:
    return _agg_normalize_stage_variables(rendered_stage)


def _resolve_variable_csv_path(*, stage_dir: Path, spec: dict[str, str]) -> Path | None:
    return _agg_resolve_variable_csv_path(stage_dir=stage_dir, spec=spec)


def _sort_values(values: list[Any]) -> list[Any]:
    return _agg_sort_values(values)


def _to_plot_value(v: Any):
    return _agg_to_plot_value(v)


def _make_aggregate_plot(
    grouped_rows: list[dict[str, Any]],
    *,
    params: list[str],
    metric_mean_col: str,
    metric_std_col: str,
    out_path: Path,
) -> tuple[bool, str]:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        return False, f"matplotlib unavailable: {exc}"

    if not grouped_rows:
        return False, "no grouped rows"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 5))
    if len(params) == 1:
        p = params[0]
        xs = [_to_plot_value(r[p]) for r in grouped_rows]
        ys = [float(r[metric_mean_col]) for r in grouped_rows]
        yerr = [float(r[metric_std_col]) for r in grouped_rows]
        plt.errorbar(xs, ys, yerr=yerr, fmt="o-", capsize=4)
        plt.xlabel(p)
        plt.ylabel(metric_mean_col)
        plt.title(f"{metric_mean_col} vs {p}")
        plt.grid(True, alpha=0.3)
    else:
        x_param, y_param = params[0], params[1]
        x_vals = _sort_values(list({r[x_param] for r in grouped_rows}))
        y_vals = _sort_values(list({r[y_param] for r in grouped_rows}))
        x_index = {v: i for i, v in enumerate(x_vals)}
        y_index = {v: i for i, v in enumerate(y_vals)}
        z = [[math.nan for _ in x_vals] for _ in y_vals]
        for row in grouped_rows:
            xi = x_index[row[x_param]]
            yi = y_index[row[y_param]]
            z[yi][xi] = float(row[metric_mean_col])
        import numpy as np

        arr = np.array(z, dtype=float)
        im = plt.imshow(arr, aspect="auto", origin="lower", cmap="viridis")
        plt.colorbar(im, label=metric_mean_col)
        plt.xticks(range(len(x_vals)), [str(v) for v in x_vals], rotation=45, ha="right")
        plt.yticks(range(len(y_vals)), [str(v) for v in y_vals])
        plt.xlabel(x_param)
        plt.ylabel(y_param)
        plt.title(f"{metric_mean_col} heatmap")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    return True, "ok"


def _aggregate_from_definition(
    *,
    study_root: Path,
    study_manifest: dict[str, Any],
    aggregate_def: AggregateDef,
    analysis_def: AnalysisDef,
    stage_filter: str | None,
) -> dict[str, Any]:
    if stage_filter and analysis_def.run_stage != stage_filter:
        raise ValueError(
            f"Aggregate '{aggregate_def.title}' targets run_stage '{analysis_def.run_stage}', "
            f"but --stage '{stage_filter}' was requested."
        )

    param_names = list((study_manifest.get("parameters") or {}).keys())
    variables = _normalize_analysis_variables(analysis_def.payload)
    x_name = aggregate_def.x
    y_names = list(aggregate_def.y)
    if x_name not in variables:
        raise KeyError(f"Aggregate '{aggregate_def.title}': x variable '{x_name}' not found in analysis '{analysis_def.title}'.")
    for y in y_names:
        if y not in variables:
            raise KeyError(f"Aggregate '{aggregate_def.title}': y variable '{y}' not found in analysis '{analysis_def.title}'.")

    raw_rows: list[dict[str, Any]] = []
    missing_rows: list[dict[str, Any]] = []
    for case in study_manifest.get("cases") or []:
        if not isinstance(case, dict):
            continue
        case_id = str(case.get("case_id") or "")
        case_path = Path(str(case.get("path") or ""))
        case_params = case.get("parameters") if isinstance(case.get("parameters"), dict) else {}
        case_manifest = _read_json(_run_case_manifest_path(case_path))
        for rep in case_manifest.get("replicates") or []:
            if not isinstance(rep, dict):
                continue
            rep_id = str(rep.get("replicate_id") or "")
            rep_path = Path(str(rep.get("path") or ""))
            rep_manifest = _read_json(_run_replicate_manifest_path(rep_path))
            stage_entries = rep_manifest.get("stages") or []
            stage_entry = next(
                (
                    entry
                    for entry in stage_entries
                    if isinstance(entry, dict) and str(entry.get("stage") or "").strip() == analysis_def.run_stage
                ),
                None,
            )
            if stage_entry is None:
                missing_rows.append({"case_id": case_id, "replicate_id": rep_id, "reason": "run_stage_not_found"})
                continue

            stage_dir = Path(str(stage_entry.get("path") or ""))
            stage_status = _load_stage_status(stage_dir)
            if str(stage_status.get("status") or "") != "completed":
                missing_rows.append({"case_id": case_id, "replicate_id": rep_id, "reason": "run_stage_not_completed"})
                continue

            analysis_manifest = _load_analysis_manifest(stage_dir)
            selected_run: dict[str, Any] | None = None
            for run in reversed(analysis_manifest.get("runs") or []):
                if not isinstance(run, dict):
                    continue
                if str(run.get("analysis_id") or "") != analysis_def.analysis_id:
                    continue
                if str(run.get("status") or "") != "completed":
                    continue
                selected_run = run
                break
            if selected_run is None:
                missing_rows.append({"case_id": case_id, "replicate_id": rep_id, "reason": "analysis_not_completed"})
                continue
            result_dirs = [str(d) for d in (selected_run.get("result_dirs") or []) if str(d).strip()]

            x_spec = variables[x_name]
            x_col = str(x_spec.get("column") or x_name).strip()
            x_file = _resolve_variable_file_for_run(stage_dir=stage_dir, spec=x_spec, result_dirs=result_dirs)
            if x_file is None or not x_file.exists():
                missing_rows.append({"case_id": case_id, "replicate_id": rep_id, "reason": f"missing_x_file:{x_name}"})
                continue
            try:
                x_vals = _load_column_values(x_file, x_col)
            except Exception as exc:
                missing_rows.append({"case_id": case_id, "replicate_id": rep_id, "reason": f"x_column_error:{exc}"})
                continue

            for y_name in y_names:
                y_spec = variables[y_name]
                y_col = str(y_spec.get("column") or y_name).strip()
                y_file = _resolve_variable_file_for_run(stage_dir=stage_dir, spec=y_spec, result_dirs=result_dirs)
                if y_file is None or not y_file.exists():
                    missing_rows.append({"case_id": case_id, "replicate_id": rep_id, "reason": f"missing_y_file:{y_name}"})
                    continue
                try:
                    y_vals = _load_column_values(y_file, y_col)
                except Exception as exc:
                    missing_rows.append({"case_id": case_id, "replicate_id": rep_id, "reason": f"y_column_error:{exc}"})
                    continue

                n = min(len(x_vals), len(y_vals))
                pairs = [(x_vals[i], y_vals[i]) for i in range(n)]
                reduced = _apply_reducer_to_series(pairs=pairs, reducer=aggregate_def.reducer)
                for x_value, y_value in reduced:
                    row = {
                        "case_id": case_id,
                        "replicate_id": rep_id,
                        "run_stage": analysis_def.run_stage,
                        "analysis_title": analysis_def.title,
                        "analysis_id": analysis_def.analysis_id,
                        "x_name": x_name,
                        "x_value": x_value,
                        "y_name": y_name,
                        "y_value": float(y_value),
                    }
                    for p in param_names:
                        row[p] = case_params.get(p)
                    raw_rows.append(row)

    if aggregate_def.on_missing == "fail" and missing_rows:
        first = missing_rows[0]
        raise RuntimeError(
            f"Aggregate '{aggregate_def.title}' missing input for case={first.get('case_id')} "
            f"replicate={first.get('replicate_id')}: {first.get('reason')}"
        )

    # Case-level outputs
    case_output_index: list[dict[str, Any]] = []
    for case in study_manifest.get("cases") or []:
        if not isinstance(case, dict):
            continue
        case_id = str(case.get("case_id") or "")
        case_path = Path(str(case.get("path") or ""))
        case_rows = [r for r in raw_rows if str(r.get("case_id") or "") == case_id]
        case_out_dir = case_path / "aggregate" / aggregate_def.title
        case_out_dir.mkdir(parents=True, exist_ok=True)

        raw_csv = case_out_dir / "raw_replicates.csv"
        raw_fields = [*param_names, "case_id", "replicate_id", "x_name", "x_value", "y_name", "y_value", "run_stage"]
        with raw_csv.open("w", encoding="utf-8", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=raw_fields)
            w.writeheader()
            for row in case_rows:
                w.writerow({k: row.get(k) for k in raw_fields})

        per_x_rows: list[dict[str, Any]] = []
        buckets: dict[tuple[str, Any], list[float]] = {}
        for row in case_rows:
            key = (str(row.get("y_name") or ""), row.get("x_value"))
            buckets.setdefault(key, []).append(float(row.get("y_value")))
        for (y_name, x_value), values in buckets.items():
            s = _compute_stats(values, aggregate_def.stats)
            out_row = {"case_id": case_id, "y_name": y_name, "x_name": x_name, "x_value": x_value}
            out_row.update({f"y_{k}": v for k, v in s.items()})
            per_x_rows.append(out_row)
        per_x_rows.sort(key=lambda r: (str(r.get("y_name")), _to_plot_value(r.get("x_value"))))
        per_x_csv = case_out_dir / "per_x_stats.csv"
        per_x_fields = ["case_id", "y_name", "x_name", "x_value", *[f"y_{k}" for k in aggregate_def.stats]]
        with per_x_csv.open("w", encoding="utf-8", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=per_x_fields)
            w.writeheader()
            for row in per_x_rows:
                write_row = dict(row)
                if "y_n" in write_row and isinstance(write_row["y_n"], float) and write_row["y_n"].is_integer():
                    write_row["y_n"] = int(write_row["y_n"])
                w.writerow({k: write_row.get(k) for k in per_x_fields})

        global_rows: list[dict[str, Any]] = []
        gb: dict[str, list[float]] = {}
        for row in case_rows:
            y_name = str(row.get("y_name") or "")
            gb.setdefault(y_name, []).append(float(row.get("y_value")))
        for y_name, values in gb.items():
            s = _compute_stats(values, aggregate_def.stats)
            out_row = {"case_id": case_id, "y_name": y_name}
            out_row.update({f"y_{k}": v for k, v in s.items()})
            global_rows.append(out_row)
        global_csv = case_out_dir / "global_stats.csv"
        global_fields = ["case_id", "y_name", *[f"y_{k}" for k in aggregate_def.stats]]
        with global_csv.open("w", encoding="utf-8", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=global_fields)
            w.writeheader()
            for row in global_rows:
                write_row = dict(row)
                if "y_n" in write_row and isinstance(write_row["y_n"], float) and write_row["y_n"].is_integer():
                    write_row["y_n"] = int(write_row["y_n"])
                w.writerow({k: write_row.get(k) for k in global_fields})
        case_output_index.append(
            {"case_id": case_id, "raw_csv": str(raw_csv), "per_x_stats_csv": str(per_x_csv), "global_stats_csv": str(global_csv)}
        )

    # Across-cases outputs
    across_dir = study_root / "cases" / "aggregate" / aggregate_def.title
    across_dir.mkdir(parents=True, exist_ok=True)
    across_raw_csv = across_dir / "raw_all_cases.csv"
    across_raw_fields = [*param_names, "case_id", "replicate_id", "x_name", "x_value", "y_name", "y_value", "run_stage"]
    with across_raw_csv.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=across_raw_fields)
        w.writeheader()
        for row in raw_rows:
            w.writerow({k: row.get(k) for k in across_raw_fields})

    per_x_buckets: dict[tuple[Any, ...], list[float]] = {}
    for row in raw_rows:
        key = tuple([row.get(p) for p in param_names] + [row.get("y_name"), row.get("x_value")])
        per_x_buckets.setdefault(key, []).append(float(row.get("y_value")))
    across_per_x_rows: list[dict[str, Any]] = []
    for key, values in per_x_buckets.items():
        out_row = {p: key[i] for i, p in enumerate(param_names)}
        y_name = key[len(param_names)]
        x_value = key[len(param_names) + 1]
        out_row.update({"y_name": y_name, "x_name": x_name, "x_value": x_value})
        out_row.update({f"y_{k}": v for k, v in _compute_stats(values, aggregate_def.stats).items()})
        across_per_x_rows.append(out_row)
    across_per_x_rows.sort(
        key=lambda r: tuple([_to_plot_value(r.get(p)) for p in param_names] + [str(r.get("y_name")), _to_plot_value(r.get("x_value"))])
    )
    across_per_x_csv = across_dir / "across_cases_per_x_stats.csv"
    across_per_x_fields = [*param_names, "y_name", "x_name", "x_value", *[f"y_{k}" for k in aggregate_def.stats]]
    with across_per_x_csv.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=across_per_x_fields)
        w.writeheader()
        for row in across_per_x_rows:
            write_row = dict(row)
            if "y_n" in write_row and isinstance(write_row["y_n"], float) and write_row["y_n"].is_integer():
                write_row["y_n"] = int(write_row["y_n"])
            w.writerow({k: write_row.get(k) for k in across_per_x_fields})

    global_buckets: dict[tuple[Any, ...], list[float]] = {}
    for row in raw_rows:
        key = tuple([row.get(p) for p in param_names] + [row.get("y_name")])
        global_buckets.setdefault(key, []).append(float(row.get("y_value")))
    across_global_rows: list[dict[str, Any]] = []
    for key, values in global_buckets.items():
        out_row = {p: key[i] for i, p in enumerate(param_names)}
        out_row["y_name"] = key[len(param_names)]
        out_row.update({f"y_{k}": v for k, v in _compute_stats(values, aggregate_def.stats).items()})
        across_global_rows.append(out_row)
    across_global_rows.sort(
        key=lambda r: tuple([_to_plot_value(r.get(p)) for p in param_names] + [str(r.get("y_name"))])
    )
    across_global_csv = across_dir / "across_cases_global_stats.csv"
    across_global_fields = [*param_names, "y_name", *[f"y_{k}" for k in aggregate_def.stats]]
    with across_global_csv.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=across_global_fields)
        w.writeheader()
        for row in across_global_rows:
            write_row = dict(row)
            if "y_n" in write_row and isinstance(write_row["y_n"], float) and write_row["y_n"].is_integer():
                write_row["y_n"] = int(write_row["y_n"])
            w.writerow({k: write_row.get(k) for k in across_global_fields})

    manifest = {
        "title": aggregate_def.title,
        "analysis_title": aggregate_def.analysis_title,
        "analysis_id": analysis_def.analysis_id,
        "run_stage": analysis_def.run_stage,
        "x": aggregate_def.x,
        "y": aggregate_def.y,
        "reducer": aggregate_def.reducer,
        "stats": aggregate_def.stats,
        "on_missing": aggregate_def.on_missing,
        "raw_count": len(raw_rows),
        "missing_count": len(missing_rows),
        "case_outputs": case_output_index,
        "across_cases": {
            "raw_csv": str(across_raw_csv),
            "per_x_stats_csv": str(across_per_x_csv),
            "global_stats_csv": str(across_global_csv),
        },
        "missing": missing_rows,
        "generated_at_utc": _utc_now(),
    }
    manifest_path = across_dir / "aggregate_manifest.json"
    _write_json(manifest_path, manifest)
    return {
        "manifest": manifest_path,
        "raw_csv": across_raw_csv,
        "grouped_csv": across_per_x_csv,
        "plot": None,
        "raw_count": len(raw_rows),
        "group_count": len(across_per_x_rows),
    }


def _iter_replicate_variable_records(
    *,
    study_manifest: dict[str, Any],
    variable_name: str,
    stage_filter: str | None,
    value_column_override: str | None,
) -> tuple[list[dict[str, Any]], str | None]:
    analysis_defs_raw = study_manifest.get("analysis") or []
    source_yaml = Path(str(study_manifest.get("source_yaml") or "")).resolve()
    if source_yaml.exists():
        try:
            study_doc = _load_study_yaml(source_yaml)
            _, _, _, _, analyses, _ = _validate_study(study_doc)
            analysis_defs_raw = [
                {
                    "analysis_id": a.analysis_id,
                    "title": a.title,
                    "run_stage": a.run_stage,
                    "payload": a.payload,
                }
                for a in analyses
            ]
        except Exception:
            # Fall back to frozen manifest definitions if current YAML is invalid.
            pass
    analysis_specs: list[dict[str, Any]] = []
    for item in analysis_defs_raw:
        if not isinstance(item, dict):
            continue
        payload = item.get("payload") if isinstance(item.get("payload"), dict) else item
        if not isinstance(payload, dict):
            continue
        title = str(item.get("title") or payload.get("title") or "").strip()
        run_stage = str(item.get("run_stage") or payload.get("run_stage") or "").strip()
        analysis_id = str(item.get("analysis_id") or "").strip()
        vars_map = _normalize_analysis_variables(payload)
        if variable_name in vars_map:
            analysis_specs.append(
                {
                    "analysis_id": analysis_id,
                    "run_stage": run_stage,
                    "variable": variable_name,
                    "spec": vars_map[variable_name],
                    "title": title,
                }
            )
            continue
        if title == variable_name and vars_map:
            first_var = next(iter(vars_map.keys()))
            analysis_specs.append(
                {
                    "analysis_id": analysis_id,
                    "run_stage": run_stage,
                    "variable": first_var,
                    "spec": vars_map[first_var],
                    "title": title,
                }
            )

    if not analysis_specs:
        raise ValueError(
            f"No analysis variable '{variable_name}' found in study manifest analysis definitions."
        )

    param_names = list((study_manifest.get("parameters") or {}).keys())
    raw_rows: list[dict[str, Any]] = []
    resolved_column: str | None = None
    case_entries = study_manifest.get("cases") or []

    for case in case_entries:
        if not isinstance(case, dict):
            continue
        case_id = str(case.get("case_id") or "")
        case_params = case.get("parameters") if isinstance(case.get("parameters"), dict) else {}
        case_path = Path(str(case.get("path") or ""))
        case_manifest = _read_json(_run_case_manifest_path(case_path))
        reps = case_manifest.get("replicates") or []
        for rep in reps:
            if not isinstance(rep, dict):
                continue
            rep_id = str(rep.get("replicate_id") or "")
            rep_path = Path(str(rep.get("path") or ""))
            rep_manifest = _read_json(_run_replicate_manifest_path(rep_path))
            stage_entries = rep_manifest.get("stages") or []
            stage_entry_by_name = {
                str(entry.get("stage") or "").strip(): entry for entry in stage_entries if isinstance(entry, dict)
            }
            for spec_item in analysis_specs:
                run_stage = str(spec_item.get("run_stage") or "").strip()
                if stage_filter and run_stage != stage_filter:
                    continue
                entry = stage_entry_by_name.get(run_stage)
                if not isinstance(entry, dict):
                    continue
                stage_dir = Path(str(entry.get("path") or ""))
                stage_status = _load_stage_status(stage_dir)
                if str(stage_status.get("status") or "") != "completed":
                    continue
                spec = spec_item.get("spec")
                if not isinstance(spec, dict):
                    continue
                analysis_manifest = _load_analysis_manifest(stage_dir)
                latest_result_dirs: list[str] = []
                wanted_analysis_id = str(spec_item.get("analysis_id") or "").strip()
                for run_entry in reversed(analysis_manifest.get("runs") or []):
                    if not isinstance(run_entry, dict):
                        continue
                    if str(run_entry.get("status") or "") != "completed":
                        continue
                    if wanted_analysis_id and str(run_entry.get("analysis_id") or "") != wanted_analysis_id:
                        continue
                    dirs = run_entry.get("result_dirs")
                    if isinstance(dirs, list):
                        latest_result_dirs = [str(d) for d in dirs if str(d).strip()]
                    break

                csv_path: Path | None = None
                file_value = str(spec.get("file") or "").strip()
                if file_value and latest_result_dirs:
                    for d in latest_result_dirs:
                        candidate = Path(d) / file_value
                        if candidate.exists():
                            csv_path = candidate
                            break
                if csv_path is None:
                    csv_path = _resolve_variable_csv_path(stage_dir=stage_dir, spec=spec)
                if csv_path is None or not csv_path.exists():
                    continue
                spec_col = str(spec.get("column") or "").strip() or None
                scalar, used_col = _extract_scalar_from_csv(
                    csv_path,
                    value_column=value_column_override or spec_col,
                )
                if used_col and resolved_column is None:
                    resolved_column = used_col
                if scalar is None:
                    continue
                metric_name = str(spec_item.get("variable") or variable_name)
                row = {
                    "case_id": case_id,
                    "replicate": rep_id,
                    "stage": run_stage,
                    variable_name: scalar,
                    "source_file": str(csv_path),
                    "source_column": used_col,
                    "analysis_title": str(spec_item.get("title") or ""),
                    "analysis_variable": metric_name,
                }
                for param in param_names:
                    row[param] = case_params.get(param)
                raw_rows.append(row)
                break

    return raw_rows, resolved_column


def _aggregate_study_analysis(
    *,
    study_root: Path,
    analysis_name: str,
    value_column: str | None,
    stage_filter: str | None,
) -> dict[str, Any]:
    return _engine_aggregate_study_analysis(
        study_root=study_root,
        analysis_name=analysis_name,
        value_column=value_column,
        stage_filter=stage_filter,
        read_json_fn=_read_json,
        load_source_study_doc_fn=_load_source_study_doc,
        aggregate_defs_from_doc_fn=_aggregate_defs_from_doc,
        analysis_defs_from_doc_fn=_analysis_defs_from_doc,
        aggregate_from_definition_fn=_aggregate_from_definition,
        iter_replicate_variable_records_fn=_iter_replicate_variable_records,
        to_plot_value_fn=_to_plot_value,
        make_aggregate_plot_fn=_make_aggregate_plot,
        utc_now_fn=_utc_now,
        write_json_fn=_write_json,
    )


def _aggregate_study_all(
    *,
    study_root: Path,
    aggregate_title_filter: str | None,
    stage_filter: str | None,
    value_column: str | None,
    legacy_analysis_name: str | None,
) -> list[dict[str, Any]]:
    return _engine_aggregate_study_all(
        study_root=study_root,
        aggregate_title_filter=aggregate_title_filter,
        stage_filter=stage_filter,
        value_column=value_column,
        legacy_analysis_name=legacy_analysis_name,
        read_json_fn=_read_json,
        load_source_study_doc_fn=_load_source_study_doc,
        aggregate_defs_from_doc_fn=_aggregate_defs_from_doc,
        aggregate_study_analysis_fn=_aggregate_study_analysis,
        local_now_fn=_local_now,
        utc_now_fn=_utc_now,
        duration_minutes_fn=_duration_minutes,
        log_task_event_fn=_log_task_event,
        write_named_status_fn=_write_named_status,
        aggregate_status_csv_file=AGGREGATE_STATUS_CSV_FILE,
        aggregate_status_json_file=AGGREGATE_STATUS_JSON_FILE,
    )


def _sort_plot_x(values: list[Any]) -> list[Any]:
    return _present_sort_plot_x(values, sort_values_fn=_sort_values)


def _make_errorbar_plots_for_case_aggregate(case_agg_dir: Path, *, aggregate_title: str) -> list[Path]:
    return _present_make_errorbar_plots_for_case_aggregate(
        case_agg_dir,
        aggregate_title=aggregate_title,
        load_csv_rows_fn=_load_csv_rows,
        to_plot_value_fn=_to_plot_value,
        safe_float_fn=_safe_float,
        slug_underscore_fn=_slug_underscore,
        render_plot_fn=render_plot,
    )


def _make_boxplots_for_case_aggregate(case_agg_dir: Path, *, aggregate_title: str) -> list[Path]:
    return _present_make_boxplots_for_case_aggregate(
        case_agg_dir,
        aggregate_title=aggregate_title,
        load_csv_rows_fn=_load_csv_rows,
        safe_float_fn=_safe_float,
        slug_underscore_fn=_slug_underscore,
        sort_plot_x_fn=_sort_plot_x,
        render_plot_fn=render_plot,
    )


def _row_key_from_params(row: dict[str, str], param_cols: list[str]) -> str:
    return _present_row_key_from_params(row, param_cols)


def _find_param_columns(rows: list[dict[str, str]]) -> list[str]:
    return _present_find_param_columns(rows)


def _make_all_cases_errorbar_plots(across_dir: Path, *, aggregate_title: str) -> list[Path]:
    return _present_make_all_cases_errorbar_plots(
        across_dir,
        aggregate_title=aggregate_title,
        load_csv_rows_fn=_load_csv_rows,
        find_param_columns_fn=_find_param_columns,
        row_key_from_params_fn=_row_key_from_params,
        to_plot_value_fn=_to_plot_value,
        safe_float_fn=_safe_float,
        stat_value_fn=_stat_value,
        slug_underscore_fn=_slug_underscore,
        render_plot_fn=render_plot,
    )


def _make_heatmap_from_rows(
    *,
    rows: list[dict[str, str]],
    x_param: str,
    y_param: str,
    value_col: str,
    out_path: Path,
    title: str,
) -> Path | None:
    return _present_make_heatmap_from_rows(
        rows=rows,
        x_param=x_param,
        y_param=y_param,
        value_col=value_col,
        out_path=out_path,
        title=title,
        safe_float_fn=_safe_float,
        stat_value_fn=_stat_value,
        render_plot_fn=render_plot,
    )


def _make_all_cases_heatmaps(across_dir: Path, *, aggregate_title: str) -> list[Path]:
    return _present_make_all_cases_heatmaps(
        across_dir,
        aggregate_title=aggregate_title,
        load_csv_rows_fn=_load_csv_rows,
        find_param_columns_fn=_find_param_columns,
        safe_float_fn=_safe_float,
        slug_underscore_fn=_slug_underscore,
        make_heatmap_from_rows_fn=_make_heatmap_from_rows,
    )


def _make_all_cases_per_iter_boxplots(across_dir: Path, *, aggregate_title: str) -> list[Path]:
    return _present_make_all_cases_per_iter_boxplots(
        across_dir,
        aggregate_title=aggregate_title,
        load_csv_rows_fn=_load_csv_rows,
        find_param_columns_fn=_find_param_columns,
        sort_plot_x_fn=_sort_plot_x,
        row_key_from_params_fn=_row_key_from_params,
        safe_float_fn=_safe_float,
        slug_underscore_fn=_slug_underscore,
        render_plot_fn=render_plot,
    )


def _plot_study_aggregates(
    *,
    study_root: Path,
    aggregate_title_filter: str | None,
    case_filter: str | None,
) -> dict[str, Any]:
    return _engine_plot_study_aggregates(
        study_root=study_root,
        aggregate_title_filter=aggregate_title_filter,
        case_filter=case_filter,
        read_json_fn=_read_json,
        load_source_study_doc_fn=_load_source_study_doc,
        aggregate_defs_from_doc_fn=_aggregate_defs_from_doc,
        case_matches_selector_fn=_case_matches_selector,
        make_errorbar_plots_for_case_aggregate_fn=_make_errorbar_plots_for_case_aggregate,
        make_boxplots_for_case_aggregate_fn=_make_boxplots_for_case_aggregate,
        make_all_cases_errorbar_plots_fn=_make_all_cases_errorbar_plots,
        make_all_cases_heatmaps_fn=_make_all_cases_heatmaps,
        make_all_cases_per_iter_boxplots_fn=_make_all_cases_per_iter_boxplots,
        local_now_fn=_local_now,
        duration_minutes_fn=_duration_minutes,
        log_task_event_fn=_log_task_event,
        write_named_status_fn=_write_named_status,
        plot_status_csv_file=PLOT_STATUS_CSV_FILE,
        plot_status_json_file=PLOT_STATUS_JSON_FILE,
    )


def _remove_analysis_outputs(
    *,
    study_root: Path,
    analysis_title: str | None,
    case_filter: str | None = None,
    replicate_filter: str | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    study_manifest = _read_json(study_root / "study_manifest.json")
    doc = _load_source_study_doc(study_manifest)
    analysis_defs = _analysis_defs_from_doc(doc)
    target_title = str(analysis_title or "").strip()
    if target_title:
        if target_title not in analysis_defs:
            raise KeyError(f"Unknown analysis title: {target_title}")
        target_ids = {analysis_defs[target_title].analysis_id}
    else:
        target_ids = {a.analysis_id for a in analysis_defs.values()}

    removed_dirs: list[str] = []
    selected_runs = 0
    updated_stage_manifests = 0
    touched_cases = 0
    touched_reps = 0
    for case in study_manifest.get("cases") or []:
        if not isinstance(case, dict):
            continue
        if not _case_matches_selector(case, case_filter):
            continue
        touched_cases += 1
        case_path = Path(str(case.get("path") or ""))
        case_manifest = _read_json(_run_case_manifest_path(case_path))
        for rep in case_manifest.get("replicates") or []:
            if not isinstance(rep, dict):
                continue
            rep_id = str(rep.get("replicate_id") or "")
            if not _replicate_matches_selector(rep_id, replicate_filter):
                continue
            touched_reps += 1
            rep_path = Path(str(rep.get("path") or ""))
            rep_manifest = _read_json(_run_replicate_manifest_path(rep_path))
            for stage_entry in rep_manifest.get("stages") or []:
                if not isinstance(stage_entry, dict):
                    continue
                stage_dir = Path(str(stage_entry.get("path") or ""))
                analysis_manifest = _load_analysis_manifest(stage_dir)
                runs = analysis_manifest.get("runs") or []
                kept_runs: list[dict[str, Any]] = []
                for run in runs:
                    if not isinstance(run, dict):
                        continue
                    analysis_id = str(run.get("analysis_id") or "")
                    if analysis_id in target_ids:
                        selected_runs += 1
                        for d in run.get("result_dirs") or []:
                            p = Path(str(d))
                            if p.exists():
                                try:
                                    if not dry_run:
                                        if p.is_dir():
                                            shutil.rmtree(p)
                                        else:
                                            p.unlink()
                                    removed_dirs.append(str(p))
                                except Exception:
                                    pass
                    else:
                        kept_runs.append(run)
                if len(kept_runs) != len(runs):
                    if not dry_run:
                        analysis_manifest["runs"] = kept_runs
                        _write_analysis_manifest(stage_dir, analysis_manifest)
                    updated_stage_manifests += 1

                    if not dry_run:
                        analysis_status = _load_analysis_status(stage_dir)
                        analyses_obj = analysis_status.get("analyses") if isinstance(analysis_status.get("analyses"), dict) else {}
                        for aid in list(analyses_obj.keys()):
                            if aid in target_ids:
                                analyses_obj.pop(aid, None)
                        analysis_status["analyses"] = analyses_obj
                        _write_analysis_status(stage_dir, analysis_status)

    # Remove top-level analyze status files only for global, unfiltered cleanup.
    if not dry_run and not case_filter and not replicate_filter:
        for p in [
            study_root / ANALYSIS_STATUS_CSV_FILE,
            study_root / ANALYSIS_STATUS_JSON_FILE,
            study_root / LEGACY_ANALYSIS_STATUS_CSV_FILE,
            study_root / LEGACY_ANALYSIS_STATUS_JSON_FILE,
        ]:
            if p.exists():
                try:
                    p.unlink()
                except Exception:
                    pass

    return {
        "analysis_title": target_title or "all",
        "selected_runs": selected_runs,
        "removed_dirs": removed_dirs,
        "updated_stage_manifests": updated_stage_manifests,
        "cases_scanned": touched_cases,
        "replicates_scanned": touched_reps,
        "dry_run": bool(dry_run),
    }


def _remove_aggregate_outputs(
    *,
    study_root: Path,
    aggregate_title: str | None,
    case_filter: str | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    study_manifest = _read_json(study_root / "study_manifest.json")
    doc = _load_source_study_doc(study_manifest)
    aggregate_defs = _aggregate_defs_from_doc(doc)
    target = str(aggregate_title or "").strip()
    if not target:
        titles = set(aggregate_defs.keys())
        # include existing dirs even if not in current YAML
        extra = study_root / "cases" / "aggregate"
        if extra.exists():
            for d in extra.iterdir():
                if d.is_dir():
                    titles.add(d.name)
    else:
        if target not in aggregate_defs and not (study_root / "cases" / "aggregate" / target).exists():
            raise KeyError(f"Unknown aggregate title: {target}")
        titles = {target}

    removed_dirs: list[str] = []
    for case in study_manifest.get("cases") or []:
        if not isinstance(case, dict):
            continue
        if not _case_matches_selector(case, case_filter):
            continue
        case_path = Path(str(case.get("path") or ""))
        for title in titles:
            d = case_path / "aggregate" / title
            if d.exists():
                if not dry_run:
                    shutil.rmtree(d)
                removed_dirs.append(str(d))
    if not case_filter:
        for title in titles:
            d = study_root / "cases" / "aggregate" / title
            if d.exists():
                if not dry_run:
                    shutil.rmtree(d)
                removed_dirs.append(str(d))

    if not dry_run and not case_filter:
        for p in [
            study_root / AGGREGATE_STATUS_CSV_FILE,
            study_root / AGGREGATE_STATUS_JSON_FILE,
        ]:
            if p.exists():
                try:
                    p.unlink()
                except Exception:
                    pass

    return {"aggregate_title": target or "all", "removed_dirs": removed_dirs, "dry_run": bool(dry_run)}


def _remove_cache_outputs(
    *,
    study_root: Path,
    case_filter: str | None = None,
    replicate_filter: str | None = None,
    stage_filter: str | None = None,
    older_than_days: int | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    study_manifest = _read_json(study_root / "study_manifest.json")
    cutoff_ts: float | None = None
    if older_than_days is not None and older_than_days >= 0:
        cutoff_ts = datetime.now().timestamp() - float(older_than_days) * 86400.0

    removed_dirs: list[str] = []
    skipped_age: list[str] = []
    stage_dirs_scanned = 0
    for case in study_manifest.get("cases") or []:
        if not isinstance(case, dict):
            continue
        if not _case_matches_selector(case, case_filter):
            continue
        case_path = Path(str(case.get("path") or ""))
        case_manifest = _read_json(_run_case_manifest_path(case_path))
        for rep in case_manifest.get("replicates") or []:
            if not isinstance(rep, dict):
                continue
            rep_id = str(rep.get("replicate_id") or "")
            if not _replicate_matches_selector(rep_id, replicate_filter):
                continue
            rep_path = Path(str(rep.get("path") or ""))
            rep_manifest = _read_json(_run_replicate_manifest_path(rep_path))
            for stage_entry in rep_manifest.get("stages") or []:
                if not isinstance(stage_entry, dict):
                    continue
                sname = str(stage_entry.get("stage") or "").strip()
                if stage_filter and sname != stage_filter:
                    continue
                stage_dirs_scanned += 1
                stage_dir = Path(str(stage_entry.get("path") or ""))
                for cand in (stage_dir / "reaxkit_workspace" / "cache", stage_dir / ".reaxkit_cache"):
                    if not cand.exists():
                        continue
                    if cutoff_ts is not None and cand.stat().st_mtime > cutoff_ts:
                        skipped_age.append(str(cand))
                        continue
                    if not dry_run:
                        shutil.rmtree(cand, ignore_errors=True)
                    removed_dirs.append(str(cand))
    return {
        "removed_dirs": removed_dirs,
        "skipped_age": skipped_age,
        "stage_dirs_scanned": stage_dirs_scanned,
        "dry_run": bool(dry_run),
    }


def _remove_status_outputs(
    *,
    study_root: Path,
    target: str,
    dry_run: bool = False,
) -> dict[str, Any]:
    target_map: dict[str, list[Path]] = {
        "run-status": [study_root / RUN_STATUS_CSV_FILE, study_root / RUN_STATUS_JSON_FILE],
        "analysis-status": [study_root / ANALYSIS_STATUS_CSV_FILE, study_root / ANALYSIS_STATUS_JSON_FILE],
        "aggregate-status": [study_root / AGGREGATE_STATUS_CSV_FILE, study_root / AGGREGATE_STATUS_JSON_FILE],
        "plot-status": [study_root / PLOT_STATUS_CSV_FILE, study_root / PLOT_STATUS_JSON_FILE],
    }
    files = target_map.get(target, [])
    removed: list[str] = []
    for p in files:
        if p.exists():
            if not dry_run:
                try:
                    p.unlink()
                except Exception:
                    continue
            removed.append(str(p))
    return {"target": target, "removed_files": removed, "dry_run": bool(dry_run)}


def _manage_study(
    *,
    study_root: Path,
    action: str,
    targets: list[str],
    case_filter: str | None,
    replicate_filter: str | None,
    stage_filter: str | None,
    analysis_title: str | None,
    aggregate_title: str | None,
    older_than_days: int | None,
    dry_run: bool,
) -> dict[str, Any]:
    action_norm = str(action or "").strip().lower()
    if action_norm not in {"update-paths", "rename-cases", "remove"}:
        raise ValueError("--manage requires --action update-paths|rename-cases|remove.")
    target_set = [str(t).strip().lower() for t in (targets or []) if str(t).strip()]
    if not target_set:
        if action_norm == "update-paths":
            target_set = ["paths"]
        elif action_norm == "rename-cases":
            target_set = ["case-names"]
        else:
            target_set = ["analysis", "aggregate", "cache"]

    details: list[dict[str, Any]] = []
    for target in target_set:
        started_at = _local_now()
        status = "done"
        message = ""
        data: dict[str, Any] = {}
        try:
            if action_norm == "update-paths":
                if target != "paths":
                    raise ValueError("Action 'update-paths' supports only target 'paths'.")
                data = _update_study_directory_paths(
                    study_root=study_root,
                    case_filter=case_filter,
                    replicate_filter=replicate_filter,
                    dry_run=dry_run,
                )
                message = f"json_updated={data.get('json_files_updated')}"
            elif action_norm == "rename-cases":
                if target != "case-names":
                    raise ValueError("Action 'rename-cases' supports only target 'case-names'.")
                data = _rename_case_directories(
                    study_root=study_root,
                    case_filter=case_filter,
                    dry_run=dry_run,
                )
                message = f"renamed={len(data.get('renamed') or [])} json_updated={data.get('json_files_updated')}"
            else:
                if target == "analysis":
                    data = _remove_analysis_outputs(
                        study_root=study_root,
                        analysis_title=analysis_title,
                        case_filter=case_filter,
                        replicate_filter=replicate_filter,
                        dry_run=dry_run,
                    )
                    message = f"removed_dirs={len(data.get('removed_dirs') or [])}"
                elif target == "aggregate":
                    data = _remove_aggregate_outputs(
                        study_root=study_root,
                        aggregate_title=aggregate_title,
                        case_filter=case_filter,
                        dry_run=dry_run,
                    )
                    message = f"removed_dirs={len(data.get('removed_dirs') or [])}"
                elif target == "cache":
                    data = _remove_cache_outputs(
                        study_root=study_root,
                        case_filter=case_filter,
                        replicate_filter=replicate_filter,
                        stage_filter=stage_filter,
                        older_than_days=older_than_days,
                        dry_run=dry_run,
                    )
                    message = f"removed_dirs={len(data.get('removed_dirs') or [])}"
                elif target in {"run-status", "analysis-status", "aggregate-status", "plot-status"}:
                    data = _remove_status_outputs(study_root=study_root, target=target, dry_run=dry_run)
                    message = f"removed_files={len(data.get('removed_files') or [])}"
                else:
                    raise ValueError(f"Unsupported remove target: {target}")
        except Exception as exc:
            status = "fail"
            message = str(exc)
        finished_at = _local_now()
        details.append(
            {
                "action": action_norm,
                "target": target,
                "status": status,
                "message": message,
                "started_at": started_at,
                "finished_at": finished_at,
                "duration_min": _duration_minutes(started_at, finished_at),
                "dry_run": bool(dry_run),
                "data": data,
            }
        )
        _log_task_event(status.upper(), f"manage {action_norm}:{target}", message)

    summary = {
        "total": len(details),
        "done": sum(1 for d in details if d.get("status") == "done"),
        "fail": sum(1 for d in details if d.get("status") == "fail"),
        "dry_run": bool(dry_run),
    }
    rows = [
        {
            "action": d["action"],
            "target": d["target"],
            "status": d["status"],
            "message": d["message"],
            "started_at": d["started_at"],
            "finished_at": d["finished_at"],
            "duration_min": d["duration_min"],
            "dry_run": d["dry_run"],
        }
        for d in details
    ]
    csv_path, json_path = _write_named_status(
        study_root=study_root,
        rows=rows,
        summary=summary,
        csv_name="manage_status.csv",
        json_name="manage_status.json",
    )
    return {"summary": summary, "details": details, "csv": csv_path, "json": json_path}


def build_parser(parser: argparse.ArgumentParser, *, command: str) -> argparse.ArgumentParser:
    _ = command
    parser.set_defaults(command=STUDY_COMMAND)
    parser.formatter_class = argparse.RawTextHelpFormatter
    parser.description = (
        "Create and initialize parameter-sweep study layouts.\n\n"
        "Examples:\n"
        "  reaxkit study --make-yaml study.yaml\n"
        "  reaxkit study --gen-yaml\n"
        "  reaxkit study --init study.yaml --root .\n"
        "  reaxkit study --init study.yaml --root studies --force\n"
        "  reaxkit study --manage study_MgTemp/ --action update-paths --target paths\n"
        "  reaxkit study --manage study_MgTemp/ --action rename-cases --target case-names\n"
        "  reaxkit study --manage study_MgTemp/ --action remove --target cache --older-than 14 --dry-run\n"
        "  reaxkit study --run study_MgTemp/\n"
        "  reaxkit study --run study_MgTemp/ --parallel-workers 4\n"
        "  reaxkit study --run study_MgTemp/ --rerun-failed\n"
        "  reaxkit study --run study_MgTemp/ --stage MM\n"
        "  reaxkit study --run study_MgTemp/ --case mg_05__temp_300\n"
        "  reaxkit study --analyze study_MgTemp/\n"
        "  reaxkit study --analyze study_MgTemp/ --rerun-failed\n"
        "  reaxkit study --analyze study_MgTemp/ --analysis msd\n"
        "  reaxkit study --aggregate study_MgTemp/\n"
        "  reaxkit study --aggregate study_MgTemp/ --aggregate msd_atom1_aggregation\n"
        "  reaxkit study --aggregate study_MgTemp/ --aggregate msd_atom1_aggregation --stage NVT\n"
        "  reaxkit study --present study_MgTemp/\n"
        "  reaxkit study --present study_MgTemp/ --present msd_atom1_aggregation"
    )

    mode = parser.add_mutually_exclusive_group(required=False)
    mode.add_argument(
        "--init",
        metavar="PATH",
        help="Initialize a study from a YAML file and generate folders/manifests.",
    )
    mode.add_argument(
        "--make-yaml",
        nargs="?",
        const="study.yaml",
        metavar="PATH",
        help="Write a starter study YAML template (default: study.yaml).",
    )
    mode.add_argument(
        "--gen-yaml",
        nargs="?",
        const="study.yaml",
        metavar="PATH",
        help="Alias for --make-yaml.",
    )
    mode.add_argument(
        "--manage",
        metavar="STUDY_ROOT",
        help="Manage study metadata/artifacts (path update and removals).",
    )
    mode.add_argument(
        "--run",
        metavar="STUDY_ROOT",
        help="Execute study stages from an initialized study root folder.",
    )
    mode.add_argument(
        "--analyze",
        metavar="STUDY_ROOT",
        help="Execute analysis pipelines declared in top-level study 'analysis'.",
    )
    mode.add_argument(
        "--aggregate",
        action="append",
        metavar="VALUE",
        help="Aggregate mode. First value is STUDY_ROOT; optional second value is aggregate title filter.",
    )
    mode.add_argument(
        "--present",
        action="append",
        metavar="VALUE",
        help="Presentation mode. First value is STUDY_ROOT; optional second value is aggregate title filter.",
    )
    mode.add_argument(
        "--plot",
        action="append",
        metavar="VALUE",
        help="Deprecated alias for --present.",
    )
    parser.add_argument(
        "--root",
        default=".",
        help="Root folder where the generated <study_name>/ tree will be created.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Allow overwriting existing template file or reusing non-empty study directory.",
    )
    parser.add_argument(
        "--artifact-transfer",
        choices=sorted(SUPPORTED_ARTIFACT_TRANSFER),
        default=DEFAULT_ARTIFACT_TRANSFER,
        help="How consumed artifacts are propagated into downstream stage folders.",
    )
    parser.add_argument(
        "--run-geometry-generator",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Execute geometry_generator.cli_template during study initialization (default: true).",
    )
    parser.add_argument(
        "--strict-actions",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Fail immediately when geometry generation or artifact propagation actions fail.",
    )
    parser.add_argument(
        "--stage",
        default=None,
        help="Run only one stage name (e.g. MM, NPT, NVT).",
    )
    parser.add_argument(
        "--case",
        default=None,
        help="Optional case selector (case_id, combo slug, or shorthand like mg_05__temp_300).",
    )
    parser.add_argument(
        "--replicate",
        default=None,
        help="Optional replicate selector (e.g. rep_01).",
    )
    parser.add_argument(
        "--parallel-workers",
        type=int,
        default=1,
        help="Number of replicate pipelines to run in parallel for --run (default: 1).",
    )
    parser.add_argument(
        "--rerun-failed",
        action="store_true",
        help="For --run, rerun only replicates with fail>0 or wait>0 in run_status.csv; cleans stage artifacts before rerun.",
    )
    parser.add_argument(
        "--analysis",
        default=None,
        help="Analysis title filter for --analyze. For legacy aggregate mode only, this can be a variable/title name.",
    )
    parser.add_argument("--action", choices=["update-paths", "rename-cases", "remove"], default=None, help="Manager action for --manage.")
    parser.add_argument(
        "--target",
        action="append",
        choices=["paths", "case-names", "analysis", "aggregate", "cache", "run-status", "analysis-status", "aggregate-status", "plot-status"],
        default=None,
        help="Manager target(s) for --manage. Can be repeated.",
    )
    parser.add_argument("--analysis-title", default=None, help="Manager filter: analysis title.")
    parser.add_argument("--aggregate-title", default=None, help="Manager filter: aggregate title.")
    parser.add_argument("--dry-run", action="store_true", help="Show what --manage would change without writing/removing.")
    parser.add_argument("--older-than", type=int, default=None, help="For cache removal: only remove entries older than N days.")
    parser.add_argument(
        "--value-column",
        default=None,
        help="For aggregate: explicit numeric column to extract from per-run analysis CSV exports.",
    )
    return parser


def run_main(command: str, args: argparse.Namespace) -> int:
    _ = command

    def _clean_opt(value: Any) -> str | None:
        if value is None:
            return None
        text = str(value).strip()
        return text or None

    yaml_out = getattr(args, "make_yaml", None) or getattr(args, "gen_yaml", None)
    if yaml_out is not None:
        out_path = _write_study_template(Path(str(yaml_out)), force=bool(getattr(args, "force", False)))
        print(f"Created study template: {out_path.resolve()}")
        return 0

    manage_root = getattr(args, "manage", None)
    if manage_root is not None:
        result = _manage_study(
            study_root=Path(str(manage_root)).resolve(),
            action=str(getattr(args, "action", "") or ""),
            targets=list(getattr(args, "target", None) or []),
            case_filter=_clean_opt(getattr(args, "case", None)),
            replicate_filter=_clean_opt(getattr(args, "replicate", None)),
            stage_filter=_clean_opt(getattr(args, "stage", None)),
            analysis_title=_clean_opt(getattr(args, "analysis_title", None)),
            aggregate_title=_clean_opt(getattr(args, "aggregate_title", None)),
            older_than_days=getattr(args, "older_than", None),
            dry_run=bool(getattr(args, "dry_run", False)),
        )
        print(
            "Manage summary: "
            f"done={result['summary']['done']} failed={result['summary']['fail']} "
            f"total={result['summary']['total']} dry_run={result['summary']['dry_run']}"
        )
        print(f"Status CSV: {result['csv']}")
        print(f"Status JSON: {result['json']}")
        return 1 if result["summary"]["fail"] > 0 else 0

    aggregate_values = getattr(args, "aggregate", None)
    if aggregate_values is not None:
        if not isinstance(aggregate_values, list) or len(aggregate_values) == 0:
            raise ValueError("--aggregate requires at least one value: study root.")
        if len(aggregate_values) > 2:
            raise ValueError("--aggregate accepts at most two values: study root and optional aggregate title.")
        target = _clean_opt(aggregate_values[0])
        if target is None:
            raise ValueError("--aggregate requires a study root path.")
        aggregate_title_filter = _clean_opt(aggregate_values[1]) if len(aggregate_values) == 2 else None
        results = _aggregate_study_all(
            study_root=Path(target).resolve(),
            aggregate_title_filter=aggregate_title_filter,
            stage_filter=_clean_opt(getattr(args, "stage", None)),
            value_column=_clean_opt(getattr(args, "value_column", None)),
            legacy_analysis_name=_clean_opt(getattr(args, "analysis", None)),
        )
        for result in results:
            print(f"Aggregate: {result.get('title')}")
            print(f"Raw CSV: {result['raw_csv']}")
            print(f"Grouped CSV: {result['grouped_csv']}")
            if result["plot"] is not None:
                print(f"Plot: {result['plot']}")
            print(f"Manifest: {result['manifest']}")
        print(f"Status CSV: {Path(target).resolve() / AGGREGATE_STATUS_CSV_FILE}")
        print(f"Status JSON: {Path(target).resolve() / AGGREGATE_STATUS_JSON_FILE}")
        return 0

    present_values = getattr(args, "present", None)
    plot_values = present_values if present_values is not None else getattr(args, "plot", None)
    if plot_values is not None:
        flag_name = "--present" if present_values is not None else "--plot"
        if not isinstance(plot_values, list) or len(plot_values) == 0:
            raise ValueError(f"{flag_name} requires at least one value: study root.")
        if len(plot_values) > 2:
            raise ValueError(f"{flag_name} accepts at most two values: study root and optional aggregate title.")
        target = _clean_opt(plot_values[0])
        if target is None:
            raise ValueError(f"{flag_name} requires a study root path.")
        aggregate_title_filter = _clean_opt(plot_values[1]) if len(plot_values) == 2 else None
        result = _plot_study_aggregates(
            study_root=Path(target).resolve(),
            aggregate_title_filter=aggregate_title_filter,
            case_filter=_clean_opt(getattr(args, "case", None)),
        )
        print(f"Status CSV: {Path(target).resolve() / PLOT_STATUS_CSV_FILE}")
        print(f"Status JSON: {Path(target).resolve() / PLOT_STATUS_JSON_FILE}")
        print(f"Plot manifest: {result['manifest']}")
        print(f"Generated plots: {len(result['generated'])}")
        return 0

    run_root = getattr(args, "run", None)
    if run_root is not None:
        run_root_path = Path(str(run_root)).resolve()
        counts = _run_study(
            study_root=run_root_path,
            stage_filter=_clean_opt(getattr(args, "stage", None)),
            case_filter=_clean_opt(getattr(args, "case", None)),
            replicate_filter=_clean_opt(getattr(args, "replicate", None)),
            artifact_transfer=str(getattr(args, "artifact_transfer", DEFAULT_ARTIFACT_TRANSFER)),
            run_geometry_generator=bool(getattr(args, "run_geometry_generator", True)),
            strict_actions=bool(getattr(args, "strict_actions", False)),
            parallel_workers=int(getattr(args, "parallel_workers", 1)),
            rerun_failed=bool(getattr(args, "rerun_failed", False)),
        )
        print(
            "Run summary: "
            f"completed={counts['completed']} skipped={counts['skipped']} "
            f"not_ready={counts['not_ready']} failed={counts['failed']}"
        )
        print(f"Status CSV: {run_root_path / RUN_STATUS_CSV_FILE}")
        print(f"Status JSON: {run_root_path / RUN_STATUS_JSON_FILE}")
        return 1 if counts["failed"] > 0 else 0

    analyze_root = getattr(args, "analyze", None)
    if analyze_root is not None:
        result = _analyze_study(
            study_root=Path(str(analyze_root)).resolve(),
            analysis_filter=_clean_opt(getattr(args, "analysis", None)),
            case_filter=_clean_opt(getattr(args, "case", None)),
            replicate_filter=_clean_opt(getattr(args, "replicate", None)),
            strict_actions=bool(getattr(args, "strict_actions", False)),
            rerun_failed=bool(getattr(args, "rerun_failed", False)),
        )
        counts = result["counts"]
        print(
            "Analyze summary: "
            f"completed={counts['completed']} skipped={counts['skipped']} failed={counts['failed']}"
        )
        print(f"Status CSV: {result['csv']}")
        print(f"Status JSON: {result['manifest']}")
        return 1 if counts["failed"] > 0 else 0

    init_path = getattr(args, "init", None)
    if init_path is None:
        raise ValueError("Missing required mode. Use --init, --manage, --run, --analyze, --aggregate, --present, or --make-yaml.")

    study_root = _init_study(
        Path(str(init_path)),
        root=Path(str(getattr(args, "root", "."))),
        force=bool(getattr(args, "force", False)),
        run_geometry_generator=bool(getattr(args, "run_geometry_generator", True)),
        artifact_transfer=str(getattr(args, "artifact_transfer", DEFAULT_ARTIFACT_TRANSFER)),
        strict_actions=bool(getattr(args, "strict_actions", False)),
    )
    print(f"Study initialized at: {study_root}")
    print(f"Manifest: {(study_root / 'study_manifest.json')}")
    return 0
