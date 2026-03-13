"""Study planning workflow: generate sweep/replicate folder structures from YAML."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import itertools
import json
import math
import shutil
import statistics
import subprocess
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

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
  mg_percent: [40, 60]
  temperature: [300, 500]
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


@dataclass(frozen=True)
class StageDef:
    name: str
    payload: dict[str, Any]


@dataclass(frozen=True)
class AnalysisDef:
    analysis_id: str
    title: str
    run_stage: str
    payload: dict[str, Any]


@dataclass(frozen=True)
class AggregateDef:
    title: str
    analysis_title: str
    x: str
    y: list[str]
    reducer: str
    stats: list[str]
    on_missing: str


@dataclass(frozen=True)
class ArtifactRef:
    stage: str
    artifact: str


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _local_now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _duration_minutes(started_at: str | None, finished_at: str | None) -> float | None:
    if not started_at or not finished_at:
        return None
    try:
        start_dt = datetime.strptime(started_at, "%Y-%m-%d %H:%M:%S")
        end_dt = datetime.strptime(finished_at, "%Y-%m-%d %H:%M:%S")
    except Exception:
        return None
    delta_min = (end_dt - start_dt).total_seconds() / 60.0
    return round(delta_min, 3) if delta_min >= 0 else None


def _stage_label_width(max_stage_chars: int) -> int:
    # Requested format width: 23 + max_stage_len + 2
    return 23 + int(max_stage_chars) + 2


def _analysis_label_width(max_analysis_chars: int) -> int:
    # Requested format width: 24 + max_analysis_len + 2
    return 24 + int(max_analysis_chars) + 2


def _log_stage_event(
    case_id: str,
    replicate_id: str,
    stage_name: str,
    tag: str,
    detail: str | None = None,
    *,
    stage_block_width: int | None = None,
) -> None:
    tag_block = f"[{str(tag).upper()}]".ljust(8)
    stage_block = f"{case_id} {replicate_id} {stage_name}"
    if stage_block_width is not None:
        stage_block = stage_block.ljust(stage_block_width)
    line = f"{tag_block}{stage_block}{_local_now()}"
    if detail:
        line = f"{line}  {detail}"
    print(line)


def _log_task_event(tag: str, info: str, detail: str | None = None) -> None:
    tag_block = f"[{str(tag).upper()}]".ljust(8)
    line = f"{tag_block}{str(info)}  {_local_now()}"
    if detail:
        line = f"{line}  {detail}"
    print(line)


def _write_study_run_status(
    *,
    study_root: Path,
    rows: list[dict[str, Any]],
    summary: dict[str, Any],
) -> tuple[Path, Path]:
    status_dir = study_root
    status_dir.mkdir(parents=True, exist_ok=True)

    json_path = status_dir / RUN_STATUS_JSON_FILE
    csv_path = status_dir / RUN_STATUS_CSV_FILE

    payload = {
        "generated_at_utc": _utc_now(),
        "summary": summary,
        "rows": rows,
    }
    _write_json(json_path, payload)

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


def _write_named_status(
    *,
    study_root: Path,
    rows: list[dict[str, Any]],
    summary: dict[str, Any],
    csv_name: str,
    json_name: str,
) -> tuple[Path, Path]:
    json_path = study_root / json_name
    csv_path = study_root / csv_name
    payload = {
        "generated_at_utc": _utc_now(),
        "summary": summary,
        "rows": rows,
    }
    _write_json(json_path, payload)

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


def _resolve_existing_file(primary: Path, *fallbacks: str) -> Path:
    if primary.exists():
        return primary
    for name in fallbacks:
        candidate = primary.parent / name
        if candidate.exists():
            return candidate
    return primary


def _run_stage_manifest_path(stage_dir: Path) -> Path:
    return _resolve_existing_file(stage_dir / RUN_STAGE_MANIFEST_FILE, LEGACY_STAGE_MANIFEST_FILE)


def _run_replicate_manifest_path(rep_dir: Path) -> Path:
    return _resolve_existing_file(rep_dir / RUN_REPLICATE_MANIFEST_FILE, LEGACY_REPLICATE_MANIFEST_FILE)


def _run_case_manifest_path(case_dir: Path) -> Path:
    return _resolve_existing_file(case_dir / RUN_CASE_MANIFEST_FILE, LEGACY_CASE_MANIFEST_FILE)


def _run_status_csv_path(study_root: Path) -> Path:
    return _resolve_existing_file(study_root / RUN_STATUS_CSV_FILE, LEGACY_RUN_STATUS_CSV_FILE)


def _analysis_status_csv_path(study_root: Path) -> Path:
    return _resolve_existing_file(study_root / ANALYSIS_STATUS_CSV_FILE, LEGACY_ANALYSIS_STATUS_CSV_FILE)


def _analysis_status_json_path(study_root: Path) -> Path:
    return _resolve_existing_file(study_root / ANALYSIS_STATUS_JSON_FILE, LEGACY_ANALYSIS_STATUS_JSON_FILE)


def _load_study_run_status_rows(study_root: Path) -> list[dict[str, str]]:
    csv_path = _run_status_csv_path(study_root)
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Cannot use rerun mode because status file does not exist: {csv_path}. "
            "Run 'reaxkit study --run <study_root>' once first."
        )
    with csv_path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        return [dict(row) for row in reader]


def _load_analysis_status_rows(study_root: Path) -> list[dict[str, str]]:
    csv_path = _analysis_status_csv_path(study_root)
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Cannot use analyze rerun mode because status file does not exist: {csv_path}. "
            "Run 'reaxkit study --analyze <study_root>' once first."
        )
    with csv_path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        return [dict(row) for row in reader]


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


def _slug(value: Any) -> str:
    text = str(value).strip()
    out = []
    for ch in text:
        if ch.isalnum() or ch in {"-", "_", "."}:
            out.append(ch)
        else:
            out.append("_")
    slug = "".join(out).strip("_")
    return slug or "value"


def _slug_underscore(value: Any) -> str:
    return _slug(value).replace("-", "_").replace(".", "_")


def _load_study_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Study file not found: {path}")
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError("Study YAML must be a mapping at top level.")
    return raw


def _load_source_study_doc(study_manifest: dict[str, Any]) -> dict[str, Any]:
    source_yaml = Path(str(study_manifest.get("source_yaml") or "")).resolve()
    if not source_yaml.exists():
        raise FileNotFoundError(f"Study source YAML not found: {source_yaml}")
    return _load_study_yaml(source_yaml)


def _analysis_defs_from_doc(doc: dict[str, Any]) -> dict[str, AnalysisDef]:
    _, _, _, _, analyses, _ = _validate_study(doc)
    out: dict[str, AnalysisDef] = {}
    for a in analyses:
        out[a.title] = a
    return out


def _aggregate_defs_from_doc(doc: dict[str, Any]) -> dict[str, AggregateDef]:
    raw = doc.get("aggregate") or []
    if not isinstance(raw, list):
        raise ValueError("aggregate must be a list when provided.")
    out: dict[str, AggregateDef] = {}
    for item in raw:
        if not isinstance(item, dict):
            raise ValueError("Each aggregate entry must be a mapping.")
        title = str(item.get("title") or "").strip()
        analysis_title = str(item.get("analysis_title") or "").strip()
        x = str(item.get("x") or "").strip()
        y_raw = item.get("y")
        reducer = str(item.get("reducer") or "identity").strip().lower()
        stats_raw = item.get("stats") or ["mean", "std", "min", "max", "sem", "n"]
        on_missing = str(item.get("on_missing") or "skip").strip().lower()

        if not title:
            raise ValueError("Each aggregate entry requires non-empty 'title'.")
        if title in out:
            raise ValueError(f"Duplicate aggregate title: {title}")
        if not analysis_title:
            raise ValueError(f"aggregate.{title}: analysis_title is required.")
        if not x:
            raise ValueError(f"aggregate.{title}: x is required.")

        if isinstance(y_raw, str):
            y = [y_raw.strip()] if y_raw.strip() else []
        elif isinstance(y_raw, list):
            y = [str(v).strip() for v in y_raw if str(v).strip()]
        else:
            y = []
        if not y:
            raise ValueError(f"aggregate.{title}: y must be a non-empty string or list.")

        if not isinstance(stats_raw, list) or not stats_raw:
            raise ValueError(f"aggregate.{title}: stats must be a non-empty list.")
        stats = [str(v).strip().lower() for v in stats_raw if str(v).strip()]
        if not stats:
            raise ValueError(f"aggregate.{title}: stats must contain at least one entry.")
        if on_missing not in {"skip", "fail"}:
            raise ValueError(f"aggregate.{title}: on_missing must be 'skip' or 'fail'.")

        out[title] = AggregateDef(
            title=title,
            analysis_title=analysis_title,
            x=x,
            y=y,
            reducer=reducer,
            stats=stats,
            on_missing=on_missing,
        )
    return out


def _validate_study(
    doc: dict[str, Any],
) -> tuple[str, dict[str, list[Any]], int, list[StageDef], list[AnalysisDef], str | None]:
    study_name = str(doc.get("study_name") or "").strip()
    if not study_name:
        raise ValueError("study_name is required.")

    params_raw = doc.get("parameters") or {}
    if not isinstance(params_raw, dict) or not params_raw:
        raise ValueError("parameters must be a non-empty mapping.")
    parameters: dict[str, list[Any]] = {}
    for key, values in params_raw.items():
        key_s = str(key).strip()
        if not key_s:
            raise ValueError("Parameter names cannot be empty.")
        if not isinstance(values, list) or not values:
            raise ValueError(f"Parameter '{key_s}' must be a non-empty list.")
        parameters[key_s] = list(values)

    replicates = int(doc.get("replicates", 1))
    if replicates < 1:
        raise ValueError("replicates must be >= 1.")

    run_raw = doc.get("run")
    if run_raw is None:
        run_raw = doc.get("workflow")  # backward-compatible
    if not isinstance(run_raw, list) or not run_raw:
        raise ValueError("run must be a non-empty list (or provide legacy workflow).")
    stages: list[StageDef] = []
    seen: set[str] = set()
    for item in run_raw:
        if not isinstance(item, dict):
            raise ValueError("Each run stage must be a mapping.")
        stage_name = str(item.get("stage") or "").strip()
        if not stage_name:
            raise ValueError("Each run stage requires a non-empty 'stage' name.")
        if stage_name in seen:
            raise ValueError(f"Duplicate run stage: {stage_name}")
        seen.add(stage_name)
        stages.append(StageDef(name=stage_name, payload=dict(item)))

    analysis_raw = doc.get("analysis") or []
    if not isinstance(analysis_raw, list):
        raise ValueError("analysis must be a list when provided.")
    analyses: list[AnalysisDef] = []
    seen_analysis_ids: set[str] = set()
    for idx, item in enumerate(analysis_raw, start=1):
        if not isinstance(item, dict):
            raise ValueError("Each analysis entry must be a mapping.")
        title = str(item.get("title") or "").strip()
        if not title:
            # Backward-compatible fallback.
            title = str(item.get("command") or "").strip()
        run_stage = str(item.get("run_stage") or "").strip()
        if not title:
            raise ValueError("Each analysis entry requires non-empty 'title'.")
        if not run_stage:
            raise ValueError("Each analysis entry requires non-empty 'run_stage'.")
        analysis_id = str(item.get("analysis_id") or f"analysis_{idx:02d}_{_slug_underscore(title)}").strip()
        if analysis_id in seen_analysis_ids:
            raise ValueError(f"Duplicate analysis_id: {analysis_id}")
        seen_analysis_ids.add(analysis_id)
        analyses.append(
            AnalysisDef(
                analysis_id=analysis_id,
                title=title,
                run_stage=run_stage,
                payload=dict(item),
            )
        )

    template_raw = doc.get("template")
    template_dir: str | None = None
    if template_raw is not None:
        template_dir = str(template_raw).strip() or None

    return study_name, parameters, replicates, stages, analyses, template_dir


def _render_value(value: Any, context: dict[str, Any]) -> Any:
    if isinstance(value, str):
        try:
            return value.format(**context)
        except KeyError as exc:
            missing = str(exc).strip("'")
            raise ValueError(f"Missing placeholder '{missing}' in study context.") from exc
    if isinstance(value, dict):
        return {str(k): _render_value(v, context) for k, v in value.items()}
    if isinstance(value, list):
        return [_render_value(v, context) for v in value]
    return value


def _enumerate_cases(parameters: dict[str, list[Any]]) -> list[dict[str, Any]]:
    keys = list(parameters.keys())
    values_grid = [parameters[k] for k in keys]
    combos = []
    for vals in itertools.product(*values_grid):
        combos.append({k: v for k, v in zip(keys, vals)})
    return combos


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _resolve_template_path(template_value: str, *, study_dir: Path) -> Path:
    candidate = Path(str(template_value))
    if candidate.is_absolute():
        return candidate
    return (study_dir / candidate).resolve()


def _copy_template_into_replicate(*, template_root: Path, replicate_dir: Path) -> None:
    if not template_root.exists():
        raise FileNotFoundError(f"Study template directory not found: {template_root}")
    if not template_root.is_dir():
        raise NotADirectoryError(f"Study template path is not a directory: {template_root}")

    for source in template_root.iterdir():
        destination = replicate_dir / source.name
        if source.is_dir():
            shutil.copytree(source, destination, dirs_exist_ok=True)
        else:
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)


def _apply_stage_slurm_job_name(*, stage_dir: Path, case_number: int, replicate_number: int, stage_name: str) -> None:
    submit_script = stage_dir / "submit_and_wait.sh"
    if not submit_script.exists():
        return

    job_name = f"C{case_number:02d}_R{replicate_number:02d}_{stage_name}"
    lines = submit_script.read_text(encoding="utf-8").splitlines()
    out_lines: list[str] = []
    replaced = False
    for line in lines:
        if line.strip().startswith("#SBATCH --job-name="):
            out_lines.append(f"#SBATCH --job-name={job_name}")
            replaced = True
        else:
            out_lines.append(line)

    if not replaced:
        insert_at = 1 if out_lines and out_lines[0].startswith("#!") else 0
        out_lines.insert(insert_at, f"#SBATCH --job-name={job_name}")

    submit_script.write_text("\n".join(out_lines) + "\n", encoding="utf-8")


def _collect_template_stage_relpaths(*, template_root: Path, stage_names: list[str]) -> dict[str, Path]:
    stage_to_relpath: dict[str, Path] = {}
    for stage_name in stage_names:
        matches: list[Path] = []
        for candidate in template_root.rglob(stage_name):
            if candidate.is_dir():
                matches.append(candidate)
        if not matches:
            raise ValueError(
                f"Template validation failed: stage folder '{stage_name}' not found under template '{template_root}'."
            )
        if len(matches) > 1:
            rels = ", ".join(str(p.relative_to(template_root)) for p in matches)
            raise ValueError(
                f"Template validation failed: stage folder '{stage_name}' is ambiguous under template '{template_root}': {rels}"
            )
        stage_to_relpath[stage_name] = matches[0].relative_to(template_root)
    return stage_to_relpath


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
            rep_id = f"replicate_{rep_idx:02d}"
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
    if not path.exists():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}.")
    return payload


def _stage_status_path(stage_dir: Path) -> Path:
    return stage_dir / STAGE_STATUS_FILE


def _load_stage_status(stage_dir: Path) -> dict[str, Any]:
    path = _stage_status_path(stage_dir)
    if not path.exists():
        return {
            "status": "pending",
            "jobs": [],
            "updated_at_utc": None,
        }
    payload = _read_json(path)
    payload.setdefault("jobs", [])
    payload.setdefault("status", "pending")
    return payload


def _write_stage_status(stage_dir: Path, status: dict[str, Any]) -> None:
    payload = dict(status)
    payload["updated_at_utc"] = _utc_now()
    _write_json(_stage_status_path(stage_dir), payload)


def _analysis_status_path(stage_dir: Path) -> Path:
    return stage_dir / ANALYSIS_STATUS_FILE


def _load_analysis_status(stage_dir: Path) -> dict[str, Any]:
    path = _analysis_status_path(stage_dir)
    if not path.exists():
        return {"analyses": {}, "updated_at_utc": None}
    payload = _read_json(path)
    if not isinstance(payload.get("analyses"), dict):
        payload["analyses"] = {}
    return payload


def _write_analysis_status(stage_dir: Path, status: dict[str, Any]) -> None:
    payload = dict(status)
    payload["updated_at_utc"] = _utc_now()
    _write_json(_analysis_status_path(stage_dir), payload)


def _analysis_manifest_path(stage_dir: Path) -> Path:
    return stage_dir / ANALYSIS_MANIFEST_FILE


def _load_analysis_manifest(stage_dir: Path) -> dict[str, Any]:
    path = _analysis_manifest_path(stage_dir)
    if not path.exists():
        return {"runs": [], "updated_at_utc": None}
    payload = _read_json(path)
    if not isinstance(payload.get("runs"), list):
        payload["runs"] = []
    return payload


def _write_analysis_manifest(stage_dir: Path, payload: dict[str, Any]) -> None:
    out = dict(payload)
    out["updated_at_utc"] = _utc_now()
    _write_json(_analysis_manifest_path(stage_dir), out)


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


def _canonical_token(text: str) -> str:
    return "".join(ch.lower() for ch in str(text) if ch.isalnum())


def _short_param_name(name: str) -> str:
    low = str(name).strip().lower()
    if low.endswith("_percent"):
        return low[: -len("_percent")]
    if low == "temperature":
        return "temp"
    return low


def _format_param_value_for_case(value: Any) -> str:
    try:
        f = float(value)
        if f.is_integer():
            i = int(f)
            return f"{i:02d}" if 0 <= i < 100 else str(i)
    except Exception:
        pass
    return _slug(value)


def _case_label_from_params(params: dict[str, Any]) -> str:
    parts = [f"{_short_param_name(k)}_{_format_param_value_for_case(v)}" for k, v in params.items()]
    return "__".join(parts)


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
    manifest = _read_json(study_root / "study_manifest.json")
    case_entries = manifest.get("cases") or []
    if not isinstance(case_entries, list):
        raise ValueError("Invalid study_manifest.json: cases must be a list.")

    counts = {"completed": 0, "skipped": 0, "failed": 0, "not_ready": 0}
    tasks: list[tuple[dict[str, Any], dict[str, Any]]] = []
    row_map: dict[tuple[str, str], dict[str, Any]] = {}

    rerun_targets: set[tuple[str, str]] | None = None
    if rerun_failed:
        rerun_targets = set()
        for row in _load_study_run_status_rows(study_root):
            case_id = str(row.get("case_id") or "").strip()
            rep_id = str(row.get("replicate_id") or "").strip()
            fail_n = _to_int(row.get("fail"))
            wait_n = _to_int(row.get("wait"))
            status = str(row.get("status") or "").strip().lower()
            if not case_id or not rep_id:
                continue
            if fail_n > 0 or wait_n > 0 or status in {"failed", "waiting"}:
                rerun_targets.add((case_id, rep_id))

    for case in case_entries:
        if not isinstance(case, dict):
            continue
        if not _case_matches_selector(case, case_filter):
            continue
        case_id = str(case.get("case_id") or "")
        case_path = Path(str(case.get("path") or ""))
        case_manifest = _read_json(_run_case_manifest_path(case_path))
        reps = case_manifest.get("replicates") or []
        for rep in reps:
            if not isinstance(rep, dict):
                continue
            rep_id = str(rep.get("replicate_id") or "")
            if replicate_filter and rep_id != replicate_filter:
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
        _write_study_run_status(study_root=study_root, rows=rows, summary=summary)

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
            rec = _run_single_replicate_pipeline(
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
                _run_single_replicate_pipeline,
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
    manifest = _read_json(study_root / "study_manifest.json")
    source_yaml = Path(str(manifest.get("source_yaml") or "")).resolve()
    if not source_yaml.exists():
        raise FileNotFoundError(f"Study source YAML not found: {source_yaml}")
    study_doc = _load_study_yaml(source_yaml)
    _, _, _, _, analyses, _ = _validate_study(study_doc)
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
    analysis_block_width = _analysis_label_width(max_analysis_chars=max_analysis_chars)

    records: list[dict[str, Any]] = []
    counts = {"completed": 0, "failed": 0, "skipped": 0}
    study_dir = source_yaml.parent
    rerun_targets: set[tuple[str, str, str]] | None = None
    if rerun_failed:
        rerun_targets = set()
        for row in _load_analysis_status_rows(study_root):
            case_id = str(row.get("case_id") or "").strip()
            rep_id = str(row.get("replicate_id") or "").strip()
            analysis_id = str(row.get("analysis_id") or "").strip()
            fail_n = _to_int(row.get("fail"))
            wait_n = _to_int(row.get("wait"))
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
        if not _case_matches_selector(case, case_filter):
            continue
        case_id = str(case.get("case_id") or "")
        case_path = Path(str(case.get("path") or ""))
        case_manifest = _read_json(_run_case_manifest_path(case_path))
        for rep in case_manifest.get("replicates") or []:
            if not isinstance(rep, dict):
                continue
            rep_id = str(rep.get("replicate_id") or "")
            if replicate_filter and rep_id != replicate_filter:
                continue
            rep_path = Path(str(rep.get("path") or ""))
            rep_manifest = _read_json(_run_replicate_manifest_path(rep_path))
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
                    _log_stage_event(
                        case_id,
                        rep_id,
                        title,
                        "SKIP",
                        reason,
                        stage_block_width=analysis_block_width,
                    )
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
                stage_status = _load_stage_status(stage_dir)
                if str(stage_status.get("status") or "") != "completed":
                    counts["skipped"] += 1
                    status_value = "skipped"
                    reason = f"run_stage_not_completed:{stage_status.get('status')}"
                    ctr = _analysis_counters(status=status_value, reason=reason)
                    _log_stage_event(
                        case_id,
                        rep_id,
                        title,
                        "WAIT",
                        reason,
                        stage_block_width=analysis_block_width,
                    )
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
                rendered_analysis = _render_value((analysis_def.get("payload") or analysis_def), context)
                started = _utc_now()
                started_local = _local_now()
                _log_stage_event(case_id, rep_id, title, "RUN", stage_block_width=analysis_block_width)
                try:
                    step_records = _run_analysis_steps(
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

                finished = _utc_now()
                finished_local = _local_now()
                duration_min = _duration_minutes(started_local, finished_local)
                result_dirs = _collect_result_dirs_from_step_records(step_records)
                analysis_status = _load_analysis_status(stage_dir)
                analysis_status.setdefault("analyses", {})
                analysis_status["analyses"][analysis_id] = {
                    "status": status_value,
                    "title": title,
                    "run_stage": run_stage,
                    "started_at_utc": started,
                    "finished_at_utc": finished,
                    "result_dirs": result_dirs,
                }
                _write_analysis_status(stage_dir, analysis_status)
                analysis_manifest = _load_analysis_manifest(stage_dir)
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
                _write_analysis_manifest(stage_dir, analysis_manifest)
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
                    _log_stage_event(case_id, rep_id, title, "DONE", stage_block_width=analysis_block_width)
                else:
                    counts["failed"] += 1
                    _log_stage_event(case_id, rep_id, title, "FAIL", stage_block_width=analysis_block_width)
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
        "generated_at_utc": _utc_now(),
        "counts": counts,
        "records": records,
        "analysis_filter": analysis_filter,
    }
    manifest_path = out_dir / ANALYSIS_STATUS_JSON_FILE
    _write_json(manifest_path, payload)

    csv_path = out_dir / ANALYSIS_STATUS_CSV_FILE
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


def _safe_float(value: Any) -> float | None:
    try:
        fv = float(value)
    except Exception:
        return None
    if not math.isfinite(fv):
        return None
    return fv


def _load_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        return [dict(row) for row in reader]


def _extract_scalar_from_csv(
    csv_path: Path,
    *,
    value_column: str | None,
) -> tuple[float | None, str | None]:
    rows = _load_csv_rows(csv_path)
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
                vals = [_safe_float(row.get(col)) for row in rows]
                vals = [v for v in vals if v is not None]
                if vals:
                    target_col = col
                    break

    if target_col is None:
        return None, None
    vals = [_safe_float(row.get(target_col)) for row in rows]
    vals = [v for v in vals if v is not None]
    if not vals:
        return None, target_col
    return float(sum(vals) / len(vals)), target_col


def _load_column_values(csv_path: Path, column: str) -> list[Any]:
    rows = _load_csv_rows(csv_path)
    if not rows:
        return []
    if column not in rows[0]:
        raise KeyError(f"Column '{column}' not found in {csv_path}.")
    return [row.get(column) for row in rows]


def _resolve_variable_file_for_run(
    *,
    stage_dir: Path,
    spec: dict[str, str],
    result_dirs: list[str] | None,
) -> Path | None:
    file_value = str(spec.get("file") or "").strip()
    if not file_value:
        return None
    if result_dirs:
        for result_dir in result_dirs:
            candidate = Path(str(result_dir)) / file_value
            if candidate.exists():
                return candidate
    return _resolve_variable_csv_path(stage_dir=stage_dir, spec=spec)


def _apply_reducer_to_series(
    *,
    pairs: list[tuple[Any, Any]],
    reducer: str,
) -> list[tuple[Any, float]]:
    vals: list[tuple[Any, float]] = []
    for x, y in pairs:
        fy = _safe_float(y)
        if fy is None:
            continue
        vals.append((x, fy))
    if not vals:
        return []
    r = str(reducer or "identity").strip().lower()
    if r == "identity":
        return vals
    ys = [v for _, v in vals]
    xs_num = [_safe_float(x) for x, _ in vals]
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


def _compute_stats(values: list[float], wanted: list[str]) -> dict[str, float]:
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


def _normalize_stage_variables(rendered_stage: dict[str, Any]) -> dict[str, dict[str, str]]:
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
        # Backward-compatible legacy spec with just file+column.
        if "file" not in payload:
            continue
        variables[name] = payload
    return variables


def _resolve_variable_csv_path(*, stage_dir: Path, spec: dict[str, str]) -> Path | None:
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


def _sort_values(values: list[Any]) -> list[Any]:
    def _key(v: Any):
        fv = _safe_float(v)
        if fv is not None:
            return (0, fv, str(v))
        return (1, str(v), str(v))

    return sorted(values, key=_key)


def _to_plot_value(v: Any):
    fv = _safe_float(v)
    return fv if fv is not None else str(v)


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
        out_row.update(_compute_stats(values, aggregate_def.stats))
        across_per_x_rows.append(out_row)
    across_per_x_rows.sort(
        key=lambda r: tuple([_to_plot_value(r.get(p)) for p in param_names] + [str(r.get("y_name")), _to_plot_value(r.get("x_value"))])
    )
    across_per_x_csv = across_dir / "across_cases_per_x_stats.csv"
    across_per_x_fields = [*param_names, "y_name", "x_name", "x_value", *aggregate_def.stats]
    with across_per_x_csv.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=across_per_x_fields)
        w.writeheader()
        for row in across_per_x_rows:
            write_row = dict(row)
            if "n" in write_row and isinstance(write_row["n"], float) and write_row["n"].is_integer():
                write_row["n"] = int(write_row["n"])
            w.writerow({k: write_row.get(k) for k in across_per_x_fields})

    global_buckets: dict[tuple[Any, ...], list[float]] = {}
    for row in raw_rows:
        key = tuple([row.get(p) for p in param_names] + [row.get("y_name")])
        global_buckets.setdefault(key, []).append(float(row.get("y_value")))
    across_global_rows: list[dict[str, Any]] = []
    for key, values in global_buckets.items():
        out_row = {p: key[i] for i, p in enumerate(param_names)}
        out_row["y_name"] = key[len(param_names)]
        out_row.update(_compute_stats(values, aggregate_def.stats))
        across_global_rows.append(out_row)
    across_global_rows.sort(
        key=lambda r: tuple([_to_plot_value(r.get(p)) for p in param_names] + [str(r.get("y_name"))])
    )
    across_global_csv = across_dir / "across_cases_global_stats.csv"
    across_global_fields = [*param_names, "y_name", *aggregate_def.stats]
    with across_global_csv.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=across_global_fields)
        w.writeheader()
        for row in across_global_rows:
            write_row = dict(row)
            if "n" in write_row and isinstance(write_row["n"], float) and write_row["n"].is_integer():
                write_row["n"] = int(write_row["n"])
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
    study_manifest = _read_json(study_root / "study_manifest.json")
    try:
        study_doc = _load_source_study_doc(study_manifest)
        aggregate_defs = _aggregate_defs_from_doc(study_doc)
        analysis_defs_by_title = _analysis_defs_from_doc(study_doc)
    except Exception:
        aggregate_defs = {}
        analysis_defs_by_title = {}

    # New aggregation model: aggregate definitions by title.
    if analysis_name in aggregate_defs:
        agg_def = aggregate_defs[analysis_name]
        if agg_def.analysis_title not in analysis_defs_by_title:
            raise KeyError(
                f"Aggregate '{agg_def.title}' references unknown analysis_title '{agg_def.analysis_title}'."
            )
        return _aggregate_from_definition(
            study_root=study_root,
            study_manifest=study_manifest,
            aggregate_def=agg_def,
            analysis_def=analysis_defs_by_title[agg_def.analysis_title],
            stage_filter=stage_filter,
        )

    # Backward-compatible legacy mode.
    param_names = list((study_manifest.get("parameters") or {}).keys())
    raw_rows, chosen_column = _iter_replicate_variable_records(
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

    grouped_rows.sort(key=lambda row: tuple(_to_plot_value(row[p]) for p in param_names))
    grouped_csv = out_dir / f"{analysis_name}_grouped.csv"
    grouped_fields = [*param_names, mean_col, std_col, sem_col, "n"]
    with grouped_csv.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=grouped_fields)
        writer.writeheader()
        for row in grouped_rows:
            writer.writerow({k: row.get(k) for k in grouped_fields})

    plot_path = out_dir / f"{analysis_name}_plot.png"
    plot_ok, plot_msg = _make_aggregate_plot(
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
        "generated_at_utc": _utc_now(),
    }
    manifest_path = out_dir / "aggregate_manifest.json"
    _write_json(manifest_path, manifest)
    return {
        "manifest": manifest_path,
        "raw_csv": raw_csv,
        "grouped_csv": grouped_csv,
        "plot": plot_path if plot_ok else None,
        "raw_count": len(raw_rows),
        "group_count": len(grouped_rows),
    }


def _aggregate_study_all(
    *,
    study_root: Path,
    aggregate_title_filter: str | None,
    stage_filter: str | None,
    value_column: str | None,
    legacy_analysis_name: str | None,
) -> list[dict[str, Any]]:
    study_manifest = _read_json(study_root / "study_manifest.json")
    aggregate_defs: dict[str, AggregateDef] = {}
    try:
        study_doc = _load_source_study_doc(study_manifest)
        aggregate_defs = _aggregate_defs_from_doc(study_doc)
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
            started_at = _local_now()
            started_utc = _utc_now()
            res = _aggregate_study_analysis(
                study_root=study_root,
                analysis_name=title,
                value_column=value_column,
                stage_filter=stage_filter,
            )
            status = "done"
            reason = ""
            finished_at = _local_now()
            duration_min = _duration_minutes(started_at, finished_at)
            out = dict(res)
            out["title"] = title
            results.append(out)
            _log_task_event("DONE", f"aggregate {title}")
            status_rows.append(
                {
                    "title": title,
                    "status": status,
                    "reason": reason,
                    "raw_csv": str(out.get("raw_csv") or ""),
                    "grouped_csv": str(out.get("grouped_csv") or ""),
                    "manifest": str(out.get("manifest") or ""),
                    "raw_count": int(out.get("raw_count", 0) or 0),
                    "group_count": int(out.get("group_count", 0) or 0),
                    "started_at": started_at,
                    "finished_at": finished_at,
                    "duration_min": duration_min,
                    "started_at_utc": started_utc,
                    "finished_at_utc": _utc_now(),
                }
            )
        summary = {
            "total": len(status_rows),
            "done": sum(1 for r in status_rows if r.get("status") == "done"),
            "failed": sum(1 for r in status_rows if r.get("status") == "failed"),
        }
        _write_named_status(
            study_root=study_root,
            rows=status_rows,
            summary=summary,
            csv_name=AGGREGATE_STATUS_CSV_FILE,
            json_name=AGGREGATE_STATUS_JSON_FILE,
        )
        return results

    # Legacy fallback when no aggregate definitions exist.
    if legacy_analysis_name is None:
        raise ValueError(
            "No top-level aggregate definitions found in study.yaml. "
            "Provide --analysis <name> to use legacy variable aggregation."
        )
    res = _aggregate_study_analysis(
        study_root=study_root,
        analysis_name=legacy_analysis_name,
        value_column=value_column,
        stage_filter=stage_filter,
    )
    out = dict(res)
    out["title"] = legacy_analysis_name
    _log_task_event("DONE", f"aggregate {legacy_analysis_name}")
    _write_named_status(
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
                "finished_at": _local_now(),
                "duration_min": None,
                "started_at_utc": None,
                "finished_at_utc": _utc_now(),
            }
        ],
        summary={"total": 1, "done": 1, "failed": 0},
        csv_name=AGGREGATE_STATUS_CSV_FILE,
        json_name=AGGREGATE_STATUS_JSON_FILE,
    )
    return [out]


def _sort_plot_x(values: list[Any]) -> list[Any]:
    return _sort_values(list(values))


def _make_errorbar_plots_for_case_aggregate(case_agg_dir: Path, *, aggregate_title: str) -> list[Path]:
    per_x_csv = case_agg_dir / "per_x_stats.csv"
    if not per_x_csv.exists():
        return []
    rows = _load_csv_rows(per_x_csv)
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
        items_sorted = sorted(items, key=lambda r: _to_plot_value(r.get("x_value")))
        x_vals: list[Any] = []
        y_vals: list[float] = []
        err_vals: list[float] = []
        for r in items_sorted:
            x_raw = r.get("x_value")
            x_num = _safe_float(x_raw)
            x_vals.append(x_num if x_num is not None else str(x_raw))
            y_mean = _safe_float(r.get("y_mean"))
            y_std = _safe_float(r.get("y_std"))
            y_sem = _safe_float(r.get("y_sem"))
            if y_mean is None:
                continue
            y_vals.append(y_mean)
            err_vals.append(y_sem if y_sem is not None else (y_std if y_std is not None else 0.0))
        if not y_vals:
            continue
        out_path = case_agg_dir / f"errorbar_{_slug_underscore(y_name)}.png"
        render_plot(
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


def _make_boxplots_for_case_aggregate(case_agg_dir: Path, *, aggregate_title: str) -> list[Path]:
    raw_csv = case_agg_dir / "raw_replicates.csv"
    if not raw_csv.exists():
        return []
    rows = _load_csv_rows(raw_csv)
    if not rows:
        return []

    grouped_y: dict[str, dict[Any, list[float]]] = {}
    for row in rows:
        y_name = str(row.get("y_name") or "").strip()
        if not y_name:
            continue
        x_val = row.get("x_value")
        y_val = _safe_float(row.get("y_value"))
        if y_val is None:
            continue
        grouped_y.setdefault(y_name, {}).setdefault(x_val, []).append(y_val)

    outputs: list[Path] = []
    for y_name, xmap in grouped_y.items():
        x_keys = _sort_plot_x(list(xmap.keys()))
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
        out_path = case_agg_dir / f"boxplot_{_slug_underscore(y_name)}.png"
        render_plot(
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


def _row_key_from_params(row: dict[str, str], param_cols: list[str]) -> str:
    parts = [f"{p}={row.get(p)}" for p in param_cols]
    return ", ".join(parts)


def _find_param_columns(rows: list[dict[str, str]]) -> list[str]:
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


def _make_all_cases_errorbar_plots(across_dir: Path, *, aggregate_title: str) -> list[Path]:
    src = across_dir / "across_cases_per_x_stats.csv"
    if not src.exists():
        return []
    rows = _load_csv_rows(src)
    if not rows:
        return []
    param_cols = _find_param_columns(rows)
    grouped_by_y: dict[str, dict[str, list[dict[str, str]]]] = {}
    for row in rows:
        y_name = str(row.get("y_name") or "").strip()
        if not y_name:
            continue
        curve_key = _row_key_from_params(row, param_cols)
        grouped_by_y.setdefault(y_name, {}).setdefault(curve_key, []).append(row)

    outputs: list[Path] = []
    for y_name, curve_map in grouped_by_y.items():
        series = []
        for curve_key, items in curve_map.items():
            items_sorted = sorted(items, key=lambda r: _to_plot_value(r.get("x_value")))
            xs: list[Any] = []
            ys: list[float] = []
            yerrs: list[float] = []
            for r in items_sorted:
                xv_raw = r.get("x_value")
                xv_num = _safe_float(xv_raw)
                x_val: Any = xv_num if xv_num is not None else str(xv_raw)
                ym = _safe_float(r.get("y_mean"))
                ys_err = _safe_float(r.get("y_sem"))
                ystd = _safe_float(r.get("y_std"))
                if ym is None:
                    continue
                xs.append(x_val)
                ys.append(ym)
                yerrs.append(ys_err if ys_err is not None else (ystd if ystd is not None else 0.0))
            if ys:
                series.append({"x": xs, "y": ys, "yerr": yerrs, "label": curve_key})
        if not series:
            continue
        out_path = across_dir / f"errorbar_all_cases_{_slug_underscore(y_name)}.png"
        render_plot(
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


def _make_heatmap_from_rows(
    *,
    rows: list[dict[str, str]],
    x_param: str,
    y_param: str,
    value_col: str,
    out_path: Path,
    title: str,
) -> Path | None:
    coords: list[list[float]] = []
    values: list[float] = []
    for row in rows:
        xv = _safe_float(row.get(x_param))
        yv = _safe_float(row.get(y_param))
        zv = _safe_float(row.get(value_col))
        if xv is None or yv is None or zv is None:
            continue
        coords.append([xv, yv, 0.0])
        values.append(zv)
    if not values:
        return None
    render_plot(
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


def _make_all_cases_heatmaps(across_dir: Path, *, aggregate_title: str) -> list[Path]:
    outputs: list[Path] = []
    per_x_csv = across_dir / "across_cases_per_x_stats.csv"
    global_csv = across_dir / "across_cases_global_stats.csv"

    # Heatmaps at final x for mean/std/sem from per-x stats.
    if per_x_csv.exists():
        rows = _load_csv_rows(per_x_csv)
        if rows:
            param_cols = _find_param_columns(rows)
            if len(param_cols) >= 2:
                x_param, y_param = param_cols[0], param_cols[1]
                # Group by y_name and pick final x_value slice.
                by_y: dict[str, list[dict[str, str]]] = {}
                for row in rows:
                    y_name = str(row.get("y_name") or "").strip()
                    if y_name:
                        by_y.setdefault(y_name, []).append(row)
                for y_name, y_rows in by_y.items():
                    x_vals = [_safe_float(r.get("x_value")) for r in y_rows]
                    x_vals = [v for v in x_vals if v is not None]
                    if not x_vals:
                        continue
                    final_x = max(x_vals)
                    final_rows = [r for r in y_rows if _safe_float(r.get("x_value")) == final_x]
                    for metric in ("y_mean", "y_std", "y_sem"):
                        out_path = across_dir / f"heatmap_final_x_{_slug_underscore(y_name)}_{metric}.png"
                        made = _make_heatmap_from_rows(
                            rows=final_rows,
                            x_param=x_param,
                            y_param=y_param,
                            value_col=metric,
                            out_path=out_path,
                            title=f"{aggregate_title} - {y_name} final x {metric}",
                        )
                        if made is not None:
                            outputs.append(made)

    # Global heatmaps from across_cases_global_stats.csv
    if global_csv.exists():
        rows = _load_csv_rows(global_csv)
        if rows:
            param_cols = _find_param_columns(rows)
            if len(param_cols) >= 2:
                x_param, y_param = param_cols[0], param_cols[1]
                by_y: dict[str, list[dict[str, str]]] = {}
                for row in rows:
                    y_name = str(row.get("y_name") or "").strip()
                    if y_name:
                        by_y.setdefault(y_name, []).append(row)
                for y_name, y_rows in by_y.items():
                    for metric in ("y_mean", "y_std", "y_sem"):
                        out_path = across_dir / f"heatmap_global_{_slug_underscore(y_name)}_{metric}.png"
                        made = _make_heatmap_from_rows(
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


def _make_all_cases_per_iter_boxplots(across_dir: Path, *, aggregate_title: str) -> list[Path]:
    src = across_dir / "raw_all_cases.csv"
    if not src.exists():
        return []
    rows = _load_csv_rows(src)
    if not rows:
        return []
    param_cols = _find_param_columns(rows)
    if not param_cols:
        return []
    by_y: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        y_name = str(row.get("y_name") or "").strip()
        if y_name:
            by_y.setdefault(y_name, []).append(row)

    outputs: list[Path] = []
    for y_name, y_rows in by_y.items():
        x_values = _sort_plot_x(list({row.get("x_value") for row in y_rows}))
        for xv in x_values:
            subset = [r for r in y_rows if str(r.get("x_value")) == str(xv)]
            dist: dict[str, list[float]] = {}
            for r in subset:
                case_key = _row_key_from_params(r, param_cols)
                yv = _safe_float(r.get("y_value"))
                if yv is None:
                    continue
                dist.setdefault(case_key, []).append(yv)
            labels = [k for k, v in dist.items() if v]
            data = [v for v in dist.values() if v]
            if not data:
                continue
            out_path = across_dir / f"boxplot_all_cases_{_slug_underscore(y_name)}_x_{_slug_underscore(xv)}.png"
            render_plot(
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


def _plot_study_aggregates(
    *,
    study_root: Path,
    aggregate_title_filter: str | None,
    case_filter: str | None,
) -> dict[str, Any]:
    study_manifest = _read_json(study_root / "study_manifest.json")
    study_doc = _load_source_study_doc(study_manifest)
    aggregate_defs = _aggregate_defs_from_doc(study_doc)
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
        if not _case_matches_selector(case, case_filter):
            continue
        case_path = Path(str(case.get("path") or ""))
        for agg_title in target_titles:
            case_agg_dir = case_path / "aggregate" / agg_title
            if not case_agg_dir.exists():
                missing.append(str(case_agg_dir))
                continue
            generated.extend(_make_errorbar_plots_for_case_aggregate(case_agg_dir, aggregate_title=agg_title))
            generated.extend(_make_boxplots_for_case_aggregate(case_agg_dir, aggregate_title=agg_title))

    # All-cases level plots from aggregate outputs under cases/aggregate/<title>.
    for agg_title in target_titles:
        started_at = _local_now()
        before_count = len(generated)
        across_dir = study_root / "cases" / "aggregate" / agg_title
        if not across_dir.exists():
            if agg_title in target_set:
                missing.append(str(across_dir))
            finished_at = _local_now()
            status_rows.append(
                {
                    "title": agg_title,
                    "status": "skip",
                    "reason": "missing_aggregate_dir",
                    "generated_count": 0,
                    "missing_count": 1,
                    "started_at": started_at,
                    "finished_at": finished_at,
                    "duration_min": _duration_minutes(started_at, finished_at),
                }
            )
            _log_task_event("SKIP", f"plot {agg_title}", "missing aggregate directory")
            continue
        generated.extend(_make_all_cases_errorbar_plots(across_dir, aggregate_title=agg_title))
        generated.extend(_make_all_cases_heatmaps(across_dir, aggregate_title=agg_title))
        generated.extend(_make_all_cases_per_iter_boxplots(across_dir, aggregate_title=agg_title))
        created_n = len(generated) - before_count
        finished_at = _local_now()
        status_rows.append(
            {
                "title": agg_title,
                "status": "done",
                "reason": "",
                "generated_count": created_n,
                "missing_count": 0,
                "started_at": started_at,
                "finished_at": finished_at,
                "duration_min": _duration_minutes(started_at, finished_at),
            }
        )
        _log_task_event("DONE", f"plot {agg_title}", f"generated={created_n}")

    generated_files = [str(p) for p in generated]
    summary = {
        "total": len(status_rows),
        "done": sum(1 for r in status_rows if r.get("status") == "done"),
        "skip": sum(1 for r in status_rows if r.get("status") == "skip"),
        "generated_count": len(generated),
        "generated_files": generated_files,
        "missing_dirs": missing,
    }
    csv_path, json_path = _write_named_status(
        study_root=study_root,
        rows=status_rows,
        summary=summary,
        csv_name=PLOT_STATUS_CSV_FILE,
        json_name=PLOT_STATUS_JSON_FILE,
    )
    return {"generated": generated, "missing": missing, "manifest": json_path, "csv": csv_path}


def _remove_analysis_outputs(
    *,
    study_root: Path,
    remove_target: str,
) -> dict[str, Any]:
    study_manifest = _read_json(study_root / "study_manifest.json")
    doc = _load_source_study_doc(study_manifest)
    analysis_defs = _analysis_defs_from_doc(doc)
    target = str(remove_target or "").strip()
    if not target:
        raise ValueError("--remove requires a value: all or <analysis_title>.")

    if target.lower() == "all":
        target_ids = {a.analysis_id for a in analysis_defs.values()}
    else:
        if target not in analysis_defs:
            raise KeyError(f"Unknown analysis title: {target}")
        target_ids = {analysis_defs[target].analysis_id}

    removed_dirs: list[str] = []
    updated_stage_manifests = 0
    for case in study_manifest.get("cases") or []:
        if not isinstance(case, dict):
            continue
        case_path = Path(str(case.get("path") or ""))
        case_manifest = _read_json(_run_case_manifest_path(case_path))
        for rep in case_manifest.get("replicates") or []:
            if not isinstance(rep, dict):
                continue
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
                        for d in run.get("result_dirs") or []:
                            p = Path(str(d))
                            if p.exists():
                                try:
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
                    analysis_manifest["runs"] = kept_runs
                    _write_analysis_manifest(stage_dir, analysis_manifest)
                    updated_stage_manifests += 1

                    analysis_status = _load_analysis_status(stage_dir)
                    analyses_obj = analysis_status.get("analyses") if isinstance(analysis_status.get("analyses"), dict) else {}
                    for aid in list(analyses_obj.keys()):
                        if aid in target_ids:
                            analyses_obj.pop(aid, None)
                    analysis_status["analyses"] = analyses_obj
                    _write_analysis_status(stage_dir, analysis_status)

    # Remove top-level analyze status files; they represent prior runs.
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
        "target": target,
        "removed_dirs": removed_dirs,
        "updated_stage_manifests": updated_stage_manifests,
    }


def _remove_aggregate_outputs(
    *,
    study_root: Path,
    remove_target: str,
) -> dict[str, Any]:
    study_manifest = _read_json(study_root / "study_manifest.json")
    doc = _load_source_study_doc(study_manifest)
    aggregate_defs = _aggregate_defs_from_doc(doc)
    target = str(remove_target or "").strip()
    if not target:
        raise ValueError("--remove requires a value: all or <aggregate_title>.")

    if target.lower() == "all":
        titles = set(aggregate_defs.keys())
        # include existing dirs even if not in current YAML
        extra = study_root / "cases" / "aggregate"
        if extra.exists():
            for d in extra.iterdir():
                if d.is_dir():
                    titles.add(d.name)
    else:
        if target not in aggregate_defs:
            raise KeyError(f"Unknown aggregate title: {target}")
        titles = {target}

    removed_dirs: list[str] = []
    for case in study_manifest.get("cases") or []:
        if not isinstance(case, dict):
            continue
        case_path = Path(str(case.get("path") or ""))
        for title in titles:
            d = case_path / "aggregate" / title
            if d.exists():
                shutil.rmtree(d)
                removed_dirs.append(str(d))
    for title in titles:
        d = study_root / "cases" / "aggregate" / title
        if d.exists():
            shutil.rmtree(d)
            removed_dirs.append(str(d))

    for p in [
        study_root / AGGREGATE_STATUS_CSV_FILE,
        study_root / AGGREGATE_STATUS_JSON_FILE,
    ]:
        if p.exists():
            try:
                p.unlink()
            except Exception:
                pass

    return {"target": target, "removed_dirs": removed_dirs}


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
        "  reaxkit study --run study_MgTemp/\n"
        "  reaxkit study --run study_MgTemp/ --parallel-workers 4\n"
        "  reaxkit study --run study_MgTemp/ --rerun-failed\n"
        "  reaxkit study --run study_MgTemp/ --stage MM\n"
        "  reaxkit study --run study_MgTemp/ --case mg_05__temp_300\n"
        "  reaxkit study --analyze study_MgTemp/\n"
        "  reaxkit study --analyze study_MgTemp/ --rerun-failed\n"
        "  reaxkit study --analyze study_MgTemp/ --remove all\n"
        "  reaxkit study --analyze study_MgTemp/ --analysis msd\n"
        "  reaxkit study --aggregate study_MgTemp/\n"
        "  reaxkit study --aggregate study_MgTemp/ --aggregate msd_atom1_aggregation\n"
        "  reaxkit study --aggregate study_MgTemp/ --remove msd_atom1_aggregation\n"
        "  reaxkit study --aggregate study_MgTemp/ --aggregate msd_atom1_aggregation --stage NVT\n"
        "  reaxkit study --plot study_MgTemp/\n"
        "  reaxkit study --plot study_MgTemp/ --plot msd_atom1_aggregation"
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
        "--plot",
        action="append",
        metavar="VALUE",
        help="Plot mode. First value is STUDY_ROOT; optional second value is aggregate title filter.",
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
        help="Optional replicate selector (e.g. replicate_01).",
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
    parser.add_argument(
        "--remove",
        default=None,
        help="For --analyze/--aggregate: remove outputs by title or 'all'.",
    )
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
        remove_target = _clean_opt(getattr(args, "remove", None))
        if remove_target is not None:
            if aggregate_title_filter is not None:
                raise ValueError("When using --remove with --aggregate, do not pass aggregate title as second --aggregate value.")
            removed = _remove_aggregate_outputs(
                study_root=Path(target).resolve(),
                remove_target=remove_target,
            )
            _log_task_event("DONE", "aggregate remove", f"target={removed['target']} removed={len(removed['removed_dirs'])}")
            return 0
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

    plot_values = getattr(args, "plot", None)
    if plot_values is not None:
        if not isinstance(plot_values, list) or len(plot_values) == 0:
            raise ValueError("--plot requires at least one value: study root.")
        if len(plot_values) > 2:
            raise ValueError("--plot accepts at most two values: study root and optional aggregate title.")
        target = _clean_opt(plot_values[0])
        if target is None:
            raise ValueError("--plot requires a study root path.")
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
        remove_target = _clean_opt(getattr(args, "remove", None))
        if remove_target is not None:
            removed = _remove_analysis_outputs(
                study_root=Path(str(analyze_root)).resolve(),
                remove_target=remove_target,
            )
            _log_task_event(
                "DONE",
                "analyze remove",
                f"target={removed['target']} removed_dirs={len(removed['removed_dirs'])}",
            )
            return 0
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
        raise ValueError("Missing required mode. Use --init, --run, --analyze, --aggregate, --plot, or --make-yaml.")

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
