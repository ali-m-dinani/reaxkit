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

STUDY_COMMAND = "study"
CONTROL_PARAMETER_ALIASES = {
    "temperature": "mdtemp",
}
DEFAULT_ARTIFACT_TRANSFER = "copy"
SUPPORTED_ARTIFACT_TRANSFER = {"copy", "symlink", "hardlink"}
STAGE_STATUS_FILE = ".stage_status.json"
AGGREGATE_ACTION = "aggregate"

STUDY_TEMPLATE_YAML = """study_name: mg_temp_sweep
template: "./template"
parameters:
  mg_percent: [5, 10]
  temperature: [300, 600]
replicates: 2

workflow:
  - stage: MM
    steps:
      - "python make_structure.py --mg-percent {mg_percent}"
      - "sbatch submit_and_wait.sh"
    produces:
      final_geometry: "outputs/final_geo.xyz"

  - stage: NPT
    consumes:
      initial_geometry:
        from: MM.final_geometry
        to: inputs/initial_geometry.xyz
    steps:
      - "reaxkit write-control --parameter mdtemp --value {temperature} --output control"
      - "sbatch submit_and_wait.sh"
    produces:
      final_geometry: "outputs/final_geo.xyz"
      control_file: "control"

  - stage: NVT
    consumes:
      initial_geometry:
        from: NPT.final_geometry
        to: inputs/initial_geometry.xyz
    steps:
      - "reaxkit write-control --parameter mdtemp --value {temperature} --output control"
      - "sbatch submit_and_wait.sh"
      - "reaxkit polarization --frame 0"
    variables:
      field:
        directory: "reaxkit_workspace/analysis/polarization"
        folder_id: "latest"
        file: "results.csv"
        column: "field_z"
      polarization:
        directory: "reaxkit_workspace/analysis/polarization"
        folder_id: "latest"
        file: "results.csv"
        column: "P_z (uC/cm^2)"
    produces:
      trajectory: "outputs/traj.xyz"
      final_geometry: "outputs/final_geo.xyz"
      control_file: "control"
"""


@dataclass(frozen=True)
class StageDef:
    name: str
    payload: dict[str, Any]


@dataclass(frozen=True)
class ArtifactRef:
    stage: str
    artifact: str


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _local_now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _log_stage_event(case_id: str, replicate_id: str, stage_name: str, tag: str, detail: str | None = None) -> None:
    label = str(tag).upper()[:5].ljust(5)
    print(f"{case_id} {replicate_id} {stage_name}")
    line = f"[{label}] {_local_now()}"
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

    json_path = status_dir / "study_run_status.json"
    csv_path = status_dir / "study_run_status.csv"

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
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in fieldnames})

    return json_path, csv_path


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


def _validate_study(doc: dict[str, Any]) -> tuple[str, dict[str, list[Any]], int, list[StageDef], str | None]:
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

    workflow_raw = doc.get("workflow") or []
    if not isinstance(workflow_raw, list) or not workflow_raw:
        raise ValueError("workflow must be a non-empty list.")
    stages: list[StageDef] = []
    seen: set[str] = set()
    for item in workflow_raw:
        if not isinstance(item, dict):
            raise ValueError("Each workflow stage must be a mapping.")
        stage_name = str(item.get("stage") or "").strip()
        if not stage_name:
            raise ValueError("Each workflow stage requires a non-empty 'stage' name.")
        if stage_name in seen:
            raise ValueError(f"Duplicate workflow stage: {stage_name}")
        seen.add(stage_name)
        stages.append(StageDef(name=stage_name, payload=dict(item)))
    template_raw = doc.get("template")
    template_dir: str | None = None
    if template_raw is not None:
        template_dir = str(template_raw).strip() or None

    return study_name, parameters, replicates, stages, template_dir


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
    study_name, parameters, replicates, stages, template_dir_raw = _validate_study(study_doc)
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
                _write_json(stage_dir / "stage_manifest.json", stage_manifest)
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
            _write_json(rep_dir / "replicate_manifest.json", rep_manifest)
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
        _write_json(case_dir / "case_manifest.json", case_manifest)
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
        "workflow_stages": stage_names,
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
    _write_json(stage_dir / "stage_manifest.json", stage_manifest)

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
) -> dict[str, Any]:
    counts = {"completed": 0, "skipped": 0, "failed": 0, "not_ready": 0}
    started_at = _local_now()
    rep_id = str(rep.get("replicate_id") or "")
    case_id = str(case.get("case_id") or "")
    rep_path = Path(str(rep.get("path") or ""))
    rep_manifest = _read_json(rep_path / "replicate_manifest.json")
    produced_map = _build_declared_produced_map(rep_manifest)

    stage_entries = rep_manifest.get("stages") or []
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
        stage_manifest = _read_json(stage_dir / "stage_manifest.json")
        stage_manifest["study_source_dir"] = str(Path(str(manifest.get("source_yaml") or "")).parent)

        status = _load_stage_status(stage_dir)
        current_status = str(status.get("status") or "pending")
        if current_status == "completed":
            counts["skipped"] += 1
            _log_stage_event(str(case.get("case_id") or ""), rep_id, stage_name, "SKIP", "already completed")
            continue

        ready, reason = _is_stage_ready(
            stage_manifest,
            status_by_stage=status_by_stage,
            produced_artifacts=produced_map,
        )
        if not ready:
            counts["not_ready"] += 1
            _log_stage_event(str(case.get("case_id") or ""), rep_id, stage_name, "WAIT", reason)
            continue

        _log_stage_event(str(case.get("case_id") or ""), rep_id, stage_name, "RUN")
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
            _log_stage_event(str(case.get("case_id") or ""), rep_id, stage_name, "DONE")
        else:
            counts["failed"] += 1
            _log_stage_event(case_id, rep_id, stage_name, "FAIL")
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
) -> dict[str, int]:
    manifest = _read_json(study_root / "study_manifest.json")
    case_entries = manifest.get("cases") or []
    if not isinstance(case_entries, list):
        raise ValueError("Invalid study_manifest.json: cases must be a list.")

    counts = {"completed": 0, "skipped": 0, "failed": 0, "not_ready": 0}
    tasks: list[tuple[dict[str, Any], dict[str, Any]]] = []
    row_map: dict[tuple[str, str], dict[str, Any]] = {}

    for case in case_entries:
        if not isinstance(case, dict):
            continue
        if not _case_matches_selector(case, case_filter):
            continue
        case_path = Path(str(case.get("path") or ""))
        case_manifest = _read_json(case_path / "case_manifest.json")
        reps = case_manifest.get("replicates") or []
        for rep in reps:
            if not isinstance(rep, dict):
                continue
            rep_id = str(rep.get("replicate_id") or "")
            if replicate_filter and rep_id != replicate_filter:
                continue
            tasks.append((case, rep))
            case_id = str(case.get("case_id") or "")
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
            }
            persist_status()
    return counts


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


def _iter_replicate_variable_records(
    *,
    study_manifest: dict[str, Any],
    variable_name: str,
    stage_filter: str | None,
    value_column_override: str | None,
) -> tuple[list[dict[str, Any]], str | None]:
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
        case_manifest = _read_json(case_path / "case_manifest.json")
        reps = case_manifest.get("replicates") or []
        for rep in reps:
            if not isinstance(rep, dict):
                continue
            rep_id = str(rep.get("replicate_id") or "")
            rep_path = Path(str(rep.get("path") or ""))
            rep_manifest = _read_json(rep_path / "replicate_manifest.json")
            stage_entries = rep_manifest.get("stages") or []
            for stage_entry in stage_entries:
                if not isinstance(stage_entry, dict):
                    continue
                stage_name = str(stage_entry.get("stage") or "").strip()
                if not stage_name:
                    continue
                if stage_filter and stage_name != stage_filter:
                    continue

                stage_dir = Path(str(stage_entry.get("path") or ""))
                stage_status = _load_stage_status(stage_dir)
                if str(stage_status.get("status") or "") != "completed":
                    continue
                stage_manifest = _read_json(stage_dir / "stage_manifest.json")
                rendered_stage = stage_manifest.get("rendered_stage")
                if not isinstance(rendered_stage, dict):
                    continue
                variables = _normalize_stage_variables(rendered_stage)
                spec = variables.get(variable_name)
                if not isinstance(spec, dict):
                    continue

                csv_path = _resolve_variable_csv_path(stage_dir=stage_dir, spec=spec)
                if csv_path is None:
                    continue
                if not csv_path.exists():
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

                row = {
                    "case_id": case_id,
                    "replicate": rep_id,
                    "stage": stage_name,
                    variable_name: scalar,
                    "source_file": str(csv_path),
                    "source_column": used_col,
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
    raw_fields = [*param_names, "replicate", analysis_name, "case_id", "stage", "source_file", "source_column"]
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


def build_parser(parser: argparse.ArgumentParser, *, command: str) -> argparse.ArgumentParser:
    _ = command
    parser.set_defaults(command=STUDY_COMMAND)
    parser.formatter_class = argparse.RawTextHelpFormatter
    parser.add_argument(
        "study_action",
        nargs="?",
        choices=[AGGREGATE_ACTION],
        help="Study sub-action. Use 'aggregate' to summarize analysis outputs.",
    )
    parser.add_argument(
        "study_action_target",
        nargs="?",
        help="Target path for study sub-action (e.g. study root for aggregate).",
    )
    parser.description = (
        "Create and initialize parameter-sweep study layouts.\n\n"
        "Examples:\n"
        "  reaxkit study --make-yaml study.yaml\n"
        "  reaxkit study --gen-yaml\n"
        "  reaxkit study --init study.yaml --root .\n"
        "  reaxkit study --init study.yaml --root studies --force\n"
        "  reaxkit study --run study_MgTemp/\n"
        "  reaxkit study --run study_MgTemp/ --parallel-workers 4\n"
        "  reaxkit study --run study_MgTemp/ --stage MM\n"
        "  reaxkit study --run study_MgTemp/ --case mg_05__temp_300\n"
        "  reaxkit study aggregate study_MgTemp/ --analysis polarization\n"
        "  reaxkit study aggregate study_MgTemp/ --analysis polarization --stage NVT"
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
        "--analysis",
        default=None,
        help="Variable name for study aggregate (e.g. polarization, field).",
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

    action = _clean_opt(getattr(args, "study_action", None))
    if action == AGGREGATE_ACTION:
        target = _clean_opt(getattr(args, "study_action_target", None))
        if target is None:
            raise ValueError("study aggregate requires a study root path: reaxkit study aggregate <study_root> --analysis ...")
        variable_name = _clean_opt(getattr(args, "analysis", None))
        if variable_name is None:
            raise ValueError("--analysis is required for study aggregate.")
        result = _aggregate_study_analysis(
            study_root=Path(target).resolve(),
            analysis_name=variable_name,
            value_column=_clean_opt(getattr(args, "value_column", None)),
            stage_filter=_clean_opt(getattr(args, "stage", None)),
        )
        print(f"Raw CSV: {result['raw_csv']}")
        print(f"Grouped CSV: {result['grouped_csv']}")
        if result["plot"] is not None:
            print(f"Plot: {result['plot']}")
        print(f"Manifest: {result['manifest']}")
        return 0

    run_root = getattr(args, "run", None)
    if run_root is not None:
        counts = _run_study(
            study_root=Path(str(run_root)).resolve(),
            stage_filter=_clean_opt(getattr(args, "stage", None)),
            case_filter=_clean_opt(getattr(args, "case", None)),
            replicate_filter=_clean_opt(getattr(args, "replicate", None)),
            artifact_transfer=str(getattr(args, "artifact_transfer", DEFAULT_ARTIFACT_TRANSFER)),
            run_geometry_generator=bool(getattr(args, "run_geometry_generator", True)),
            strict_actions=bool(getattr(args, "strict_actions", False)),
            parallel_workers=int(getattr(args, "parallel_workers", 1)),
        )
        print(
            "Run summary: "
            f"completed={counts['completed']} skipped={counts['skipped']} "
            f"not_ready={counts['not_ready']} failed={counts['failed']}"
        )
        return 1 if counts["failed"] > 0 else 0

    init_path = getattr(args, "init", None)
    if init_path is None:
        raise ValueError("Missing required mode. Use --init, --run, study aggregate, or --make-yaml.")

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
