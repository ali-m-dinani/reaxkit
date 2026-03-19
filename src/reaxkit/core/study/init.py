"""Initialization helpers for study folder generation."""

from __future__ import annotations

import itertools
import shutil
from pathlib import Path
from typing import Any


def render_value(value: Any, context: dict[str, Any]) -> Any:
    if isinstance(value, str):
        try:
            return value.format(**context)
        except KeyError as exc:
            missing = str(exc).strip("'")
            raise ValueError(f"Missing placeholder '{missing}' in study context.") from exc
    if isinstance(value, dict):
        return {str(k): render_value(v, context) for k, v in value.items()}
    if isinstance(value, list):
        return [render_value(v, context) for v in value]
    return value


def enumerate_cases(parameters: dict[str, list[Any]]) -> list[dict[str, Any]]:
    keys = list(parameters.keys())
    values_grid = [parameters[k] for k in keys]
    combos = []
    for vals in itertools.product(*values_grid):
        combos.append({k: v for k, v in zip(keys, vals)})
    return combos


def resolve_template_path(template_value: str, *, study_dir: Path) -> Path:
    candidate = Path(str(template_value))
    if candidate.is_absolute():
        return candidate
    return (study_dir / candidate).resolve()


def copy_template_into_replicate(*, template_root: Path, replicate_dir: Path) -> None:
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


def apply_stage_slurm_job_name(*, stage_dir: Path, case_number: int, replicate_number: int, stage_name: str) -> None:
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


def collect_template_stage_relpaths(*, template_root: Path, stage_names: list[str]) -> dict[str, Path]:
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

