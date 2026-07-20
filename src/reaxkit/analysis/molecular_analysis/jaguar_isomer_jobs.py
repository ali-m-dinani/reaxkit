"""Create Jaguar jobs for detected isomer representatives.

This helper prepares Jaguar input files and Slurm submit scripts from isomer
folders produced by representative detection. It does not track job completion,
parse Jaguar outputs, or diagnose runtime/license/software failures.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
import re
import shutil
import subprocess
from typing import Any

import yaml


JAGUAR_SETTINGS_WARNING = (
    "WARNING: hf_base.in contains Jaguar/DFT settings that are molecular-system specific. "
    "Review charge, multiplicity, method, basis set, constraints, solvent, and related settings before submission."
)
SUBMISSION_WARNING = (
    "WARNING: Slurm submission only means sbatch accepted the job. Jaguar/module/license/runtime failures "
    "must be checked later from Slurm and Jaguar logs; troubleshooting those failures is outside this workflow."
)


@dataclass(frozen=True)
class SlurmJaguarJobConfig:
    """Slurm settings for generated Jaguar jobs."""

    partition: str
    time: str
    cpus_per_task: int
    jaguar_command: str
    job_system: str = "slurm"
    account: str | None = None
    nodes: int = 1
    ntasks_per_node: int = 1
    mem: str | None = None
    module_load: list[str] = field(default_factory=list)
    environment: dict[str, str] = field(default_factory=dict)
    pre_commands: list[str] = field(default_factory=list)
    jaguar_args: list[str] = field(default_factory=lambda: ["run", "hf.in", "-WAIT"])


@dataclass(frozen=True)
class JaguarIsomerJobRecord:
    """One generated or skipped Jaguar isomer job."""

    structure_name: str
    source_xmolout: Path
    job_dir: Path
    hf_input: Path
    submit_script: Path
    submitted: bool = False
    slurm_job_id: str = ""
    skipped: bool = False
    skip_reason: str = ""


@dataclass(frozen=True)
class JaguarIsomerJobResult:
    """Result for Jaguar isomer job creation."""

    records: list[JaguarIsomerJobRecord]
    manifest_path: Path
    warnings: list[str]


def load_slurm_jaguar_job_config(path: str | Path) -> SlurmJaguarJobConfig:
    """Load and validate Slurm Jaguar job settings from YAML."""
    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"job config file not found: {config_path}")
    data = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError("job config must be a YAML mapping.")

    job_system = str(data.get("job_system", "slurm")).strip().lower()
    if job_system != "slurm":
        raise ValueError("Only job_system='slurm' is supported for Jaguar isomer jobs.")

    missing = [
        key
        for key in ("partition", "time", "cpus_per_task", "jaguar_command")
        if data.get(key) in (None, "")
    ]
    if missing:
        raise ValueError(f"job config is missing required keys: {missing}")

    cpus_per_task = int(data["cpus_per_task"])
    if cpus_per_task < 1:
        raise ValueError("cpus_per_task must be a positive integer.")

    module_load = _as_string_list(data.get("module_load") or [], field_name="module_load")
    pre_commands = _as_string_list(data.get("pre_commands") or [], field_name="pre_commands")
    jaguar_args = _as_string_list(
        data.get("jaguar_args") or ["run", "hf.in", "-WAIT"],
        field_name="jaguar_args",
    )
    environment = _as_string_dict(data.get("environment") or {}, field_name="environment")

    return SlurmJaguarJobConfig(
        job_system=job_system,
        partition=str(data["partition"]),
        account=str(data["account"]) if data.get("account") not in (None, "") else None,
        nodes=int(data.get("nodes", 1)),
        ntasks_per_node=int(data.get("ntasks_per_node", 1)),
        cpus_per_task=cpus_per_task,
        time=str(data["time"]),
        mem=str(data["mem"]) if data.get("mem") not in (None, "") else None,
        module_load=module_load,
        environment=environment,
        pre_commands=pre_commands,
        jaguar_command=str(data["jaguar_command"]),
        jaguar_args=jaguar_args,
    )


def _as_string_list(value: Any, *, field_name: str) -> list[str]:
    """Normalize a YAML list field to strings."""
    if isinstance(value, str) or not isinstance(value, list | tuple):
        raise ValueError(f"job config field {field_name!r} must be a list.")
    return [str(item) for item in value]


def _as_string_dict(value: Any, *, field_name: str) -> dict[str, str]:
    """Normalize a YAML mapping field to string keys and values."""
    if not isinstance(value, dict):
        raise ValueError(f"job config field {field_name!r} must be a mapping.")
    return {str(key): str(item) for key, item in value.items()}


def discover_isomer_xmolouts(isomer_dir: str | Path) -> list[tuple[str, Path]]:
    """Return sorted ``(structure_name, xmolout_path)`` pairs from an isomer directory."""
    root = Path(isomer_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"isomer directory not found: {root}")
    subdirs = sorted(path for path in root.iterdir() if path.is_dir())
    if not subdirs:
        raise ValueError(f"isomer directory contains no isomer subfolders: {root}")

    missing = [path.name for path in subdirs if not (path / "xmolout").is_file()]
    if missing:
        raise FileNotFoundError(f"isomer subfolders missing xmolout files: {missing}")

    pairs = [(path.name, path / "xmolout") for path in subdirs]
    if not pairs:
        raise ValueError(f"no isomer xmolout files found under: {root}")
    return pairs


def _merge_hf_base_and_xmolout(hf_base: Path, xmolout: Path) -> str:
    """Return Jaguar input text using legacy ``hf_base.in + xmolout`` behavior."""
    hf_text = hf_base.read_text(encoding="utf-8")
    xmol_text = xmolout.read_text(encoding="utf-8")
    if hf_text and not hf_text.endswith("\n"):
        hf_text += "\n"
    return hf_text + xmol_text


def _format_arg(value: str, config: SlurmJaguarJobConfig) -> str:
    """Format a Jaguar argument using known config placeholders."""
    return str(value).format(
        cpus_per_task=config.cpus_per_task,
        partition=config.partition,
        time=config.time,
    )


def render_slurm_jaguar_script(*, structure_name: str, config: SlurmJaguarJobConfig) -> str:
    """Render a Slurm script for one Jaguar input."""
    lines = [
        "#!/bin/bash",
        "",
        f"#SBATCH --partition={config.partition}",
        f"#SBATCH --time={config.time}",
        f"#SBATCH --nodes={config.nodes}",
        f"#SBATCH --ntasks-per-node={config.ntasks_per_node}",
        f"#SBATCH --cpus-per-task={config.cpus_per_task}",
    ]
    if config.account:
        lines.append(f"#SBATCH --account={config.account}")
    if config.mem:
        lines.append(f"#SBATCH --mem={config.mem}")
    lines.extend(
        [
            f"#SBATCH --job-name={structure_name}",
            "#SBATCH --output=slurm-%j.out",
            "#SBATCH --error=slurm-%j.err",
            "",
            'echo "Job started on $(hostname) at $(date)"',
            'echo " "',
        ]
    )
    for module_name in config.module_load:
        lines.append(f"module load {module_name}")
    for key, value in config.environment.items():
        lines.append(f"export {key}={value}")
    lines.extend(config.pre_commands)
    if config.module_load or config.environment or config.pre_commands:
        lines.append("")
    args = [_format_arg(value, config) for value in config.jaguar_args]
    lines.append(" ".join([config.jaguar_command, *args]).strip())
    lines.extend(
        [
            "",
            'echo " "',
            'echo "Job ended on $(hostname) at $(date)"',
            "",
        ]
    )
    return "\n".join(lines)


def _parse_sbatch_job_id(stdout: str) -> str:
    """Parse a Slurm job id from sbatch stdout."""
    match = re.search(r"Submitted batch job\s+(\S+)", str(stdout))
    return match.group(1) if match else ""


def _submit_slurm_script(script_path: Path) -> str:
    """Submit one Slurm script and return the accepted job id if available."""
    if shutil.which("sbatch") is None:
        raise FileNotFoundError("sbatch command not found; cannot submit Slurm jobs.")
    proc = subprocess.run(
        ["sbatch", str(script_path.name)],
        cwd=str(script_path.parent),
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"sbatch failed for {script_path}: {proc.stderr.strip() or proc.stdout.strip()}"
        )
    return _parse_sbatch_job_id(proc.stdout)


def _slurm_job_exists(job_name: str) -> bool:
    """Return whether Slurm currently has a job with this name."""
    if shutil.which("squeue") is None:
        return False
    proc = subprocess.run(
        ["squeue", f"--name={job_name}", "--noheader"],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"squeue failed for job {job_name}: {proc.stderr.strip() or proc.stdout.strip()}")
    return bool(proc.stdout.strip())


def _has_completed_hf_output(*directories: Path) -> bool:
    """Return whether any directory has an hf.out containing Jaguar final geometry."""
    for directory in directories:
        hf_output = directory / "hf.out"
        if hf_output.is_file() and "final geometry:" in hf_output.read_text(encoding="utf-8", errors="replace"):
            return True
    return False


def _write_manifest(records: list[JaguarIsomerJobRecord], manifest_path: Path) -> None:
    """Write generated job records to CSV."""
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "structure_name",
                "source_xmolout",
                "job_dir",
                "hf_input",
                "submit_script",
                "submitted",
                "slurm_job_id",
                "skipped",
                "skip_reason",
            ],
        )
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    "structure_name": record.structure_name,
                    "source_xmolout": str(record.source_xmolout),
                    "job_dir": str(record.job_dir),
                    "hf_input": str(record.hf_input),
                    "submit_script": str(record.submit_script),
                    "submitted": str(bool(record.submitted)).lower(),
                    "slurm_job_id": record.slurm_job_id,
                    "skipped": str(bool(record.skipped)).lower(),
                    "skip_reason": record.skip_reason,
                }
            )


def create_jaguar_isomer_jobs(
    *,
    isomer_dir: str | Path,
    hf_base_path: str | Path,
    job_config_path: str | Path,
    output_dir: str | Path,
    submit: bool = False,
    force: bool = False,
    skip_completed: bool = True,
    skip_queued: bool = True,
) -> JaguarIsomerJobResult:
    """Create Jaguar inputs and Slurm scripts for detected isomer folders."""
    hf_base = Path(hf_base_path)
    if not hf_base.is_file():
        raise FileNotFoundError(f"hf_base.in file not found: {hf_base}")
    if not hf_base.read_text(encoding="utf-8").strip():
        raise ValueError(f"hf_base.in file is empty: {hf_base}")

    config = load_slurm_jaguar_job_config(job_config_path)
    pairs = discover_isomer_xmolouts(isomer_dir)

    out_dir = Path(output_dir)
    expected_names = {structure_name for structure_name, _ in pairs}
    if out_dir.exists() and not force:
        unexpected_entries = [
            path.name
            for path in out_dir.iterdir()
            if path.name != "jaguar_job_manifest.csv" and (not path.is_dir() or path.name not in expected_names)
        ]
        if unexpected_entries:
            raise FileExistsError(
                f"output directory is non-empty and contains files outside expected isomer job folders: {unexpected_entries}. "
                "Use force=True to overwrite."
            )
    out_dir.mkdir(parents=True, exist_ok=True)

    warnings = [JAGUAR_SETTINGS_WARNING]
    if submit:
        warnings.append(SUBMISSION_WARNING)

    records: list[JaguarIsomerJobRecord] = []
    for structure_name, xmolout in pairs:
        job_dir = out_dir / structure_name
        copied_xmolout = job_dir / "xmolout"
        hf_input = job_dir / "hf.in"
        submit_script = job_dir / "jaguar.job"

        if skip_queued and submit and _slurm_job_exists(structure_name):
            records.append(
                JaguarIsomerJobRecord(
                    structure_name=structure_name,
                    source_xmolout=xmolout,
                    job_dir=job_dir,
                    hf_input=hf_input,
                    submit_script=submit_script,
                    skipped=True,
                    skip_reason="queued_slurm_job",
                )
            )
            continue

        if skip_completed and _has_completed_hf_output(xmolout.parent, job_dir):
            records.append(
                JaguarIsomerJobRecord(
                    structure_name=structure_name,
                    source_xmolout=xmolout,
                    job_dir=job_dir,
                    hf_input=hf_input,
                    submit_script=submit_script,
                    skipped=True,
                    skip_reason="completed_hf_output",
                )
            )
            continue

        if job_dir.exists() and any(job_dir.iterdir()) and not force:
            raise FileExistsError(f"job directory exists and is non-empty: {job_dir}. Use force=True to overwrite.")

        job_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(xmolout, copied_xmolout)

        hf_input.write_text(_merge_hf_base_and_xmolout(hf_base, xmolout), encoding="utf-8")

        submit_script.write_text(
            render_slurm_jaguar_script(structure_name=structure_name, config=config),
            encoding="utf-8",
        )

        slurm_job_id = _submit_slurm_script(submit_script) if submit else ""
        records.append(
            JaguarIsomerJobRecord(
                structure_name=structure_name,
                source_xmolout=xmolout,
                job_dir=job_dir,
                hf_input=hf_input,
                submit_script=submit_script,
                submitted=bool(submit),
                slurm_job_id=slurm_job_id,
            )
        )

    manifest_path = out_dir / "jaguar_job_manifest.csv"
    _write_manifest(records, manifest_path)
    return JaguarIsomerJobResult(records=records, manifest_path=manifest_path, warnings=warnings)


__all__ = [
    "JAGUAR_SETTINGS_WARNING",
    "SUBMISSION_WARNING",
    "JaguarIsomerJobRecord",
    "JaguarIsomerJobResult",
    "SlurmJaguarJobConfig",
    "create_jaguar_isomer_jobs",
    "discover_isomer_xmolouts",
    "load_slurm_jaguar_job_config",
    "render_slurm_jaguar_script",
]
