"""Workflow for creating Jaguar jobs from isomer representative folders."""

from __future__ import annotations

import argparse
from pathlib import Path

from reaxkit.analysis.molecular_analysis.jaguar_isomer_jobs import (
    create_jaguar_isomer_jobs,
)
from reaxkit.core.resolve.command_alias_resolver import resolve_command_name

ALL_COMMANDS = ("create-jaguar-isomer-jobs",)
ALL_LEGACY_COMMANDS = ("create_jaguar_isomer_jobs", "make-jaguar-isomer-jobs", "make_jaguar_isomer_jobs")
COMMAND_ALIASES = {"create-jaguar-isomer-jobs": ALL_LEGACY_COMMANDS}


EXAMPLE_CONFIG = """# Slurm settings for `reaxkit create-jaguar-isomer-jobs`.
# Review every value for the target cluster before using --submit.
job_system: slurm
partition: open
time: "24:00:00"
nodes: 1
ntasks_per_node: 1
cpus_per_task: 8
mem: "4G"
module_load:
  - schrodinger
jaguar_command: jaguar
jaguar_args:
  - run
  - hf.in
  - -WAIT
  - -PARALLEL
  - "{cpus_per_task}"
"""


def write_example_config(path: str | Path, *, force: bool = False) -> Path:
    """Write an editable Slurm/Jaguar job config template."""
    out = Path(path)
    if out.exists() and not force:
        raise FileExistsError(f"example config already exists: {out}. Use --force to overwrite.")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(EXAMPLE_CONFIG, encoding="utf-8")
    return out


def build_parser(parser: argparse.ArgumentParser, *, command: str) -> argparse.ArgumentParser:
    """Build the parser for Jaguar isomer job creation."""
    canonical = resolve_command_name(command, task_names=ALL_COMMANDS, aliases=COMMAND_ALIASES)
    parser.set_defaults(command=canonical)
    parser.formatter_class = argparse.RawTextHelpFormatter
    parser.description = (
        "Create Jaguar input files and Slurm submit scripts from detected isomer folders.\n"
        "This workflow expects stage-1 per-isomer folders containing `xmolout` files.\n"
        "It prepares jobs and can optionally call `sbatch`, but it does not track job completion\n"
        "or diagnose Jaguar/runtime/license failures.\n\n"
        "Examples:\n"
        "   reaxkit create-jaguar-isomer-jobs --write-example-config slurm_job_config.yaml\n"
        "   reaxkit create-jaguar-isomer-jobs --isomer-dir isomer_outputs/isomers --hf-base hf_base.in --job-config slurm_job_config.yaml --output-dir jaguar_jobs\n"
        "   reaxkit create-jaguar-isomer-jobs --isomer-dir isomer_outputs/isomers --hf-base hf_base.in --job-config slurm_job_config.yaml --output-dir jaguar_jobs --submit"
    )
    parser.add_argument(
        "--isomer-dir",
        default=None,
        help="Directory containing per-isomer folders from detect-isomer-representatives --write-isomer-dirs.",
    )
    parser.add_argument(
        "--hf-base",
        default=None,
        help="Jaguar hf_base.in template. This is molecular-system specific and must be reviewed.",
    )
    parser.add_argument(
        "--job-config",
        default=None,
        help="Slurm/Jaguar YAML settings file. Use --write-example-config to create a template.",
    )
    parser.add_argument(
        "--output-dir",
        default="jaguar_jobs",
        help="Output directory for generated Jaguar job folders and manifest.",
    )
    parser.add_argument(
        "--submit",
        action="store_true",
        help="Submit generated jobs with sbatch. Submission success does not mean Jaguar completed successfully.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Allow overwriting files in an existing non-empty output directory or example config.",
    )
    parser.add_argument(
        "--no-skip-completed",
        action="store_true",
        help="Regenerate jobs even when an existing hf.out contains 'final geometry:'.",
    )
    parser.add_argument(
        "--no-skip-queued",
        action="store_true",
        help="With --submit, do not skip structures that already have a matching Slurm job name in squeue.",
    )
    parser.add_argument(
        "--write-example-config",
        default=None,
        help="Write an editable Slurm/Jaguar YAML config template and exit.",
    )
    return parser


def _require_arg(args: argparse.Namespace, name: str, flag: str) -> str:
    """Return a required argument value or raise a clear parser-style error."""
    value = getattr(args, name)
    if value in (None, ""):
        raise ValueError(f"{flag} is required unless --write-example-config is used.")
    return str(value)


def run_main(command: str, args: argparse.Namespace) -> int:
    """Run Jaguar isomer job creation."""
    canonical = resolve_command_name(command, task_names=ALL_COMMANDS, aliases=COMMAND_ALIASES)

    example_config = getattr(args, "write_example_config", None)
    if example_config:
        out = write_example_config(example_config, force=bool(getattr(args, "force", False)))
        print(f"{canonical}: wrote example Slurm/Jaguar config to {out}")
        return 0

    result = create_jaguar_isomer_jobs(
        isomer_dir=_require_arg(args, "isomer_dir", "--isomer-dir"),
        hf_base_path=_require_arg(args, "hf_base", "--hf-base"),
        job_config_path=_require_arg(args, "job_config", "--job-config"),
        output_dir=getattr(args, "output_dir"),
        submit=bool(getattr(args, "submit", False)),
        force=bool(getattr(args, "force", False)),
        skip_completed=not bool(getattr(args, "no_skip_completed", False)),
        skip_queued=not bool(getattr(args, "no_skip_queued", False)),
    )
    for warning in result.warnings:
        print(warning)
    submitted_count = sum(1 for record in result.records if record.submitted)
    skipped_count = sum(1 for record in result.records if record.skipped)
    generated_count = len(result.records) - skipped_count
    print(
        f"{canonical}: processed {len(result.records)} isomers"
        f"; generated {generated_count} jobs; submitted {submitted_count}; skipped {skipped_count}; "
        f"manifest {result.manifest_path}"
    )
    return 0


__all__ = [
    "ALL_COMMANDS",
    "ALL_LEGACY_COMMANDS",
    "EXAMPLE_CONFIG",
    "build_parser",
    "run_main",
    "write_example_config",
]
