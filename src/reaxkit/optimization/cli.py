"""Command-line interface for force-field campaigns and output conversion."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from .config import UniqueLoader, generate_template, load_config
from .conversion import collect_native
from .models import OptimizationError
from .orchestrator import collect, prepare, status, submit
from .registry import OPTIMIZERS, specification
from .schedulers.slurm import generate_job_template


def build_parser(parser):
    parser.description = (
        "Prepare and submit independent force-field optimization runs through Slurm.\n"
        "Use verified native command mappings or an explicit external runner contract.\n"
        "Optimizers/engines keep separate environments. GUI integration is deferred."
    )
    actions = parser.add_mutually_exclusive_group(required=True)
    for flag, help_text in (
        ("--generate-template", "Write a commented YAML settings template; refuse overwrite."),
        ("--generate-job-template", "Write the site-neutral Slurm worker-contract template."),
        ("--settings", "Validate and submit a YAML campaign; use --dry-run to prepare only."),
        ("--submit", "Submit an existing prepared campaign once; never retry attempted jobs."),
        ("--status", "Inspect campaign scheduler and worker status without resubmission."),
        ("--collect", "Collect current native outputs from every campaign run."),
        ("--smoke-report", "Inspect cluster smoke-test evidence without submitting or certifying scientific validity."),
        ("--convert", "Convert one existing native output directory without running an optimizer."),
    ):
        action = actions.add_argument(flag, type=Path, help=help_text)
        action.help_metadata = ("Input and file selection", "short")
    action = actions.add_argument("--list-optimizers", action="store_true", help="Show audited adapter capabilities and native setting mappings.")
    action.help_metadata = ("Scientific choices", "short")
    mode = parser.add_mutually_exclusive_group()
    for flag, help_text in (
        ("--validate", "Check settings, inputs, and job contract without staging or submission."),
        ("--dry-run", "Prepare reviewable run files without calling Slurm."),
    ):
        action = mode.add_argument(flag, action="store_true", help=help_text)
        action.help_metadata = ("Execution", "short")
    for key in ("ffield", "params", "trainset", "geo", "control", "job"):
        help_text = ("Default Slurm job template; per-optimizer job paths retain precedence."
                     if key == "job" else f"Override the {key} input path; relative to the current directory.")
        action = parser.add_argument("--" + key, type=Path, help=help_text)
        action.help_metadata = ("Input and file selection", "short")
    action = parser.add_argument("--optimizer", choices=OPTIMIZERS, help="Optimizer format for --convert.")
    action.help_metadata = ("Scientific choices", "short")
    action = parser.add_argument("--output", type=Path, help="Normalized output directory for --convert.")
    action.help_metadata = ("Outputs and plots", "short")
    action = parser.add_argument("--conversion-config", type=Path, help="Optional YAML converter options, including version-specific columns.")
    action.help_metadata = ("Input and file selection", "full")
    return parser


def run_main(args):
    if getattr(args, "resume", None):
        raise OptimizationError("--resume is an analysis option; native optimizer recovery requires an explicitly prepared new run")
    overrides = {key: str(getattr(args, key)) for key in ("ffield", "params", "trainset", "geo", "control", "job") if getattr(args, key, None)}
    if (args.validate or args.dry_run or overrides) and not args.settings:
        raise OptimizationError("Input overrides, --validate and --dry-run require --settings")
    if (args.optimizer or args.output or args.conversion_config) and not args.convert:
        raise OptimizationError("--optimizer, --output and --conversion-config require --convert")
    if args.generate_template:
        result = {"template": str(generate_template(args.generate_template))}
    elif args.generate_job_template:
        result = {"job_template": str(generate_job_template(args.generate_job_template))}
    elif args.list_optimizers:
        result = [specification(optimizer) for optimizer in OPTIMIZERS]
    elif args.settings:
        config = load_config(args.settings, overrides)
        if args.validate:
            result = {"valid": True, "optimizers": config["optimizers"], "replicas": config["replicas"],
                      "execution_contracts": {optimizer: config["optimizer_settings"][optimizer]["runtime"]["mode"]
                                              for optimizer in config["optimizers"]},
                      "cluster_validated": False}
        else:
            if not args.dry_run:
                import platform
                if platform.system() != "Linux":
                    raise OptimizationError("Submit on Roar Linux; use --dry-run for local preparation")
            campaign = prepare(config)
            result = {"campaign": str(campaign), "dry_run": args.dry_run}
            if not args.dry_run:
                result["submissions"] = submit(campaign)
    elif args.submit:
        result = {"submissions": submit(args.submit)}
    elif args.status:
        result = status(args.status)
    elif args.collect:
        result = collect(args.collect)
    elif args.smoke_report:
        from .validation import smoke_report
        result = smoke_report(args.smoke_report)
    else:
        if not args.optimizer or not args.output:
            raise OptimizationError("--convert requires --optimizer and --output")
        options = {}
        if args.conversion_config:
            try:
                options = yaml.load(args.conversion_config.read_text(encoding="utf-8"), Loader=UniqueLoader)
            except yaml.YAMLError as error:
                raise OptimizationError(f"Invalid conversion YAML: {error}") from error
        result = collect_native(args.optimizer, args.convert, args.output, options)
    print(json.dumps(result, indent=2, allow_nan=False))
    if isinstance(result, dict) and result.get("automated_checks_pass") is False:
        return 1
    rows = result if isinstance(result, list) else result.get("submissions", [])
    return 1 if any("error" in row or row.get("state") in ("unknown", "submission_failed", "submitting") for row in rows) else 0
