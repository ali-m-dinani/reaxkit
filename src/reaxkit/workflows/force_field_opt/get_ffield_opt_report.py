"""Generate a complete force-field optimization report from one run directory.

This composite workflow delegates to the existing focused commands instead of
reimplementing their analysis logic. ``REPORT_COMMANDS`` is intentionally kept
near the top of the module so developers can see the complete report pipeline
without tracing the runner.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import ModuleType

from reaxkit.core.resolve.command_alias_resolver import resolve_command_name
from reaxkit.core.storage.storage_layout import (
    ReaxkitStorageLayout,
    add_storage_cli_arguments,
    normalize_storage_args,
)
from reaxkit.workflows.file_tools import ffield_workflow, trainset_workflow
from reaxkit.workflows.force_field_opt import get_ffield_opt_plots as plots_workflow

ALL_COMMANDS = ("get-ffield-opt-report",)
ALL_LEGACY_COMMANDS = ("get_ffield_opt_report",)

REPORT_COMMANDS = (
    "get_trainset_data",
    "get_trainset_group_comments",
    "get_ffield_opt_results",
    "get_ffield_opt_plots",
    "get_ffield_opt_bulk_modulus",
)

_COMMAND_MODULES: dict[str, ModuleType] = {
    "get_trainset_data": trainset_workflow,
    "get_trainset_group_comments": trainset_workflow,
    "get_ffield_opt_results": ffield_workflow,
    "get_ffield_opt_plots": plots_workflow,
    "get_ffield_opt_bulk_modulus": ffield_workflow,
}


def _positive_int(value: str) -> int:
    """Return a strictly positive integer for argparse options."""
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be at least 1")
    return parsed


def _canonical_command(command: str) -> str:
    """Resolve the report command and its underscore spelling."""
    return resolve_command_name(command, task_names=ALL_COMMANDS)


def build_parser(
    parser: argparse.ArgumentParser,
    *,
    command: str,
) -> argparse.ArgumentParser:
    """Configure the ``get-ffield-opt-report`` command parser."""
    canonical = _canonical_command(command)
    parser.formatter_class = argparse.RawTextHelpFormatter
    parser.set_defaults(command=canonical, progress=True)
    parser.description = (
        "Generate a complete force-field optimization report by running the existing\n"
        "trainset-data, trainset-comment, optimization-result, plot, and bulk-modulus\n"
        "workflows in sequence. Run this command inside a force-field optimization\n"
        "directory to use fort.99, fort.74, trainset.in, and geo automatically.\n\n"
        "By default, results are written under:\n"
        "  reaxkit_workspace/analysis/get-ffield-opt-report/<run-id>/\n\n"
        "Examples:\n"
        "  1. Build a report from the current optimization directory:\n"
        "     reaxkit get-ffield-opt-report\n\n"
        "  2. Build a report from another run and choose the output directory:\n"
        "     reaxkit get-ffield-opt-report --run-dir run --output report\n"
    )
    parser.add_argument("--engine", choices=["reaxff", "ams", "lammps"], default=None)
    parser.add_argument("--input", default=None, help="Input path used for engine detection")
    parser.add_argument(
        "--run-dir",
        "--dir",
        dest="run_dir",
        default=".",
        help="Force-field optimization run directory",
    )
    parser.add_argument("--fort99", default="fort.99", help="Path to fort.99")
    parser.add_argument("--fort74", default="fort.74", help="Path to fort.74")
    parser.add_argument("--trainset", default="trainset.in", help="Path to trainset file")
    parser.add_argument("--geo", default="geo", help="Path to the geo file")
    parser.add_argument(
        "--entry-per-figure",
        type=_positive_int,
        default=6,
        help="Maximum entries per grouped-bar figure (default: 6)",
    )
    parser.add_argument(
        "--min-points",
        type=_positive_int,
        default=6,
        help="Minimum EOS points required for each bulk-modulus fit (default: 6)",
    )
    parser.add_argument(
        "--flip-sign",
        action="store_true",
        help="Flip energy signs during bulk-modulus fitting",
    )
    parser.add_argument(
        "--no-shift-min-to-zero",
        action="store_true",
        help="Do not shift minimum energy to zero during bulk-modulus fitting",
    )
    parser.add_argument(
        "--output",
        "--outdir",
        dest="output",
        default=None,
        help="Optional report-directory override",
    )
    parser.add_argument("--log", choices=["verbose", "quiet"], default=None)
    add_storage_cli_arguments(parser)
    return parser


def _storage_argv(args: argparse.Namespace) -> list[str]:
    """Build shared storage arguments for one delegated command."""
    argv = [
        "--run-id",
        str(args.run_id),
        "--project-root",
        str(args.project_root),
    ]
    if getattr(args, "analysis_id", None):
        argv.extend(["--analysis-id", str(args.analysis_id)])
    return argv


def _runtime_argv(args: argparse.Namespace) -> list[str]:
    """Build common force-field runtime arguments."""
    argv = [
        "--input",
        str(args.input),
        "--run-dir",
        str(args.run_dir),
        "--fort99",
        str(args.fort99),
        "--fort74",
        str(args.fort74),
        "--trainset",
        str(args.trainset),
    ]
    if args.engine:
        argv.extend(["--engine", str(args.engine)])
    if args.log:
        argv.extend(["--log", str(args.log)])
    return argv


def _delegated_argv(
    command: str,
    args: argparse.Namespace,
    report_root: Path,
) -> list[str]:
    """Return parser arguments for one delegated report command."""
    storage = _storage_argv(args)
    if command == "get_trainset_data":
        return [
            "--run-dir",
            str(args.run_dir),
            "--trainset",
            str(args.trainset),
            "--section",
            "all",
            "--export",
            str(report_root / "trainset_data"),
            *storage,
        ]
    if command == "get_trainset_group_comments":
        return [
            "--run-dir",
            str(args.run_dir),
            "--trainset",
            str(args.trainset),
            "--section",
            "all",
            "--export",
            str(report_root / "trainset_group_comments.csv"),
            *storage,
        ]
    if command == "get_ffield_opt_results":
        return [
            *_runtime_argv(args),
            "--export",
            str(report_root / "ffield_opt_results.csv"),
            *storage,
        ]
    if command == "get_ffield_opt_plots":
        return [
            *_runtime_argv(args),
            "--geo",
            str(args.geo),
            "--entry-per-figure",
            str(args.entry_per_figure),
            "--output",
            str(report_root / "plots"),
            *storage,
        ]
    if command == "get_ffield_opt_bulk_modulus":
        argv = [
            *_runtime_argv(args),
            "--iden",
            "all",
            "--min-points",
            str(args.min_points),
            "--export",
            str(report_root / "bulk_modulus.csv"),
            *storage,
        ]
        if args.flip_sign:
            argv.append("--flip-sign")
        if args.no_shift_min_to_zero:
            argv.append("--no-shift-min-to-zero")
        return argv
    raise KeyError(f"Unsupported delegated report command: {command}")


def _report_root(args: argparse.Namespace) -> Path:
    """Return the explicit or run-scoped report output directory."""
    if args.output:
        root = Path(args.output).expanduser()
    else:
        layout = ReaxkitStorageLayout(project_root=Path(args.project_root))
        root = (
            layout.analysis_root
            / ALL_COMMANDS[0]
            / str(args.analysis_id or args.run_id)
        )
    root.mkdir(parents=True, exist_ok=True)
    return root


def _run_delegated_command(
    command: str,
    args: argparse.Namespace,
    report_root: Path,
) -> int:
    """Parse and execute one existing workflow with report-specific outputs."""
    module = _COMMAND_MODULES[command]
    parser = argparse.ArgumentParser(prog=f"reaxkit {command}", add_help=False)
    module.build_parser(parser, command=command)
    child_args = parser.parse_args(_delegated_argv(command, args, report_root))
    child_args.progress = bool(getattr(args, "progress", True))
    return int(module.run_main(command, child_args))


def run_main(command: str, args: argparse.Namespace) -> int:
    """Run every report command and write a status manifest."""
    canonical = _canonical_command(command)
    uses_default_geo = Path(str(args.geo)).parent == Path(".") and str(args.geo) == "geo"
    normalized = normalize_storage_args(vars(args))
    for key, value in normalized.items():
        setattr(args, key, value)
    if uses_default_geo:
        args.geo = str(Path(args._snapshot_source_dir) / "geo")
    report_root = _report_root(args)

    statuses: list[dict[str, object]] = []
    for index, delegated in enumerate(REPORT_COMMANDS, start=1):
        print(f"[Report] ({index}/{len(REPORT_COMMANDS)}) Running {delegated}")
        try:
            exit_code = _run_delegated_command(delegated, args, report_root)
            statuses.append({"command": delegated, "exit_code": exit_code})
        except Exception as exc:
            statuses.append(
                {
                    "command": delegated,
                    "exit_code": 1,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            print(f"[Warning] {delegated} failed: {exc}")

    manifest = {
        "command": canonical,
        "delegated_commands": list(REPORT_COMMANDS),
        "run_id": args.run_id,
        "analysis_id": args.analysis_id,
        "source_directory": str(args._snapshot_source_dir),
        "output_directory": str(report_root),
        "statuses": statuses,
    }
    manifest_path = report_root / "report_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )

    failures = [status for status in statuses if int(status["exit_code"]) != 0]
    if failures:
        print(
            f"[Warning] Report completed with {len(failures)} failed command(s). "
            f"See {manifest_path}"
        )
        print(f"Results saved in:\n  {report_root}")
        return 1

    print(f"[Done] Force-field optimization report manifest: {manifest_path}")
    print(f"Results saved in:\n  {report_root}")
    return 0


__all__ = [
    "ALL_COMMANDS",
    "ALL_LEGACY_COMMANDS",
    "REPORT_COMMANDS",
    "build_parser",
    "run_main",
]
