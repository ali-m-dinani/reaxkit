"""
Top-level command-line interface for ReaxKit.

This module defines the ``reaxkit`` entry point and routes each top-level command
to either:
- a direct analysis/generator command module, or
- a workflow command module (command-level or task-subcommand based).
"""

from __future__ import annotations

import argparse
import sys
from importlib import import_module

from reaxkit.core.analysis_cli_routing_registry import get_registered_analysis_commands
from reaxkit.core.command_alias_resolver import resolve_command_name
from reaxkit.core.command_catalog import get_registered_commands
from reaxkit.core.exceptions import AnalysisError, ParseError
from reaxkit.core.generator_cli_routing_registry import get_registered_generators
from reaxkit.core.workflow_cli_routing_registry import get_registered_workflows


def _print_error_with_hints(title: str, message: str, *, hints: list[str] | None = None) -> None:
    print(f"[{title}] {message}", file=sys.stderr)
    for hint in hints or []:
        print(f"  - {hint}", file=sys.stderr)


def _hints_for_filenotfound(exc: FileNotFoundError, args: argparse.Namespace) -> list[str]:
    msg = str(exc)
    command = str(getattr(args, "command", "") or "")
    hints: list[str] = []
    if "Study file not found:" in msg and command == "study":
        hints.append("Ensure the YAML path is correct and relative to your current directory.")
        hints.append(r"Example: reaxkit study --init .\study.yaml --root .")
        hints.append(r"Create template first if needed: reaxkit study --make-yaml study.yaml")
    return hints


def _hints_for_oserror(exc: OSError, args: argparse.Namespace) -> list[str]:
    msg = str(exc)
    lower = msg.lower()
    if getattr(exc, "winerror", None) == 206 or "filename or extension is too long" in lower:
        command = str(getattr(args, "command", "") or "")
        hints = [
            r"Use a short project root for analysis cache/output paths, e.g. add: --project-root C:\rk",
            r"Shorten case folder names in a study: reaxkit study --manage <study_root> --action rename-cases --target case-names",
            "Optionally enable Windows long paths policy and restart.",
        ]
        if command and command != "study":
            hints.insert(0, f"Re-run with a shorter root, e.g.: reaxkit {command} ... --project-root C:\\rk")
        return hints
    return []


def _command_help_text(name: str, fallback: str) -> str:
    """Return help text for a top-level command."""
    spec = get_registered_commands(include_analysis_tasks=True).get(name)
    return spec.help_text or fallback if spec is not None else fallback


def _intspec_runner(args: argparse.Namespace) -> int:
    """Run the ``intspec`` command-level workflow."""
    workflow = import_module("reaxkit.workflows.meta.introspection_workflow")
    return workflow.run_main(
        getattr(args, "file", None),
        getattr(args, "folder", None),
    )


def _canonicalize_direct_command(argv: list[str]) -> list[str]:
    """Rewrite direct-command aliases to canonical names before parsing."""
    out = list(argv)
    if len(out) < 2:
        return out

    direct_commands = {
        **get_registered_analysis_commands(),
        **get_registered_generators(),
    }
    try:
        out[1] = resolve_command_name(out[1], task_names=direct_commands.keys())
    except KeyError:
        pass
    return out


def _direct_command_runner(module, command: str):
    """Create an argparse runner for direct-command modules."""

    def _runner(args: argparse.Namespace) -> int:
        return module.run_main(command, args)

    return _runner


def main() -> int:
    """
    Build and execute the ``reaxkit`` CLI dispatcher.

    Examples
    --------
    - reaxkit connection_list --fort7 fort.7 --export connections.csv
    - reaxkit help "fort.7"
    - reaxkit intspec --folder workflows
    """
    sys_argv = _canonicalize_direct_command(sys.argv)

    probe = argparse.ArgumentParser(add_help=False)
    probe.add_argument("command", nargs="?")
    command_ns, _ = probe.parse_known_args(sys_argv[1:])
    selected_command = getattr(command_ns, "command", None)

    parser = argparse.ArgumentParser("reaxkit CLI")
    parser.add_argument(
        "--timing",
        action="store_true",
        help="Print per-task timing to console (timing is always persisted to logs/timing.log).",
    )
    parser.add_argument(
        "--progress",
        action="store_true",
        help="Enable progress reporting for supported handlers and analysis tasks.",
    )
    parser.add_argument(
        "--log-in-terminal",
        action="store_true",
        help="Show [ReaxKit] runtime log lines in terminal output.",
    )

    sub = parser.add_subparsers(dest="command", required=True)

    direct_commands = {
        **get_registered_analysis_commands(),
        **get_registered_generators(),
    }
    workflow_commands = get_registered_workflows()

    for command, spec in direct_commands.items():
        cp = sub.add_parser(command, help=_command_help_text(command, f"{command} command"))

        if command != selected_command:
            continue

        module = import_module(spec.module_path)
        if hasattr(module, "build_parser"):
            module.build_parser(cp, command=command)
        cp.set_defaults(_run=_direct_command_runner(module, command))

    for command, spec in workflow_commands.items():
        wp = sub.add_parser(command, help=_command_help_text(command, f"{command} workflows"))

        if command != selected_command:
            continue

        module = import_module(spec.module_path)

        if spec.dispatch_mode == "intspec_runner":
            if hasattr(module, "build_parser"):
                module.build_parser(wp)
            wp.set_defaults(_run=_intspec_runner)
        elif spec.dispatch_mode == "kind_runner":
            if hasattr(module, "build_parser"):
                module.build_parser(wp)
            wp.set_defaults(_run=module.run_main)
        else:
            tasks = wp.add_subparsers(dest="task", required=True)
            module.register_tasks(tasks)

    args = parser.parse_args(sys_argv[1:])
    try:
        return args._run(args)
    except ParseError as exc:
        print(f"[Parse error] {exc}", file=sys.stderr)
        return 2
    except AnalysisError as exc:
        print(f"[Analysis error] {exc}", file=sys.stderr)
        return 3
    except FileNotFoundError as exc:
        _print_error_with_hints("File not found", str(exc), hints=_hints_for_filenotfound(exc, args))
        return 4
    except OSError as exc:
        _print_error_with_hints("OS error", str(exc), hints=_hints_for_oserror(exc, args))
        return 5
    except Exception as exc:
        _print_error_with_hints("Error", str(exc))
        return 1
