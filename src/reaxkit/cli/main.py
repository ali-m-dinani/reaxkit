"""
Top-level command-line interface for ReaxKit workflows.

Works on: ReaxKit CLI (`reaxkit`) → dispatches to workflow modules under `reaxkit.workflows`.

This module defines the `reaxkit` entry point and routes each top-level subcommand
("kind") to its corresponding workflow module, which then registers task-level
subcommands (or runs as a kind-level workflow such as `help` / `intspec`).
"""

import argparse
import sys
from importlib import import_module

from reaxkit.core.analysis_cli_routing_registry import get_registered_analysis_commands
from reaxkit.core.command_catalog import get_registered_commands
from reaxkit.core.command_alias_resolver import resolve_command_name
from reaxkit.core.exceptions import ParseError, AnalysisError
from reaxkit.core.generator_cli_routing_registry import get_registered_generators
from reaxkit.core.workflow_cli_routing_registry import get_registered_workflows


# Workflows that are allowed to omit an explicit task; a default task
# will be injected for them if the user does not provide one.
DEFAULTABLE = {}

# Names that are treated as "known tasks" when deciding whether to inject
# a synthetic `_default` task for DEFAULTABLE workflows.
DEFAULT_TASKS = {}


def _command_help_text(name: str, fallback: str) -> str:
    """Return help text for a top-level command."""
    spec = get_registered_commands(include_analysis_tasks=True).get(name)
    return spec.help_text or fallback if spec is not None else fallback


def _preinject(argv):
    """
    Inject a default task for selected workflows before argparse parsing.

    Works on: CLI argv preprocessing for workflows that allow omitting an explicit task.

    Parameters
    ----------
    argv : Sequence[str]
        Raw argument vector (typically `sys.argv`).

    Returns
    -------
    list[str]
        A rewritten argv where a synthetic `_default` task is inserted for
        workflows in `DEFAULTABLE` when the next token is not a known task.

    Examples
    --------
    >>> _preinject(["reaxkit", "gplot", "file.txt"])
    ['reaxkit', 'gplot', '_default', 'file.txt']
    """
    out = list(argv)
    for i, tok in enumerate(out):
        if tok in DEFAULTABLE:
            j = i + 1
            if j >= len(out):
                break
            nxt = out[j]
            if nxt in ("-h", "--help"):
                break  # allow `reaxkit gplot -h` to show help normally
            if nxt not in DEFAULT_TASKS:  # filename or option → inject default task
                out.insert(j, "_default")
            break
    return out


def _intspec_default_runner(args):
    """
    Run the `intspec` workflow without requiring a task subcommand.

    Works on: `intspec` kind-level workflow (introspection).

    Parameters
    ----------
    args : argparse.Namespace
        Parsed CLI arguments; may include `file` and/or `folder`.

    Returns
    -------
    int
        Workflow exit code returned by `introspection_workflow.run_main(...)`.

    Examples
    --------
    # Equivalent CLI calls:
    # reaxkit intspec --file fort7_analyzer
    # reaxkit intspec --folder workflow
    """
    introspection_workflow = import_module("reaxkit.workflows.meta.introspection_workflow")
    return introspection_workflow.run_main(
        getattr(args, "file", None),
        getattr(args, "folder", None),
    )


def _canonicalize_direct_command(argv):
    """Rewrite direct command aliases to canonical names before parsing."""
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


def _direct_command_runner(module, command):
    def _runner(args):
        return module.run_main(command, args)

    return _runner


def main():
    """
    Build and execute the `reaxkit` CLI dispatcher.

    Works on: ReaxKit CLI (`reaxkit`) and registered workflow modules.

    This function:
      1. Preprocesses argv to inject default tasks where needed.
      2. Creates the top-level parser and the `kind` subparsers.
      3. Lets each workflow module register its own `task` subparsers.
      4. Parses the CLI and dispatches to the selected task's `_run` function.

    Returns
    -------
    int
        Exit code returned by the selected workflow task runner.

    Examples
    --------
    Direct command:

      - reaxkit connection_list --fort7 fort.7 --export connections.csv

    Kind-level workflow without task:

      - reaxkit help --query "fort.7"

      - reaxkit intspec --folder workflows
    """
    # Preprocess argv so DEFAULTABLE workflows can omit an explicit task.
    sys_argv = _canonicalize_direct_command(_preinject(sys.argv))

    probe = argparse.ArgumentParser(add_help=False)
    probe.add_argument("command", nargs="?")
    kind_ns, _ = probe.parse_known_args(sys_argv[1:])
    selected_command = getattr(kind_ns, "command", None)

    # Top-level parser for `reaxkit`
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

    # First level of subcommands: direct commands and legacy workflow kinds.
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

    # For each workflow module, create its own subparser and let it
    # register its internal tasks (second-level subcommands).
    for kind, spec in workflow_commands.items():
        # e.g. `reaxkit help ...`, `reaxkit timeseries ...`
        kp = sub.add_parser(kind, help=_command_help_text(kind, f"{kind} workflows"))

        if kind != selected_command:
            continue

        module = import_module(spec.module_path)

        if spec.dispatch_mode == "intspec_runner":
            # Kind-level workflow: no subcommands
            if hasattr(module, "build_parser"):
                module.build_parser(kp)
            kp.set_defaults(_run=_intspec_default_runner)

        elif spec.dispatch_mode == "kind_runner":
            # Kind-level workflow: no subcommands
            if hasattr(module, "build_parser"):
                module.build_parser(kp)
            kp.set_defaults(_run=module.run_main)

        else:
            # Normal workflows: require a task unless the kind is in DEFAULTABLE.
            # e.g. `reaxkit help ...` would not reach this branch
            tasks = kp.add_subparsers(dest="task", required=kind not in DEFAULTABLE)
            module.register_tasks(tasks)

    # Parse the CLI (minus the program name) and dispatch to the chosen task.
    args = parser.parse_args(sys_argv[1:])
    try:
        return args._run(args)
    except ParseError as exc:
        print(f"[Parse error] {exc}", file=sys.stderr)
        return 2
    except AnalysisError as exc:
        print(f"[Analysis error] {exc}", file=sys.stderr)
        return 3
