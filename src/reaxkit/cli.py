"""
Top-level command-line interface for ReaxKit workflows.

Works on: ReaxKit CLI (`reaxkit`) → dispatches to workflow modules under `reaxkit.workflows`.

This module defines the `reaxkit` entry point and routes each top-level subcommand
("kind") to its corresponding workflow module, which then registers task-level
subcommands (or runs as a kind-level workflow such as `help` / `intspec`).
"""

import argparse
import sys

from reaxkit.workflows.meta import make_video_workflow, plotter_workflow, help_workflow, introspection_workflow
from reaxkit.workflows.per_file import fort57_workflow, summary_workflow, geo_workflow, trainset_workflow, \
    fort76_workflow, fort78_workflow, xmolout_workflow, fort83_workflow, eregime_workflow, fort79_workflow, \
    fort99_workflow, control_workflow, ffield_workflow, vels_workflow, fort13_workflow, tregime_workflow, \
    params_workflow, fort7_workflow, vregime_workflow, molfra_workflow, fort73_workflow, fort74_workflow
from reaxkit.workflows.composed import coordination_workflow, xmolout_fort7_workflow, electrostatics_workflow

# Mapping from top-level CLI "kind" (subcommand) to the workflow module
# that knows how to register its own tasks and arguments.

# important note: energylog and fort.58 have exactly the same structure as fort.73, hence they should map to fort.73
# also, fort.8 is similar to fort.7 in the same way
# same for molsav and moldyn which are similar to vels
WORKFLOW_MODULES = {
    "fort78": fort78_workflow, "xmolout": xmolout_workflow, "summary": summary_workflow,
    "eregime": eregime_workflow, "molfra": molfra_workflow, "fort13": fort13_workflow,
    "fort79": fort79_workflow, "fort7": fort7_workflow, "xmolfort7": xmolout_fort7_workflow,
    "coord": coordination_workflow, "intspec": introspection_workflow, "geo": geo_workflow,
    "fort99": fort99_workflow, "trainset": trainset_workflow, "fort83": fort83_workflow,
    "fort73": fort73_workflow, "elect": electrostatics_workflow, "video": make_video_workflow,
    "plotter": plotter_workflow, "control": control_workflow, "fort76": fort76_workflow,
    "fort74.md": fort74_workflow, "ffield": ffield_workflow, "params": params_workflow,
    "energylog": fort73_workflow, "fort58": fort73_workflow, "fort57.md": fort57_workflow,
    "vels": vels_workflow, "help": help_workflow, "fort8": fort7_workflow,
    "moldyn": vels_workflow, "molsav": vels_workflow, "tregime": tregime_workflow,
    "vregime": vregime_workflow,
}


# Workflows that are allowed to omit an explicit task; a default task
# will be injected for them if the user does not provide one.
DEFAULTABLE = {}

# Names that are treated as "known tasks" when deciding whether to inject
# a synthetic `_default` task for DEFAULTABLE workflows.
DEFAULT_TASKS = {}


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
    return introspection_workflow.run_main(
        getattr(args, "file", None),
        getattr(args, "folder", None),
    )


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
    Per-file workflow with task:

      - reaxkit fort7 get --file fort.7

    Kind-level workflow without task:

      - reaxkit help --query "fort.7"

      - reaxkit intspec --folder workflows
    """
    # Preprocess argv so DEFAULTABLE workflows can omit an explicit task.
    sys_argv = _preinject(sys.argv)

    # Top-level parser for `reaxkit`
    parser = argparse.ArgumentParser("reaxkit CLI")

    # First level of subcommands: the workflow "kind" (summary, xmolout, etc.)
    sub = parser.add_subparsers(dest="kind", required=True)

    # For each workflow module, create its own subparser and let it
    # register its internal tasks (second-level subcommands).
    for kind, module in WORKFLOW_MODULES.items():
        # e.g. `reaxkit summary ...`, `reaxkit xmolout ...`
        kp = sub.add_parser(kind, help=f"{kind} workflows")

        if kind == "intspec":
            # Kind-level workflow: no subcommands
            if hasattr(module, "build_parser"):
                module.build_parser(kp)
            kp.set_defaults(_run=_intspec_default_runner)

        elif kind == "help":
            # Kind-level workflow: no subcommands
            if hasattr(module, "build_parser"):
                module.build_parser(kp)
            kp.set_defaults(_run=module.run_main)

        else:
            # Normal workflows: require a task unless the kind is in DEFAULTABLE.
            # e.g. `reaxkit summary get ...`
            tasks = kp.add_subparsers(dest="task", required=kind not in DEFAULTABLE)
            module.register_tasks(tasks)

    # Parse the CLI (minus the program name) and dispatch to the chosen task.
    args = parser.parse_args(sys_argv[1:])
    return args._run(args)
