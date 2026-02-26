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

# Mapping from top-level CLI "kind" (subcommand) to the workflow module
# that knows how to register its own tasks and arguments.

# important note: energylog and fort.58 have exactly the same structure as fort.73, hence they should map to fort.73
# also, fort.8 is similar to fort.7 in the same way
# same for molsav and moldyn which are similar to vels
WORKFLOW_MODULES = {
    "fort78": "reaxkit.workflows.per_file.fort78_workflow",
    "xmolout": "reaxkit.workflows.per_file.xmolout_workflow",
    "summary": "reaxkit.workflows.per_file.summary_workflow",
    "eregime": "reaxkit.workflows.per_file.eregime_workflow",
    "molfra": "reaxkit.workflows.per_file.molfra_workflow",
    "fort13": "reaxkit.workflows.per_file.fort13_workflow",
    "fort79": "reaxkit.workflows.per_file.fort79_workflow",
    "fort7": "reaxkit.workflows.per_file.fort7_workflow",
    "xmolfort7": "reaxkit.workflows.composed.xmolout_fort7_workflow",
    "coord": "reaxkit.workflows.composed.coordination_workflow",
    "intspec": "reaxkit.workflows.meta.introspection_workflow",
    "geo": "reaxkit.workflows.per_file.geo_workflow",
    "fort99": "reaxkit.workflows.per_file.fort99_workflow",
    "trainset": "reaxkit.workflows.per_file.trainset_workflow",
    "fort83": "reaxkit.workflows.per_file.fort83_workflow",
    "fort73": "reaxkit.workflows.per_file.fort73_workflow",
    "elect": "reaxkit.workflows.composed.electrostatics_workflow",
    "video": "reaxkit.workflows.meta.make_video_workflow",
    "plotter": "reaxkit.workflows.meta.plotter_workflow",
    "control": "reaxkit.workflows.per_file.control_workflow",
    "fort76": "reaxkit.workflows.per_file.fort76_workflow",
    "fort74.md": "reaxkit.workflows.per_file.fort74_workflow",
    "ffield": "reaxkit.workflows.per_file.ffield_workflow",
    "params": "reaxkit.workflows.per_file.params_workflow",
    "energylog": "reaxkit.workflows.per_file.fort73_workflow",
    "fort58": "reaxkit.workflows.per_file.fort73_workflow",
    "fort57.md": "reaxkit.workflows.per_file.fort57_workflow",
    "vels": "reaxkit.workflows.per_file.vels_workflow",
    "help": "reaxkit.workflows.meta.help_workflow",
    "fort8": "reaxkit.workflows.per_file.fort7_workflow",
    "moldyn": "reaxkit.workflows.per_file.vels_workflow",
    "molsav": "reaxkit.workflows.per_file.vels_workflow",
    "tregime": "reaxkit.workflows.per_file.tregime_workflow",
    "vregime": "reaxkit.workflows.per_file.vregime_workflow",
    "analysis": "reaxkit.workflows.diffusion_workflow",
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
    introspection_workflow = import_module("reaxkit.workflows.meta.introspection_workflow")
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

    probe = argparse.ArgumentParser(add_help=False)
    probe.add_argument("kind", nargs="?")
    kind_ns, _ = probe.parse_known_args(sys_argv[1:])
    selected_kind = getattr(kind_ns, "kind", None)

    # Top-level parser for `reaxkit`
    parser = argparse.ArgumentParser("reaxkit CLI")

    # First level of subcommands: the workflow "kind" (summary, xmolout, etc.)
    sub = parser.add_subparsers(dest="kind", required=True)

    # For each workflow module, create its own subparser and let it
    # register its internal tasks (second-level subcommands).
    for kind, module_path in WORKFLOW_MODULES.items():
        # e.g. `reaxkit summary ...`, `reaxkit xmolout ...`
        kp = sub.add_parser(kind, help=f"{kind} workflows")

        if kind != selected_kind:
            continue

        module = import_module(module_path)

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
