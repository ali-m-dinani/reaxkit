"""Command-line interface entry point for running ReaxKit workflows.

This module builds the top-level `reaxkit` CLI and delegates per-file/feature
commands to individual workflow modules (summary, xmolout, fort7, etc.).

"""
import argparse
import sys

from reaxkit.workflows import (
    eregime_workflow, fort13_workflow, fort78_workflow,
    fort79_workflow, molfra_workflow, summary_workflow,
    xmolout_workflow, fort7_workflow, coordination_workflow,
    xmolout_fort7_workflow, geo_workflow, fort99_workflow,
    trainset_workflow, fort83_workflow, fort73_workflow,
    electrostatics_workflow, make_video_workflow, plotter_workflow,
    control_workflow, fort76_workflow, fort74_workflow,
    ffield_workflow, params_workflow, fort57_workflow,
    vels_workflow, introspection_workflow, help_workflow,
    tregime_workflow,
)

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
    "fort74": fort74_workflow, "ffield": ffield_workflow, "params": params_workflow,
    "energylog": fort73_workflow, "fort58": fort73_workflow, "fort57": fort57_workflow,
    "vels": vels_workflow, "help": help_workflow, "fort8": fort7_workflow,
    "moldyn": vels_workflow, "molsav": vels_workflow, "tregime": tregime_workflow,

}


# Workflows that are allowed to omit an explicit task; a default task
# will be injected for them if the user does not provide one.
DEFAULTABLE = {}

# Names that are treated as "known tasks" when deciding whether to inject
# a synthetic `_default` task for DEFAULTABLE workflows.
DEFAULT_TASKS = {}


def _preinject(argv):
    """Optionally inject a default task for certain workflows.

    For workflows listed in DEFAULTABLE, if the user calls, e.g.,
        reaxkit gplot somefile
    this rewrites the argument list to:
        reaxkit gplot _default somefile
    so that argparse sees an explicit task name (`_default`).
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
    """Default runner for the `intspec` (introspection) workflow.

    If the user supplies `--file` or `--folder`, pass those through to
    `introspection.run_main` and let that module decide what to do.
    """
    return introspection_workflow.run_main(
        getattr(args, "file", None),
        getattr(args, "folder", None),
    )


def main():
    """Build and execute the top-level `reaxkit` CLI.

    This function:
      1. Preprocesses argv to inject default tasks where needed.
      2. Creates the top-level parser and the `kind` subparsers.
      3. Lets each workflow module register its own `task` subparsers.
      4. Parses the CLI and dispatches to the selected task's `_run` function.
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
            # Introspection is a bit special: it can run without an explicit task
            # and accepts either a file or a folder (mutually exclusive).
            g = kp.add_mutually_exclusive_group(required=False)
            g.add_argument("--file")
            g.add_argument("--folder")

            # Optional second-level tasks for `intspec`, if provided.
            tasks = kp.add_subparsers(dest="task", required=False)
            module.register_tasks(tasks)

            # If no explicit task is given, use the default introspection runner.
            kp.set_defaults(_run=_intspec_default_runner)

        elif kind == "help":
            # Like `intspec`: runs without a second-level task.
            # Accept an optional query string.
            kp.add_argument(
                "query",
                nargs="?",
                help='Search phrase, e.g. "restraint", "bond order", "bulk modulus".',
            )
            kp.add_argument("--top", type=int, default=8, help="Max number of results to show.")
            kp.add_argument("--min-score", type=float, default=35.0, help="Minimum match score threshold.")
            kp.add_argument("--why", action="store_true", help="Show brief reasons for each match.")
            kp.add_argument("--examples", action="store_true", help="Show one example command per match.")
            kp.add_argument("--tags", action="store_true", help="Show tags for each matched file.")
            kp.add_argument("--core-vars", action="store_true", help="Show core_vars for each matched file.")
            kp.add_argument("--optional-vars", action="store_true", help="Show optional_vars for each matched file.")
            kp.add_argument("--derived-vars", action="store_true", help="Show derived_vars for each matched file.")
            kp.add_argument("--notes", action="store_true", help="Show notes for each matched file.")
            kp.add_argument("--all-info", action="store_true",
                            help="Show why/examples/tags/core/optional/derived/notes.")

            # Call the help workflow's runner directly (no tasks).
            kp.set_defaults(_run=help_workflow.run_main)

        else:
            # Normal workflows: require a task unless the kind is in DEFAULTABLE.
            # e.g. `reaxkit summary get ...`
            tasks = kp.add_subparsers(dest="task", required=kind not in DEFAULTABLE)
            module.register_tasks(tasks)

    # Parse the CLI (minus the program name) and dispatch to the chosen task.
    args = parser.parse_args(sys_argv[1:])
    return args._run(args)
