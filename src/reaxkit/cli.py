"""Command-line interface entry point for running ReaxKit modules and workflows."""
import argparse, sys
from reaxkit.workflows import (
    eregime_workflow, fort13_workflow, fort78_workflow, fort79_workflow,
    molfra_workflow, summary_workflow, xmolout_workflow, fort7_workflow,
    coordination_workflow, xmolout_fort7_workflow, geo_workflow,
    fort99_workflow, trainset_workflow, fort83_workflow,
    fort73_workflow, electrostatics_workflow, make_video_workflow,
    plotter_workflow,
)
from reaxkit import introspection

WORKFLOW_MODULES = {
    "fort78": fort78_workflow, "xmolout": xmolout_workflow, "summary": summary_workflow,
    "eregime": eregime_workflow, "molfra": molfra_workflow, "fort13": fort13_workflow,
    "fort79": fort79_workflow, "fort7": fort7_workflow, "xmolfort7": xmolout_fort7_workflow,
    "coord": coordination_workflow, "intspec": introspection, "geo": geo_workflow,
    "fort99": fort99_workflow, "trainset": trainset_workflow, "fort83": fort83_workflow,
    "fort73": fort73_workflow, "elect": electrostatics_workflow, 'video': make_video_workflow,
    "plotter": plotter_workflow,
}
DEFAULTABLE = {"gplot"}
DEFAULT_TASKS = {"plot", "extreme", "_default"}

def _preinject(argv):
    out = list(argv)
    for i, tok in enumerate(out):
        if tok in DEFAULTABLE:
            j = i + 1
            if j >= len(out): break
            nxt = out[j]
            if nxt in ("-h","--help"): break            # allow workflow help
            if nxt not in DEFAULT_TASKS:                # filename or option → inject
                out.insert(j, "_default")
            break
    return out

def _intspec_default_runner(args):
    return introspection.run_main(getattr(args, "file", None), getattr(args, "folder", None))

def main():
    sys_argv = _preinject(sys.argv)

    parser = argparse.ArgumentParser("reaxkit CLI")
    sub = parser.add_subparsers(dest="kind", required=True)

    for kind, module in WORKFLOW_MODULES.items():
        kp = sub.add_parser(kind, help=f"{kind} workflows")

        if kind == "intspec":
            g = kp.add_mutually_exclusive_group(required=False)
            g.add_argument("--file")
            g.add_argument("--folder")
            tasks = kp.add_subparsers(dest="task", required=False)
            module.register_tasks(tasks)
            kp.set_defaults(_run=_intspec_default_runner)
        else:
            tasks = kp.add_subparsers(dest="task", required=kind not in DEFAULTABLE)
            module.register_tasks(tasks)

    args = parser.parse_args(sys_argv[1:])
    return args._run(args)
