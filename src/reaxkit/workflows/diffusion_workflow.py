"""Task-first analysis workflow with dynamic task registry dispatch."""

from __future__ import annotations

import argparse

from reaxkit.analysis import trajectory as _trajectory_tasks  # noqa: F401 (registration side effects)
from reaxkit.core.analysis_executor import AnalysisExecutor
from reaxkit.core.task_registry import TASK_REGISTRY
from reaxkit.analysis.trajectory.msd_task import MSDRequest


def _run_msd(args: argparse.Namespace) -> int:
    task_cls = TASK_REGISTRY["msd"]
    task = task_cls()
    request = MSDRequest(atom_ids=args.atom_ids)

    executor = AnalysisExecutor()
    result = executor.run(
        task,
        request,
        {
            "engine": args.engine,
            "input": args.input,
            "xmolout": args.xmolout,
            "run_dir": args.run_dir,
        },
    )

    print(result.table.to_string(index=False))
    return 0


def register_tasks(subparsers):
    """Register task subcommands for `reaxkit analysis ...`."""

    if "msd" in TASK_REGISTRY:
        p = subparsers.add_parser("msd", help="Compute mean-squared displacement")
        p.add_argument("--engine", choices=["reaxff", "ams", "lammps"], default=None)
        p.add_argument("--input", default=".", help="Input file or directory for engine resolution")
        p.add_argument("--run-dir", default=".", help="Run directory (fallback for detection)")
        p.add_argument("--xmolout", default="xmolout", help="ReaxFF xmolout path")
        p.add_argument("--atom-ids", type=int, nargs="*", default=None, help="1-based atom ids")
        p.set_defaults(_run=_run_msd)
