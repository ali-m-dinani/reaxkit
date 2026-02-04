"""
Template workflow for building new CLI workflows.

This module demonstrates a minimal ReaxKit workflow structure: load a handler,
run an analysis function, and optionally plot/export results.

Typical use cases include:

- defining small task functions that operate on parsed handler data
- registering subcommands under a workflow "kind" via ``register_tasks()``
"""

from __future__ import annotations

import argparse

from reaxkit.io.base_handler import BaseHandler


def metric_task(args: argparse.Namespace) -> int:
    """
    Run the example metric task (extract + plot energy vs iteration).

    Works on
    --------
    TemplateHandler — ``<filetype>``

    Parameters
    ----------
    args : argparse.Namespace
        Parsed CLI args containing at least ``file`` and optional ``save``.

    Returns
    -------
    int
        Process exit code (0 for success).

    Examples
    --------
    >>> # From CLI:
    >>> # reaxkit template metric --file <filetype> --save energy.png
    """
    handler = BaseHandler(args.file)

    # Some analysis is done here using analyzers
    # .....

    # then the results are plotted
    # single_plot(x, y, title="Energy vs Iteration", xlabel="Iteration", ylabel="Energy", save=args.save)

    if args.save:
        print(f"[Done] Saved plot to {args.save}")
    return 0


def register_tasks(subparsers: argparse._SubParsersAction) -> None:
    """
    Register workflow tasks under the given argparse subparser collection.

    Parameters
    ----------
    subparsers : argparse._SubParsersAction
        Subparser collection created by the parent workflow parser.

    Examples
    --------
    >>> # Typically called by the top-level CLI during workflow registration.
    >>> # register_tasks(subparsers)
    """
    p = subparsers.add_parser("metric", help="Plot example metric")
    p.add_argument("--file", required=True, help="Path to <filetype> file")
    p.add_argument("--save", default=None, help="Path to save plot image (optional)")
    p.set_defaults(_run=metric_task)
