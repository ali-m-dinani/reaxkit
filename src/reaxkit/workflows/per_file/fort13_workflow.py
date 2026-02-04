"""
fort.13 error-analysis workflow for ReaxKit.

This workflow provides utilities for reading and analyzing ReaxFF `fort.13` files,
which store force-field training and optimization error metrics as a function
of training epoch.

It supports:
- Extracting total force-field error values versus epoch.
- Visualizing error convergence during force-field optimization.
- Exporting error data to CSV for post-processing or comparison across runs.

The workflow is intended for monitoring ReaxFF parameter optimization and
assessing convergence behavior in training or fitting workflows.
"""


import argparse
from reaxkit.io.handlers.fort13_handler import Fort13Handler
from reaxkit.analysis.per_file.fort13_analyzer import get_fort13_data
from reaxkit.utils.media.plotter import single_plot
from reaxkit.utils.path import resolve_output_path

def _fort13_get_task(args: argparse.Namespace) -> int:
    """
    Handle 'reaxkit fort13 get ...'
    to plot, export, or save error data vs epoch.
    """
    handler = Fort13Handler(args.file)
    df = get_fort13_data(handler)

    workflow_name = args.kind
    # Export if requested
    if args.export:
        out = resolve_output_path(args.export, workflow_name)
        df.to_csv(out, index=False)
        print(f'Successfully exported data to {out}.')

    if args.plot or args.save:
        x = df["epoch"]
        y = df["total_ff_error"]

        if args.plot:
            single_plot(
                x,
                y,
                title="Force Field Error vs Epoch",
                xlabel="Epoch",
                ylabel="Total Force Field Error",
                save=None,
            )
        elif args.save:
            out = resolve_output_path(args.save, workflow_name)
            single_plot(
                x,
                y,
                title="Force Field Error vs Epoch",
                xlabel="Epoch",
                ylabel="Total Force Field Error",
                save=out,
            )

    if not (args.plot or args.save or args.export):
        print("ℹ️ Nothing to do. Use --plot, --save <path>, --export <csv>.")
    return 0


def register_tasks(subparsers: argparse._SubParsersAction) -> None:
    """
    CLI registration for:
        reaxkit fort13 get ...
    """
    p = subparsers.add_parser(
        "get",
        help="Plot, export, or save fort.13 error data vs epoch.\n",
        description=(
            "Examples:\n"
            "  reaxkit fort13 get --save total_ff_error_vs_epoch.png\n"
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    p.add_argument("--file", default="fort.13", help="Path to fort.13 file")
    p.add_argument("--plot", action="store_true", help="Plot error vs epoch")
    p.add_argument("--save", default=None, help="Save plot image to path")
    p.add_argument("--export", default=None, help="Export data to CSV file")
    p.set_defaults(_run=_fort13_get_task)
