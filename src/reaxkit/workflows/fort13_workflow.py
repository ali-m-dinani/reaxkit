"""used to get the data in fort.13 file"""
import argparse
from reaxkit.io.fort13_handler import Fort13Handler
from reaxkit.analysis.fort13_analyzer import get_errors
from reaxkit.analysis.plotter import single_plot


def fort13_get_task(args: argparse.Namespace) -> int:
    """
    Handle 'reaxkit fort13 get ...'
    to plot, export, or save error data vs epoch.
    """
    handler = Fort13Handler(args.file)
    df = get_errors(handler)

    # Export if requested
    if args.export:
        df.to_csv(args.export, index=False)
        print(f'Successfully exported data to {args.export}.')

    if args.plot or args.save:
        x = df["epoch"]
        y = df["total_ff_error"]
        single_plot(
            x,
            y,
            title="Force Field Error vs Epoch",
            xlabel="Epoch",
            ylabel="Total FF Error",
            save=args.save,
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
        "get", help="Plot, export, or save fort.13 error data vs epoch.\n"
                    "reaxkit fort13 get --save total_ff_error_vs_epoch.png"
    )
    p.add_argument("--file", default="fort.13", help="Path to fort.13 file")
    p.add_argument("--plot", action="store_true", help="Plot error vs epoch")
    p.add_argument("--save", default=None, help="Save plot image to path")
    p.add_argument("--export", default=None, help="Export data to CSV file")
    p.set_defaults(_run=fort13_get_task)
