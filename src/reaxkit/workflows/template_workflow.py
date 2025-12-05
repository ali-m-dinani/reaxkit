"""a template workflow for future workflow development"""
import argparse
from reaxkit.io.template_handler import TemplateHandler
from reaxkit.analysis.template_analyzer import example_metric
from reaxkit.utils.plotter import single_plot


def metric_task(args: argparse.Namespace) -> int:
    handler = TemplateHandler(args.file)
    df = example_metric(handler)
    x = df["iteration"]
    y = df["energy"]
    single_plot(x, y, title="Energy vs Iteration",
                xlabel="Iteration", ylabel="Energy", save=args.save)
    if args.save:
        print(f"[Done] Saved plot to {args.save}")
    return 0


def register_tasks(subparsers: argparse._SubParsersAction) -> None:
    """
    in the CLI structure of:
        reaxkit kind task --flags
    task should be placed here as a parser not the kind (workflow).
    moreover, if multiple tasks have common flags like plot, etc., they should be grouped in a format like:
        def _add_common_xmolout_io_args(
            p: argparse.ArgumentParser,
            *,
            include_plot: bool = False,
        ) -> None:
            p.add_argument("--file", default="xmolout", help="Path to xmolout file.")
            if include_plot:
                p.add_argument("--plot", action="store_true", help="Show plot interactively.")
            p.add_argument("--save", default=None, help="Path to save plot image.")
            p.add_argument("--export", default=None, help="Path to export CSV data.")

    furthermore, examples of each task are added in a format as follows:
        description=(
            "Examples:\n"
            "  reaxkit workflow task --flags\n"
        ),
        formatter_class=argparse.RawTextHelpFormatter,

    finally, for the sake of readability, add_arguments as written as compact as possible. so, instead of:
        pt.add_argument(
        "--atoms",
        default=None,
        help="Comma/space separated 1-based atom indices, e.g. '1,5,12'."
        )
    it's better to write it as:
        pt.add_argument("--atoms", default=None,
        help="Comma/space separated 1-based atom indices, e.g. '1,5,12'.")

    """
    p = subparsers.add_parser("metric", help="Plot example metric")
    p.add_argument("--file", required=True, help="Path to <filetype> file")
    p.add_argument("--save", default=None, help="Path to save plot (optional)")
    p.set_defaults(_run=metric_task)
