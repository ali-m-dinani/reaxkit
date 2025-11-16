"""a template workflow for future workflow development"""
import argparse
from reaxkit.io.template_handler import TemplateHandler
from reaxkit.analysis.template_analyzer import example_metric
from reaxkit.analysis.plotter import single_plot


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
    p = subparsers.add_parser("metric", help="Plot example metric")
    p.add_argument("--file", required=True, help="Path to <filetype> file")
    p.add_argument("--save", default=None, help="Path to save plot (optional)")
    p.set_defaults(_run=metric_task)
