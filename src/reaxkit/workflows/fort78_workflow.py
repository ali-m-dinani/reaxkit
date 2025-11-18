"""used to get the data in fort.78 file"""
import argparse
from pathlib import Path
from reaxkit.utils.units import UNITS
from reaxkit.io.fort78_handler import Fort78Handler
from reaxkit.analysis.fort78_analyzer import get_iter_vs
from reaxkit.analysis.plotter import single_plot

# x-axis conversion (iter → frame/time); optional dependency
try:
    from reaxkit.analysis.convert import convert_xaxis  # expects ControlHandler under reaxkit.io
except Exception:  # graceful fallback if module not available
    def convert_xaxis(iters, xaxis: str, control_file: str = "control"):
        if xaxis != "iter":
            print("⚠️ convert_xaxis not available; using 'iter' for x-axis.")
        return iters, "iter"


def _export_csv(x, x_label: str, y, y_label: str, path: str) -> None:
    import pandas as pd
    df = pd.DataFrame({x_label: x, y_label: y})
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"[Done] exported data → {out}")


def get_task(args: argparse.Namespace) -> int:
    """
    Get/plot/export a single column from fort.78 vs iter/frame/time.
    """
    handler = Fort78Handler(args.file)

    # Pull requested column from fort.78 summary.
    # Analyzer will *rename* the output column to the *requested alias*.
    ykey: str = args.column.strip()
    df = get_iter_vs(handler, [ykey])  # DataFrame: columns ['iter', ykey]

    if ykey not in df.columns:
        raise KeyError(f"❌ Column '{ykey}' not found in fort.78 data.")

    # Select and sanitize x/y
    import pandas as pd
    iters = pd.to_numeric(df["iter"], errors="coerce")
    iters = iters.dropna().to_numpy()

    xvals, xlabel = convert_xaxis(iters, args.xaxis, control_file=args.control)

    yvals = pd.to_numeric(df[ykey], errors="coerce").to_numpy()

    # Export CSV if requested
    if args.export:
        _export_csv(xvals, xlabel, yvals, ykey, args.export)

    # Plot and/or save plot
    if args.plot or args.save:
        # Ensure plain numeric arrays before plotting
        import numpy as np
        x_plot = np.asarray(xvals, dtype=float)
        y_plot = np.asarray(yvals, dtype=float)

        single_plot(
            x=x_plot,
            y=y_plot,
            title=f"{ykey} vs {xlabel} (fort.78)",
            xlabel=xlabel,
            ylabel=f"{ykey} ({UNITS.get_sections_data(ykey, '') or ''})",
            save=args.save,       # file path OR directory; handled by plotter._save_or_show
            legend=False,
        )

    # If neither plot nor export was requested, print a short tip
    if not (args.plot or args.save or args.export or args.head):
        print("ℹ️ Nothing plotted or exported. Use --plot, --save <path>, --export <csv>, or --head N to preview rows.")

    return 0


def register_tasks(subparsers: argparse._SubParsersAction) -> None:
    """
    Register only one task under the already-created 'fort78' workflow:
        reaxkit fort78 get ...
    """
    g = subparsers.add_parser(
        "get",
        help="Get/plot/export a single column from fort.78 vs iter/frame/time.\n"
             "reaxkit fort78 get --column E_field_x --save E_field_x.png\n"
             "reaxkit fort78 get --column E_field --save molfa_E_field.png",
    )
    g.add_argument("--file", default="fort.78", help="Path to fort.78 file")
    g.add_argument(
        "--column",
        required=True,
        help="Name of the fort.78 column to extract (e.g., 'E_field_x')",
    )
    g.add_argument(
        "--xaxis",
        choices=("iter", "frame", "time"),
        default="iter",
        help="X-axis for plotting/export (default: iter). 'time' may require a control file.",
    )
    g.add_argument(
        "--control",
        default="control",
        help="Path to control file (only used when --xaxis time).",
    )
    # Output options
    g.add_argument("--plot", action="store_true", help="Show a plot in a window")
    g.add_argument("--save", default=None, help="Save plot to a path or directory")
    g.add_argument("--export", default=None, help="Export data to CSV at this path")

    g.set_defaults(_run=get_task)
