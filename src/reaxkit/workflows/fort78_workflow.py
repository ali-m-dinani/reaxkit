"""workflow used to get the data in fort.78 file"""

import argparse
from pathlib import Path
from reaxkit.utils.units import UNITS
from reaxkit.io.fort78_handler import Fort78Handler
from reaxkit.analysis.fort78_analyzer import get_iter_vs_fort78_data
from reaxkit.utils.plotter import single_plot
from reaxkit.utils.convert import convert_xaxis
from reaxkit.utils.path import resolve_output_path

def _export_csv(x, x_label: str, y, y_label: str, path: str) -> None:
    import pandas as pd
    df = pd.DataFrame({x_label: x, y_label: y})
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"[Done] exported data to {out}")


def get_task(args: argparse.Namespace) -> int:
    """
    Get/plot/export a single yaxis from fort.78 vs iter/frame/time.
    """
    handler = Fort78Handler(args.file)

    # Pull requested yaxis from fort.78 summary.
    # Analyzer will *rename* the output yaxis to the *requested alias*.
    ykey: str = args.yaxis.strip()
    df = get_iter_vs_fort78_data(handler, [ykey])  # DataFrame: columns ['iter', ykey]

    if ykey not in df.columns:
        raise KeyError(f"❌ yaxis '{ykey}' not found in fort.78 data.")

    # Select and sanitize x/y
    import pandas as pd
    iters = pd.to_numeric(df["iter"], errors="coerce")
    iters = iters.dropna().to_numpy()

    xvals, xlabel = convert_xaxis(iters, args.xaxis, control_file=args.control)

    yvals = pd.to_numeric(df[ykey], errors="coerce").to_numpy()

    workflow_name = args.kind
    # Export CSV if requested
    if args.export:
        out = resolve_output_path(args.export, workflow_name)
        _export_csv(xvals, xlabel, yvals, ykey, out)

    # Plot and/or save plot
    if args.plot or args.save:
        # Ensure plain numeric arrays before plotting
        import numpy as np
        x_plot = np.asarray(xvals, dtype=float)
        y_plot = np.asarray(yvals, dtype=float)

        if args.save:
            out = resolve_output_path(args.save, workflow_name)
            single_plot(
                x=x_plot,
                y=y_plot,
                title=f"{ykey} vs {xlabel}",
                xlabel=xlabel,
                ylabel=f"{ykey} ({UNITS.get(ykey, '') or ''})",
                save=out,
                legend=False,
            )
        elif args.plot:
            single_plot(
                x=x_plot,
                y=y_plot,
                title=f"{ykey} vs {xlabel}",
                xlabel=xlabel,
                ylabel=f"{ykey} ({UNITS.get_sections_data(ykey, '') or ''})",
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
        help="Get/plot/export a single yaxis from fort.78 vs iter/frame/time.\n",
        description=(
            "Examples:\n"
            "  reaxkit fort78 get --xaxis time --yaxis E_field_x --save E_field_x.png --export E_field_x.csv\n"
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    g.add_argument("--file", default="fort.78", help="Path to fort.78 file")
    g.add_argument("--yaxis",
        required=True,
        help="Name of the fort.78 yaxis to extract (e.g., 'E_field_x')",
    )
    g.add_argument("--xaxis",
        choices=("iter", "frame", "time"), default="iter",
        help="X-axis for plotting/export (default: iter). 'time' may require a control file.",
    )
    g.add_argument("--control", default="control",
        help="Path to control file (only used when --xaxis time).",
    )
    # Output options
    g.add_argument("--plot", action="store_true", help="Show a plot in a window")
    g.add_argument("--save", default=None, help="Save plot to a path or directory")
    g.add_argument("--export", default=None, help="Export data to CSV at this path")

    g.set_defaults(_run=get_task)
