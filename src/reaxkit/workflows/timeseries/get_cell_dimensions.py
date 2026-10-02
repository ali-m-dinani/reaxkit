"""Dedicated workflow for simulation-cell dimensions."""

from reaxkit.workflows.timeseries.common import build_cell_dimensions_request, configure_parser, run_task

COMMAND = "get_cell_dimensions"


def build_parser(parser, *, command: str):
    configure_parser(
        parser,
        command=command,
        description=(
            "Get selected cell lengths and angles as time series.\n"
            "\n"
            "Compare cell expansion and angular distortion in existing trajectory data.\n"
            "\n"
            "Examples:\n"
            "  1. Cell lengths:\n"
            "     reaxkit get-cell-dimensions --xmolout runs/heating/xmolout --fields a b c --plot subplot\n"
            "\n"
            "  2. Cell angles:\n"
            "     reaxkit get-cell-dimensions --xmolout runs/heating/xmolout --fields alpha beta gamma --export cell_angles.csv"
        ),
        inputs=("xmolout", "summary"),
    )
    parser.add_argument(
        "--fields",
        nargs="+",
        choices=["a", "b", "c", "alpha", "beta", "gamma"],
        default=("a", "b", "c", "alpha", "beta", "gamma"),
        help="Cell lengths and angles to include. Example: --fields a b c, which extracts lengths and excludes angles.",
    )
    return parser


def build_request(args):
    return build_cell_dimensions_request(args)


def run_main(command: str, args) -> int:
    return run_task(COMMAND, "cell_dimensions", build_request(args), args)

