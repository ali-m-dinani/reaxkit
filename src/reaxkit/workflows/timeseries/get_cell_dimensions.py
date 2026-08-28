"""Dedicated workflow for simulation-cell dimensions."""

from reaxkit.workflows.timeseries.common import build_cell_dimensions_request, configure_parser, run_task

COMMAND = "get_cell_dimensions"


def build_parser(parser, *, command: str):
    configure_parser(
        parser,
        command=command,
        description="Get selected cell lengths and angles as time series.",
        inputs=("xmolout", "summary"),
    )
    parser.add_argument(
        "--fields",
        nargs="+",
        choices=["a", "b", "c", "alpha", "beta", "gamma"],
        default=("a", "b", "c", "alpha", "beta", "gamma"),
    )
    return parser


def build_request(args):
    return build_cell_dimensions_request(args)


def run_main(command: str, args) -> int:
    return run_task(COMMAND, "cell_dimensions", build_request(args), args)

