"""Dedicated workflow for per-atom charge time series."""

from reaxkit.workflows.timeseries.common import build_charge_request, configure_parser, run_task

COMMAND = "get_charge"


def build_parser(parser, *, command: str):
    configure_parser(
        parser,
        command=command,
        description=(
            "Get charge time series for all atoms or a selected subset.\n\n"
            "Examples:\n"
            "  reaxkit get_charge --frames 0 --export charges.csv\n"
            "  reaxkit get_charge --atom-ids 1 2 --fort7 fort.7 --export charges.csv"
        ),
        inputs=("fort7", "xmolout", "summary"),
    )
    parser.add_argument(
        "--atom-ids",
        type=int,
        nargs="+",
        default=None,
        help="Optional 1-based atom IDs. If omitted, all atoms are included.",
    )
    return parser


def build_request(args):
    return build_charge_request(args)


def run_main(command: str, args) -> int:
    return run_task(COMMAND, "charge_series", build_request(args), args)

