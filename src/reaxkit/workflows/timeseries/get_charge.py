"""Dedicated workflow for per-atom charge time series."""

from reaxkit.workflows.timeseries.common import build_charge_request, configure_parser, run_task

COMMAND = "get_charge"


def build_parser(parser, *, command: str):
    configure_parser(
        parser,
        command=command,
        description=(
            "Get charge time series for all atoms or a selected subset.\n"
            "\n"
            "Inspect charge transfer for all atoms or a selected subset of an existing run.\n"
            "\n"
            "Examples:\n"
            "  1. All charges at frame zero:\n"
            "     reaxkit get-charge --fort7 runs/heating/fort.7 --frames 0 --export charges.csv\n"
            "\n"
            "  2. Selected atom charges:\n"
            "     reaxkit get-charge --fort7 runs/heating/fort.7 --atom-ids 1 2 --plot single"
        ),
        inputs=("fort7", "xmolout", "summary"),
    )
    parser.add_argument(
        "--atom-ids",
        type=int,
        nargs="+",
        default=None,
        help="One-based atom IDs to include; omitted IDs allow all atoms or the type filter. Example: --atom-ids 1 2, which includes only atoms 1 and 2.",
    )
    return parser


def build_request(args):
    return build_charge_request(args)


def run_main(command: str, args) -> int:
    return run_task(COMMAND, "charge_series", build_request(args), args)

