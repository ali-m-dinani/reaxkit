"""Dedicated workflow for partial-energy time series."""

from reaxkit.workflows.timeseries.common import build_partial_energy_request, configure_parser, run_task

COMMAND = "get_partial_energy"


def build_parser(parser, *, command: str):
    configure_parser(
        parser,
        command=command,
        description="Get selected partial-energy components; omit --components to get every available component.",
        inputs=("fort73",),
    )
    parser.add_argument("--components", nargs="*", default=None)
    return parser


def build_request(args):
    return build_partial_energy_request(args)


def run_main(command: str, args) -> int:
    return run_task(COMMAND, "partial_energy_series", build_request(args), args)

