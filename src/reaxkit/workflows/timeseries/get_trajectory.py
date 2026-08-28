"""Dedicated workflow for trajectory-coordinate time series."""

from reaxkit.workflows.timeseries.common import build_trajectory_request, configure_parser, run_task

COMMAND = "get_trajectory"


def build_parser(parser, *, command: str):
    configure_parser(
        parser,
        command=command,
        description=(
            "Get trajectory coordinates for selected atoms and dimensions.\n\n"
            "Example:\n  reaxkit get_trajectory --atom-ids 1 2 --dims z --xaxis time --plot single"
        ),
        inputs=("xmolout",),
    )
    parser.add_argument("--atom-ids", type=int, nargs="*", default=None, help="1-based atom IDs; defaults to all atoms.")
    parser.add_argument("--atom-types", nargs="*", default=None, help="Atom types used when atom IDs are omitted.")
    parser.add_argument("--dims", nargs="+", choices=["x", "y", "z", "xy", "xz", "yz", "xyz"], default=("x", "y", "z"))
    return parser


def build_request(args):
    return build_trajectory_request(args)


def run_main(command: str, args) -> int:
    return run_task(COMMAND, "trajectory_coordinate_series", build_request(args), args)

