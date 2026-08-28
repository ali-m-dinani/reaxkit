"""Dedicated workflow for trajectory-displacement time series."""

from reaxkit.workflows.timeseries.common import build_displacement_request, configure_parser, run_task

COMMAND = "get_displacement"


def build_parser(parser, *, command: str):
    configure_parser(
        parser,
        command=command,
        description=(
            "Get atom displacement relative to a reference frame.\n\n"
            "Example:\n  reaxkit get_displacement --atom-ids 1 2 --dims xy --reference-frame 0 --plot single"
        ),
        inputs=("xmolout",),
    )
    parser.add_argument("--atom-ids", type=int, nargs="*", default=None, help="1-based atom IDs; defaults to all atoms.")
    parser.add_argument("--atom-types", nargs="*", default=None)
    parser.add_argument("--dims", nargs="+", choices=["x", "y", "z", "xy", "xz", "yz", "xyz"], default=("xyz",))
    parser.add_argument("--reference-frame", type=int, default=0)
    return parser


def build_request(args):
    return build_displacement_request(args)


def run_main(command: str, args) -> int:
    return run_task(COMMAND, "trajectory_displacement_series", build_request(args), args)

