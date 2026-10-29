"""Dedicated workflow for trajectory-displacement time series."""

from reaxkit.workflows.timeseries.common import build_displacement_request, configure_parser, run_task

COMMAND = "get_displacement"


def build_parser(parser, *, command: str):
    configure_parser(
        parser,
        command=command,
        description=(
            "Get atom displacement relative to a reference frame.\n"
            "\n"
            "Measure coordinate changes relative to a source frame in an existing trajectory.\n"
            "Combined dimensions report displacement magnitudes; single axes retain signed components.\n"
            "\n"
            "Examples:\n"
            "  1. Signed components:\n"
            "     reaxkit get-displacement --xmolout runs/heating/xmolout --atom-ids 1 2 --dims x y --reference-frame 0 --plot subplot\n"
            "\n"
            "  2. In-plane magnitude:\n"
            "     reaxkit get-displacement --xmolout runs/heating/xmolout --atom-ids 1 2 --dims xy --reference-frame 0 --export displacement.csv"
        ),
        inputs=("xmolout",),
    )
    parser.add_argument("--atom-ids", type=int, nargs="*", default=None, help="One-based atom IDs to include; omitted IDs allow all atoms or the type filter. Example: --atom-ids 1 2, which includes only atoms 1 and 2.")
    parser.add_argument("--atom-types", nargs="*", default=None, help="Atom types to include when atom IDs are omitted. Example: --atom-types O H, which includes oxygen and hydrogen atoms.")
    parser.add_argument("--dims", nargs="+", choices=["x", "y", "z", "xy", "xz", "yz", "xyz"], default=("xyz",), help="Coordinate components to extract; combined axes give vector magnitudes. Example: --dims x y, which keeps x and y as separate series.")
    parser.add_argument("--reference-frame", type=int, default=0, help="Zero-based source frame used as the displacement reference. Example: --reference-frame 10, which subtracts frame 10 coordinates from each selected frame.")
    return parser


def build_request(args):
    return build_displacement_request(args)


def run_main(command: str, args) -> int:
    return run_task(COMMAND, "trajectory_displacement_series", build_request(args), args)

