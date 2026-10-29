"""Dedicated workflow for trajectory-coordinate time series."""

from reaxkit.workflows.timeseries.common import build_trajectory_request, configure_parser, run_task

COMMAND = "get_trajectory"


def build_parser(parser, *, command: str):
    configure_parser(
        parser,
        command=command,
        description=(
            "Get trajectory coordinates for selected atoms and dimensions.\n"
            "\n"
            "Inspect selected atom coordinates in an existing trajectory.\n"
            "Combined dimensions report coordinate-vector magnitudes; single axes retain coordinates.\n"
            "\n"
            "Examples:\n"
            "  1. Separate coordinates:\n"
            "     reaxkit get-trajectory --xmolout runs/heating/xmolout --atom-ids 1 2 --dims x y z --plot subplot\n"
            "\n"
            "  2. Coordinate magnitude:\n"
            "     reaxkit get-trajectory --xmolout runs/heating/xmolout --atom-types O --dims xyz --export oxygen_coordinates.csv"
        ),
        inputs=("xmolout",),
    )
    parser.add_argument("--atom-ids", type=int, nargs="*", default=None, help="One-based atom IDs to include; omitted IDs allow all atoms or the type filter. Example: --atom-ids 1 2, which includes only atoms 1 and 2.")
    parser.add_argument("--atom-types", nargs="*", default=None, help="Atom types to include when atom IDs are omitted. Example: --atom-types O H, which includes oxygen and hydrogen atoms.")
    parser.add_argument("--dims", nargs="+", choices=["x", "y", "z", "xy", "xz", "yz", "xyz"], default=("x", "y", "z"), help="Coordinate components to extract; combined axes give vector magnitudes. Example: --dims x y, which keeps x and y as separate series.")
    return parser


def build_request(args):
    return build_trajectory_request(args)


def run_main(command: str, args) -> int:
    return run_task(COMMAND, "trajectory_coordinate_series", build_request(args), args)

