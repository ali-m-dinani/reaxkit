"""Dedicated workflow for geometry-optimization progress."""

from reaxkit.workflows.timeseries.common import build_geometry_optimization_request, configure_parser, run_task

COMMAND = "get_geometry_optimization"
COMPONENTS = ("E_pot", "T", "T_set", "RMSG", "nfc")


def build_parser(parser, *, command: str):
    configure_parser(
        parser,
        command=command,
        description="Get selected geometry-optimization progress components; omit --components to get all.",
        inputs=("fort57",),
    )
    parser.add_argument("--components", nargs="*", choices=list(COMPONENTS), default=None)
    parser.add_argument("--include-geo-descriptor", action="store_true")
    return parser


def build_request(args):
    return build_geometry_optimization_request(args)


def run_main(command: str, args) -> int:
    return run_task(COMMAND, "geometry_optimization_data", build_request(args), args)

