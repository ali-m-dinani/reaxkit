"""Dedicated workflow for electric-field time series."""

from reaxkit.workflows.timeseries.common import build_electric_field_request, configure_parser, run_task

COMMAND = "get_electric_field"


def build_parser(parser, *, command: str):
    configure_parser(
        parser,
        command=command,
        description=(
            "Get applied-field or field-energy components as time series.\n\n"
            "For --xaxis frame, cadence is read from control/iout2, then from "
            "xmolout headers, or finally from an explicit --frame-count.\n\n"
            "Example:\n  reaxkit get_electric_field --components field_z --fort78 fort.78 "
            "--xaxis frame --plot single"
        ),
        inputs=("fort78",),
    )
    parser.add_argument("--components", nargs="+", required=True)
    parser.add_argument("--field-kind", choices=["applied", "energy", "auto"], default="auto")
    parser.add_argument(
        "--copy-to-dot",
        action="store_true",
        help=(
            "Also copy explicitly saved plot or CSV artifacts to the current "
            "directory."
        ),
    )
    return parser


def build_request(args):
    return build_electric_field_request(args)


def run_main(command: str, args) -> int:
    return run_task(COMMAND, "electric_field_series", build_request(args), args)

