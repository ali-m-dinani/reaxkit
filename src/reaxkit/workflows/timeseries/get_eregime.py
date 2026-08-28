"""Dedicated workflow for electric-field regime time series."""

from reaxkit.workflows.timeseries.common import build_eregime_request, configure_parser, run_task

COMMAND = "get_eregime"


def build_parser(parser, *, command: str):
    configure_parser(parser, command=command, description="Get one eregime column as a time series.", inputs=("eregime",))
    parser.add_argument("--field", required=True, help="Eregime field name, such as field, field_zones, or field_dir.")
    return parser


def build_request(args):
    return build_eregime_request(args)


def run_main(command: str, args) -> int:
    return run_task(COMMAND, "eregime_series", build_request(args), args)

