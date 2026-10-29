"""Dedicated workflow for electric-field regime time series."""

from reaxkit.workflows.timeseries.common import build_eregime_request, configure_parser, run_task

COMMAND = "get_eregime"


def build_parser(parser, *, command: str):
    configure_parser(parser, command=command, description=(
            "Get one eregime column as a time series.\n"
            "\n"
            "Inspect the prescribed field program in an existing eregime.in file.\n"
            "This reads the field schedule; it does not run a simulation.\n"
            "\n"
            "Examples:\n"
            "  1. Field strength:\n"
            "     reaxkit get-eregime --eregime runs/heating/eregime.in --field field --plot single"
        ), inputs=("eregime",))
    parser.add_argument("--field", required=True, help="Electric-field regime column to extract. Example: --field field, which returns the prescribed field strength.")
    return parser


def build_request(args):
    return build_eregime_request(args)


def run_main(command: str, args) -> int:
    return run_task(COMMAND, "eregime_series", build_request(args), args)

