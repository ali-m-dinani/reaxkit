"""Dedicated workflow for restraint time series."""

from reaxkit.workflows.timeseries.common import build_restraint_request, configure_parser, run_task

COMMAND = "get_restraint"


def build_parser(parser, *, command: str):
    configure_parser(
        parser,
        command=command,
        description=(
            "Get restraint energies or target/actual values.\n\n"
            "Examples:\n"
            "  reaxkit get_restraint --fields E_res --fort76 fort.76\n"
            "  reaxkit get_restraint --restraint-index 1 --fort76 fort.76"
        ),
        inputs=("fort76",),
    )
    parser.add_argument("--fields", nargs="*", default=None)
    parser.add_argument("--restraint-index", type=int, nargs="*", default=None)
    parser.add_argument("--dropna-rows", action="store_true")
    return parser


def build_request(args):
    if not args.fields and not args.restraint_index:
        raise ValueError("Provide --fields and/or --restraint-index.")
    return build_restraint_request(args)


def run_main(command: str, args) -> int:
    return run_task(COMMAND, "restraint_series", build_request(args), args)

