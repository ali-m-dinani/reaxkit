"""Dedicated workflow for restraint time series."""

from reaxkit.workflows.timeseries.common import build_restraint_request, configure_parser, run_task

COMMAND = "get_restraint"


def build_parser(parser, *, command: str):
    configure_parser(
        parser,
        command=command,
        description=(
            "Get restraint energies or target/actual values.\n"
            "\n"
            "Inspect restraint energies or target/actual values in existing fort.76 output.\n"
            "Provide --fields and/or --restraint-index to select the values.\n"
            "\n"
            "Examples:\n"
            "  1. Restraint energy:\n"
            "     reaxkit get-restraint --fort76 runs/heating/fort.76 --fields E_res --plot single\n"
            "\n"
            "  2. Target and actual values:\n"
            "     reaxkit get-restraint --fort76 runs/heating/fort.76 --restraint-index 1 --export restraint_1.csv"
        ),
        inputs=("fort76",),
    )
    parser.add_argument("--fields", nargs="*", default=None, help="Named restraint columns to include. Example: --fields E_res, which extracts restraint energy.")
    parser.add_argument("--restraint-index", type=int, nargs="*", default=None, help="Restraint indices whose target and actual values are selected. Example: --restraint-index 1, which extracts values for restraint 1.")
    parser.add_argument("--dropna-rows", action="store_true", help="Remove rows missing all selected restraint values. Example: --dropna-rows, which excludes empty records from the output.")
    return parser


def build_request(args):
    if not args.fields and not args.restraint_index:
        raise ValueError("Provide --fields and/or --restraint-index.")
    return build_restraint_request(args)


def run_main(command: str, args) -> int:
    return run_task(COMMAND, "restraint_series", build_request(args), args)

