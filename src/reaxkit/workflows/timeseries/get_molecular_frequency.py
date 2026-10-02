"""Dedicated workflow for molecular-frequency time series."""

from reaxkit.workflows.timeseries.common import build_molecular_frequency_request, configure_parser, run_task

COMMAND = "get_molecular_frequency"


def build_parser(parser, *, command: str):
    configure_parser(
        parser,
        command=command,
        description=(
            "Get molecular frequency time series for selected formulas.\n"
            "\n"
            "Track selected species counts in existing molecular-analysis output.\n"
            "\n"
            "Examples:\n"
            "  1. Water and hydroxyl:\n"
            "     reaxkit get-molecular-frequency --molfra runs/heating/molfra.out --molecules H2O OH --plot single"
        ),
        inputs=("molfra",),
    )
    parser.add_argument("--molecules", nargs="+", required=True, help="Molecular formulas to include in frequency series. Example: --molecules H2O OH, which includes water and hydroxyl counts only.")
    return parser


def build_request(args):
    return build_molecular_frequency_request(args)


def run_main(command: str, args) -> int:
    return run_task(COMMAND, "molecular_frequency_series", build_request(args), args)

