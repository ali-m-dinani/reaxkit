"""Dedicated workflow for electric-field time series."""

from reaxkit.workflows.timeseries.common import build_electric_field_request, configure_parser, run_task

COMMAND = "get_electric_field"


def build_parser(parser, *, command: str):
    configure_parser(
        parser,
        command=command,
        description=(
            "Get applied-field or field-energy components as time series.\n"
            "\n"
            "Compare applied-field and field-energy channels in existing fort.78 data.\n"
            "Frame conversion uses control/iout2, then trajectory headers, then an explicit frame count.\n"
            "\n"
            "Examples:\n"
            "  1. Applied field:\n"
            "     reaxkit get-electric-field --fort78 runs/heating/fort.78 --components field_z --field-kind applied --plot single\n"
            "\n"
            "  2. Field energy:\n"
            "     reaxkit get-electric-field --fort78 runs/heating/fort.78 --components E_field_z --field-kind energy --export field_energy.csv\n"
            "\n"
            "  3. Automatic channel detection:\n"
            "     reaxkit get-electric-field --fort78 runs/heating/fort.78 --components field_x --field-kind auto --plot single"
        ),
        inputs=("fort78",),
    )
    parser.add_argument("--components", nargs="+", required=True, help="Applied-field or field-energy components to include. Example: --components field_z, which extracts the applied z-directed field.")
    parser.add_argument("--field-kind", choices=["applied", "energy", "auto"], default="auto", help="Electric-field group: applied, energy, or automatic component detection. Example: --field-kind energy, which reads field-energy channels.")
    parser.add_argument(
        "--copy-to-dot",
        action="store_true",
        help="Also copy explicitly saved plots and CSV artifacts to the current directory. Example: --copy-to-dot, which creates local copies of those outputs.",
    )
    return parser


def build_request(args):
    return build_electric_field_request(args)


def run_main(command: str, args) -> int:
    return run_task(COMMAND, "electric_field_series", build_request(args), args)

