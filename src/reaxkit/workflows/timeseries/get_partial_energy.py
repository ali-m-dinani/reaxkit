"""Dedicated workflow for partial-energy time series."""

from reaxkit.workflows.timeseries.common import build_partial_energy_request, configure_parser, run_task

COMMAND = "get_partial_energy"


def build_parser(parser, *, command: str):
    configure_parser(
        parser,
        command=command,
        description=(
            "Extract partial-energy components from a fort.73 time series.\n"
            "\n"
            "Inspect individual energy contributions in existing fort.73 output.\n"
            "Omit --components to include all available components.\n"
            "Use --relative for differences from --reference-frame (default: 0), before sampling.\n"
            "Use --num-atoms N for per-atom energies, including relative energies.\n"
            "\n"
            "Examples:\n"
            "  1. Combined plot:\n"
            "     reaxkit get-partial-energy --fort73 runs/heating/fort.73 --components Ebond Eatom --plot single\n"
            "\n"
            "  2. Separate panels:\n"
            "     reaxkit get-partial-energy --fort73 runs/heating/fort.73 --components Ebond Eatom --plot subplot\n"
            "\n"
            "  3. One image per component:\n"
            "     reaxkit get-partial-energy --fort73 runs/heating/fort.73 --plot separate --save partial_energies"
        ),
        inputs=("fort73",),
    )
    parser.add_argument(
        "--components",
        nargs="*",
        default=None,
        help=(
            "Partial-energy components to include. Example: --components Ebond Eatom, "
            "which includes only the bond and atomic energy series; omit this option "
            "to include all available components."
        ),
    )
    parser.add_argument("--relative", action="store_true", help="Also output each component's energy minus its reference-frame energy. Example: --relative.")
    parser.add_argument("--reference-frame", type=int, default=0, help="Zero-based source frame for --relative, before sampling (default: 0). Example: --reference-frame 10.")
    parser.add_argument("--num-atoms", type=int, default=None, help="Positive atom count used to normalize all components; implies --per-atom. Example: --num-atoms 100.")
    parser.add_argument("--per-atom", action="store_true", help="Report energies per atom; requires --num-atoms. Example: --per-atom --num-atoms 100 --relative.")
    return parser


def build_request(args):
    return build_partial_energy_request(args)


def run_main(command: str, args) -> int:
    return run_task(COMMAND, "partial_energy_series", build_request(args), args)
