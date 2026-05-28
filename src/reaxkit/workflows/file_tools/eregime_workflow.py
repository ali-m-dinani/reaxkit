"""Direct command workflow for generating ``eregime.in`` files.

This module implements CLI workflow orchestration for its command family, including argument parsing, request construction, execution dispatch, and result presentation handoff.

**Usage context**

- Command routing: Resolve CLI aliases and normalized command names.
- Task execution: Build request objects and invoke registered tasks.
- Output handling: Forward results to table, plot, export, or report flows.
"""

from __future__ import annotations

import argparse
from typing import Any, Callable

from reaxkit.core.generator_runtime import (
    maybe_copy_output_to_dot,
    persist_generator_metadata,
    prepare_generator_output,
    print_saved_dirs,
)
from reaxkit.core.storage_layout import add_storage_cli_arguments
from reaxkit.engine.reaxff.generators.eregime_generator import gen_eregime

ALL_COMMANDS = ("gen_eregime",)
ALL_LEGACY_COMMANDS = ("gen-eregime", "make-eregime", "make_eregime")


def _safe_build_func(expr: str) -> Callable[[float], float]:
    """Safe build func."""
    import math
    import numpy as np

    allowed: dict[str, Any] = {k: getattr(math, k) for k in dir(math) if not k.startswith("_")}
    allowed["np"] = np
    code = compile(expr, "<func-expr>", "eval")

    def f(t: float) -> float:
        """F.

        Execute the workflow function for this command path and return the
        computed result for downstream CLI handling.

        Parameters
        -----
        t : Any
            Function argument.

        Returns
        -----
        float
            Function return value.

        Examples
        -----
        >>> # See workflow CLI usage for concrete examples.
        """
        return float(eval(code, {"__builtins__": {}}, {**allowed, "t": float(t)}))

    return f


def build_parser(parser: argparse.ArgumentParser, *, command: str) -> argparse.ArgumentParser:
    """Build parser.

    Execute the workflow function for this command path and return the
    computed result for downstream CLI handling.

    Parameters
    -----
    parser : Any
        Function argument.
    command : Any
        Function argument.

    Returns
    -----
    argparse.ArgumentParser
        Function return value.

    Examples
    -----
    >>> # See workflow CLI usage for concrete examples.
    """
    _ = command
    parser.set_defaults(command="gen_eregime")
    parser.formatter_class = argparse.RawTextHelpFormatter
    parser.description = (
        "Generate a ReaxFF `eregime.in` file from a selected electric-field profile.\n"
        "This command writes sampled field values for three profile types:\n"
        "  1. `sin`  -> sinusoidal waveform\n"
        "  2. `pulse` -> pulse waveform with rise/flat/fall regions\n"
        "  3. `func` -> custom expression in `t`\n"
        "It only generates the input file and does not execute the simulation.\n\n"
        "Examples:\n"
        "  1. Sinusoidal profile:\n"
        "   reaxkit gen_eregime --type sin --output eregime.in --max-magnitude 0.004 --step-angle 0.05 --iteration-step 500 --num-cycles 2 --direction z --V 1\n\n"
        "  2. Pulse profile:\n"
        "   reaxkit gen_eregime --type pulse --output eregime.in --amplitude 0.003 --width 50 --period 200 --slope 20 --iteration-step 250 --num-cycles 5 --direction z --V 1\n\n"
        "  3. Custom function profile:\n"
        "   reaxkit gen_eregime --type func --output eregime.in --expr '0.003*cos(2*pi*t/100)' --t-end 1000 --dt 1 --iteration-step 250 --direction z --V 1"
    )

    parser.add_argument(
        "--type",
        choices=["sin", "pulse", "func"],
        required=True,
        help="Generator profile type. Example: --type sin, which selects sinusoidal waveform generation.",
    )
    parser.add_argument(
        "--output",
        default="eregime.in",
        help="Output file path. Example: --output eregime_custom.in, which writes the generated file with that name.",
    )
    parser.add_argument(
        "--copy-to-dot",
        action="store_true",
        help="Also copy generated output to current directory. Example: --copy-to-dot, which keeps a convenience copy where you run the command.",
    )
    parser.add_argument(
        "--direction",
        default="z",
        help="Field direction: x|y|z. Example: --direction x, which applies the field along x-axis.",
    )
    parser.add_argument(
        "--V",
        type=int,
        default=1,
        help="Voltage index. Example: --V 2, which writes the field under voltage channel/index 2.",
    )
    parser.add_argument(
        "--start-iter",
        type=int,
        default=0,
        help="Starting iteration. Example: --start-iter 1000, which starts the generated schedule at iteration 1000.",
    )

    parser.add_argument(
        "--max-magnitude",
        type=float,
        default=None,
        help="Peak amplitude for sin profile (V/A). Example: --max-magnitude 0.004, which sets the sine peak field strength.",
    )
    parser.add_argument(
        "--step-angle",
        type=float,
        default=None,
        help="Angular sampling step for sin profile (radians). Example: --step-angle 0.05, which controls sine sampling density per cycle.",
    )
    parser.add_argument(
        "--num-cycles",
        type=float,
        default=None,
        help="Number of cycles for sin or pulse profile. Example: --num-cycles 3, which repeats the waveform for three cycles.",
    )
    parser.add_argument(
        "--phase",
        type=float,
        default=0.0,
        help="Phase offset for sin profile (radians). Example: --phase 1.57, which shifts the sine wave by roughly pi/2.",
    )
    parser.add_argument(
        "--dc-offset",
        type=float,
        default=0.0,
        help="DC offset for sin profile (V/A). Example: --dc-offset 0.001, which adds a constant baseline to the sine waveform.",
    )

    parser.add_argument(
        "--amplitude",
        type=float,
        default=None,
        help="Peak amplitude for pulse profile (V/A). Example: --amplitude 0.003, which sets the pulse peak field strength.",
    )
    parser.add_argument(
        "--width",
        type=float,
        default=None,
        help="Flat-top width for pulse profile. Example: --width 50, which sets how long each pulse stays at peak level.",
    )
    parser.add_argument(
        "--period",
        type=float,
        default=None,
        help="Full-cycle period for pulse profile. Example: --period 200, which sets one pulse cycle duration.",
    )
    parser.add_argument(
        "--slope",
        type=float,
        default=None,
        help="Ramp duration for pulse profile. Example: --slope 20, which sets rise/fall transition duration.",
    )
    parser.add_argument(
        "--step-size",
        type=float,
        default=0.1,
        help="Temporal resolution for pulse profile. Example: --step-size 0.1, which samples the pulse every 0.1 time unit.",
    )
    parser.add_argument(
        "--baseline",
        type=float,
        default=0.0,
        help="Baseline for pulse profile (V/A). Example: --baseline 0.0005, which shifts the pulse around a non-zero base field.",
    )

    parser.add_argument(
        "--expr",
        default=None,
        help="Python expression in t for func profile. Example: --expr '0.003*cos(2*pi*t/100)', which defines field value as a function of t.",
    )
    parser.add_argument(
        "--t-end",
        type=float,
        default=None,
        help="End time for func profile. Example: --t-end 1000, which sets the final time point for function sampling.",
    )
    parser.add_argument(
        "--dt",
        type=float,
        default=None,
        help="Time step for func profile. Example: --dt 1, which samples the function every 1 time unit.",
    )

    parser.add_argument(
        "--iteration-step",
        type=int,
        required=True,
        help="Iterations per sample. Example: --iteration-step 250, which maps each generated sample to 250 MD iterations.",
    )
    add_storage_cli_arguments(parser)
    return parser


def run_main(command: str, args: argparse.Namespace) -> int:
    """Run main.

    Execute the workflow function for this command path and return the
    computed result for downstream CLI handling.

    Parameters
    -----
    command : Any
        Function argument.
    args : Any
        Function argument.

    Returns
    -----
    int
        Function return value.

    Examples
    -----
    >>> # See workflow CLI usage for concrete examples.
    """
    out_path, layout = prepare_generator_output(args, command=command, output_value=str(args.output))

    gen_eregime(
        out_path,
        profile_type=args.type,
        iteration_step=args.iteration_step,
        direction=args.direction,
        voltage_idx=args.V,
        start_iter=args.start_iter,
        max_magnitude=args.max_magnitude,
        step_angle=args.step_angle,
        num_cycles=args.num_cycles,
        phase=args.phase,
        dc_offset=args.dc_offset,
        amplitude=args.amplitude,
        width=args.width,
        period=args.period,
        slope=args.slope,
        step_size=args.step_size,
        baseline=args.baseline,
        func=_safe_build_func(args.expr) if args.expr is not None else None,
        t_end=args.t_end,
        dt=args.dt,
    )

    persist_generator_metadata(
        args,
        command=command,
        output_path=out_path,
        layout=layout,
        copy_to_dot=bool(getattr(args, "copy_to_dot", False)),
    )
    copied = maybe_copy_output_to_dot(out_path, enabled=bool(getattr(args, "copy_to_dot", False)))
    dirs = [out_path.parent]
    if copied is not None:
        dirs.append(copied.parent)
    print_saved_dirs(dirs)
    return 0
