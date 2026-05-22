"""Direct command workflow for generating ``eregime.in`` files."""

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

EREGIME_FILE_TOOL_COMMANDS = ("gen_eregime", "make-eregime")


def _safe_build_func(expr: str) -> Callable[[float], float]:
    import math
    import numpy as np

    allowed: dict[str, Any] = {k: getattr(math, k) for k in dir(math) if not k.startswith("_")}
    allowed["np"] = np
    code = compile(expr, "<func-expr>", "eval")

    def f(t: float) -> float:
        return float(eval(code, {"__builtins__": {}}, {**allowed, "t": float(t)}))

    return f


def build_parser(parser: argparse.ArgumentParser, *, command: str) -> argparse.ArgumentParser:
    _ = command
    parser.set_defaults(command="gen_eregime")
    parser.formatter_class = argparse.RawTextHelpFormatter
    parser.description = (
        "Generate ReaxFF eregime.in files from standard field profiles.\n\n"
        "Examples:\n"
        "  reaxkit gen_eregime --type sin --output eregime.in --max-magnitude 0.004 --step-angle 0.05 --iteration-step 500 --num-cycles 2 --direction z --V 1\n"
        "  reaxkit gen_eregime --type pulse --output eregime.in --amplitude 0.003 --width 50 --period 200 --slope 20 --iteration-step 250 --num-cycles 5 --direction z --V 1\n"
        "  reaxkit gen_eregime --type func --output eregime.in --expr '0.003*cos(2*pi*t/100)' --t-end 1000 --dt 1 --iteration-step 250 --direction z --V 1"
    )

    parser.add_argument("--type", choices=["sin", "pulse", "func"], required=True, help="Generator profile type")
    parser.add_argument("--output", default="eregime.in", help="Output file path")
    parser.add_argument("--copy-to-dot", action="store_true", help="Also copy generated output to current directory")
    parser.add_argument("--direction", default="z", help="Field direction: x|y|z")
    parser.add_argument("--V", type=int, default=1, help="Voltage index")
    parser.add_argument("--start-iter", type=int, default=0, help="Starting iteration")

    parser.add_argument("--max-magnitude", type=float, default=None, help="Peak amplitude for sin profile (V/A)")
    parser.add_argument("--step-angle", type=float, default=None, help="Angular sampling step for sin profile (radians)")
    parser.add_argument("--num-cycles", type=float, default=None, help="Number of cycles for sin or pulse profile")
    parser.add_argument("--phase", type=float, default=0.0, help="Phase offset for sin profile (radians)")
    parser.add_argument("--dc-offset", type=float, default=0.0, help="DC offset for sin profile (V/A)")

    parser.add_argument("--amplitude", type=float, default=None, help="Peak amplitude for pulse profile (V/A)")
    parser.add_argument("--width", type=float, default=None, help="Flat-top width for pulse profile")
    parser.add_argument("--period", type=float, default=None, help="Full-cycle period for pulse profile")
    parser.add_argument("--slope", type=float, default=None, help="Ramp duration for pulse profile")
    parser.add_argument("--step-size", type=float, default=0.1, help="Temporal resolution for pulse profile")
    parser.add_argument("--baseline", type=float, default=0.0, help="Baseline for pulse profile (V/A)")

    parser.add_argument("--expr", default=None, help="Python expression in t for func profile")
    parser.add_argument("--t-end", type=float, default=None, help="End time for func profile")
    parser.add_argument("--dt", type=float, default=None, help="Time step for func profile")

    parser.add_argument("--iteration-step", type=int, required=True, help="Iterations per sample")
    add_storage_cli_arguments(parser)
    return parser


def run_main(command: str, args: argparse.Namespace) -> int:
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
