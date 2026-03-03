"""Direct command workflow for generating ``eregime.in`` files."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Callable

from reaxkit.engine.reaxff.generators.eregime_generator import (
    write_eregime_from_function,
    write_eregime_sinusoidal,
    write_eregime_smooth_pulse,
)

EREGIME_FILE_TOOL_COMMANDS = ("make-eregime",)


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
    parser.set_defaults(command="make-eregime")
    parser.formatter_class = argparse.RawTextHelpFormatter
    parser.description = (
        "Generate ReaxFF eregime.in files from standard field profiles.\n\n"
        "Examples:\n"
        "  reaxkit make-eregime --type sin --output eregime.in --max-magnitude 0.004 --step-angle 0.05 --iteration-step 500 --num-cycles 2 --direction z --V 1\n"
        "  reaxkit make-eregime --type pulse --output eregime.in --amplitude 0.003 --width 50 --period 200 --slope 20 --iteration-step 250 --num-cycles 5 --direction z --V 1\n"
        "  reaxkit make-eregime --type func --output eregime.in --expr '0.003*cos(2*pi*t/100)' --t-end 1000 --dt 1 --iteration-step 250 --direction z --V 1"
    )

    parser.add_argument("--type", choices=["sin", "pulse", "func"], required=True, help="Generator profile type")
    parser.add_argument("--output", default="eregime.in", help="Output file path")
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
    return parser


def run_main(command: str, args: argparse.Namespace) -> int:
    _ = command
    out_path = Path(args.output)

    if args.type == "sin":
        if args.max_magnitude is None or args.step_angle is None or args.num_cycles is None:
            raise ValueError("--type sin requires --max-magnitude, --step-angle, and --num-cycles.")
        write_eregime_sinusoidal(
            out_path,
            max_magnitude=args.max_magnitude,
            step_angle=args.step_angle,
            iteration_step=args.iteration_step,
            num_cycles=args.num_cycles,
            direction=args.direction,
            voltage_idx=args.V,
            phase=args.phase,
            dc_offset=args.dc_offset,
            start_iter=args.start_iter,
        )
    elif args.type == "pulse":
        required = {
            "--amplitude": args.amplitude,
            "--width": args.width,
            "--period": args.period,
            "--slope": args.slope,
            "--num-cycles": args.num_cycles,
        }
        missing = [name for name, value in required.items() if value is None]
        if missing:
            raise ValueError(f"--type pulse requires {', '.join(missing)}.")
        write_eregime_smooth_pulse(
            out_path,
            amplitude=args.amplitude,
            width=args.width,
            period=args.period,
            slope=args.slope,
            iteration_step=args.iteration_step,
            num_of_cycles=args.num_cycles,
            step_size=args.step_size,
            direction=args.direction,
            voltage_idx=args.V,
            baseline=args.baseline,
            start_iter=args.start_iter,
        )
    else:
        if args.expr is None or args.t_end is None or args.dt is None:
            raise ValueError("--type func requires --expr, --t-end, and --dt.")
        write_eregime_from_function(
            out_path,
            func=_safe_build_func(args.expr),
            t_end=args.t_end,
            dt=args.dt,
            iteration_step=args.iteration_step,
            direction=args.direction,
            voltage_idx=args.V,
            start_iter=args.start_iter,
        )

    print(f"Wrote eregime file to {out_path.resolve()}")
    return 0
